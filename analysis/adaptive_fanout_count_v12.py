from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np


def stable_int(*parts: object) -> int:
    s = "|".join(map(str, parts))
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest()[:16], 16)


def norm_text(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def sigmoid(x: float) -> float:
    x = float(max(-40.0, min(40.0, x)))
    return 1.0 / (1.0 + math.exp(-x))


def logit(p: float) -> float:
    p = float(min(1.0 - 1e-8, max(1e-8, p)))
    return math.log(p / (1.0 - p))


def entropy01(p: float) -> float:
    p = float(min(1.0 - 1e-12, max(1e-12, p)))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    na = max(float(np.linalg.norm(aa)), 1e-8)
    nb = max(float(np.linalg.norm(bb)), 1e-8)
    return float(np.clip(np.dot(aa, bb) / (na * nb), -1.0, 1.0))


def tree_sha256(path: str | Path) -> str:
    root = Path(path)
    if not root.exists():
        return "MISSING"
    h = hashlib.sha256()
    for p in sorted(x for x in root.rglob("*") if x.is_file()):
        rel = p.relative_to(root).as_posix().encode("utf-8")
        h.update(len(rel).to_bytes(8, "big"))
        h.update(rel)
        data = p.read_bytes()
        h.update(len(data).to_bytes(8, "big"))
        h.update(data)
    return h.hexdigest()


def json_hash(obj: object) -> str:
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class JsonlTargetCache:
    """Public black-box response cache only: (system_id, query) -> final output."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, str] = {}
        if self.path.exists():
            for line in self.path.open(encoding="utf-8"):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    self.data[str(r["key"])] = str(r["output"])
                except Exception:
                    continue

    @staticmethod
    def _key(system_id: str, query: str) -> str:
        return hashlib.sha256(f"{system_id}\0{query}".encode("utf-8")).hexdigest()

    def run(self, system_id: str, query: str, fn: Callable[[str], str]) -> tuple[str, bool]:
        k = self._key(system_id, query)
        if k in self.data:
            return self.data[k], True
        out = str(fn(query))
        rec = {"key": k, "system_id": system_id, "query": query, "output": out}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
        self.data[k] = out
        return out, False


class SharedPresenceDetector:
    """Role-conditioned detector trained only from V11 TRAIN counterfactual outputs.

    Runtime features contain only public candidate description, generated query and
    final external output. Branch identity / true presence is used only as an offline
    training label and never as a runtime feature.
    """

    def __init__(self, embedder, role_descriptions: dict[str, str], *, c: float = 0.05):
        self.embedder = embedder
        self.role_descriptions = role_descriptions
        self.c = float(c)
        self.model = None
        self.threshold_score = 0.0
        self.oof = {}

    def _style(self, text: str) -> np.ndarray:
        t = str(text or "")
        words = re.findall(r"\b[\w'-]+\b", t, flags=re.UNICODE)
        chars = max(len(t), 1)
        return np.asarray(
            [
                math.log1p(len(t)),
                math.log1p(len(words)),
                math.log1p(t.count("\n")),
                math.log1p(t.count("- ") + t.count("* ")),
                float("```" in t),
                sum(ch.isdigit() for ch in t) / chars,
                t.count(":") / chars,
                t.count("|") / chars,
                (t.count("{") + t.count("}")) / chars,
            ],
            dtype=np.float32,
        )

    def features(self, role: str, query: str, output: str) -> np.ndarray:
        desc = self.role_descriptions.get(role, role)
        r = self.embedder.encode(desc).astype(np.float32)
        q = self.embedder.encode(query).astype(np.float32)
        o = self.embedder.encode(output).astype(np.float32)
        # Embeddings are normalized. Interactions make one shared classifier
        # role-conditioned without handing it a hard-coded role id.
        scalars = np.asarray(
            [
                cosine(o, r),
                cosine(o, q),
                cosine(q, r),
                float(np.linalg.norm(o - r)),
                float(np.linalg.norm(o - q)),
                cosine(o - q, r),
            ],
            dtype=np.float32,
        )
        return np.concatenate(
            [
                scalars,
                self._style(output),
                (o * r).astype(np.float32),
                (o * q).astype(np.float32),
                np.abs(o - r).astype(np.float32),
                np.abs(o - q).astype(np.float32),
            ]
        )

    @staticmethod
    def _fit_pipe(X: np.ndarray, y: np.ndarray, w: np.ndarray, c: float):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=float(c),
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ),
        )
        m.fit(X, y, logisticregression__sample_weight=w)
        return m

    @staticmethod
    def _best_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
        from sklearn.metrics import balanced_accuracy_score

        candidates = np.unique(np.quantile(scores, np.linspace(0.05, 0.95, 181)))
        best_t, best_ba = 0.0, -1.0
        for t in candidates:
            ba = float(balanced_accuracy_score(y, (scores >= t).astype(int)))
            if ba > best_ba:
                best_t, best_ba = float(t), ba
        return best_t, best_ba

    def fit(self, examples: list[dict], *, folds: int = 5) -> dict:
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold

        if not examples:
            raise RuntimeError("No detector training examples were extracted from the V11 log")
        X = np.stack([self.features(x["role"], x["query"], x["output"]) for x in examples])
        y = np.asarray([int(x["y"]) for x in examples], dtype=int)
        w = np.asarray([float(x.get("weight", 1.0)) for x in examples], dtype=float)
        groups = np.asarray([str(x["group"]) for x in examples], dtype=object)
        if len(np.unique(y)) < 2:
            raise RuntimeError("Detector training examples contain only one class")

        unique_groups = np.unique(groups)
        nfold = min(max(2, int(folds)), len(unique_groups))
        oof = np.full(len(y), np.nan, dtype=float)
        gkf = GroupKFold(n_splits=nfold)
        for tr, te in gkf.split(X, y, groups):
            m = self._fit_pipe(X[tr], y[tr], w[tr], self.c)
            oof[te] = m.decision_function(X[te])
        mask = np.isfinite(oof)
        if not mask.all():
            raise RuntimeError("OOF detector scores are incomplete")
        auc = float(roc_auc_score(y, oof))
        thr, ba = self._best_threshold(y, oof)

        # Per-role OOF AUC is diagnostic only. The deployed detector remains shared.
        per_role = {}
        roles = np.asarray([x["role"] for x in examples], dtype=object)
        for role in sorted(set(roles.tolist())):
            rm = roles == role
            if rm.sum() >= 6 and len(np.unique(y[rm])) == 2:
                per_role[role] = float(roc_auc_score(y[rm], oof[rm]))
            else:
                per_role[role] = None

        self.model = self._fit_pipe(X, y, w, self.c)
        self.threshold_score = float(thr)
        self.oof = {
            "examples": int(len(y)),
            "groups": int(len(unique_groups)),
            "positive_rate": float(y.mean()),
            "auc": auc,
            "balanced_accuracy_at_oof_threshold": ba,
            "threshold_score": float(thr),
            "per_role_auc": per_role,
        }
        return dict(self.oof)

    def score(self, role: str, query: str, output: str) -> tuple[float, float]:
        if self.model is None:
            raise RuntimeError("Detector is not fitted")
        x = self.features(role, query, output).reshape(1, -1)
        raw = float(self.model.decision_function(x)[0])
        centered = raw - self.threshold_score
        return centered, sigmoid(centered)


def load_detector_examples(path: str | Path, allowed_roles: set[str]) -> list[dict]:
    """Extract TRAIN-only labels from V11's counterfactual log.

    The test-time detector never receives these privileged fields. This function is
    offline training/calibration only.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    out: list[dict] = []
    seen: set[tuple[str, str, int, str]] = set()
    for line_no, line in enumerate(p.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        role = str(rec.get("target_role", ""))
        query = str(rec.get("query", "")).strip()
        if rec.get("phase") != "train" or rec.get("status") != "ok" or role not in allowed_roles or not query:
            continue

        rb = rec.get("reward_breakdown_training_only") or {}
        strict = bool(rb.get("strict_specific_pass", False))
        confirm = bool(rec.get("confirmation_pass_training_only", False))
        # Keep weaker probes for calibration breadth, but make robust/confirmed probes
        # more influential. No truth from the future test is used here.
        weight = 1.0 + (0.75 if strict else 0.0) + (0.75 if confirm else 0.0)
        group = f"{rec.get('group_id','g')}|{rec.get('candidate_index',line_no)}|{hashlib.sha1(query.encode()).hexdigest()[:10]}"

        branches = list(rec.get("branches_training_only") or [])
        branches += list(rec.get("confirmation_branches_training_only") or [])
        for b in branches:
            for field, y in (("plus_output", 1), ("sham_output", 1), ("minus_output", 0)):
                text = str(b.get(field, "") or "").strip()
                if not text:
                    continue
                key = (role, query, y, hashlib.sha1(text.encode("utf-8")).hexdigest())
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    {
                        "role": role,
                        "query": query,
                        "output": text,
                        "y": y,
                        "weight": weight,
                        "group": group,
                    }
                )
    return out


def composition_key(roles: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(map(str, roles)))


def make_fresh_fanout_specs(candidate_roles: Sequence[str], *, systems_per_cell: int, seed: int):
    from targets.single_router_fanout_mas import SingleRouterSystemSpec
    from targets.single_router_zoo_v11 import get_v11_split

    rng = random.Random(seed)
    existing = {
        composition_key(s.worker_roles)
        for s in list(get_v11_split("train")) + list(get_v11_split("dev_unseen_roles"))
    }
    used = set(existing)
    counts = Counter({r: 0 for r in candidate_roles})
    specs = []
    styles = ("s1", "s2", "s3")
    idx = 0
    for mode in ("fanout2", "fanout3"):
        for k in (3, 4, 5, 6):
            for rep in range(int(systems_per_cell)):
                chosen = None
                for _ in range(5000):
                    ordered = sorted(candidate_roles, key=lambda r: (counts[r], rng.random()))
                    # Mostly choose under-represented roles, with light randomization.
                    pool = ordered[: max(k + 4, len(candidate_roles) // 2)]
                    trial = tuple(rng.sample(pool, k))
                    key = composition_key(trial)
                    if key not in used:
                        chosen = trial
                        break
                if chosen is None:
                    raise RuntimeError("Could not construct a fresh composition-disjoint FANOUT test system")
                used.add(composition_key(chosen))
                counts.update(chosen)
                specs.append(
                    SingleRouterSystemSpec(
                        system_id=f"v12_count_{mode}_k{k}_r{rep:02d}",
                        split="v12_fanout_count_development",
                        execution_mode=mode,
                        style=styles[idx % len(styles)],
                        worker_roles=tuple(chosen),
                    )
                )
                idx += 1
    return specs


def sample_probe_fast(policy, observation: dict, *, seed: int, temperature: float, top_p: float, max_new_tokens: int = 96):
    """Frozen-policy generation without log-prob work needed only during RL training."""
    import torch
    from v8.query import extract_one_question

    torch.manual_seed(int(seed) % (2**31 - 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) % (2**31 - 1))
    prompt = policy.format_discovery_prompt(observation)
    encoded = policy.tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max(96, policy.max_seq_length - max_new_tokens),
    ).to(policy.device)
    kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=policy.tokenizer.pad_token_id,
        eos_token_id=policy.tokenizer.eos_token_id,
        do_sample=True,
        temperature=max(float(temperature), 0.05),
        top_p=float(top_p),
    )
    policy.model.eval()
    with torch.no_grad():
        seq = policy.model.generate(**encoded, **kwargs)
    new_ids = seq[0, encoded["input_ids"].shape[1]:]
    raw = policy.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    norm = extract_one_question(raw)
    return norm.query, bool(norm.valid), str(norm.reason), raw


def recent_agrees(evidence: Sequence[float], positive: bool, n: int = 2) -> bool:
    if len(evidence) < n:
        return False
    tail = list(evidence)[-n:]
    return all(x > 0 for x in tail) if positive else all(x < 0 for x in tail)


def run_blackbox_session(
    *,
    system_id: str,
    query_fn: Callable[[str], tuple[str, bool]],
    candidate_roles: Sequence[str],
    role_descriptions: dict[str, str],
    anchor_probe_fn: Callable[[str], str],
    detector: SharedPresenceDetector,
    policy,
    gate,
    embedder,
    seed: int,
    min_probes_per_role: int,
    max_probes_per_role: int,
    max_total_queries: int,
    confidence: float,
    evidence_discount: float,
    evidence_clip: float,
    temperature: float,
    top_p: float,
    generation_retries: int,
) -> dict:
    """Adaptive inference. No true roles, K, mode, routing, traces or branch ids enter here."""

    states = {
        role: {
            "n": 0,
            "log_odds": 0.0,
            "belief": 0.5,
            "queries": [],
            "raw_scores": [],
            "probe_probabilities": [],
            "evidence": [],
            "resolved": False,
            "decision": None,
            "generation_failures": 0,
            "exhausted": False,
        }
        for role in candidate_roles
    }
    history: list[dict] = []
    total_queries = 0

    def update_resolution(role: str) -> None:
        s = states[role]
        if s["n"] < min_probes_per_role:
            return
        b = float(s["belief"])
        if b >= confidence and recent_agrees(s["evidence"], True):
            s["resolved"] = True
            s["decision"] = True
        elif b <= 1.0 - confidence and recent_agrees(s["evidence"], False):
            s["resolved"] = True
            s["decision"] = False

    while total_queries < max_total_queries:
        # First guarantee minimum evidence for every candidate; afterwards focus on
        # highest-entropy unresolved candidates. This is the "long interrogation"
        # part: extra queries are spent only where confidence is still weak.
        need_min = [
            r for r in candidate_roles
            if not states[r]["exhausted"] and states[r]["n"] < min_probes_per_role
        ]
        if need_min:
            role = min(need_min, key=lambda r: (states[r]["n"], candidate_roles.index(r)))
        else:
            unresolved = [
                r for r in candidate_roles
                if not states[r]["resolved"] and not states[r]["exhausted"]
                and states[r]["n"] < max_probes_per_role
            ]
            if not unresolved:
                break
            role = max(unresolved, key=lambda r: (entropy01(states[r]["belief"]), -states[r]["n"]))

        s = states[role]
        prev = list(s["queries"])
        query = ""
        valid = False
        raw = ""
        reason = ""
        for retry in range(max(1, generation_retries)):
            obs = {
                "candidate_role": role,
                "candidate_description": role_descriptions.get(role, role),
                "anchor_probe": anchor_probe_fn(role),
                "avoid_probes": prev[-6:],
            }
            qseed = stable_int(seed, system_id, role, s["n"] + 1, retry)
            query, valid, reason, raw = sample_probe_fast(
                policy,
                obs,
                seed=qseed,
                temperature=temperature,
                top_p=top_p,
            )
            if not valid or not query.strip():
                continue
            if norm_text(query) in {norm_text(x) for x in prev}:
                valid = False
                reason = "duplicate_previous_probe"
                prev.append(query)
                continue
            if prev:
                qv = embedder.encode(query)
                maxsim = max(cosine(qv, embedder.encode(x)) for x in prev)
                if maxsim >= 0.965:
                    valid = False
                    reason = f"semantic_duplicate:{maxsim:.3f}"
                    prev.append(query)
                    continue
            gres = gate.check(query)
            if not gres.allowed:
                valid = False
                reason = f"blocked:{getattr(gres,'stage','gate')}:{getattr(gres,'category','')}"
                prev.append(query)
                continue
            valid = True
            break

        if not valid:
            s["generation_failures"] += 1
            # Prevent a pathological role from blocking the entire benchmark.
            if s["generation_failures"] >= 3:
                s["exhausted"] = True
            history.append({"role": role, "event": "generation_failure", "reason": reason, "raw": raw})
            continue

        output, cache_hit = query_fn(query)
        centered_score, p = detector.score(role, query, output)
        ev = float(np.clip(centered_score, -evidence_clip, evidence_clip)) * float(evidence_discount)
        s["n"] += 1
        s["queries"].append(query)
        s["raw_scores"].append(centered_score)
        s["probe_probabilities"].append(p)
        s["evidence"].append(ev)
        s["log_odds"] = float(s["log_odds"] + ev)
        s["belief"] = sigmoid(s["log_odds"])
        total_queries += 1
        update_resolution(role)

        history.append(
            {
                "step": total_queries,
                "role": role,
                "query": query,
                "output": output,
                "target_cache_hit": cache_hit,
                "detector_centered_score": centered_score,
                "single_probe_probability": p,
                "discounted_evidence": ev,
                "belief_after": s["belief"],
                "resolved_after": s["resolved"],
                "decision_after": s["decision"],
            }
        )
        print(
            f"[{system_id}] q={total_queries:03d} role={role:<26} "
            f"probe_p={p:.3f} belief={s['belief']:.3f} n={s['n']} resolved={s['resolved']}",
            flush=True,
        )

    # Budget-end decision is intentionally simple and uses no knowledge of K.
    for role, s in states.items():
        if s["decision"] is None:
            s["decision"] = bool(s["belief"] >= 0.5)
    present = [r for r in candidate_roles if states[r]["decision"] is True]
    unresolved = [r for r in candidate_roles if not states[r]["resolved"]]
    return {
        "system_id": system_id,
        "total_queries": total_queries,
        "predicted_roles": present,
        "k_hat_workers": len(present),
        "unresolved_roles": unresolved,
        "states": states,
        "history": history,
    }


def set_metrics(true_roles: set[str], pred_roles: set[str]) -> dict:
    tp = len(true_roles & pred_roles)
    fp = len(pred_roles - true_roles)
    fn = len(true_roles - pred_roles)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-12)
    jac = tp / max(len(true_roles | pred_roles), 1)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "jaccard": jac}


def aggregate_results(rows: list[dict], candidate_roles: Sequence[str]) -> dict:
    n = len(rows)
    if not n:
        return {}
    k_err = [abs(int(r["k_hat_workers"]) - int(r["k_true_workers_eval_only"])) for r in rows]
    exact_sets = [set(r["predicted_roles"]) == set(r["true_roles_eval_only"]) for r in rows]
    tp = sum(r["role_metrics_eval_only"]["tp"] for r in rows)
    fp = sum(r["role_metrics_eval_only"]["fp"] for r in rows)
    fn = sum(r["role_metrics_eval_only"]["fn"] for r in rows)
    micro_p = tp / max(tp + fp, 1)
    micro_r = tp / max(tp + fn, 1)
    micro_f1 = 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12)

    per_mode = {}
    for mode in ("fanout2", "fanout3"):
        rr = [r for r in rows if r["execution_mode_eval_only"] == mode]
        if rr:
            ee = [abs(r["k_hat_workers"] - r["k_true_workers_eval_only"]) for r in rr]
            per_mode[mode] = {
                "n": len(rr),
                "k_exact_accuracy": sum(x == 0 for x in ee) / len(rr),
                "k_mae": float(np.mean(ee)),
                "mean_queries": float(np.mean([x["total_queries"] for x in rr])),
            }

    per_role = {}
    for role in candidate_roles:
        yt = np.asarray([int(role in r["true_roles_eval_only"]) for r in rows])
        yp = np.asarray([int(role in r["predicted_roles"]) for r in rows])
        tp_r = int(((yt == 1) & (yp == 1)).sum())
        fp_r = int(((yt == 0) & (yp == 1)).sum())
        fn_r = int(((yt == 1) & (yp == 0)).sum())
        per_role[role] = {
            "support": int(yt.sum()),
            "tp": tp_r,
            "fp": fp_r,
            "fn": fn_r,
            "precision": tp_r / max(tp_r + fp_r, 1),
            "recall": tp_r / max(tp_r + fn_r, 1),
        }

    return {
        "systems": n,
        "k_exact_accuracy": sum(e == 0 for e in k_err) / n,
        "k_mae": float(np.mean(k_err)),
        "k_within_1_accuracy": sum(e <= 1 for e in k_err) / n,
        "exact_role_set_accuracy": sum(exact_sets) / n,
        "mean_role_set_jaccard": float(np.mean([r["role_metrics_eval_only"]["jaccard"] for r in rows])),
        "micro_role_precision": micro_p,
        "micro_role_recall": micro_r,
        "micro_role_f1": micro_f1,
        "mean_queries": float(np.mean([r["total_queries"] for r in rows])),
        "mean_unresolved_roles": float(np.mean([len(r["unresolved_roles"]) for r in rows])),
        "per_mode": per_mode,
        "per_role": per_role,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V12 adaptive end-to-end FANOUT worker-count inference using frozen V11")
    ap.add_argument("--execute", action="store_true", help="Actually query the fresh FANOUT systems. Without this flag: plan only.")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="artifacts/v11_policy")
    ap.add_argument("--v11-log", default="artifacts/v11_probe_discovery.jsonl")
    ap.add_argument("--output-dir", default="artifacts/v12_fanout_count")
    ap.add_argument("--candidate-universe", choices=["train", "all"], default="train")
    ap.add_argument("--systems-per-cell", type=int, default=1)
    ap.add_argument("--seed", type=int, default=12012)
    ap.add_argument("--min-probes-per-role", type=int, default=3)
    ap.add_argument("--max-probes-per-role", type=int, default=8)
    ap.add_argument("--max-total-queries", type=int, default=120)
    ap.add_argument("--confidence", type=float, default=0.92)
    ap.add_argument("--evidence-discount", type=float, default=0.55)
    ap.add_argument("--evidence-clip", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--generation-retries", type=int, default=6)
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hard")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--detector-c", type=float, default=0.05)
    ap.add_argument("--detector-folds", type=int, default=5)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    if not (0.5 < args.confidence < 1.0):
        raise SystemExit("--confidence must be between 0.5 and 1")
    if args.min_probes_per_role < 1 or args.max_probes_per_role < args.min_probes_per_role:
        raise SystemExit("invalid min/max probes per role")

    # Importing the zoo registers all V11 worker prompts into the shared runtime.
    from targets.single_router_zoo_v11 import get_v11_split  # noqa: F401
    from v11.role_universe import ALL_ROLES, TRAIN_ROLES, ROLE_DESCRIPTIONS, anchor_probe

    candidates = tuple(TRAIN_ROLES if args.candidate_universe == "train" else ALL_ROLES)
    specs = make_fresh_fanout_specs(candidates, systems_per_cell=args.systems_per_cell, seed=args.seed)
    examples = load_detector_examples(args.v11_log, set(TRAIN_ROLES))

    manifest = [
        {
            "system_id": s.system_id,
            "execution_mode": s.execution_mode,
            "style": s.style,
            "k_true_workers_eval_only": len(s.worker_roles),
            "true_roles_eval_only": list(s.worker_roles),
        }
        for s in specs
    ]
    config = {
        "candidate_universe": args.candidate_universe,
        "candidate_roles": list(candidates),
        "systems_per_cell": args.systems_per_cell,
        "seed": args.seed,
        "min_probes_per_role": args.min_probes_per_role,
        "max_probes_per_role": args.max_probes_per_role,
        "max_total_queries": args.max_total_queries,
        "confidence": args.confidence,
        "evidence_discount": args.evidence_discount,
        "evidence_clip": args.evidence_clip,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "gate_mode": args.gate_mode,
        "detector_c": args.detector_c,
        "detector_folds": args.detector_folds,
        "v11_policy_sha256": tree_sha256(args.adapter),
        "manifest_sha256": json_hash(manifest),
    }
    run_hash = json_hash(config)

    print("=== V12 ADAPTIVE FANOUT COUNT / ROLE-SET TEST ===")
    print("frozen V11 adapter:", args.adapter)
    print("V11 policy sha256:", config["v11_policy_sha256"])
    print("detector source: V11 TRAIN counterfactual outputs only")
    print("detector training examples:", len(examples))
    print("candidate roles:", len(candidates), candidates)
    print("fresh systems:", len(specs), "= fanout2/fanout3 x K{3,4,5,6} x", args.systems_per_cell)
    print("manifest sha256:", config["manifest_sha256"])
    print("adaptive stopping: min", args.min_probes_per_role, "max", args.max_probes_per_role,
          "per role; confidence", args.confidence, "total query cap/system", args.max_total_queries)
    print("IMPORTANT: V11 weights are FROZEN. Test responses update belief/confidence only, never policy weights.")
    print("Policy/controller observation excludes true roles, K, mode, routing, traces and counterfactual branch identity.")
    print("Primary K is number of hidden specialist workers. In this synthetic FANOUT runtime, total hidden nodes = K + router + aggregator = K+2.")
    for s in specs:
        print(f"  EVAL-ONLY manifest {s.system_id}: mode={s.execution_mode} K={len(s.worker_roles)} roles={list(s.worker_roles)}")

    if not args.execute:
        print("PLAN ONLY. Add --execute to run.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not Path(args.adapter).exists():
        raise SystemExit(f"Frozen V11 adapter missing: {args.adapter}")

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to load/generate with the frozen V11 policy")

    from v8.embedding import FrozenOutputEmbedderV8
    from v10.gate import BenignGateV10
    from v10.policy import SeededProbePolicyV10
    from targets.single_router_fanout_mas import SingleRouterFanoutMAS

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    config_path = outdir / "run_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text(encoding="utf-8"))
        if old.get("run_hash") != run_hash:
            raise SystemExit(
                f"Existing {outdir} was created with different settings. Use a new --output-dir or archive the old one."
            )
    else:
        config_path.write_text(json.dumps({**config, "run_hash": run_hash}, indent=2), encoding="utf-8")
    (outdir / "test_manifest_eval_only.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\n=== FIT SHARED BLACK-BOX PRESENCE DETECTOR (OFFLINE) ===", flush=True)
    embedder = FrozenOutputEmbedderV8(device="cpu")
    detector = SharedPresenceDetector(embedder, ROLE_DESCRIPTIONS, c=args.detector_c)
    oof = detector.fit(examples, folds=args.detector_folds)
    (outdir / "detector_oof.json").write_text(json.dumps(oof, indent=2), encoding="utf-8")
    print(
        f"detector grouped-OOF AUC={oof['auc']:.3f} BA={oof['balanced_accuracy_at_oof_threshold']:.3f} "
        f"examples={oof['examples']} groups={oof['groups']} threshold={oof['threshold_score']:+.3f}",
        flush=True,
    )

    gate = BenignGateV10(mode=args.gate_mode, guard_model=args.guard_model)
    policy = SeededProbePolicyV10(
        args.model_name,
        adapter_path=args.adapter,
        trainable=False,
        use_4bit=not args.no_4bit,
        max_seq_length=1536,
    )
    cache = JsonlTargetCache(outdir / "target_public_cache.jsonl")

    sessions_path = outdir / "sessions.jsonl"
    completed: dict[str, dict] = {}
    if sessions_path.exists():
        for line in sessions_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                if r.get("run_hash") == run_hash and r.get("complete"):
                    completed[str(r["system_id"])] = r
            except Exception:
                pass

    results = []
    for si, spec in enumerate(specs, 1):
        if spec.system_id in completed:
            print(f"\n=== SYSTEM {si}/{len(specs)} {spec.system_id}: resume-skip complete ===", flush=True)
            results.append(completed[spec.system_id])
            continue

        print(f"\n=== SYSTEM {si}/{len(specs)} {spec.system_id} ===", flush=True)
        # The controller below receives only this public query function, not spec/truth.
        mas = SingleRouterFanoutMAS(spec)

        def public_query(q: str, _sid=spec.system_id, _mas=mas):
            return cache.run(_sid, q, _mas.query)

        session = run_blackbox_session(
            system_id=spec.system_id,
            query_fn=public_query,
            candidate_roles=candidates,
            role_descriptions=ROLE_DESCRIPTIONS,
            anchor_probe_fn=anchor_probe,
            detector=detector,
            policy=policy,
            gate=gate,
            embedder=embedder,
            seed=args.seed,
            min_probes_per_role=args.min_probes_per_role,
            max_probes_per_role=args.max_probes_per_role,
            max_total_queries=args.max_total_queries,
            confidence=args.confidence,
            evidence_discount=args.evidence_discount,
            evidence_clip=args.evidence_clip,
            temperature=args.temperature,
            top_p=args.top_p,
            generation_retries=args.generation_retries,
        )

        # Evaluation happens strictly after the black-box session has stopped.
        # These fields are never fed back to the controller/policy/detector.
        true_roles = set(spec.worker_roles)
        pred_roles = set(session["predicted_roles"])
        session.update(
            {
                "complete": True,
                "run_hash": run_hash,
                "execution_mode_eval_only": spec.execution_mode,
                "true_roles_eval_only": sorted(true_roles),
                "k_true_workers_eval_only": len(true_roles),
                "k_true_total_nodes_eval_only": len(true_roles) + 2,
                "k_hat_total_nodes": session["k_hat_workers"] + 2,
                "role_metrics_eval_only": set_metrics(true_roles, pred_roles),
            }
        )
        with sessions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(session, ensure_ascii=False) + "\n")
            f.flush()
        results.append(session)
        print(
            f"RESULT {spec.system_id}: K_true={len(true_roles)} K_hat={session['k_hat_workers']} "
            f"true={sorted(true_roles)} pred={sorted(pred_roles)} "
            f"queries={session['total_queries']} unresolved={len(session['unresolved_roles'])}",
            flush=True,
        )

    summary = {
        "protocol": "frozen_v11_policy_plus_shared_train_only_presence_detector_with_adaptive_evidence",
        "development_test": True,
        "not_final_untouched_paper_test": True,
        "run_hash": run_hash,
        "v11_policy_sha256": config["v11_policy_sha256"],
        "manifest_sha256": config["manifest_sha256"],
        "detector_oof": oof,
        "metrics": aggregate_results(results, candidates),
        "systems": [
            {
                "system_id": r["system_id"],
                "mode": r["execution_mode_eval_only"],
                "k_true": r["k_true_workers_eval_only"],
                "k_hat": r["k_hat_workers"],
                "queries": r["total_queries"],
                "unresolved": len(r["unresolved_roles"]),
                "role_f1": r["role_metrics_eval_only"]["f1"],
                "jaccard": r["role_metrics_eval_only"]["jaccard"],
            }
            for r in results
        ],
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    m = summary["metrics"]
    print("\n" + "=" * 80)
    print("V12 FANOUT COUNT FINAL DEVELOPMENT SUMMARY")
    print("=" * 80)
    print(f"K exact accuracy      : {m['k_exact_accuracy']:.3f}")
    print(f"K MAE                 : {m['k_mae']:.3f}")
    print(f"K within-1 accuracy   : {m['k_within_1_accuracy']:.3f}")
    print(f"Exact role-set acc    : {m['exact_role_set_accuracy']:.3f}")
    print(f"Mean role-set Jaccard : {m['mean_role_set_jaccard']:.3f}")
    print(f"Micro role F1         : {m['micro_role_f1']:.3f}")
    print(f"Mean queries/system   : {m['mean_queries']:.1f}")
    print(f"Mean unresolved roles : {m['mean_unresolved_roles']:.2f}")
    print("Per mode:", json.dumps(m["per_mode"], indent=2))
    print("summary:", outdir / "summary.json")
    print("NOTE: development test only. Freeze the full pipeline before creating a fresh untouched final set.")


if __name__ == "__main__":
    main()
