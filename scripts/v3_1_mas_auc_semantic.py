from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES


# =============================================================================
# Reproducibility
# =============================================================================

def seed_all(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_auc(role_probs: dict[str, float], true_roles: tuple[str, ...]) -> float:
    truth = set(true_roles)
    y = [int(r in truth) for r in TRAIN_ROLES]
    p = [float(role_probs[r]) for r in TRAIN_ROLES]
    return float(roc_auc_score(y, p))


def turn_text(h: dict) -> str:
    return (
        "[USER QUERY]\n" + str(h.get("query", "")) +
        "\n[VISIBLE MAS RESPONSE]\n" + str(h.get("answer", ""))
    )


def query_text(q: str) -> str:
    return "[CANDIDATE USER QUERY]\n" + str(q)


# =============================================================================
# Dataset construction from REAL MAS executions
# =============================================================================

@dataclass
class Candidate:
    idx: int
    query: str
    answer: str
    delta_auc: float
    after_auc: float


@dataclass
class Group:
    sid: str
    split: str
    round: int
    history: list[dict]
    role_probs: dict[str, float]
    true_roles: tuple[str, ...]
    before_auc: float
    candidates: list[Candidate]
    old_chosen: int | None


def load_groups(root: Path, manifest_path: Path, belief_path: Path, split: str, rounds: list[int]):
    manifest = json.loads(manifest_path.read_text())
    truth = {
        str(r["session_id"]): r
        for r in manifest["systems"]
        if r.get("split") == split
    }
    belief = FrozenBeliefBundle.load(belief_path)
    groups: list[Group] = []

    for rd in rounds:
        state_path = root / f"states_{split}_round_{rd-1:02d}.jsonl"
        cand_path = root / f"candidates_{split}_round_{rd:02d}.jsonl"
        resp_path = root / f"responses_{split}_round_{rd:02d}.jsonl"
        scored_path = root / f"scored_{split}_round_{rd:02d}.jsonl"
        metrics_path = root / f"metrics_{split}_round_{rd:02d}.jsonl"

        if not all(p.exists() for p in [state_path, cand_path, resp_path, scored_path]):
            print(f"SKIP round={rd} split={split}: missing files", flush=True)
            continue

        states = {
            str(r["session_id"]): PublicBeliefState.from_public_dict(r)
            for r in read_jsonl(state_path)
        }
        candidates = read_jsonl(cand_path)
        responses = read_jsonl(resp_path)
        scored = read_jsonl(scored_path)

        cand_by: dict[str, list[dict]] = defaultdict(list)
        for c in candidates:
            cand_by[str(c["session_id"])].append(c)

        resp_by = {
            (str(r["session_id"]), int(r["candidate_index"])): r
            for r in responses
        }

        score_by = {}
        for r in scored:
            sid = str(r.get("system_id", r.get("session_id", "")))
            idx = int(dict(r.get("action", {})).get("variant", r.get("candidate_index", 0)))
            score_by[(sid, idx)] = r

        old_chosen = {}
        if metrics_path.exists():
            old_chosen = {
                str(r["session_id"]): int(r["chosen_candidate"])
                for r in read_jsonl(metrics_path)
                if "chosen_candidate" in r
            }

        for sid, cs in sorted(cand_by.items()):
            if sid not in states or sid not in truth:
                continue

            state = states[sid]
            true_roles = tuple(map(str, truth[sid]["roles"]))
            before_auc = safe_auc(dict(state.role_probs), true_roles)
            evaluated: list[Candidate] = []

            for c in sorted(cs, key=lambda x: int(x["candidate_index"])):
                idx = int(c["candidate_index"])
                r = resp_by.get((sid, idx))
                sc = score_by.get((sid, idx))

                if not c.get("valid") or r is None or r.get("execution_status") != "ok" or sc is None:
                    continue

                answer = str(r.get("final_output", ""))
                if not answer.strip():
                    continue

                frame = EvidenceFrame(
                    query=str(c["query"]),
                    answer=answer,
                    v135={str(k): float(v) for k, v in dict(sc.get("evidence", {})).items()},
                    v134={str(k): float(v) for k, v in dict(sc.get("metadata", {}).get("evidence_v134", {})).items()},
                )
                _, new_roles = role_only_predict(belief, [*state.history, frame])
                after_auc = safe_auc(new_roles, true_roles)

                evaluated.append(Candidate(
                    idx=idx,
                    query=str(c["query"]),
                    answer=answer,
                    delta_auc=float(after_auc - before_auc),
                    after_auc=float(after_auc),
                ))

            if len(evaluated) >= 2:
                groups.append(Group(
                    sid=sid,
                    split=split,
                    round=rd,
                    history=[{
                        "query": str(h.query),
                        "answer": str(h.answer),
                    } for h in state.history],
                    role_probs={str(k): float(v) for k, v in state.role_probs.items()},
                    true_roles=true_roles,
                    before_auc=before_auc,
                    candidates=evaluated,
                    old_chosen=old_chosen.get(sid),
                ))

    return groups


# =============================================================================
# Frozen semantic encoder: reuse locally cached V14 backbone
# =============================================================================

class SemanticEmbedder:
    def __init__(self, root: Path, device: str, max_length: int = 384):
        from transformers import AutoModel, AutoTokenizer

        policy_dir = root / "checkpoints" / "round_06" / "policy"
        adapter_cfg = json.loads((policy_dir / "adapter_config.json").read_text())
        base = adapter_cfg.get("base_model_name_or_path")
        if not base:
            raise RuntimeError("adapter_config.json has no base_model_name_or_path")

        print(f"SEMANTIC BASE MODEL: {base}", flush=True)
        print("LOCAL_FILES_ONLY=True", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(
            policy_dir,
            local_files_only=True,
            trust_remote_code=True,
        )

        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.model = AutoModel.from_pretrained(
            base,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )
        self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = device
        self.max_length = max_length
        self.dim = int(self.model.config.hidden_size)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int = 8) -> torch.Tensor:
        all_vecs = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            tok = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tok = {k: v.to(self.device) for k, v in tok.items()}
            out = self.model(**tok, return_dict=True)
            h = out.last_hidden_state.float()
            mask = tok["attention_mask"].unsqueeze(-1).float()
            vec = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
            vec = F.normalize(vec, dim=-1)
            all_vecs.append(vec.cpu())
        return torch.cat(all_vecs, dim=0)


def collect_texts(groups: list[Group]) -> list[str]:
    texts = set()
    for g in groups:
        for h in g.history:
            texts.add(turn_text(h))
        for c in g.candidates:
            texts.add(query_text(c.query))
    return sorted(texts)


def load_or_build_embeddings(root: Path, outdir: Path, groups: list[Group], device: str):
    cache_path = outdir / "semantic_embeddings.pt"
    wanted = collect_texts(groups)

    if cache_path.exists():
        obj = torch.load(cache_path, map_location="cpu")
        cache = obj.get("embeddings", {})
        missing = [t for t in wanted if t not in cache]
        if not missing:
            print(f"EMBEDDING CACHE HIT: {len(cache)} texts", flush=True)
            return cache, int(obj["dim"])
        print(f"CACHE PARTIAL: have={len(cache)} missing={len(missing)}", flush=True)
    else:
        cache = {}
        missing = wanted

    embedder = SemanticEmbedder(root, device)
    print(f"ENCODING {len(missing)} UNIQUE TEXTS", flush=True)
    vecs = embedder.encode(missing, batch_size=8)
    for text, vec in zip(missing, vecs):
        cache[text] = vec

    dim = embedder.dim
    torch.save({"dim": dim, "embeddings": cache}, cache_path)
    print(f"EMBEDDINGS SAVED: {cache_path}", flush=True)
    del embedder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return cache, dim


# =============================================================================
# Selector dataset
# =============================================================================

@dataclass
class SelectorItem:
    group: Group
    turns: torch.Tensor
    belief: torch.Tensor
    queries: torch.Tensor
    utility: torch.Tensor


def build_selector_items(groups: list[Group], cache: dict[str, torch.Tensor], emb_dim: int):
    items = []
    role_order = list(TRAIN_ROLES)

    for g in groups:
        if g.history:
            turns = torch.stack([cache[turn_text(h)].float() for h in g.history])
        else:
            turns = torch.zeros(0, emb_dim)

        belief = torch.tensor([float(g.role_probs[r]) for r in role_order], dtype=torch.float32)
        queries = torch.stack([cache[query_text(c.query)].float() for c in g.candidates])
        utility = torch.tensor([c.delta_auc for c in g.candidates], dtype=torch.float32)
        items.append(SelectorItem(g, turns, belief, queries, utility))

    return items


# =============================================================================
# Models / ablations
# =============================================================================

class AUCSelector(nn.Module):
    def __init__(self, emb_dim: int, mode: str):
        super().__init__()
        assert mode in {"query_only", "belief_query", "history_belief_query"}
        self.mode = mode

        self.turn_proj = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.GELU(), nn.LayerNorm(256)
        )
        self.gru = nn.GRU(256, 192, batch_first=True)
        self.belief_proj = nn.Sequential(
            nn.Linear(len(TRAIN_ROLES), 64), nn.GELU()
        )
        self.state_proj = nn.Sequential(
            nn.Linear(192 + 64, 160), nn.GELU(), nn.LayerNorm(160)
        )
        self.belief_state_proj = nn.Sequential(
            nn.Linear(64, 160), nn.GELU(), nn.LayerNorm(160)
        )
        self.qproj = nn.Sequential(
            nn.Linear(emb_dim, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 160), nn.GELU(), nn.LayerNorm(160)
        )
        self.head = nn.Sequential(
            nn.Linear(160 * 4, 192), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(192, 64), nn.GELU(), nn.Linear(64, 1)
        )
        self.query_only_head = nn.Sequential(
            nn.Linear(160, 128), nn.GELU(), nn.Dropout(0.10), nn.Linear(128, 1)
        )

    def state(self, turns: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        if self.mode == "query_only":
            raise RuntimeError("query_only has no state")

        b = self.belief_proj(belief.unsqueeze(0)).squeeze(0)
        if self.mode == "belief_query":
            return self.belief_state_proj(b)

        if turns.shape[0] == 0:
            h = torch.zeros(192, device=belief.device)
        else:
            x = self.turn_proj(turns).unsqueeze(0)
            _, hn = self.gru(x)
            h = hn[-1, 0]
        return self.state_proj(torch.cat([h, b], dim=-1))

    def score(self, turns: torch.Tensor, belief: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        q = self.qproj(queries)
        if self.mode == "query_only":
            return self.query_only_head(q).squeeze(-1)

        s = self.state(turns, belief)
        ss = s.unsqueeze(0).expand(q.shape[0], -1)
        feat = torch.cat([ss, q, ss * q, (ss - q).abs()], dim=-1)
        return self.head(feat).squeeze(-1)


def group_loss(scores: torch.Tensor, utility: torch.Tensor) -> torch.Tensor:
    # Relative ranking is the primary target; regression stabilizes scale.
    uz = (utility - utility.mean()) / utility.std().clamp_min(1e-4)
    target = F.softmax(uz / 0.70, dim=0)
    rank = -(target * F.log_softmax(scores, dim=0)).sum()
    sz = (scores - scores.mean()) / scores.std().clamp_min(1e-4)
    reg = F.smooth_l1_loss(sz, uz)
    return rank + 0.20 * reg


def fit_once(items: list[SelectorItem], emb_dim: int, mode: str, seed: int, epochs: int):
    seed_all(seed)
    model = AUCSelector(emb_dim, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=4e-4)

    for epoch in range(epochs):
        order = list(range(len(items)))
        random.shuffle(order)
        model.train()
        total = 0.0
        for i in order:
            x = items[i]
            scores = model.score(x.turns, x.belief, x.queries)
            loss = group_loss(scores, x.utility)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"train mode={mode} epoch={epoch+1:03d} loss={total/max(1,len(items)):.5f}", flush=True)
    return model.eval()


@torch.no_grad()
def ranking_score(model: AUCSelector, items: list[SelectorItem]) -> float:
    # Validation objective: mean fraction of oracle gain above random.
    vals = []
    model.eval()
    for x in items:
        s = model.score(x.turns, x.belief, x.queries)
        pick = int(s.argmax())
        u = x.utility
        rand = float(u.mean())
        oracle = float(u.max())
        learned = float(u[pick])
        denom = oracle - rand
        if abs(denom) > 1e-8:
            vals.append((learned - rand) / denom)
    return float(sum(vals) / max(1, len(vals)))


def train_with_internal_validation(all_items: list[SelectorItem], emb_dim: int, mode: str, seed: int):
    sids = sorted({x.group.sid for x in all_items})
    rng = random.Random(seed + 991)
    rng.shuffle(sids)
    n_val = max(3, round(0.17 * len(sids)))
    val_sids = set(sids[:n_val])
    fit_items = [x for x in all_items if x.group.sid not in val_sids]
    val_items = [x for x in all_items if x.group.sid in val_sids]

    seed_all(seed)
    model = AUCSelector(emb_dim, mode)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=4e-4)
    best_epoch = 1
    best_score = -1e9
    best_state = None
    bad = 0

    for epoch in range(1, 81):
        order = list(range(len(fit_items)))
        random.shuffle(order)
        model.train()
        total = 0.0
        for i in order:
            x = fit_items[i]
            scores = model.score(x.turns, x.belief, x.queries)
            loss = group_loss(scores, x.utility)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())

        sc = ranking_score(model, val_items)
        if sc > best_score + 1e-5:
            best_score = sc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"train mode={mode} seed={seed} epoch={epoch:03d} "
                f"loss={total/max(1,len(fit_items)):.5f} val_gap_closed={sc:.4f} "
                f"best_epoch={best_epoch}",
                flush=True,
            )
        if epoch >= 15 and bad >= 15:
            break

    # Refit on every TRAIN session for the selected epoch count.
    final = fit_once(all_items, emb_dim, mode, seed + 10000, best_epoch)
    print(
        f"SELECTED mode={mode} seed={seed} epoch={best_epoch} "
        f"internal_gap_closed={best_score:.4f}",
        flush=True,
    )
    return final, best_epoch, best_score


# =============================================================================
# Evaluation: one-step counterfactual only
# =============================================================================

@torch.no_grad()
def evaluate(model: AUCSelector, items: list[SelectorItem]):
    rows = []
    for x in items:
        scores = model.score(x.turns, x.belief, x.queries)
        pick = int(scores.argmax())
        u = x.utility.tolist()
        oracle_idx = max(range(len(u)), key=lambda i: u[i])
        order = sorted(range(len(u)), key=lambda i: u[i], reverse=True)
        top20 = set(order[:max(1, math.ceil(0.20 * len(order)))])

        old_delta = None
        if x.group.old_chosen is not None:
            for j, c in enumerate(x.group.candidates):
                if c.idx == x.group.old_chosen:
                    old_delta = u[j]
                    break

        rows.append({
            "before": x.group.before_auc,
            "learned_delta": u[pick],
            "random_delta": sum(u) / len(u),
            "oracle_delta": u[oracle_idx],
            "old_delta": old_delta,
            "exact": float(pick == oracle_idx),
            "top20": float(pick in top20),
            "positive": float(u[pick] > 0),
            "regret": u[oracle_idx] - u[pick],
        })

    def mean(k):
        xs = [r[k] for r in rows if r[k] is not None]
        return float(sum(xs) / max(1, len(xs)))

    random_delta = mean("random_delta")
    learned_delta = mean("learned_delta")
    oracle_delta = mean("oracle_delta")
    before = mean("before")
    old_delta = mean("old_delta")

    return {
        "groups": len(rows),
        "mean_auc_before": before,
        "mean_delta_auc_random": random_delta,
        "mean_delta_auc_old_v14": old_delta,
        "mean_delta_auc_learned": learned_delta,
        "mean_delta_auc_oracle": oracle_delta,
        "mean_auc_after_random": before + random_delta,
        "mean_auc_after_old_v14": before + old_delta,
        "mean_auc_after_learned": before + learned_delta,
        "mean_auc_after_oracle": before + oracle_delta,
        "oracle_gap_closed": (learned_delta - random_delta) / (oracle_delta - random_delta + 1e-12),
        "oracle_exact_hit_rate": mean("exact"),
        "top20_hit_rate": mean("top20"),
        "positive_delta_rate": mean("positive"),
        "mean_oracle_regret": mean("regret"),
    }


def aggregate_seed_results(results: list[dict]):
    out = {}
    modes = results[0]["models"].keys()
    for mode in modes:
        keys = [k for k in results[0]["models"][mode] if k != "groups"]
        out[mode] = {}
        for k in keys:
            vals = [float(r["models"][mode][k]) for r in results]
            mean = sum(vals) / len(vals)
            std = math.sqrt(sum((v - mean) ** 2 for v in vals) / len(vals))
            out[mode][k] = {"mean": mean, "std": std, "values": vals}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/v14_adaptive_role_v1")
    ap.add_argument("--manifest", default="artifacts/v14_adaptive_role_v1/manifest_privileged.json")
    ap.add_argument("--belief", default="artifacts/v14_adaptive_role_v1/frozen_belief.joblib")
    ap.add_argument("--outdir", default="artifacts/eqs_v3_1_mas_auc")
    ap.add_argument("--rounds", default="1,2,3,4,5,6")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rounds = [int(x) for x in args.rounds.split(",")]

    print("=" * 96)
    print("V3.1-MAS — SEMANTIC HISTORY-CONDITIONED QUERY SELECTION FOR ROLE AUC")
    print("=" * 96)
    print("PRIMARY TARGET: delta ROC-AUC over installed-role inference")
    print("DATA: existing REAL MAS candidate executions")
    print("EVAL: one-step counterfactual on held-out DEV systems")
    print("NO NEW MAS CALLS")

    train_groups = load_groups(root, Path(args.manifest), Path(args.belief), "train", rounds)
    dev_groups = load_groups(root, Path(args.manifest), Path(args.belief), "dev", rounds)
    print(f"GROUPS train={len(train_groups)} dev={len(dev_groups)}")
    print(f"SESSIONS train={len(set(g.sid for g in train_groups))} dev={len(set(g.sid for g in dev_groups))}")
    assert train_groups and dev_groups

    all_groups = train_groups + dev_groups
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"EMBED DEVICE={device}")
    cache, emb_dim = load_or_build_embeddings(root, outdir, all_groups, device)
    print(f"SEMANTIC EMBEDDING DIM={emb_dim}")

    # Free backbone GPU memory before selector training.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_items = build_selector_items(train_groups, cache, emb_dim)
    dev_items = build_selector_items(dev_groups, cache, emb_dim)

    modes = ["query_only", "belief_query", "history_belief_query"]
    seeds = [7, 17, 29]
    all_results = []

    for seed in seeds:
        print("\n" + "=" * 96)
        print(f"SEED {seed}")
        print("=" * 96)
        model_results = {}
        meta = {}

        for mode in modes:
            model, best_epoch, val_score = train_with_internal_validation(
                train_items, emb_dim, mode, seed
            )
            result = evaluate(model, dev_items)
            model_results[mode] = result
            meta[mode] = {"selected_epoch": best_epoch, "internal_val_gap_closed": val_score}
            print(mode.upper(), json.dumps(result, sort_keys=True), flush=True)

            torch.save({
                "state_dict": model.state_dict(),
                "emb_dim": emb_dim,
                "mode": mode,
                "roles": list(TRAIN_ROLES),
                "seed": seed,
            }, outdir / f"selector_{mode}_seed{seed}.pt")

        all_results.append({"seed": seed, "models": model_results, "training": meta})

    aggregate = aggregate_seed_results(all_results)
    final = {
        "experiment": "V3.1-MAS semantic one-step delta-AUC query selection",
        "primary_metric": "role ROC-AUC",
        "train_groups": len(train_items),
        "dev_groups": len(dev_items),
        "train_sessions": len(set(g.sid for g in train_groups)),
        "dev_sessions": len(set(g.sid for g in dev_groups)),
        "semantic_embedding_dim": emb_dim,
        "aggregate": aggregate,
        "seeds": all_results,
        "important_limitation": (
            "Offline evaluation is one-step counterfactual at logged histories. "
            "Multi-step adaptive AUC trajectories require new closed-loop MAS executions."
        ),
    }
    out = outdir / "results.json"
    out.write_text(json.dumps(final, indent=2))

    print("\n" + "=" * 96)
    print("FINAL V3.1-MAS AUC RESULTS")
    print("=" * 96)
    for mode in modes:
        a = aggregate[mode]
        print(f"\n{mode.upper()}")
        for key in [
            "mean_delta_auc_learned",
            "mean_auc_after_learned",
            "oracle_gap_closed",
            "top20_hit_rate",
            "positive_delta_rate",
            "mean_oracle_regret",
        ]:
            x = a[key]
            print(f"  {key}: {x['mean']:.6f} +/- {x['std']:.6f}")

    # Shared baselines are identical across selector modes; print from full mode.
    a = aggregate["history_belief_query"]
    print("\nSHARED BASELINES")
    for key in [
        "mean_auc_before",
        "mean_delta_auc_random",
        "mean_delta_auc_old_v14",
        "mean_delta_auc_oracle",
        "mean_auc_after_random",
        "mean_auc_after_old_v14",
        "mean_auc_after_oracle",
    ]:
        x = a[key]
        print(f"  {key}: {x['mean']:.6f} +/- {x['std']:.6f}")

    print(f"\nSAVED: {out}")


if __name__ == "__main__":
    main()
