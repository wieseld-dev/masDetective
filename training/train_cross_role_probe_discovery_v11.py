from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from v11.role_universe import TRAIN_ROLES, DEV_UNSEEN_ROLES, ALL_ROLES, ROLE_DESCRIPTIONS, anchor_probe
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_zoo_v11 import get_v11_split
from v8.counterfactual import (
    build_counterfactual_pairs,
    get_k,
    get_mode,
    get_roles,
    split_base_specs,
)
from v8.embedding import FrozenOutputEmbedderV8
from v8.execution import execute_cached
from v10.checkpoint import atomic_write_json, commit_round_snapshot, load_committed_state
from v10.gate import BenignGateV10
from v10.policy import SeededProbePolicyV10
from v10.reward import PairwiseRobustCounterfactualGraderV10, group_relative_advantages
from v11.hashcheck import assert_exact_v10_start


@dataclass
class TrainSample:
    role: str
    prompt: str
    completion: str
    old_logp: float
    ref_logp: float
    advantage: float
    reward: float
    elite: bool
    group_id: str


def _norm_text(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def _word_jaccard(a: str, b: str) -> float:
    wa = set(_norm_text(a).split())
    wb = set(_norm_text(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / max(1, len(wa | wb))


def _kl_k3(new_lp, ref_lp, torch):
    delta = ref_lp - new_lp
    return torch.exp(torch.clamp(delta, -6.0, 6.0)) - delta - 1.0


def _stable_jitter(seed: int, *parts: object) -> float:
    txt = "|".join(map(str, (seed,) + parts))
    h = hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
    return int(h, 16) / float(16**12)


def _pair_signature(p: object) -> tuple[str, int, str, tuple[str, ...]]:
    return (
        get_mode(p.plus_spec),
        get_k(p.plus_spec),
        str(getattr(p, "negative_replacement", "")),
        tuple(sorted(get_roles(p.plus_spec))),
    )


def _build_stratified_schedule(
    by_role: dict[str, list[object]],
    roles: Sequence[str],
    *,
    rounds: int,
    bundle_size: int,
    confirmation_size: int,
    seed: int,
) -> tuple[dict[tuple[int, str], list[object]], dict[tuple[int, str], list[object]]]:
    """Deterministic primary+confirmation schedules with broad composition coverage.

    Primary reward bundles prioritize pairs never before used as PRIMARY evidence,
    then mode/K/replacement/composition diversity. Confirmation bundles are disjoint
    from the current primary bundle and prioritize globally under-used pairs.
    """
    primary_usage: dict[str, int] = {}
    overall_usage: dict[str, int] = {}
    primary: dict[tuple[int, str], list[object]] = {}
    confirm: dict[tuple[int, str], list[object]] = {}

    def choose(
        pairs: list[object],
        n: int,
        round_i: int,
        role: str,
        blocked: set[str],
        usage_map: dict[str, int],
    ) -> list[object]:
        chosen: list[object] = []
        used_modes: set[str] = set()
        used_k: set[int] = set()
        used_neg: set[str] = set()
        used_comp: set[tuple[str, ...]] = set()
        available_n = len([p for p in pairs if str(getattr(p, "pair_id")) not in blocked])
        while len(chosen) < min(max(0, n), available_n):
            remaining = [
                p for p in pairs
                if p not in chosen and str(getattr(p, "pair_id")) not in blocked
            ]
            if not remaining:
                break

            def score(p: object):
                pid = str(getattr(p, "pair_id"))
                mode, kval, neg, comp = _pair_signature(p)
                return (
                    1 if usage_map.get(pid, 0) == 0 else 0,
                    1 if mode not in used_modes else 0,
                    1 if kval not in used_k else 0,
                    1 if neg not in used_neg else 0,
                    1 if comp not in used_comp else 0,
                    -usage_map.get(pid, 0),
                    -overall_usage.get(pid, 0),
                    _stable_jitter(seed, round_i, role, pid),
                )

            best = max(remaining, key=score)
            chosen.append(best)
            mode, kval, neg, comp = _pair_signature(best)
            used_modes.add(mode)
            used_k.add(kval)
            used_neg.add(neg)
            used_comp.add(comp)
        return chosen

    for round_i in range(1, rounds + 1):
        for role in roles:
            pairs = list(by_role.get(role, []))
            if not pairs:
                primary[(round_i, role)] = []
                confirm[(round_i, role)] = []
                continue

            psel = choose(pairs, int(bundle_size), round_i, role, set(), primary_usage)
            blocked = {str(getattr(p, "pair_id")) for p in psel}
            csel = choose(
                pairs, int(confirmation_size), round_i + 10000, role, blocked, overall_usage
            )

            for p in psel:
                pid = str(getattr(p, "pair_id"))
                primary_usage[pid] = primary_usage.get(pid, 0) + 1
                overall_usage[pid] = overall_usage.get(pid, 0) + 1
            for p in csel:
                pid = str(getattr(p, "pair_id"))
                overall_usage[pid] = overall_usage.get(pid, 0) + 1

            primary[(round_i, role)] = psel
            confirm[(round_i, role)] = csel
    return primary, confirm


def _balanced_minibatches(
    by_role_samples: dict[str, list[TrainSample]],
    roles: Sequence[str],
    *,
    batches: int,
    seed: int,
) -> list[list[TrainSample]]:
    """Create several optimizer batches while preserving role balance in each batch."""
    n_batches = max(1, int(batches))
    rng = random.Random(seed)
    shuffled: dict[str, list[TrainSample]] = {}
    for role in roles:
        xs = list(by_role_samples.get(role, []))
        rng.shuffle(xs)
        shuffled[role] = xs

    out: list[list[TrainSample]] = [[] for _ in range(n_batches)]
    for role in roles:
        xs = shuffled[role]
        if not xs:
            continue
        for i, x in enumerate(xs):
            out[i % n_batches].append(x)
        # If a role has fewer samples than minibatches, cycle one sample so every
        # optimizer step still receives evidence from that role.
        for b in range(n_batches):
            if not any(t.role == role for t in out[b]):
                clone = xs[b % len(xs)]
                out[b].append(clone)

    for b in out:
        rng.shuffle(b)
    return [b for b in out if b]


def _load_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _truncate_log_for_resume(path: str | Path, *, phase: str, start_round: int) -> None:
    p = Path(path)
    if not p.exists():
        return
    rows = _load_jsonl(p)
    keep: list[dict] = []
    for r in rows:
        if r.get("phase") == "train":
            if int(r.get("round", 0)) < int(start_round):
                keep.append(r)
        elif phase == "complete":
            keep.append(r)
        # Holdout is deliberately regenerated deterministically after a resume.
    tmp = p.with_suffix(p.suffix + ".resume-tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in keep:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, p)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    aa = aa / max(float(np.linalg.norm(aa)), 1e-8)
    bb = bb / max(float(np.linalg.norm(bb)), 1e-8)
    return float(np.clip(float(np.dot(aa, bb)), -1.0, 1.0))


def _load_seed_elites(path: str, roles: Sequence[str], max_per_role: int = 3) -> dict[str, list[str]]:
    out = {r: [] for r in roles}
    p = Path(path)
    if not p.exists():
        return out
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        raw = obj.get("elite_probes", {})
        for role in roles:
            rows = list(raw.get(role, []))
            rows.sort(key=lambda x: float(x.get("reward", -1e9)), reverse=True)
            seen: set[str] = set()
            for row in rows:
                q = str(row.get("query", "")).strip()
                k = _norm_text(q)
                if not q or k in seen:
                    continue
                seen.add(k)
                out[role].append(q)
                if len(out[role]) >= max_per_role:
                    break
    except Exception as exc:
        print("WARNING: could not load initial elite summary:", type(exc).__name__, flush=True)
    return out


def _load_logged_elites(path: str, roles: Sequence[str]) -> dict[str, list[str]]:
    out = {r: [] for r in roles}
    p = Path(path)
    if not p.exists():
        return out
    try:
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("phase") == "train" and r.get("elite") and r.get("target_role") in out:
                q = str(r.get("query", "")).strip()
                if q and _norm_text(q) not in {_norm_text(x) for x in out[r["target_role"]]}:
                    out[r["target_role"]].append(q)
    except Exception as exc:
        print("WARNING: could not reload prior V10 elites:", type(exc).__name__, flush=True)
    return out


def _merge_unique(a: Sequence[str], b: Sequence[str], limit: int = 8) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for x in list(a) + list(b):
        q = str(x).strip()
        k = _norm_text(q)
        if q and k not in seen:
            seen.add(k)
            out.append(q)
    return out[-limit:]


def _copy_cache_seed(src: str, dst: str) -> None:
    s, d = Path(src), Path(dst)
    if s.exists() and not d.exists():
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        print(f"seeded cache {d} from {s}", flush=True)


def main() -> None:
    import torch

    ap = argparse.ArgumentParser(description="V11 cross-role benign probe discovery continued from frozen V10.1")
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--profile", default="config/application_profile_v11.json")
    ap.add_argument("--rounds-per-role", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--bundle-size", type=int, default=4)
    ap.add_argument("--confirmation-size", type=int, default=2)
    ap.add_argument("--confirm-top-k", type=int, default=2)
    ap.add_argument("--balanced-minibatches", type=int, default=3)
    ap.add_argument("--grpo-epochs", type=int, default=2)
    ap.add_argument("--elite-nll-weight", type=float, default=0.06)
    ap.add_argument("--lr", type=float, default=1.0e-5)
    ap.add_argument("--clip", type=float, default=0.20)
    ap.add_argument("--kl-beta", type=float, default=0.0)
    ap.add_argument("--temperature", type=float, default=0.95)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--gate-mode", choices=["hard", "hybrid"], default="hybrid")
    ap.add_argument("--guard-model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=1010)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--resume-adapter", default="artifacts/v10_policy")
    ap.add_argument("--initial-elite-summary", default="artifacts/v10_summary.json")
    ap.add_argument("--baseline-summary", default="artifacts/v10_summary.json")
    ap.add_argument("--max-avoid-probes", type=int, default=4)
    ap.add_argument("--min-activation-rate", type=float, default=0.75)
    ap.add_argument("--novelty-free-similarity", type=float, default=0.90)
    ap.add_argument("--novelty-weight", type=float, default=0.25)
    ap.add_argument("--semantic-hard-duplicate", type=float, default=0.97)
    ap.add_argument("--holdout-max-similarity", type=float, default=0.94)
    ap.add_argument("--holdout-generation-retries", type=int, default=6)
    ap.add_argument("--output-dir", default="artifacts/v11_policy")
    ap.add_argument("--log", default="artifacts/v11_probe_discovery.jsonl")
    ap.add_argument("--summary", default="artifacts/v11_summary.json")
    ap.add_argument("--response-public-cache", default="artifacts/v11_response_public.jsonl")
    ap.add_argument("--response-truth-cache", default="artifacts/v11_response_truth.jsonl")
    ap.add_argument("--seed-response-public-cache", default="artifacts/v11_seed_none_public.jsonl")
    ap.add_argument("--seed-response-truth-cache", default="artifacts/v11_seed_none_truth.jsonl")
    ap.add_argument("--holdout-eval", action="store_true")
    ap.add_argument("--holdout-probes-per-role", type=int, default=3)
    ap.add_argument("--syntax-retries", type=int, default=3)
    ap.add_argument("--start-round", type=int, default=1)
    ap.add_argument("--checkpoint-state", default="artifacts/v11_checkpoint.json")
    ap.add_argument("--checkpoint-root", default="artifacts/v11_checkpoints")
    ap.add_argument("--optimizer-state", default="artifacts/v11_optimizer.pt")
    ap.add_argument("--optimizer-mirror", default="artifacts/v11_optimizer.pt")
    ap.add_argument("--resume-optimizer", action="store_true")
    ap.add_argument("--resume-from-checkpoint", action="store_true")
    ap.add_argument("--v10-frozen-summary", default="artifacts/v10_summary.json")
    ap.add_argument("--allow-v10-hash-mismatch", action="store_true", help="DEBUG ONLY; never use for reported runs")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    # Prove that continuation starts from the exact trained V10.1 adapter that was
    # frozen and evaluated previously. This check occurs before any API call or update.
    start_fingerprint = assert_exact_v10_start(
        args.resume_adapter,
        args.v10_frozen_summary,
        strict=not args.allow_v10_hash_mismatch,
    )

    if args.execute and not args.resume_from_checkpoint and Path(args.checkpoint_state).exists():
        raise SystemExit(
            f"Existing V11 checkpoint found at {args.checkpoint_state}. "
            "Use the resume sbatch, or explicitly remove/archive old V11 artifacts before a fresh run."
        )

    resume_state = None
    resume_phase = "train"
    if args.resume_from_checkpoint:
        resume_state = load_committed_state(args.checkpoint_state)
        resume_phase = str(resume_state["phase"])
        if resume_phase == "complete":
            print("V11 is already complete according to", args.checkpoint_state)
            return
        args.resume_adapter = str(resume_state["adapter_dir"])
        args.optimizer_state = str(resume_state["optimizer_state"])
        args.resume_optimizer = True
        args.start_round = int(resume_state.get("next_round", 1))
        _truncate_log_for_resume(args.log, phase=resume_phase, start_round=args.start_round)

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    train_roles = tuple(profile.get("train_candidate_roles", TRAIN_ROLES))
    dev_roles = tuple(profile.get("dev_unseen_candidate_roles", DEV_UNSEEN_ROLES))
    if tuple(train_roles) != tuple(TRAIN_ROLES):
        raise SystemExit("V11 profile training roles differ from the reviewed TRAIN_ROLES split")
    if tuple(dev_roles) != tuple(DEV_UNSEEN_ROLES):
        raise SystemExit("V11 profile development roles differ from the reviewed DEV_UNSEEN_ROLES split")
    if set(train_roles) & set(dev_roles):
        raise SystemExit("V11 role leakage: train/dev role sets overlap")

    all_specs = list(get_v11_split("train"))
    rl_specs, detector_reserved_specs = split_base_specs(all_specs, detector_fraction=1 / 8, seed=1109)
    holdout_specs = list(get_v11_split("dev_unseen_roles"))

    # CRITICAL: unseen development roles are not even in the candidate replacement
    # universe used to create training MINUS/SHAM systems. They therefore never
    # occur as positive, negative, or sham roles during V11 training.
    train_pairs = build_counterfactual_pairs(rl_specs, train_roles, include_sham=True)
    holdout_pairs_all = build_counterfactual_pairs(holdout_specs, ALL_ROLES, include_sham=True)
    holdout_pairs = [p for p in holdout_pairs_all if p.target_role in set(dev_roles)]

    by_role: dict[str, list[object]] = {r: [] for r in train_roles}
    holdout_by_role: dict[str, list[object]] = {r: [] for r in dev_roles}
    for p in train_pairs:
        by_role.setdefault(p.target_role, []).append(p)
    for p in holdout_pairs:
        holdout_by_role.setdefault(p.target_role, []).append(p)
    usable_roles = [r for r in train_roles if by_role.get(r)]
    dev_usable_roles = [r for r in dev_roles if holdout_by_role.get(r)]

    if set(dev_roles) & {r for spec in rl_specs for r in get_roles(spec)}:
        raise SystemExit("V11 leakage detected: unseen development role appears in RL training specs")
    if set(dev_roles) & {str(getattr(p, "negative_replacement", "")) for p in train_pairs}:
        raise SystemExit("V11 leakage detected: unseen development role used as training negative replacement")

    schedule, confirmation_schedule = _build_stratified_schedule(
        by_role,
        usable_roles,
        rounds=args.rounds_per_role,
        bundle_size=args.bundle_size,
        confirmation_size=args.confirmation_size,
        seed=args.seed,
    )

    training_pair_coverage: dict[str, dict] = {}
    for role in usable_roles:
        prim = [p for rd in range(1, args.rounds_per_role + 1) for p in schedule[(rd, role)]]
        conf = [p for rd in range(1, args.rounds_per_role + 1) for p in confirmation_schedule[(rd, role)]]
        training_pair_coverage[role] = {
            "available_pairs": len(by_role.get(role, [])),
            "primary_draws": len(prim),
            "primary_unique_pairs": len({p.pair_id for p in prim}),
            "confirmation_draws": len(conf),
            "confirmation_unique_pairs": len({p.pair_id for p in conf}),
            "total_exposed_unique_pairs": len({p.pair_id for p in prim + conf}),
            "primary_modes": sorted({get_mode(p.plus_spec) for p in prim}),
            "primary_k": sorted({get_k(p.plus_spec) for p in prim}),
            "negative_replacements": sorted({str(getattr(p, "negative_replacement", "")) for p in prim}),
        }

    groups = len(usable_roles) * args.rounds_per_role
    candidate_count = groups * args.group_size
    candidate_interactions = 0
    confirmation_interactions_upper = 0
    for round_i in range(1, args.rounds_per_role + 1):
        for role in usable_roles:
            candidate_interactions += args.group_size * len(schedule[(round_i, role)]) * 3
            confirmation_interactions_upper += (
                args.confirm_top_k * len(confirmation_schedule[(round_i, role)]) * 3
            )

    print("=== V11 CROSS-ROLE CONTINUATION FROM FROZEN V10.1 ===")
    print("starting policy adapter:", args.resume_adapter)
    print("starting V10.1 policy sha256:", start_fingerprint["policy_sha256"])
    print("exact frozen V10.1 match:", start_fingerprint["policy_match"] and start_fingerprint["summary_match"])
    print("TRAIN role types:", len(usable_roles), usable_roles)
    print("UNSEEN DEV role types:", len(dev_usable_roles), dev_usable_roles)
    print("resume phase:", resume_phase, "start round:", args.start_round)
    print("RL base systems:", len(rl_specs), "detector-reserved:", len(detector_reserved_specs), "unseen-role dev base systems:", len(holdout_specs))
    print("train matched pairs:", len(train_pairs), "unseen-role dev matched pairs:", len(holdout_pairs))
    print("rounds/role:", args.rounds_per_role, "group size:", args.group_size, "primary bundle:", args.bundle_size, "confirmation:", args.confirmation_size)
    print("candidate probes:", candidate_count)
    print("upper-bound primary MAS interactions:", candidate_interactions)
    print("upper-bound confirmation MAS interactions:", confirmation_interactions_upper)
    print("optimization:", args.balanced_minibatches, "balanced minibatches/round across all roles")
    print("KL-to-base disabled by default:", args.kl_beta == 0.0)
    print("reward: pairwise PRESENT-ABSENT-SHAM; robust median+Q25; robust PRESENT+SHAM activation")
    print("elite: primary strict pass + disjoint confirmation strict pass + semantic novelty")
    print("novelty: semantic hard duplicate pre-filter + soft penalty against anchor/prior elites")
    print("gate: task-language-aware semantic guard + unchanged hard gate")
    print("evaluation: exhaustive over ALL development-validation matched pairs per role:", bool(args.holdout_eval))
    print("NOTE: V11 dev roles are complete role-type holdouts. They are development validation, not a final paper test.")
    print("fresh optimizer for new objective:", not args.resume_optimizer)

    for round_i in range(1, args.rounds_per_role + 1):
        print(f"ROUND {round_i} bundles:")
        for role in usable_roles:
            b = schedule[(round_i, role)]
            c = confirmation_schedule[(round_i, role)]
            print(
                f"  {role:<10} primary n={len(b)} modes={[get_mode(p.plus_spec) for p in b]} "
                f"K={[get_k(p.plus_spec) for p in b]} neg={[getattr(p, 'negative_replacement', '') for p in b]} "
                f"ids={[p.pair_id for p in b]}"
            )
            print(
                f"  {'':<10} confirm n={len(c)} modes={[get_mode(p.plus_spec) for p in c]} "
                f"K={[get_k(p.plus_spec) for p in c]} neg={[getattr(p, 'negative_replacement', '') for p in c]} "
                f"ids={[p.pair_id for p in c]}"
            )

    if not args.execute:
        print("PLAN ONLY. Add --execute to run.")
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if args.resume_adapter and not Path(args.resume_adapter).exists():
        raise SystemExit(f"resume adapter does not exist: {args.resume_adapter}")

    _copy_cache_seed(args.seed_response_public_cache, args.response_public_cache)
    _copy_cache_seed(args.seed_response_truth_cache, args.response_truth_cache)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    gate = BenignGateV10(mode=args.gate_mode, guard_model=args.guard_model)
    embedder = FrozenOutputEmbedderV8(device="cpu")
    text_embedding_cache: dict[str, np.ndarray] = {}

    def embed_text(text: str) -> np.ndarray:
        key = str(text)
        if key not in text_embedding_cache:
            text_embedding_cache[key] = embedder.encode(key)
        return text_embedding_cache[key]

    def semantic_similarity(query: str, refs: Sequence[str]) -> float:
        clean = [str(x).strip() for x in refs if str(x).strip()]
        if not clean:
            return 0.0
        qv = embed_text(query)
        return max(_cosine_similarity(qv, embed_text(r)) for r in clean)

    grader = PairwiseRobustCounterfactualGraderV10(
        min_activation_rate=args.min_activation_rate,
        novelty_free_similarity=args.novelty_free_similarity,
        novelty_weight=args.novelty_weight,
    )
    cache = ResponseCacheV7(args.response_public_cache, args.response_truth_cache)
    policy = SeededProbePolicyV10(
        args.model_name,
        adapter_path=(args.resume_adapter or None),
        trainable=True,
        use_4bit=not args.no_4bit,
        max_seq_length=1536,
    )
    optimizer = torch.optim.AdamW(policy.trainable_parameters(), lr=args.lr, weight_decay=1e-4)
    if args.resume_optimizer:
        if not Path(args.optimizer_state).exists():
            raise SystemExit(f"resume optimizer state is missing: {args.optimizer_state}")
        optimizer.load_state_dict(torch.load(args.optimizer_state, map_location="cpu"))
        print("resumed committed V11 optimizer state:", args.optimizer_state, flush=True)

    outdir = Path(args.output_dir)
    outdir.parent.mkdir(parents=True, exist_ok=True)
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    log_mode = "a" if Path(args.log).exists() and (args.resume_from_checkpoint or args.start_round > 1) else "w"
    logf = Path(args.log).open(log_mode, encoding="utf-8")

    seed_elites = _load_seed_elites(args.initial_elite_summary, usable_roles, max_per_role=3)
    prior_v10 = _load_logged_elites(args.log, usable_roles) if args.start_round > 1 else {r: [] for r in usable_roles}
    novelty_refs: dict[str, list[str]] = {
        r: _merge_unique(seed_elites.get(r, []), prior_v10.get(r, []), limit=8) for r in usable_roles
    }

    update_idx = max(0, (args.start_round - 1) * max(1, args.balanced_minibatches))
    total_candidates = 0
    total_evaluated = 0
    strict_train_count = 0
    elite_records: dict[str, list[dict]] = {r: [] for r in usable_roles}
    group_rewards: list[float] = []

    def update(buf: list[TrainSample]):
        nonlocal update_idx
        if not buf:
            return
        update_idx += 1
        policy.model.train()
        for epoch in range(args.grpo_epochs):
            optimizer.zero_grad(set_to_none=True)
            n = max(len(buf), 1)
            loss_sum = ratio_sum = kl_sum = 0.0
            for tr in buf:
                new_lp = policy.mean_logprob(tr.prompt, tr.completion)
                old_lp = torch.tensor(tr.old_logp, dtype=new_lp.dtype, device=new_lp.device)
                ratio = torch.exp(torch.clamp(new_lp - old_lp, -4.0, 4.0))
                adv = torch.tensor(tr.advantage, dtype=new_lp.dtype, device=new_lp.device)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1.0 - args.clip, 1.0 + args.clip) * adv
                pg = -torch.minimum(unclipped, clipped)
                if math.isfinite(tr.ref_logp) and args.kl_beta > 0:
                    ref_lp = torch.tensor(tr.ref_logp, dtype=new_lp.dtype, device=new_lp.device)
                    kl = _kl_k3(new_lp, ref_lp, torch)
                else:
                    kl = new_lp * 0.0
                elite_nll = (-new_lp) if tr.elite else new_lp * 0.0
                loss = pg + args.kl_beta * kl + args.elite_nll_weight * elite_nll
                (loss / n).backward()
                loss_sum += float(loss.detach().cpu())
                ratio_sum += float(ratio.detach().cpu())
                kl_sum += float(kl.detach().cpu())
                del new_lp, old_lp, ratio, adv, unclipped, clipped, pg, kl, elite_nll, loss
            torch.nn.utils.clip_grad_norm_(policy.trainable_parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(
                f"update={update_idx} epoch={epoch+1} loss={loss_sum/n:+.4f} "
                f"ratio={ratio_sum/n:.3f} kl={kl_sum/n:.4f}",
                flush=True,
            )
        policy.model.eval()

    def eval_query(role: str, query: str, bundle: Sequence[object], anchor: str, semantic_sim: float):
        plus_q_embs: list[np.ndarray] = []
        minus_q_embs: list[np.ndarray] = []
        sham_q_embs: list[np.ndarray] = []
        plus_a_embs: list[np.ndarray] = []
        minus_a_embs: list[np.ndarray] = []
        sham_a_embs: list[np.ndarray] = []
        selected: list[bool] = []
        sham_selected: list[bool] = []
        branch_rows: list[dict] = []

        for p in bundle:
            pa = execute_cached(p.plus_spec, anchor, cache)
            ma = execute_cached(p.minus_spec, anchor, cache)
            sa = execute_cached(p.sham_spec, anchor, cache) if p.sham_spec is not None else None
            pq = execute_cached(p.plus_spec, query, cache)
            mq = execute_cached(p.minus_spec, query, cache)
            sq = execute_cached(p.sham_spec, query, cache) if p.sham_spec is not None else None

            plus_a_embs.append(embed_text(pa.output))
            minus_a_embs.append(embed_text(ma.output))
            plus_q_embs.append(embed_text(pq.output))
            minus_q_embs.append(embed_text(mq.output))
            if sa is not None and sq is not None:
                sham_a_embs.append(embed_text(sa.output))
                sham_q_embs.append(embed_text(sq.output))

            selected.append(role in set(pq.selected_roles_training_only))
            sham_selected.append(
                role in set(sq.selected_roles_training_only) if sq is not None else role in set(pq.selected_roles_training_only)
            )
            branch_rows.append(
                {
                    "pair_id": p.pair_id,
                    "mode": get_mode(p.plus_spec),
                    "k": get_k(p.plus_spec),
                    "plus_selected": list(pq.selected_roles_training_only),
                    "minus_selected": list(mq.selected_roles_training_only),
                    "sham_selected": list(sq.selected_roles_training_only) if sq is not None else [],
                    "plus_output": pq.output,
                    "minus_output": mq.output,
                    "sham_output": sq.output if sq is not None else "",
                    "plus_cache_hit": pq.cache_hit,
                    "minus_cache_hit": mq.cache_hit,
                    "sham_cache_hit": (sq.cache_hit if sq is not None else None),
                }
            )

        rb = grader.score(
            target_selected_flags=selected,
            sham_target_selected_flags=sham_selected,
            plus_query_embs=plus_q_embs,
            minus_query_embs=minus_q_embs,
            plus_anchor_embs=plus_a_embs,
            minus_anchor_embs=minus_a_embs,
            sham_query_embs=sham_q_embs if len(sham_q_embs) == len(bundle) else None,
            sham_anchor_embs=sham_a_embs if len(sham_a_embs) == len(bundle) else None,
            semantic_max_similarity=semantic_sim,
        )

        # Attach exact pairwise reward anatomy for later forensic analysis.
        for i, row in enumerate(branch_rows[: rb.pair_count]):
            row["target_distance"] = rb.target_pairwise[i]
            row["sham_distance"] = rb.sham_pairwise[i]
            row["specific_pairwise"] = rb.specific_pairwise[i]
            row["anchor_specific_pairwise"] = rb.anchor_specific_pairwise[i]
            row["delta_vs_anchor_pairwise"] = rb.delta_vs_anchor_pairwise[i]
        return rb, branch_rows

    for round_i in range(max(1, args.start_round), args.rounds_per_role + 1):
        print(f"=== TRAIN ROUND {round_i}/{args.rounds_per_role} ===", flush=True)
        round_by_role: dict[str, list[TrainSample]] = {r: [] for r in usable_roles}
        role_order = list(usable_roles)
        random.Random(args.seed + 7919 * round_i).shuffle(role_order)

        for role_pos, role in enumerate(role_order, 1):
            bundle = schedule[(round_i, role)]
            confirmation_bundle = confirmation_schedule[(round_i, role)]
            if not bundle:
                continue
            group_num = (round_i - 1) * len(usable_roles) + role_pos
            torch.manual_seed(args.seed + 100003 * round_i + 997 * role_pos)

            anchor = anchor_probe(role)
            avoid = novelty_refs.get(role, [])[-args.max_avoid_probes :]
            obs = {
                "candidate_role": role,
                "candidate_description": ROLE_DESCRIPTIONS.get(role, role),
                "anchor_probe": anchor,
                "avoid_probes": avoid,
            }

            candidates: list[dict] = []
            rewards: list[float] = []
            group_seen: list[str] = []

            for ci in range(args.group_size):
                total_candidates += 1
                sample = None
                attempts: list[dict] = []
                for _try in range(max(1, args.syntax_retries)):
                    trial = policy.sample_probe(obs, temperature=args.temperature, top_p=args.top_p)
                    attempts.append(
                        {"raw": trial.raw_completion, "valid": trial.valid, "reason": trial.invalid_reason}
                    )
                    sample = trial
                    if trial.valid:
                        break
                assert sample is not None

                status = "ok"
                rb = None
                branches: list[dict] = []
                semantic_sim = 0.0

                if not sample.valid:
                    status = f"invalid:{sample.invalid_reason}"
                    reward = -0.80
                elif _norm_text(sample.query) == _norm_text(anchor):
                    status = "exact_anchor_duplicate"
                    reward = -0.70
                elif any(_norm_text(sample.query) == _norm_text(q) for q in group_seen):
                    status = "exact_group_duplicate"
                    reward = -0.65
                elif _word_jaccard(sample.query, anchor) >= 0.92:
                    status = "near_anchor_duplicate"
                    reward = -0.60
                elif any(_word_jaccard(sample.query, q) >= 0.95 for q in group_seen):
                    status = "near_group_duplicate"
                    reward = -0.55
                else:
                    refs = [anchor] + novelty_refs.get(role, []) + group_seen
                    semantic_sim = semantic_similarity(sample.query, refs)
                    group_seen.append(sample.query)
                    if semantic_sim >= args.semantic_hard_duplicate:
                        # Local-only rejection: do not spend a guard/API/MAS call on a
                        # semantic near-copy of an already-known mechanism.
                        status = "semantic_duplicate"
                        reward = -0.60
                    else:
                        g = gate.check(sample.query)
                        if not g.allowed:
                            status = f"blocked:{g.stage}:{g.category}"
                            reward = -1.50
                        else:
                            total_evaluated += 1
                            rb, branches = eval_query(role, sample.query, bundle, anchor, semantic_sim)
                            reward = float(rb.total)
                            strict_train_count += int(rb.strict_specific_pass)

                rewards.append(float(reward))
                candidates.append(
                    {
                        "sample": sample,
                        "attempts": attempts,
                        "status": status,
                        "rb": rb,
                        "branches": branches,
                        "reward": float(reward),
                        "semantic_sim": semantic_sim,
                        "confirm_rb": None,
                        "confirm_branches": [],
                        "confirmation_pass": False,
                    }
                )
                if rb is not None:
                    print(
                        f"  cand={ci+1}/{args.group_size} role={role:<10} R={reward:+.3f} "
                        f"ACT={rb.activation_rate:.2f} (plus={rb.plus_activation_rate:.2f},sham={rb.sham_activation_rate:.2f}) "
                        f"ROB={rb.robust_specific:+.3f} med={rb.specific_median:+.3f} q25={rb.specific_q25:+.3f} "
                        f"pairs+={rb.positive_specific_pair_rate:.2f} sim={semantic_sim:.3f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  cand={ci+1}/{args.group_size} role={role:<10} {status} R={reward:+.3f}",
                        flush=True,
                    )

            adv = group_relative_advantages(rewards)

            # Confirmation is deliberately disjoint from the primary reward bundle.
            # Only the strongest primary strict candidates pay this extra API cost.
            primary_strict = [
                i for i, x in enumerate(candidates)
                if x["status"] == "ok"
                and x["rb"] is not None
                and x["rb"].strict_specific_pass
                and x["semantic_sim"] <= args.holdout_max_similarity
            ]
            primary_strict.sort(key=lambda i: candidates[i]["reward"], reverse=True)
            for i in primary_strict[: max(0, args.confirm_top_k)]:
                if not confirmation_bundle:
                    continue
                c = candidates[i]
                crb, cbranches = eval_query(
                    role, c["sample"].query, confirmation_bundle, anchor, c["semantic_sim"]
                )
                c["confirm_rb"] = crb
                c["confirm_branches"] = cbranches
                c["confirmation_pass"] = bool(crb.strict_specific_pass)
                print(
                    f"    confirm cand={i+1} role={role:<10} pass={c['confirmation_pass']} "
                    f"ACT={crb.activation_rate:.2f} ROB={crb.robust_specific:+.3f} "
                    f"med={crb.specific_median:+.3f} q25={crb.specific_q25:+.3f} "
                    f"pairs+={crb.positive_specific_pair_rate:.2f}",
                    flush=True,
                )

            confirmed = [i for i in primary_strict if candidates[i]["confirmation_pass"]]
            elite_i = max(confirmed, key=lambda i: candidates[i]["reward"]) if confirmed else None
            best_i = int(np.argmax(rewards))
            group_id = f"v10r{round_i:02d}_{role}_{group_num:03d}"

            for i, (c, a) in enumerate(zip(candidates, adv)):
                sample = c["sample"]
                elite = bool(elite_i is not None and i == elite_i)
                if sample.completion.strip():
                    round_by_role[role].append(
                        TrainSample(
                            role=role,
                            prompt=sample.prompt,
                            completion=sample.completion,
                            old_logp=sample.old_mean_logprob,
                            ref_logp=sample.ref_mean_logprob,
                            advantage=float(a),
                            reward=c["reward"],
                            elite=elite,
                            group_id=group_id,
                        )
                    )
                rec = {
                    "phase": "train",
                    "round": round_i,
                    "group_id": group_id,
                    "candidate_index": i + 1,
                    "target_role": role,
                    "anchor_probe": anchor,
                    "avoid_probes_public": avoid,
                    "query": sample.query,
                    "raw_completion": sample.raw_completion,
                    "syntax_attempts": c["attempts"],
                    "status": c["status"],
                    "reward": c["reward"],
                    "advantage": float(a),
                    "elite": elite,
                    "semantic_max_similarity": c["semantic_sim"],
                    "bundle_training_only": [
                        {
                            "pair_id": p.pair_id,
                            "mode": get_mode(p.plus_spec),
                            "k": get_k(p.plus_spec),
                            "negative_replacement": getattr(p, "negative_replacement", ""),
                        }
                        for p in bundle
                    ],
                    "confirmation_bundle_training_only": [
                        {
                            "pair_id": p.pair_id,
                            "mode": get_mode(p.plus_spec),
                            "k": get_k(p.plus_spec),
                            "negative_replacement": getattr(p, "negative_replacement", ""),
                        }
                        for p in confirmation_bundle
                    ],
                    "reward_breakdown_training_only": c["rb"].to_dict() if c["rb"] is not None else None,
                    "branches_training_only": c["branches"],
                    "confirmation_pass_training_only": c["confirmation_pass"],
                    "confirmation_reward_breakdown_training_only": (
                        c["confirm_rb"].to_dict() if c["confirm_rb"] is not None else None
                    ),
                    "confirmation_branches_training_only": c["confirm_branches"],
                }
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()
                if elite:
                    elite_records[role].append(rec)
                    novelty_refs[role] = _merge_unique(novelty_refs[role], [sample.query], limit=8)

            rb_best = candidates[best_i]["rb"]
            mean_r = float(np.mean(rewards))
            group_rewards.append(mean_r)
            if rb_best is not None:
                print(
                    f"group={group_num}/{groups} round={round_i} role={role:<10} "
                    f"Rmean={mean_r:+.3f} Rbest={max(rewards):+.3f} "
                    f"bestROB={rb_best.robust_specific:+.3f} bestACT={rb_best.activation_rate:.2f} "
                    f"strict={sum(1 for x in candidates if x['rb'] is not None and x['rb'].strict_specific_pass)}/{args.group_size} "
                    f"confirmed={sum(1 for x in candidates if x['confirmation_pass'])} "
                    f"valid={sum(1 for x in candidates if x['status']=='ok')}/{args.group_size}",
                    flush=True,
                )
            else:
                print(
                    f"group={group_num}/{groups} round={round_i} role={role:<10} "
                    f"Rmean={mean_r:+.3f} Rbest={max(rewards):+.3f} no evaluated best",
                    flush=True,
                )

        minibatches = _balanced_minibatches(
            round_by_role,
            usable_roles,
            batches=args.balanced_minibatches,
            seed=args.seed + 123457 * round_i,
        )
        print(
            f"=== BALANCED OPTIMIZATION ROUND {round_i}: minibatches={len(minibatches)} "
            f"samples={sum(len(x) for x in minibatches)} ===",
            flush=True,
        )
        for bi, mb in enumerate(minibatches, 1):
            role_counts = {r: sum(1 for x in mb if x.role == r) for r in usable_roles}
            print(f"  minibatch={bi}/{len(minibatches)} role_counts={role_counts}", flush=True)
            update(mb)

        # Transactional round checkpoint: resume trusts only this committed snapshot.
        state = commit_round_snapshot(
            policy=policy,
            optimizer=optimizer,
            round_i=round_i,
            total_rounds=args.rounds_per_role,
            checkpoint_root=args.checkpoint_root,
            checkpoint_state=args.checkpoint_state,
            public_output_dir=outdir,
            log_path=args.log,
            torch_module=torch,
        )
        # Convenience optimizer mirror for inspection/downstream tooling. Resume
        # correctness never depends on this mutable mirror.
        mirror = Path(args.optimizer_mirror)
        mirror.parent.mkdir(parents=True, exist_ok=True)
        if Path(state["optimizer_state"]).resolve() != mirror.resolve():
            shutil.copy2(state["optimizer_state"], mirror)

        print(
            f"COMMITTED round={round_i} next_round={state['next_round']} snapshot={state['round_snapshot']}",
            flush=True,
        )

    holdout_rows: list[dict] = []
    if args.holdout_eval:
        print("=== V11 UNSEEN-ROLE DEVELOPMENT EVALUATION ===", flush=True)
        policy.model.eval()
        for role in dev_usable_roles:
            hpairs = holdout_by_role.get(role, [])
            if not hpairs:
                continue
            anchor = anchor_probe(role)
            holdout_seen: list[str] = []
            role_rows: list[dict] = []

            for j in range(args.holdout_probes_per_role):
                torch.manual_seed(args.seed + 900000 + 1000 * dev_usable_roles.index(role) + j)
                sample = None
                attempts: list[dict] = []
                semantic_sim = 0.0
                status = "generation_failed"

                # Generate until the probe is syntactically valid, semantically distinct
                # from anchor/elites/other dev probes, and benign-gate allowed. This
                # prevents the V9 Writing-style duplicate triplet from wasting evaluation.
                for _try in range(max(1, args.holdout_generation_retries)):
                    dynamic_avoid = _merge_unique(
                        novelty_refs.get(role, []), holdout_seen, limit=max(args.max_avoid_probes, 6)
                    )
                    obs = {
                        "candidate_role": role,
                        "candidate_description": ROLE_DESCRIPTIONS.get(role, role),
                        "anchor_probe": anchor,
                        "avoid_probes": dynamic_avoid[-max(args.max_avoid_probes, 6):],
                    }
                    trial = policy.sample_probe(obs, temperature=0.75, top_p=0.90)
                    sample = trial
                    attempt = {
                        "raw": trial.raw_completion,
                        "valid": trial.valid,
                        "reason": trial.invalid_reason,
                    }
                    if not trial.valid:
                        attempt["decision"] = "retry_invalid"
                        attempts.append(attempt)
                        status = f"invalid:{trial.invalid_reason}"
                        continue

                    refs = [anchor] + novelty_refs.get(role, []) + holdout_seen
                    semantic_sim = semantic_similarity(trial.query, refs)
                    attempt["semantic_max_similarity"] = semantic_sim
                    if semantic_sim >= args.holdout_max_similarity:
                        attempt["decision"] = "retry_semantic_duplicate"
                        attempts.append(attempt)
                        status = "semantic_duplicate"
                        continue

                    g = gate.check(trial.query)
                    attempt["gate_allowed"] = g.allowed
                    attempt["gate_stage"] = g.stage
                    attempt["gate_category"] = g.category
                    if not g.allowed:
                        attempt["decision"] = "retry_gate_blocked"
                        attempts.append(attempt)
                        status = f"blocked:{g.stage}:{g.category}"
                        continue

                    attempt["decision"] = "accept"
                    attempts.append(attempt)
                    status = "ok"
                    holdout_seen.append(trial.query)
                    break

                if sample is None:
                    raise RuntimeError("holdout generation produced no sample object")

                rb = None
                branches: list[dict] = []
                if status == "ok":
                    rb, branches = eval_query(role, sample.query, hpairs, anchor, semantic_sim)

                rec = {
                    "phase": "holdout",
                    "evaluation_split": "development_unseen_roles_inspected",
                    "target_role": role,
                    "query": sample.query,
                    "raw_completion": sample.raw_completion,
                    "syntax_attempts": attempts,
                    "status": status,
                    "reward": float(rb.total) if rb else None,
                    "semantic_max_similarity": semantic_sim,
                    "pair_count": len(hpairs),
                    "pair_modes": [get_mode(p.plus_spec) for p in hpairs],
                    "pair_k": [get_k(p.plus_spec) for p in hpairs],
                    "reward_breakdown_training_only": rb.to_dict() if rb else None,
                    "branches_training_only": branches,
                }
                holdout_rows.append(rec)
                role_rows.append(rec)
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()

                if rb:
                    print(
                        f"dev role={role:<10} p{j+1} R={rb.total:+.3f} ACT={rb.activation_rate:.2f} "
                        f"(plus={rb.plus_activation_rate:.2f},sham={rb.sham_activation_rate:.2f}) "
                        f"ROB={rb.robust_specific:+.3f} med={rb.specific_median:+.3f} "
                        f"q25={rb.specific_q25:+.3f} pair+={rb.positive_specific_pair_rate:.2f} "
                        f"strict={rb.strict_specific_pass} sim={semantic_sim:.3f} | {sample.query}",
                        flush=True,
                    )
                else:
                    print(f"dev role={role:<10} p{j+1} {status} | {sample.query}", flush=True)

            good = [r for r in role_rows if r["reward_breakdown_training_only"] is not None]
            strict = [r for r in good if r["reward_breakdown_training_only"]["strict_specific_pass"]]
            print(
                f"DEV ROLE {role}: valid={len(good)}/{len(role_rows)} strict={len(strict)}/{len(good)} "
                f"any_strict={bool(strict)}",
                flush=True,
            )

    logf.close()

    # Rebuild the final report from the committed JSONL, not from in-memory counters.
    # This makes resumed runs scientifically/accounting-equivalent to uninterrupted runs.
    all_rows = _load_jsonl(args.log)
    train_rows = [r for r in all_rows if r.get("phase") == "train"]
    holdout_rows = [r for r in all_rows if r.get("phase") == "holdout"]
    train_evaluated_rows = [r for r in train_rows if r.get("reward_breakdown_training_only") is not None]
    train_strict_rows = [
        r for r in train_evaluated_rows
        if r["reward_breakdown_training_only"].get("strict_specific_pass")
    ]
    train_confirmed_elites = [r for r in train_rows if r.get("elite")]

    group_to_rewards: dict[str, list[float]] = {}
    for r in train_rows:
        if isinstance(r.get("reward"), (int, float)):
            group_to_rewards.setdefault(str(r.get("group_id", "")), []).append(float(r["reward"]))
    reconstructed_group_means = [float(np.mean(v)) for v in group_to_rewards.values() if v]

    holdout_role_stats: dict[str, dict] = {}
    for role in dev_usable_roles:
        rr = [r for r in holdout_rows if r.get("target_role") == role]
        good = [r for r in rr if r.get("reward_breakdown_training_only") is not None]
        rbs = [r["reward_breakdown_training_only"] for r in good]
        strict = [x for x in rbs if x.get("strict_specific_pass")]
        delta_strict = [x for x in rbs if x.get("strict_delta_pass")]
        holdout_role_stats[role] = {
            "total": len(rr),
            "valid": len(good),
            "strict_specific_count": len(strict),
            "strict_delta_count": len(delta_strict),
            "any_strict_specific": bool(strict),
            "any_strict_delta": bool(delta_strict),
            "any_activation_ge_075": any(float(x["activation_rate"]) >= 0.75 for x in rbs),
            "best_robust_specific": max((float(x["robust_specific"]) for x in rbs), default=None),
            "best_activation": max((float(x["activation_rate"]) for x in rbs), default=None),
            "best_plus_activation": max((float(x.get("plus_activation_rate", x["activation_rate"])) for x in rbs), default=None),
            "best_sham_activation": max((float(x.get("sham_activation_rate", x["activation_rate"])) for x in rbs), default=None),
            "mean_robust_specific": float(np.mean([x["robust_specific"] for x in rbs])) if rbs else None,
        }

    role_any_strict = sum(v["any_strict_specific"] for v in holdout_role_stats.values())
    role_any_delta = sum(v["any_strict_delta"] for v in holdout_role_stats.values())
    role_activation = sum(v["any_activation_ge_075"] for v in holdout_role_stats.values())

    baseline = None
    if Path(args.baseline_summary).exists():
        try:
            baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
        except Exception:
            baseline = None

    elite_by_role: dict[str, list[dict]] = {r: [] for r in usable_roles}
    for r in train_confirmed_elites:
        role = r.get("target_role")
        if role in elite_by_role:
            elite_by_role[role].append(
                {
                    "query": r.get("query"),
                    "reward": r.get("reward"),
                    "breakdown": r.get("reward_breakdown_training_only"),
                    "confirmation_breakdown": r.get("confirmation_reward_breakdown_training_only"),
                }
            )

    summary = {
        "version": "V11",
        "objective": "role-conditioned benign probe discovery with complete unseen-role-type development holdout",
        "train_roles": usable_roles,
        "dev_unseen_roles": dev_usable_roles,
        "rounds_per_role": args.rounds_per_role,
        "group_size": args.group_size,
        "bundle_size": args.bundle_size,
        "confirmation_size": args.confirmation_size,
        "balanced_minibatches_per_round": args.balanced_minibatches,
        "balanced_update_blocks": update_idx,
        "optimizer_steps": update_idx * args.grpo_epochs,
        "kl_beta": args.kl_beta,
        "train_candidates": len(train_rows),
        "train_evaluated": len(train_evaluated_rows),
        "train_strict_specific_count": len(train_strict_rows),
        "train_strict_specific_rate": len(train_strict_rows) / max(1, len(train_evaluated_rows)),
        "train_confirmed_elites": len(train_confirmed_elites),
        "mean_group_reward": float(np.mean(reconstructed_group_means)) if reconstructed_group_means else 0.0,
        "development_validation_inspected": True,
        "development_role_holdout": True,
        "unseen_roles_appeared_in_training": False,
        "final_untouched_test": False,
        "consumed_v10_final_used_for_training": False,
        "holdout_total": len(holdout_rows),
        "holdout_valid": sum(1 for r in holdout_rows if r.get("reward_breakdown_training_only") is not None),
        "holdout_role_any_strict_specific": role_any_strict,
        "holdout_role_any_strict_specific_rate": role_any_strict / max(1, len(dev_usable_roles)),
        "holdout_role_any_strict_delta": role_any_delta,
        "holdout_role_any_strict_delta_rate": role_any_delta / max(1, len(dev_usable_roles)),
        "holdout_role_any_activation_ge_075": role_activation,
        "holdout_role_any_activation_ge_075_rate": role_activation / max(1, len(dev_usable_roles)),
        "holdout_role_stats": holdout_role_stats,
        "detector_reserved_base_systems": len(detector_reserved_specs),
        "training_pair_coverage": training_pair_coverage,
        "elite_probes": elite_by_role,
        "v10_start_reference": {
            "source": args.baseline_summary,
            "v10_dev_role_any_strict_specific": baseline.get("holdout_role_any_strict_specific") if baseline else None,
            "v10_dev_role_any_strict_delta": baseline.get("holdout_role_any_strict_delta") if baseline else None,
        },
        "policy_dir": str(outdir),
        "checkpoint_root": args.checkpoint_root,
        "reward_model_used": False,
        "v65_detector_used_in_reward": False,
        "starting_adapter": args.resume_adapter,
        "starting_adapter_verified_exact_v10": True,
        "anchor_used_as_reward_hurdle": False,
    }
    atomic_write_json(args.summary, summary)

    # Mark the run complete only after summary is durably written. The adapter and
    # optimizer remain those of the last committed round snapshot.
    state = load_committed_state(args.checkpoint_state)
    complete_state = dict(state)
    complete_state.update(
        {
            "version": "V11",
            "phase": "complete",
            "summary": args.summary,
            "committed": True,
        }
    )
    atomic_write_json(args.checkpoint_state, complete_state)

    print("=== V11 COMPLETE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
