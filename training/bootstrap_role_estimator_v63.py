from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
from sklearn.model_selection import GroupKFold

from estimator.role_presence_estimator_v63 import RolePresenceEstimatorV63, load_profile
from probes.role_probe_bank import ANCHOR_PROBE, primary_probe
from targets.single_router_fanout_mas import SingleRouterFanoutMAS
from targets.single_router_zoo import get_split


def _metrics(estimator, histories, labels):
    role_correct = 0
    role_total = 0
    k_abs = []
    briers = []
    roles = estimator.candidate_roles
    for h, truth in zip(histories, labels):
        truth = set(truth)
        b = estimator.predict_proba(h)
        for r in roles:
            y = 1 if r in truth else 0
            role_correct += int((b[r] >= 0.5) == bool(y))
            role_total += 1
        k_abs.append(abs(sum(b.values()) - len(truth)))
        briers.append(sum((b[r] - (1.0 if r in truth else 0.0)) ** 2 for r in roles) / len(roles))
    return {
        "accuracy": role_correct / max(role_total, 1),
        "k_mae": float(np.mean(k_abs)) if k_abs else 999.0,
        "brier": float(np.mean(briers)) if briers else 999.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--output", default="artifacts/role_presence_estimator_v63.joblib")
    ap.add_argument("--public-log", default="artifacts/role_bootstrap_v63_public.jsonl")
    ap.add_argument("--truth-log", default="artifacts/role_bootstrap_v63_truth.jsonl")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--prefix-permutations", type=int, default=6)
    ap.add_argument("--min-cv-accuracy", type=float, default=0.70)
    ap.add_argument("--max-cv-brier", type=float, default=0.25)
    ap.add_argument("--max-cv-k-mae", type=float, default=1.60)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    candidates = list(profile["candidate_roles"])
    specs = get_split("train")
    interactions_per_system = 1 + len(candidates)
    rough_upper = sum(s.estimated_calls_per_query * interactions_per_system for s in specs)
    print("=== V6.3 FANOUT ROLE ESTIMATOR BOOTSTRAP PLAN ===")
    print("single routing layer modes:", sorted({s.execution_mode for s in specs}))
    print("train systems:", len(specs), "(balanced 4 modes x K=3..6)")
    print("candidate roles:", candidates)
    print("external interactions:", len(specs) * interactions_per_system)
    print("rough upper-bound API calls:", rough_upper)
    print("system-level CV quality gate: accuracy>=", args.min_cv_accuracy,
          "brier<=", args.max_cv_brier, "K-MAE<=", args.max_cv_k_mae)
    print("output:", args.output)
    if not args.execute:
        print("PLAN ONLY. Re-run with --execute to make API calls.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    rng = random.Random(args.seed)
    histories: List[List[Dict[str, str]]] = []
    labels: List[Set[str]] = []
    groups: List[str] = []
    public_rows: List[Dict] = []
    truth_rows: List[Dict] = []

    for si, spec in enumerate(specs, 1):
        print(f"[{si}/{len(specs)}] {spec.system_id} mode={spec.execution_mode} K={spec.k_true}")
        target = SingleRouterFanoutMAS(spec)
        interactions: Dict[str, Dict[str, str]] = {}

        q = ANCHOR_PROBE
        o = target.query(q)
        tr = target.privileged_last_trace()
        interactions["__anchor__"] = {"query": q, "output": o, "focus_role": "__anchor__"}
        public_rows.append({"system_id": spec.system_id, "focus_role": "__anchor__", "query": q, "output": o})
        truth_rows.append({"system_id": spec.system_id, "focus_role": "__anchor__", "true_roles": list(spec.worker_roles), "selected_roles": list(tr.selected_roles), "mode": spec.execution_mode})

        for role in candidates:
            q = primary_probe(role)
            o = target.query(q)
            tr = target.privileged_last_trace()
            interactions[role] = {"query": q, "output": o, "focus_role": role}
            public_rows.append({"system_id": spec.system_id, "focus_role": role, "query": q, "output": o})
            truth_rows.append({"system_id": spec.system_id, "focus_role": role, "true_roles": list(spec.worker_roles), "selected_roles": list(tr.selected_roles), "mode": spec.execution_mode})

        truth = set(spec.worker_roles)
        # Independent views make each fixed probe interpretable.
        for role in candidates:
            histories.append([interactions[role]])
            labels.append(truth)
            groups.append(spec.system_id)
        # Anchor-only prior evidence.
        histories.append([interactions["__anchor__"]]); labels.append(truth); groups.append(spec.system_id)
        # Multiple offline prefix permutations reuse the same paid interactions and teach adaptive updates.
        for _ in range(max(1, args.prefix_permutations)):
            order = candidates[:]
            rng.shuffle(order)
            h = [interactions["__anchor__"]]
            for role in order:
                h = [*h, interactions[role]]
                histories.append(h)
                labels.append(truth)
                groups.append(spec.system_id)

    Path(args.public_log).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.public_log).open("w", encoding="utf-8") as f:
        for row in public_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with Path(args.truth_log).open("w", encoding="utf-8") as f:
        for row in truth_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # System-level CV: no histories from the same MAS appear on both sides of a fold.
    unique_groups = sorted(set(groups))
    n_splits = min(4, len(unique_groups))
    gkf = GroupKFold(n_splits=n_splits)
    fold_metrics = []
    idx = np.arange(len(histories))
    group_arr = np.asarray(groups)
    for fi, (tr_idx, va_idx) in enumerate(gkf.split(idx, groups=group_arr), 1):
        est = RolePresenceEstimatorV63(candidates, random_state=args.seed + fi).fit(
            [histories[i] for i in tr_idx], [labels[i] for i in tr_idx]
        )
        m = _metrics(est, [histories[i] for i in va_idx], [labels[i] for i in va_idx])
        fold_metrics.append(m)
        print(f"CV fold {fi}: role-acc={m['accuracy']:.3f} brier={m['brier']:.3f} K-MAE={m['k_mae']:.3f}")

    cv = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    print(f"CV mean: role-acc={cv['accuracy']:.3f} brier={cv['brier']:.3f} K-MAE={cv['k_mae']:.3f}")
    if cv["accuracy"] < args.min_cv_accuracy or cv["brier"] > args.max_cv_brier or cv["k_mae"] > args.max_cv_k_mae:
        raise SystemExit("V6.3 estimator quality gate FAILED; RL job will not start. Inspect bootstrap logs before spending more API/GPU.")

    estimator = RolePresenceEstimatorV63(candidates, random_state=args.seed).fit(histories, labels)
    estimator.save(args.output)
    train_m = _metrics(estimator, histories, labels)
    print(f"full-fit sanity: role-acc={train_m['accuracy']:.3f} brier={train_m['brier']:.3f} K-MAE={train_m['k_mae']:.3f}")
    print("saved:", args.output)


if __name__ == "__main__":
    main()
