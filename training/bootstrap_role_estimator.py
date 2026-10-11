from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Set

from estimator.role_presence_estimator import RolePresenceEstimator, load_profile
from probes.role_probe_bank import ANCHOR_PROBE, primary_probe
from targets.hierarchical_mas import HierarchicalWhiteBoxMAS
from targets.topology_zoo import get_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--limit-systems", type=int, default=8)
    ap.add_argument("--output", default="artifacts/role_presence_estimator.joblib")
    ap.add_argument("--raw-log", default="artifacts/role_bootstrap.jsonl")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    candidates = list(profile["candidate_roles"])
    specs = get_split("train")[: args.limit_systems]
    interactions_per_system = 1 + len(candidates)
    upper_calls = sum((s.estimated_calls_per_query if hasattr(s, 'estimated_calls_per_query') else s.depth + 1) * interactions_per_system for s in [])
    rough_upper = sum((s.depth + 1) * interactions_per_system for s in specs)

    print("=== ROLE-PRESENCE ESTIMATOR BOOTSTRAP PLAN ===")
    print("application:", profile.get("app_name"))
    print("candidate roles:", candidates)
    print("systems:", len(specs))
    print("interactions/system:", interactions_per_system, "(1 anchor + one benign probe per candidate)")
    print("external interactions:", len(specs) * interactions_per_system)
    print("rough upper-bound API calls:", rough_upper)
    print("output:", args.output)
    if not args.execute:
        print("PLAN ONLY. Re-run with --execute to make API calls.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    rng = random.Random(args.seed)
    histories: List[List[Dict[str, str]]] = []
    labels: List[Set[str]] = []
    raw_rows: List[Dict] = []

    for si, spec in enumerate(specs, 1):
        print(f"[{si}/{len(specs)}] {spec.system_id} K={spec.k_true} roles={list(spec.worker_roles)}")
        target = HierarchicalWhiteBoxMAS(spec)
        interactions: Dict[str, Dict[str, str]] = {}

        anchor_output = target.query(ANCHOR_PROBE)
        interactions["__anchor__"] = {"query": ANCHOR_PROBE, "output": anchor_output}
        raw_rows.append({"system_id": spec.system_id, "probe_role": "__anchor__", "query": ANCHOR_PROBE, "output": anchor_output})

        for role in candidates:
            q = primary_probe(role)
            o = target.query(q)
            interactions[role] = {"query": q, "output": o}
            raw_rows.append({"system_id": spec.system_id, "probe_role": role, "query": q, "output": o})

        true_roles = set(spec.worker_roles)
        # Independent views teach role-specific response fingerprints.
        for role in candidates:
            histories.append([interactions[role]])
            labels.append(true_roles)
        # Prefix views teach the estimator how to update beliefs across an adaptive history.
        order = candidates[:]
        rng.shuffle(order)
        h = [interactions["__anchor__"]]
        histories.append(list(h)); labels.append(true_roles)
        for role in order:
            h.append(interactions[role])
            histories.append(list(h)); labels.append(true_roles)

    estimator = RolePresenceEstimator(candidates, random_state=args.seed).fit(histories, labels)
    estimator.save(args.output)
    Path(args.raw_log).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.raw_log).open("w", encoding="utf-8") as f:
        for row in raw_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Training-only bootstrap sanity metrics.
    role_correct = 0
    role_total = 0
    k_abs = []
    for h, truth in zip(histories, labels):
        b = estimator.predict_proba(h)
        pred = {r for r, p in b.items() if p >= 0.5}
        for r in candidates:
            role_correct += int((r in pred) == (r in truth))
            role_total += 1
        k_abs.append(abs(sum(b.values()) - len(truth)))
    print(f"bootstrap role-label accuracy={role_correct/max(role_total,1):.3f}")
    print(f"bootstrap expected-K MAE={sum(k_abs)/max(len(k_abs),1):.3f}")
    print("saved:", args.output)
    print("NOTE: these are training-only sanity metrics, not held-out results.")


if __name__ == "__main__":
    main()
