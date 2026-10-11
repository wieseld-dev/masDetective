from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

from estimator.k_estimator import KBeliefEstimator
from probes.seeds import BENIGN_SEEDS
from targets.hierarchical_mas import HierarchicalWhiteBoxMAS
from targets.topology_zoo import get_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "val", "test"])
    ap.add_argument("--limit-systems", type=int, default=0, help="0 = all systems")
    ap.add_argument("--max-seeds", type=int, default=5)
    ap.add_argument("--model", default=os.environ.get("MAS_MODEL", "gpt-4o-mini"))
    ap.add_argument("--output", default="artifacts/k_estimator.joblib")
    ap.add_argument("--examples", default="artifacts/k_estimator_examples.jsonl")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    specs = get_split(args.split)
    if args.limit_systems > 0:
        specs = specs[: args.limit_systems]
    seeds = BENIGN_SEEDS[: max(1, args.max_seeds)]
    upper_calls = sum((s.depth + 1) * len(seeds) for s in specs)

    print("=== K ESTIMATOR BOOTSTRAP PLAN ===")
    print("systems:", len(specs))
    print("seeds/system:", len(seeds))
    print("external interactions:", len(specs) * len(seeds))
    print("upper-bound API calls:", upper_calls)
    print("output:", args.output)
    if not args.execute:
        print("PLAN ONLY. Re-run with --execute to make API calls.")
        return
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set")

    histories: List[List[Dict[str, str]]] = []
    labels: List[int] = []
    rows: List[Dict] = []

    for si, spec in enumerate(specs, start=1):
        print(f"[{si}/{len(specs)}] {spec.system_id} K={spec.k_true} depth={spec.depth}")
        mas = HierarchicalWhiteBoxMAS(spec, model=args.model)
        history: List[Dict[str, str]] = []
        for qi, query in enumerate(seeds, start=1):
            output = mas.query(query)
            history.append({"query": query, "output": output})
            # Every prefix is a black-box training example; label is training-only truth.
            histories.append([dict(x) for x in history])
            labels.append(spec.k_true)
            rows.append({
                "system_id": spec.system_id,
                "prefix_len": qi,
                "history": [dict(x) for x in history],
                "k_true_training_only": spec.k_true,
            })

    est = KBeliefEstimator(min_k=2, max_k=8).fit(histories, labels)
    est.save(args.output)
    Path(args.examples).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.examples).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    correct = sum(est.predict(h) == y for h, y in zip(histories, labels))
    print(f"bootstrap train accuracy={correct/len(labels):.3f} ({correct}/{len(labels)})")
    print("saved:", args.output)
    print("NOTE: this accuracy is only a bootstrap sanity check, not a held-out result.")


if __name__ == "__main__":
    main()
