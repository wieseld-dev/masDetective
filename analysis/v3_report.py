#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/v3_coverage_elimination_pilot")
    ap.add_argument("--horizon", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.root)
    print("=" * 96)
    print("V3 COVERAGE + ELIMINATION PILOT")
    print("=" * 96)

    prep = root / "prepare_summary.json"
    if prep.exists():
        print("\nPREP:")
        print(prep.read_text())

    for r in range(1, args.horizon + 1):
        p = root / f"advance_train_round_{r:02d}_summary.json"
        if p.exists():
            x = json.loads(p.read_text())
            print(
                f"r={r:02d} "
                f"KMAP={x['mean_k_map_acc']:.3f} "
                f"KGAP={x['mean_gap_k_acc']:.3f} "
                f"TOPSET={x['mean_top_true_k_exact']:.3f} "
                f"P(K*)={x['mean_p_true_k']:.4f} "
                f"P(Z*)={x['mean_p_true_composition']:.4f} "
                f"COV={x['mean_coverage']:.3f} "
                f"R={x['mean_chosen_reward']}"
            )

    upd = root / "update_summary.json"
    if upd.exists():
        print("\nUPDATE:")
        print(upd.read_text())
    else:
        print("\nUPDATE: not finished yet")


if __name__ == "__main__":
    main()
