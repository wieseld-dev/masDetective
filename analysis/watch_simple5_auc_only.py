#!/usr/bin/env python3

import json
import os
import time
from pathlib import Path

ROOT = Path(
    "artifacts/v14_cleanR7_selectorV2_simple5_v1"
)

BASELINE = 0.8471
REFRESH = 20


def load(p):
    return json.loads(p.read_text())


while True:
    os.system("clear")

    print("===== SIMPLE-5 LIVE AUC =====")
    print()

    found = []

    for r in range(1, 9):
        p = ROOT / (
            f"advance_dev_round_{r:02d}_summary.json"
        )

        if p.exists() and p.stat().st_size:
            try:
                d = load(p)

                found.append(
                    (
                        r,
                        float(d["mean_role_auc"]),
                        d,
                    )
                )
            except Exception:
                pass

    if not found:
        print("No completed rounds yet.")
        print()
        print(
            "Waiting for:",
            ROOT / "advance_dev_round_01_summary.json",
        )

    else:
        prev = 0.5000

        print(
            f"{'ROUND':>5}  "
            f"{'AUC':>8}  "
            f"{'Δ ROUND':>9}  "
            f"{'Δ vs .8471':>11}"
        )

        print("-" * 43)

        for r, auc, _ in found:

            print(
                f"{r:5d}  "
                f"{auc:8.4f}  "
                f"{auc-prev:+9.4f}  "
                f"{auc-BASELINE:+11.4f}"
            )

            prev = auc

        r, auc, d = found[-1]

        print()
        print("=" * 43)
        print(f"LATEST ROUND      : {r}")
        print(f"CURRENT MEAN AUC  : {auc:.4f}")
        print(f"HISTORICAL        : {BASELINE:.4f}")
        print(f"DELTA             : {auc-BASELINE:+.4f}")

        if "mean_role_log_loss" in d:
            print(
                "LOG LOSS          : "
                f"{float(d['mean_role_log_loss']):.4f}"
            )

        if "mean_true_false_margin" in d:
            print(
                "TRUE/FALSE MARGIN : "
                f"{float(d['mean_true_false_margin']):.4f}"
            )

        if "mean_oracle_k_role_recall" in d:
            print(
                "ORACLE-K RECALL   : "
                f"{float(d['mean_oracle_k_role_recall']):.4f}"
            )

        print("=" * 43)

    print()
    print("Refreshing every 20 seconds...")
    print("Ctrl+C to stop.")

    time.sleep(REFRESH)
