from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.io import read_jsonl, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="artifacts/adaptive_v11_active_v1")
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()
    root = Path(args.root)
    report = {"version": "adaptive_v11_active_v1", "dev": {}, "train": {}}
    for split in ("train", "dev"):
        for r in range(1, args.rounds + 1):
            p = root / f"metrics_{split}_round_{r:02d}.jsonl"
            if not p.exists():
                continue
            rows = read_jsonl(p)
            report[split][str(r)] = {
                "n": len(rows),
                "k_exact": sum(x["k_correct"] for x in rows)/max(1,len(rows)),
                "role_f1": sum(x["role_f1"] for x in rows)/max(1,len(rows)),
                "exact_set": sum(x["exact_set"] for x in rows)/max(1,len(rows)),
            }
    report["primary_note"] = "Dev is the controller-development holdout for this experiment, not final paper test data."
    write_json(root / "report.json", report)
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__": main()
