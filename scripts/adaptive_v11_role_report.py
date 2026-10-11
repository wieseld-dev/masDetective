from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.io import write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, required=True)
    ap.add_argument("--root", default="artifacts/adaptive_v11_role_only_v1")
    args = ap.parse_args()
    root = Path(args.root)
    rows = []
    for split in ("train", "dev"):
        for r in range(1, args.rounds + 1):
            p = root / f"advance_{split}_round_{r:02d}_summary.json"
            if p.exists():
                rows.append(json.loads(p.read_text(encoding="utf-8")))
    out = {"version": "adaptive_v11_role_only_v1", "rounds": args.rounds, "summaries": rows}
    write_json(root / "role_adaptive_report.json", out)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
