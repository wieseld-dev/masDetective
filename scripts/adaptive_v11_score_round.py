from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from adaptive_v11_active.compat import discover_dual_scorer_cli
from adaptive_v11_active.io import read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser(description="Bridge Adaptive-V11 responses through the existing frozen V13.4+V13.5 scorer")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="train")
    ap.add_argument("--root", default="artifacts/adaptive_v11_active_v1")
    ap.add_argument("--scorer", default="adaptive_score_v135.py")
    args, extra = ap.parse_known_args()

    root = Path(args.root)
    inp = root / f"responses_{args.split}_round_{args.round:02d}.jsonl"
    out = root / f"scored_{args.split}_round_{args.round:02d}.jsonl"
    if not inp.exists():
        raise SystemExit(f"missing response file: {inp}")
    valid_rows = [r for r in read_jsonl(inp) if r.get("valid") and r.get("execution_status") == "ok"]
    if not valid_rows:
        raise RuntimeError("no valid executed responses to score")
    score_input = root / f"score_input_{args.split}_round_{args.round:02d}.jsonl"
    write_jsonl(score_input, valid_rows)
    input_flag, output_flag, _ = discover_dual_scorer_cli(args.scorer)
    cmd = [sys.executable, args.scorer, input_flag, str(score_input), output_flag, str(out), *extra]
    print("[score-bridge]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(f"dual scorer returned successfully but did not create {out}")
    print(f"[done] scored={out}", flush=True)


if __name__ == "__main__":
    main()
