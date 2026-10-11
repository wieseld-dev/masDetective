from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"[copy] {src.name} -> {dst}")


def main():
    ap = argparse.ArgumentParser(description="Seed static/adaptive arms from one shared round-1 trajectory")
    ap.add_argument("--common-root", required=True)
    ap.add_argument("--static-root", required=True)
    ap.add_argument("--adaptive-root", required=True)
    ap.add_argument("--split", choices=["train", "dev"], default="dev")
    args = ap.parse_args()

    common = Path(args.common_root)
    arms = [Path(args.static_root), Path(args.adaptive_root)]
    split = args.split

    required = [
        common / f"states_{split}_round_01.jsonl",
        common / f"candidates_{split}_round_01.jsonl",
        common / f"responses_{split}_round_01.jsonl",
        common / f"scored_{split}_round_01.jsonl",
    ]
    missing = [str(x) for x in required if not x.exists()]
    if missing:
        raise SystemExit("missing common round-1 artifacts: " + ", ".join(missing))

    names = [
        f"candidates_{split}_round_01.jsonl",
        f"responses_{split}_round_01.jsonl",
        f"score_input_{split}_round_01.jsonl",
        f"scored_{split}_round_01.jsonl",
        f"states_{split}_round_01.jsonl",
        f"kboot_metrics_{split}_round_01.jsonl",
        f"kboot_advance_{split}_round_01_summary.json",
        "target_public_cache.jsonl",
        "target_truth_cache_training_only.jsonl",
    ]

    for arm in arms:
        for name in names:
            copy_if_exists(common / name, arm / name)

    print("PASS: both arms now share the exact same round-1 query/answer/evidence/state")


if __name__ == "__main__":
    main()
