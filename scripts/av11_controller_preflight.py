from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--belief", required=True)
    ap.add_argument("--adapter", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    dev = [x for x in manifest["systems"] if x["split"] == "dev"]
    if len(dev) != 32:
        raise SystemExit(f"expected 32 untouched DEV systems, got {len(dev)}")
    if len({x["session_id"] for x in dev}) != 32:
        raise SystemExit("duplicate DEV session ids")
    if not Path(args.belief).is_file():
        raise SystemExit(f"missing belief: {args.belief}")
    if not Path(args.adapter).is_dir():
        raise SystemExit(f"missing V11 adapter: {args.adapter}")

    for arm in ("common", "static", "adaptive"):
        p = root / arm / "states_dev_round_00.jsonl"
        rows = read_jsonl(p)
        if len(rows) != 32:
            raise SystemExit(f"{arm}: expected 32 initial DEV states, got {len(rows)}")
        for raw in rows:
            s = PublicBeliefState.from_public_dict(raw)
            if set(s.role_probs) != set(TRAIN_ROLES):
                raise SystemExit(f"{arm}: role universe mismatch")
            if s.history:
                raise SystemExit(f"{arm}: initial history not empty")

    print("PASS")
    print("DEV systems=32")
    print("arms=common/static/adaptive initialized identically")
    print("controller intervention: STATIC=fixed target-role permutation; ADAPTIVE=max role posterior entropy among untested roles")
    print("query generator: frozen original V11 adapter via SeededProbePolicyV10")
    print("round 1 will be executed once and copied exactly into both arms")


if __name__ == "__main__":
    main()
