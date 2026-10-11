from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from targets.single_router_zoo_v11 import KS, MODES, validate_v11_design
from v11.role_universe import TRAIN_ROLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--belief", required=True)
    args = ap.parse_args()

    validate_v11_design()
    root = Path(args.root)
    manifest = json.loads((root / "manifest_privileged.json").read_text(encoding="utf-8"))
    bundle = FrozenBeliefBundle.load(args.belief)

    systems = manifest["systems"]
    train = [x for x in systems if x["split"] == "train"]
    dev = [x for x in systems if x["split"] == "dev"]
    if len(train) != 96 or len(dev) != 32:
        raise RuntimeError(f"expected 96/32 systems, got {len(train)}/{len(dev)}")

    for split, rows, expected_per_cell in (("train", train, 6), ("dev", dev, 2)):
        cells = Counter((x["mode"], int(x["k"])) for x in rows)
        want = {(m, int(k)): expected_per_cell for m in MODES for k in KS}
        if cells != Counter(want):
            raise RuntimeError(f"{split} cell imbalance: {cells}")
        comps = [tuple(sorted(x["roles"])) for x in rows]
        if len(comps) != len(set(comps)):
            raise RuntimeError(f"duplicate role composition in {split}")
        for x in rows:
            if set(x["roles"]) - set(TRAIN_ROLES):
                raise RuntimeError(f"unknown role in {x['session_id']}")

        sp = root / f"states_{split}_round_00.jsonl"
        states = [PublicBeliefState.from_public_dict(x) for x in read_jsonl(sp)]
        if len(states) != len(rows):
            raise RuntimeError(f"{split} initial state count mismatch")
        for st in states:
            st.validate(TRAIN_ROLES, bundle.ks)
            if any(abs(float(st.k_probs[k]) - 0.25) > 1e-9 for k in bundle.ks):
                raise RuntimeError("initial K prior is not uniform")

    overlap = {tuple(sorted(x["roles"])) for x in train} & {tuple(sorted(x["roles"])) for x in dev}
    if overlap:
        raise RuntimeError(f"train/dev composition overlap: {len(overlap)}")
    print("PASS")
    print("train=96 dev=32; 16 balanced mode×K cells; no composition overlap; K prior uniform")


if __name__ == "__main__":
    main()
