from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import write_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


def main():
    ap = argparse.ArgumentParser(description="Initialize leak-safe public belief states")
    ap.add_argument("--manifest", default="artifacts/adaptive_v11_active_v1/manifest_privileged.json")
    ap.add_argument("--belief", default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    ap.add_argument("--output-dir", default="artifacts/adaptive_v11_active_v1")
    args = ap.parse_args()

    obj = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    belief = FrozenBeliefBundle.load(args.belief)
    kp, rp = belief.initial()
    for split in ("train", "dev"):
        rows = []
        for s in obj["systems"]:
            if s["split"] != split:
                continue
            state = PublicBeliefState(
                session_id=s["session_id"],
                split=split,
                round_index=0,
                k_probs=dict(kp),
                role_probs=dict(rp),
                history=[],
            )
            state.validate(TRAIN_ROLES, belief.ks)
            rows.append(state.to_public_dict())
        out = Path(args.output_dir) / f"states_{split}_round_00.jsonl"
        write_jsonl(out, rows)
        print(f"[init] split={split} states={len(rows)} -> {out}")


if __name__ == "__main__":
    main()
