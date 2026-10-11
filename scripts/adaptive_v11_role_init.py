from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import write_jsonl
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="artifacts/adaptive_v11_active_v1/manifest_privileged.json")
    ap.add_argument("--belief", default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib")
    ap.add_argument("--root", default="artifacts/adaptive_v11_role_only_v1")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    belief = FrozenBeliefBundle.load(args.belief)
    kp, rp = role_only_predict(belief, [])
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "dev"):
        rows = []
        for s in manifest["systems"]:
            if s["split"] != split:
                continue
            st = PublicBeliefState(
                session_id=str(s["session_id"]),
                split=split,
                round_index=0,
                k_probs=dict(kp),
                role_probs=dict(rp),
                history=[],
            )
            st.validate(TRAIN_ROLES, belief.ks)
            rows.append(st.to_public_dict())
        out = root / f"states_{split}_round_00.jsonl"
        write_jsonl(out, rows)
        print(f"[init] {split}={len(rows)} -> {out}", flush=True)


if __name__ == "__main__":
    main()
