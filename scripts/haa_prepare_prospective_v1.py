from __future__ import annotations

import json
import shutil
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


SRC_MANIFEST = Path(
    "artifacts/adaptive_v11_factorial_heldout32/manifest_privileged.json"
)
BELIEF = Path(
    "artifacts/adaptive_v11_active_v1/frozen_belief.joblib"
)
ROOT = Path(
    "artifacts/haa_prospective_v1"
)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    if not SRC_MANIFEST.exists():
        raise FileNotFoundError(SRC_MANIFEST)

    if not BELIEF.exists():
        raise FileNotFoundError(BELIEF)

    manifest = json.loads(
        SRC_MANIFEST.read_text(encoding="utf-8")
    )

    systems = manifest["systems"]

    if len(systems) != 32:
        raise RuntimeError(
            f"Expected 32 held-out systems, found {len(systems)}"
        )

    # Keep all systems on dev side.
    for s in systems:
        s["split"] = "dev"

    (ROOT / "manifest_privileged.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    shutil.copy2(
        BELIEF,
        ROOT / "frozen_belief.joblib",
    )

    belief = FrozenBeliefBundle.load(
        ROOT / "frozen_belief.joblib"
    )

    kp, rp = belief.initial()

    states = []

    for s in systems:
        st = PublicBeliefState(
            session_id=str(s["session_id"]),
            split="dev",
            round_index=0,
            k_probs=dict(kp),
            role_probs=dict(rp),
            history=[],
        )

        st.validate(
            TRAIN_ROLES,
            belief.ks,
        )

        states.append(
            st.to_public_dict()
        )

    write_jsonl(
        ROOT / "states_dev_round_00.jsonl",
        states,
    )

    print("=" * 80)
    print("HAA PROSPECTIVE V1 PREPARED")
    print("=" * 80)
    print("systems:", len(states))
    print("root:", ROOT)
    print("manifest:", ROOT / "manifest_privileged.json")
    print("state0:", ROOT / "states_dev_round_00.jsonl")


if __name__ == "__main__":
    main()
