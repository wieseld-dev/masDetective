from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_curriculum.belief_utils import dummy_k_probs
from v11.role_universe import TRAIN_ROLES

from presence_3of10.config import POOL, HIDDEN, SESSION_ID


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--belief", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    bundle = FrozenBeliefBundle.load(args.belief)

    # Exactly 3 of 10 are known to be possible.
    # 0.30 is therefore the natural marginal prior.
    #
    # Roles outside the known candidate pool remain near zero so the
    # trained V11 policy spends its probing budget on these ten.
    probs = {
        r: (0.30 if r in POOL else 1e-4)
        for r in TRAIN_ROLES
    }

    state = PublicBeliefState(
        session_id=SESSION_ID,
        split="dev",
        round_index=0,
        k_probs=dummy_k_probs(bundle),
        role_probs=probs,
        history=[],
    )

    state.validate(TRAIN_ROLES, bundle.ks)

    write_jsonl(
        root / "states_dev_round_00.jsonl",
        [state.to_public_dict()],
    )

    config = {
        "session_id": SESSION_ID,
        "candidate_pool": list(POOL),
        "hidden_roles_evaluation_only": list(HIDDEN),
        "k": 3,
        "n": 10,
        "prior_pool": 0.30,
        "prior_outside_pool": 1e-4,
    }

    (root / "experiment.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print("PRESENCE 3-of-10 INITIALIZED")
    print("candidate pool =", list(POOL))
    print("K = 3")
    print("state =", root / "states_dev_round_00.jsonl")


if __name__ == "__main__":
    main()
