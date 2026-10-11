from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import (
    EvidenceFrame,
    PublicBeliefState,
)
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/haa_prospective_v1")
EPS = 1e-6


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sid_of(r):
    return str(
        r.get("system_id")
        or r.get("session_id")
        or ""
    )


def role_probs_unprojected(bundle, history):
    raw = bundle._raw_role_vector_with(
        bundle.role_model,
        history,
    )

    p = bundle._apply_platt(
        bundle.role_platt,
        raw,
    )

    return {
        role: float(
            np.clip(p[i], EPS, 1.0-EPS)
        )
        for i, role in enumerate(TRAIN_ROLES)
    }


def main():
    belief = FrozenBeliefBundle.load(
        ROOT / "frozen_belief.joblib"
    )

    states = {
        str(x["session_id"]):
            PublicBeliefState.from_public_dict(x)
        for x in read_jsonl(
            ROOT / "states_dev_round_00.jsonl"
        )
    }

    cand = {
        (
            str(x["session_id"]),
            int(x["candidate_index"]),
        ): x
        for x in read_jsonl(
            ROOT / "candidates_dev_round_01.jsonl"
        )
    }

    resp = {
        (
            str(x["session_id"]),
            int(x["candidate_index"]),
        ): x
        for x in read_jsonl(
            ROOT / "responses_dev_round_01.jsonl"
        )
    }

    scored = {}

    for x in read_jsonl(
        ROOT / "scored_dev_round_01.jsonl"
    ):
        sid = sid_of(x)

        idx = int(
            dict(x.get("action", {})).get(
                "variant",
                x.get("candidate_index", 0),
            )
        )

        scored[(sid, idx)] = x

    out = []

    for sid, state in states.items():
        key = (sid, 0)

        if key not in cand or key not in resp or key not in scored:
            raise RuntimeError(
                f"Missing bootstrap data for {sid}"
            )

        c = cand[key]
        r = resp[key]
        s = scored[key]

        if (
            not c.get("valid")
            or r.get("execution_status") != "ok"
        ):
            raise RuntimeError(
                f"Invalid bootstrap candidate for {sid}"
            )

        v135 = {
            k: float(v)
            for k, v in dict(
                s["evidence"]
            ).items()
        }

        v134 = {
            k: float(v)
            for k, v in dict(
                s.get("metadata", {}).get(
                    "evidence_v134",
                    {},
                )
            ).items()
        }

        answer = str(
            r.get("final_output")
            or r.get("output")
            or ""
        )

        frame = EvidenceFrame(
            query=str(c["query"]),
            answer=answer,
            v135=v135,
            v134=v134,
        )

        history = [
            *state.history,
            frame,
        ]

        # Role-only belief update.
        rp = role_probs_unprojected(
            belief,
            history,
        )

        # Keep K prior fixed in this experiment.
        # We are testing active ROLE probing first.
        ns = PublicBeliefState(
            session_id=sid,
            split="dev",
            round_index=1,
            k_probs=dict(state.k_probs),
            role_probs=rp,
            history=history,
        )

        ns.validate(
            TRAIN_ROLES,
            belief.ks,
        )

        out.append(
            ns.to_public_dict()
        )

    write_jsonl(
        ROOT / "states_dev_round_01.jsonl",
        out,
    )

    print(
        f"[done] bootstrap states={len(out)}"
    )


if __name__ == "__main__":
    main()
