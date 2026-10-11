#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


def auc_for_state(
    probs,
    truth,
):
    y = np.asarray(
        [
            int(r in truth)
            for r in TRAIN_ROLES
        ],
        dtype=int,
    )

    p = np.asarray(
        [
            float(
                probs[r]
            )
            for r in TRAIN_ROLES
        ],
        dtype=float,
    )

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--rounds",
        type=int,
        default=8,
    )

    a = ap.parse_args()

    root = Path(
        a.root
    )

    manifest = json.loads(
        (
            root
            / "manifest_privileged.json"
        ).read_text()
    )

    truth = {
        str(x["session_id"]):
            set(
                map(
                    str,
                    x["roles"],
                )
            )
        for x in manifest["systems"]
        if x["split"] == "dev"
    }

    trajectory = []

    for rnd in range(
        0,
        a.rounds + 1,
    ):

        p = (
            root
            / f"states_dev_round_{rnd:02d}.jsonl"
        )

        if not p.exists():
            break

        rows = read_jsonl(p)

        aucs = {}

        for raw in rows:

            st = PublicBeliefState.from_public_dict(
                raw
            )

            sid = str(
                st.session_id
            )

            aucs[sid] = auc_for_state(
                st.role_probs,
                truth[sid],
            )

        trajectory.append({
            "round":
                rnd,

            "mean_auc":
                float(
                    np.mean(
                        list(
                            aucs.values()
                        )
                    )
                ),

            "per_system_auc":
                aucs,
        })

    out = {
        "experiment":
            "belief_targeted_simple5_clean_v1",

        "rounds_completed":
            len(
                trajectory
            ) - 1,

        "trajectory":
            trajectory,

        "initial_mean_auc":
            trajectory[0][
                "mean_auc"
            ],

        "final_mean_auc":
            trajectory[-1][
                "mean_auc"
            ],

        "delta_auc":
            trajectory[-1][
                "mean_auc"
            ]
            -
            trajectory[0][
                "mean_auc"
            ],
    }

    (
        root
        / "final_report.json"
    ).write_text(
        json.dumps(
            out,
            indent=2,
        )
        + "\n"
    )

    print(
        "ROUND  MEAN_AUC"
    )

    for x in trajectory:
        print(
            f"{x['round']:>2}     "
            f"{x['mean_auc']:.6f}"
        )

    print()
    print(
        "INITIAL =",
        f"{out['initial_mean_auc']:.6f}",
    )

    print(
        "FINAL   =",
        f"{out['final_mean_auc']:.6f}",
    )

    print(
        "DELTA   =",
        f"{out['delta_auc']:+.6f}",
    )

    print(
        "SAVED   =",
        root
        / "final_report.json",
    )


if __name__ == "__main__":
    main()
