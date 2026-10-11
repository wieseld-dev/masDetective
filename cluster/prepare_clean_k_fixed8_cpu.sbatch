#!/bin/bash
#SBATCH --job-name=prep-k8
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/prep-k8-%j.out
#SBATCH --error=logs/prep-k8-%j.err

set -euo pipefail

cd "$HOME/mas_rl_detective"

source /storage/modules/packages/anaconda/etc/profile.d/conda.sh
conda activate masdetective

export PYTHONPATH="$HOME/mas_rl_detective:${PYTHONPATH:-}"

python -u - <<'PY'
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


SEED = 20260923

SRC = Path(
    "artifacts/adaptive_v11_curriculum_v2"
)

OUT = Path(
    "artifacts/clean_k_fixed8_v1"
)

BELIEF = Path(
    "artifacts/adaptive_v11_active_v1/frozen_belief.joblib"
)

KS = range(3, 9)
STYLES = ("s1", "s2", "s3")

# 14 per style => 42 systems per K.
PER_STYLE = 14


def write_jsonl(path, rows):
    with path.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


def old_compositions():
    used = set()

    paths = list(
        SRC.glob(
            "stage_*/manifest_train_cycle_*.json"
        )
    )

    paths += list(
        SRC.glob(
            "stage_*/manifest_dev.json"
        )
    )

    for p in paths:
        obj = json.loads(
            p.read_text()
        )

        for x in obj.get(
            "systems",
            [],
        ):
            roles = x.get(
                "roles"
            )

            if roles:
                used.add(
                    tuple(
                        sorted(
                            map(
                                str,
                                roles,
                            )
                        )
                    )
                )

    return used


def choose_roles(
    k,
    role_counts,
    forbidden,
    selected,
    rng,
):
    best = None

    # Large random search for a fresh composition
    # that also keeps role prevalence balanced.
    for _ in range(5000):

        roles = tuple(
            rng.sample(
                list(TRAIN_ROLES),
                k,
            )
        )

        sig = tuple(
            sorted(
                roles
            )
        )

        if (
            sig in forbidden
            or sig in selected
        ):
            continue

        score = (
            sum(
                role_counts[r]
                for r in roles
            ),
            max(
                role_counts[r]
                for r in roles
            ),
            rng.random(),
        )

        if (
            best is None
            or score < best[0]
        ):
            best = (
                score,
                roles,
            )

    if best is None:
        raise RuntimeError(
            f"Cannot find fresh composition K={k}"
        )

    return best[1]


def main():

    rng = random.Random(
        SEED
    )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    forbidden = old_compositions()
    selected = set()

    systems = []
    split_info = {}

    role_counts = {
        k: Counter({
            r: 0
            for r in TRAIN_ROLES
        })
        for k in KS
    }

    seq = 0

    for k in KS:

        for style in STYLES:

            for j in range(
                PER_STYLE
            ):

                roles = choose_roles(
                    k,
                    role_counts[k],
                    forbidden,
                    selected,
                    rng,
                )

                sig = tuple(
                    sorted(
                        roles
                    )
                )

                selected.add(
                    sig
                )

                role_counts[k].update(
                    roles
                )

                # Within EACH style and EACH K:
                # 8 train + 3 val + 3 test.
                if j < 8:
                    k_split = "train"

                elif j < 11:
                    k_split = "val"

                else:
                    k_split = "test"

                sid = (
                    f"cleank8_k{k}_{style}_"
                    f"{j:02d}_{seq:04d}"
                )

                seq += 1

                systems.append({
                    "session_id":
                        sid,

                    # Use DEV path during rollout:
                    # selector epsilon is zero.
                    "split":
                        "dev",

                    "k":
                        k,

                    # CRITICAL CONTROL:
                    # identical execution mode for all K.
                    "mode":
                        "fanout2",

                    "style":
                        style,

                    "roles":
                        list(
                            roles
                        ),
                })

                split_info[
                    sid
                ] = {
                    "k_split":
                        k_split,

                    "K":
                        k,

                    "style":
                        style,

                    "mode":
                        "fanout2",

                    "roles":
                        list(
                            roles
                        ),
                }

    base = {
        "version":
            "clean-k-fixed8-v1",

        "seed":
            SEED,

        "notes": (
            "Fresh K3-K8 fixed-budget benchmark. "
            "All systems fanout2; styles balanced; "
            "8 black-box rounds planned."
        ),

        "systems":
            systems,
    }

    digest = hashlib.sha256(
        json.dumps(
            base,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    manifest = dict(
        base
    )

    manifest[
        "sha256"
    ] = digest

    (
        OUT
        / "manifest_privileged.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
    )

    (
        OUT
        / "K_SPLIT.json"
    ).write_text(
        json.dumps(
            split_info,
            indent=2,
            sort_keys=True,
        )
    )

    # ---------------------------------------------------------
    # Identical PUBLIC initial state for every MAS.
    # No true K / roles are passed into this state.
    # ---------------------------------------------------------

    bundle = (
        FrozenBeliefBundle.load(
            BELIEF
        )
    )

    kp, rp = bundle.initial()

    states = []

    for x in systems:

        st = PublicBeliefState(
            session_id=x[
                "session_id"
            ],
            split="dev",
            round_index=0,
            k_probs=dict(
                kp
            ),
            role_probs=dict(
                rp
            ),
            history=[],
        )

        st.validate(
            TRAIN_ROLES,
            bundle.ks,
        )

        states.append(
            st.to_public_dict()
        )

    write_jsonl(
        OUT
        / "states_dev_round_00.jsonl",
        states,
    )

    # ---------------------------------------------------------
    # AUDIT
    # ---------------------------------------------------------

    by_k = Counter(
        x["k"]
        for x in systems
    )

    by_split_k = Counter(
        (
            split_info[
                x["session_id"]
            ][
                "k_split"
            ],
            x["k"],
        )
        for x in systems
    )

    by_style_k = Counter(
        (
            x["style"],
            x["k"],
        )
        for x in systems
    )

    by_mode = Counter(
        x["mode"]
        for x in systems
    )

    compositions = [
        tuple(
            sorted(
                x["roles"]
            )
        )
        for x in systems
    ]

    overlap = (
        set(
            compositions
        )
        & forbidden
    )

    prior_sigs = {
        json.dumps(
            {
                "k_probs":
                    x[
                        "k_probs"
                    ],

                "role_probs":
                    x[
                        "role_probs"
                    ],

                "round_index":
                    x[
                        "round_index"
                    ],

                "history":
                    x[
                        "history"
                    ],
            },
            sort_keys=True,
        )
        for x in states
    }

    print(
        "=" * 100
    )

    print(
        "CLEAN K FIXED-8 — PREPARATION RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        f"TOTAL SYSTEMS: {len(systems)}"
    )

    print(
        "\nBY K:"
    )

    for k in KS:
        print(
            f"K={k}: {by_k[k]}"
        )

    print(
        "\nSPLIT x K:"
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        for k in KS:
            print(
                f"{split:5s} "
                f"K={k}: "
                f"{by_split_k[(split,k)]}"
            )

    print(
        "\nSTYLE x K:"
    )

    for style in STYLES:
        for k in KS:
            print(
                f"{style} "
                f"K={k}: "
                f"{by_style_k[(style,k)]}"
            )

    print(
        "\nROLE BALANCE:"
    )

    for k in KS:

        vals = [
            role_counts[k][r]
            for r in TRAIN_ROLES
        ]

        print(
            f"K={k}: "
            f"min={min(vals)} "
            f"max={max(vals)} "
            f"mean={sum(vals)/len(vals):.2f} "
            f"spread={max(vals)-min(vals)}"
        )

    checks = [
        (
            "252 systems",
            len(systems)
            == 252,
        ),

        (
            "42 systems per K",
            all(
                by_k[k]
                == 42
                for k in KS
            ),
        ),

        (
            "24/9/9 per K",
            all(
                by_split_k[
                    ("train",k)
                ]
                == 24
                and
                by_split_k[
                    ("val",k)
                ]
                == 9
                and
                by_split_k[
                    ("test",k)
                ]
                == 9
                for k in KS
            ),
        ),

        (
            "14 per style per K",
            all(
                by_style_k[
                    (style,k)
                ]
                == 14
                for style
                in STYLES
                for k
                in KS
            ),
        ),

        (
            "fanout2 only",
            by_mode
            == Counter({
                "fanout2":
                    252
            }),
        ),

        (
            "unique compositions",
            len(
                compositions
            )
            ==
            len(
                set(
                    compositions
                )
            ),
        ),

        (
            "zero old-composition overlap",
            len(
                overlap
            )
            == 0,
        ),

        (
            "252 public initial states",
            len(
                states
            )
            == 252,
        ),

        (
            "identical public prior",
            len(
                prior_sigs
            )
            == 1,
        ),
    ]

    print(
        "\n" + "=" * 100
    )

    print(
        "FINAL CHECKS"
    )

    print(
        "=" * 100
    )

    for name, ok in checks:
        print(
            f"{name:40s}: "
            f"{'PASS' if ok else 'FAIL'}"
        )

    overall = all(
        ok
        for _, ok in checks
    )

    print()

    print(
        "OVERALL:",
        "PASS"
        if overall
        else "FAIL",
    )

    print(
        "OLD COMPOSITION OVERLAP:",
        len(
            overlap
        ),
    )

    print(
        "UNIQUE PUBLIC PRIORS:",
        len(
            prior_sigs
        ),
    )

    print(
        "FINAL INTERNAL TEST USED: NO"
    )

    if not overall:
        raise SystemExit(
            "PREPARATION AUDIT FAILED"
        )


if __name__ == "__main__":
    main()
PY
