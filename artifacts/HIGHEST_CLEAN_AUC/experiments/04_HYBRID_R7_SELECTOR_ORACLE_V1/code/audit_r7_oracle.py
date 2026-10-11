from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES


HCA = Path("artifacts/HIGHEST_CLEAN_AUC")

ROOT = (
    HCA
    / "experiments/04_HYBRID_R7_SELECTOR_ORACLE_V1"
    / "data"
)

SRC = (
    HCA
    / "experiments/01_GENERATOR_HYBRID_V2"
    / "data/r1_selector_test"
)

EXPECTED_BELIEF_SHA = (
    "d10c91b7321e5cf08ee754a0cbfcfd7c4cc6a459f4463a6e206816ecd90ae74f"
)


def sha256(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def role_auc(probs, truth):
    truth = set(truth)

    y = np.asarray(
        [
            int(r in truth)
            for r in TRAIN_ROLES
        ],
        dtype=np.int64,
    )

    p = np.asarray(
        [
            float(probs[r])
            for r in TRAIN_ROLES
        ],
        dtype=np.float64,
    )

    return float(
        roc_auc_score(
            y,
            p,
        )
    )


# ------------------------------------------------------------
# Find the EXACT frozen belief by hash.
# ------------------------------------------------------------

search_roots = [
    HCA,
    Path("artifacts/adaptive_v11_active_v1"),
]

belief_candidates = []

for base in search_roots:
    if not base.exists():
        continue

    belief_candidates.extend(
        base.rglob(
            "frozen_belief.joblib"
        )
    )

belief_path = None

for p in belief_candidates:
    try:
        if sha256(p) == EXPECTED_BELIEF_SHA:
            belief_path = p
            break
    except Exception:
        pass

if belief_path is None:
    raise RuntimeError(
        "Could not locate exact frozen belief "
        f"SHA={EXPECTED_BELIEF_SHA}"
    )

print(
    "BELIEF =",
    belief_path,
)

print(
    "BELIEF SHA =",
    sha256(belief_path),
)

belief = FrozenBeliefBundle.load(
    belief_path
)


states = {
    str(x["session_id"]):
        PublicBeliefState.from_public_dict(x)
    for x in read_jsonl(
        ROOT / "states_dev_round_06.jsonl"
    )
}

candidates = read_jsonl(
    ROOT / "candidates_dev_round_07.jsonl"
)

responses = {
    (
        str(x["system_id"]),
        int(x["candidate_index"]),
    ): x
    for x in read_jsonl(
        ROOT / "responses_dev_round_07.jsonl"
    )
}

scores = {}

for x in read_jsonl(
    ROOT / "scored_dev_round_07.jsonl"
):
    sid = str(
        x.get(
            "system_id",
            x.get("session_id", ""),
        )
    )

    idx = int(
        dict(
            x.get("action", {})
        ).get(
            "variant",
            x.get("candidate_index", 0),
        )
    )

    scores[
        (sid, idx)
    ] = x


manifest = json.loads(
    (
        ROOT / "manifest_privileged.json"
    ).read_text(
        encoding="utf-8"
    )
)

truth = {
    str(x["session_id"]):
        tuple(
            map(
                str,
                x["roles"],
            )
        )
    for x in manifest["systems"]
    if x.get("split") == "dev"
}


selected = json.loads(
    (
        ROOT
        / "original_selector_choices.json"
    ).read_text(
        encoding="utf-8"
    )
)

selected = {
    str(k): int(v)
    for k, v in selected.items()
}


grouped = defaultdict(list)

for c in candidates:
    grouped[
        str(c["session_id"])
    ].append(c)


print()
print(
    "===== HYBRID R7 SELECTOR ORACLE GAP ====="
)

system_rows = []


for sid in sorted(grouped):

    state = states[sid]

    base_auc = role_auc(
        state.role_probs,
        truth[sid],
    )

    rows = []

    for c in sorted(
        grouped[sid],
        key=lambda x:
            int(x["candidate_index"]),
    ):

        idx = int(
            c["candidate_index"]
        )

        r = responses[
            (sid, idx)
        ]

        sc = scores[
            (sid, idx)
        ]

        if (
            not c.get("valid")
            or r.get(
                "execution_status"
            ) != "ok"
        ):
            continue

        frame = EvidenceFrame(
            query=str(
                c["query"]
            ),
            answer=str(
                r.get(
                    "final_output",
                    "",
                )
            ),
            v135={
                str(k): float(v)
                for k, v in dict(
                    sc.get(
                        "evidence",
                        {},
                    )
                ).items()
            },
            v134={
                str(k): float(v)
                for k, v in dict(
                    sc.get(
                        "metadata",
                        {},
                    ).get(
                        "evidence_v134",
                        {},
                    )
                ).items()
            },
        )

        new_history = [
            *state.history,
            frame,
        ]

        _, new_roles = role_only_predict(
            belief,
            new_history,
        )

        auc = role_auc(
            new_roles,
            truth[sid],
        )

        rows.append({
            "idx": idx,
            "auc": auc,
            "delta": (
                auc - base_auc
            ),
            "query": str(
                c["query"]
            ),
            "cache_hit": bool(
                r.get(
                    "cache_hit"
                )
            ),
        })


    assert len(rows) == 4, (
        sid,
        len(rows),
    )

    chosen_idx = selected[sid]

    chosen = next(
        x
        for x in rows
        if x["idx"] == chosen_idx
    )

    oracle = max(
        rows,
        key=lambda x:
            x["auc"],
    )

    worst = min(
        rows,
        key=lambda x:
            x["auc"],
    )

    random_mean = float(
        np.mean(
            [
                x["auc"]
                for x in rows
            ]
        )
    )

    order = sorted(
        rows,
        key=lambda x:
            -x["auc"],
    )

    selector_rank = (
        1
        + next(
            i
            for i, x in enumerate(order)
            if x["idx"] == chosen_idx
        )
    )


    print()
    print("=" * 110)

    print(sid)

    print(
        f"R6 BASE AUC       = "
        f"{base_auc:.6f}"
    )

    for x in sorted(
        rows,
        key=lambda z:
            z["idx"],
    ):
        flag = (
            " <<< SELECTED"
            if x["idx"] == chosen_idx
            else ""
        )

        oracle_flag = (
            " <<< ORACLE"
            if x["idx"] == oracle["idx"]
            else ""
        )

        print(
            f"C{x['idx']} "
            f"AUC={x['auc']:.6f} "
            f"DELTA={x['delta']:+.6f} "
            f"cache={x['cache_hit']}"
            f"{flag}{oracle_flag}"
        )

        print(
            "    ",
            x["query"],
        )


    print(
        f"SELECTOR AUC      = "
        f"{chosen['auc']:.6f}"
    )

    print(
        f"RANDOM-MEAN AUC   = "
        f"{random_mean:.6f}"
    )

    print(
        f"ORACLE AUC        = "
        f"{oracle['auc']:.6f}"
    )

    print(
        f"ORACLE GAP        = "
        f"{oracle['auc'] - chosen['auc']:+.6f}"
    )

    print(
        f"SELECTOR TRUE RANK= "
        f"{selector_rank}/4"
    )


    system_rows.append({
        "sid":
            sid,

        "base":
            base_auc,

        "selected":
            chosen["auc"],

        "random":
            random_mean,

        "oracle":
            oracle["auc"],

        "oracle_gap":
            oracle["auc"]
            - chosen["auc"],

        "selector_rank":
            selector_rank,

        "oracle_idx":
            oracle["idx"],

        "selected_idx":
            chosen_idx,
    })


print()
print("=" * 110)
print(
    "===== FIVE-SYSTEM SUMMARY ====="
)

for key in (
    "base",
    "selected",
    "random",
    "oracle",
    "oracle_gap",
):

    vals = [
        x[key]
        for x in system_rows
    ]

    print(
        f"{key:12s} "
        f"mean={np.mean(vals):.6f}"
    )


print(
    "selector_top1 =",
    sum(
        x["selector_rank"] == 1
        for x in system_rows
    ),
    "/",
    len(system_rows),
)

print(
    "selector_top2 =",
    sum(
        x["selector_rank"] <= 2
        for x in system_rows
    ),
    "/",
    len(system_rows),
)


actual_expected = 0.6274509803921569

selected_mean = float(
    np.mean(
        [
            x["selected"]
            for x in system_rows
        ]
    )
)

print()
print(
    "Expected original Hybrid R7 =",
    f"{actual_expected:.6f}",
)

print(
    "Recomputed selected R7      =",
    f"{selected_mean:.6f}",
)

print(
    "Reproduction difference     =",
    f"{selected_mean - actual_expected:+.9f}",
)


out = {
    "mean_base_r6":
        float(
            np.mean(
                [
                    x["base"]
                    for x in system_rows
                ]
            )
        ),

    "mean_selector_r7":
        selected_mean,

    "mean_random_candidate_r7":
        float(
            np.mean(
                [
                    x["random"]
                    for x in system_rows
                ]
            )
        ),

    "mean_oracle_candidate_r7":
        float(
            np.mean(
                [
                    x["oracle"]
                    for x in system_rows
                ]
            )
        ),

    "mean_oracle_gap":
        float(
            np.mean(
                [
                    x["oracle_gap"]
                    for x in system_rows
                ]
            )
        ),

    "selector_top1_fraction":
        float(
            np.mean(
                [
                    x["selector_rank"] == 1
                    for x in system_rows
                ]
            )
        ),

    "systems":
        system_rows,
}

(
    ROOT.parent
    / "reports/oracle_gap.json"
).write_text(
    json.dumps(
        out,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
