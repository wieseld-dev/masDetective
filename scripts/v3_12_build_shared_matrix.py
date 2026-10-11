from __future__ import annotations

import itertools
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


ROOT = Path("artifacts/eqs_v3_11e_final_shared_panel")
V14 = Path("artifacts/v14_adaptive_role_v1")

MANIFEST = V14 / "manifest_privileged.json"
BELIEF = V14 / "frozen_belief.joblib"

OUT_ROWS = ROOT / "state_query_rows.jsonl"
OUT_SUMMARY = ROOT / "signal_summary.json"


def auc(probs, true_roles):
    truth = set(map(str, true_roles))

    y = np.asarray(
        [int(r in truth) for r in TRAIN_ROLES],
        dtype=np.int64,
    )

    p = np.asarray(
        [float(probs[r]) for r in TRAIN_ROLES],
        dtype=np.float64,
    )

    return float(roc_auc_score(y, p))


manifest = json.loads(MANIFEST.read_text())

truth = {
    str(x["session_id"]): tuple(map(str, x["roles"]))
    for x in manifest["systems"]
    if x["split"] == "train"
}

assert len(truth) == 24


belief = FrozenBeliefBundle.load(BELIEF)


# ------------------------------------------------------------
# Candidates
# ------------------------------------------------------------

candidates = read_jsonl(
    ROOT / "candidates_train_round_01.jsonl"
)

cand_by = {
    (
        str(x["session_id"]),
        int(x["candidate_index"]),
    ): x
    for x in candidates
}

query_text = {}

for x in candidates:
    query_text[int(x["candidate_index"])] = str(x["query"])

assert len(candidates) == 576
assert len(query_text) == 24


# ------------------------------------------------------------
# Responses
# ------------------------------------------------------------

responses = read_jsonl(
    ROOT / "responses_train_round_01.jsonl"
)

resp_by = {
    (
        str(x["session_id"]),
        int(x["candidate_index"]),
    ): x
    for x in responses
}

assert len(responses) == 576


# ------------------------------------------------------------
# Frozen sensor scores
# ------------------------------------------------------------

scored = read_jsonl(
    ROOT / "scored_train_round_01.jsonl"
)

score_by = {}

for x in scored:

    sid = str(
        x.get(
            "system_id",
            x.get("session_id", ""),
        )
    )

    action = dict(
        x.get("action", {})
        or {}
    )

    idx = int(
        action.get(
            "variant",
            x.get("candidate_index", -1),
        )
    )

    score_by[(sid, idx)] = x


assert len(score_by) == 576


# ------------------------------------------------------------
# Attacker states:
# states after rounds 1..5
# correspond to selecting probes for rounds 2..6.
# ------------------------------------------------------------

states = {}

for sr in range(1, 6):

    path = (
        V14
        / f"states_train_round_{sr:02d}.jsonl"
    )

    xs = read_jsonl(path)

    assert len(xs) == 24, (path, len(xs))

    for x in xs:
        sid = str(x["session_id"])

        states[(sid, sr)] = (
            PublicBeliefState.from_public_dict(x)
        )


# ------------------------------------------------------------
# Counterfactual replay
# ------------------------------------------------------------

rows = []


for sid in sorted(truth):

    true_roles = truth[sid]

    for state_round in range(1, 6):

        state = states[
            (sid, state_round)
        ]

        before = auc(
            state.role_probs,
            true_roles,
        )

        for qidx in range(24):

            c = cand_by[
                (sid, qidx)
            ]

            rr = resp_by[
                (sid, qidx)
            ]

            if rr.get("execution_status") != "ok":
                raise RuntimeError(
                    f"bad response {sid=} {qidx=}: "
                    f"{rr.get('execution_status')}"
                )

            sc = score_by[
                (sid, qidx)
            ]

            frame = EvidenceFrame(
                query=str(c["query"]),

                answer=str(
                    rr.get(
                        "final_output",
                        "",
                    )
                ),

                v135={
                    str(k): float(v)
                    for k, v in dict(
                        sc.get("evidence", {})
                        or {}
                    ).items()
                },

                v134={
                    str(k): float(v)
                    for k, v in dict(
                        dict(
                            sc.get("metadata", {})
                            or {}
                        ).get(
                            "evidence_v134",
                            {},
                        )
                        or {}
                    ).items()
                },
            )

            _, new_roles = role_only_predict(
                belief,
                [
                    *state.history,
                    frame,
                ],
            )

            after = auc(
                new_roles,
                true_roles,
            )

            rows.append(
                {
                    "session_id": sid,

                    "round":
                        state_round + 1,

                    "state_round":
                        state_round,

                    "query_index":
                        qidx,

                    "query":
                        str(c["query"]),

                    "assigned_role":
                        c.get(
                            "assigned_role"
                        ),

                    "history_len":
                        len(state.history),

                    "belief": {
                        r: float(
                            state.role_probs[r]
                        )
                        for r in TRAIN_ROLES
                    },

                    "before_auc":
                        before,

                    "after_auc":
                        after,

                    "delta_auc":
                        after - before,
                }
            )


assert len(rows) == 2880


with OUT_ROWS.open("w") as f:

    for x in rows:
        f.write(
            json.dumps(x)
            + "\n"
        )


# ============================================================
# SIGNAL ANALYSIS
# ============================================================

by_state = defaultdict(dict)
by_system = defaultdict(lambda: defaultdict(dict))
by_query = defaultdict(list)


for x in rows:

    sid = x["session_id"]
    rd = int(x["round"])
    q = int(x["query_index"])
    u = float(x["delta_auc"])

    by_state[(sid, rd)][q] = u
    by_system[sid][rd][q] = u
    by_query[q].append(u)


assert len(by_state) == 120
assert {len(x) for x in by_state.values()} == {24}


# One globally fixed query.
global_means = {
    q: float(np.mean(v))
    for q, v in by_query.items()
}

best_global_q = max(
    global_means,
    key=global_means.get,
)

global_static = global_means[
    best_global_q
]


# Best static query separately for each MAS.
per_system_static = []

# Best query separately at each state.
contextual = []

unique_per_system = []


for sid, rounds in by_system.items():

    qmeans = {
        q: float(
            np.mean([
                rounds[r][q]
                for r in sorted(rounds)
            ])
        )
        for q in range(24)
    }

    per_system_static.append(
        max(qmeans.values())
    )

    oracle_qs = []

    for rd in sorted(rounds):

        bq = max(
            rounds[rd],
            key=rounds[rd].get,
        )

        oracle_qs.append(bq)

        contextual.append(
            rounds[rd][bq]
        )

    unique_per_system.append(
        len(set(oracle_qs))
    )


per_system_static_mean = float(
    np.mean(per_system_static)
)

contextual_mean = float(
    np.mean(contextual)
)


# Same-query utility variability.
q_stds = [
    float(np.std(x))
    for x in by_query.values()
]

q_ranges = [
    float(
        np.max(x)
        - np.min(x)
    )
    for x in by_query.values()
]


# Utility-ranking correlation across states.
state_vectors = []

for key in sorted(by_state):

    state_vectors.append(
        np.asarray(
            [
                by_state[key][q]
                for q in range(24)
            ],
            dtype=float,
        )
    )


corrs = []

for a, b in itertools.combinations(
    state_vectors,
    2,
):

    ra = np.empty(24)
    rb = np.empty(24)

    ra[
        np.argsort(-a)
    ] = np.arange(24)

    rb[
        np.argsort(-b)
    ] = np.arange(24)

    corrs.append(
        float(
            np.corrcoef(
                ra,
                rb,
            )[0, 1]
        )
    )


summary = {
    "systems":
        24,

    "states_per_system":
        5,

    "shared_queries":
        24,

    "state_query_labels":
        len(rows),

    "global_static_delta_auc":
        float(global_static),

    "best_global_query_index":
        int(best_global_q),

    "best_global_query":
        query_text[best_global_q],

    "per_system_static_oracle_delta_auc":
        per_system_static_mean,

    "contextual_oracle_delta_auc":
        contextual_mean,

    "adaptive_headroom_over_global_static":
        contextual_mean
        - global_static,

    "adaptive_headroom_over_per_system_static":
        contextual_mean
        - per_system_static_mean,

    "mean_unique_oracle_queries_per_system":
        float(
            np.mean(
                unique_per_system
            )
        ),

    "mean_same_query_utility_std":
        float(np.mean(q_stds)),

    "mean_same_query_utility_range":
        float(np.mean(q_ranges)),

    "fraction_queries_range_ge_005":
        float(
            np.mean([
                x >= 0.05
                for x in q_ranges
            ])
        ),

    "fraction_queries_range_ge_010":
        float(
            np.mean([
                x >= 0.10
                for x in q_ranges
            ])
        ),

    "mean_cross_state_utility_rank_corr":
        float(np.mean(corrs)),
}


OUT_SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


print()
print("=" * 100)
print("V3.12 SHARED STATE×QUERY MATRIX")
print("=" * 100)

for k, v in summary.items():

    if isinstance(v, float):
        print(f"{k}: {v:.6f}")

    else:
        print(f"{k}: {v}")


print()
print("ROWS:", OUT_ROWS)
print("SUMMARY:", OUT_SUMMARY)
