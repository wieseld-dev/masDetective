#!/usr/bin/env python3

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.role_belief import role_only_predict
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/v14_adaptive_role_v1")
MANIFEST = ROOT / "manifest_privileged.json"
BELIEF_PATH = ROOT / "frozen_belief.joblib"
OUT = Path("artifacts/eqs_v3_6_counterfactual")
ROUNDS = range(1, 7)


def safe_auc(probs, true_roles):
    truth = set(true_roles)

    y = np.asarray(
        [int(r in truth) for r in TRAIN_ROLES],
        dtype=int,
    )

    p = np.asarray(
        [float(probs[r]) for r in TRAIN_ROLES],
        dtype=float,
    )

    pos = p[y == 1]
    neg = p[y == 0]

    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    wins = 0.0

    for a in pos:
        for b in neg:
            if a > b:
                wins += 1.0
            elif a == b:
                wins += 0.5

    return float(
        wins / (len(pos) * len(neg))
    )


def normq(q):
    return " ".join(
        str(q).strip().lower().split()
    )


manifest = json.loads(
    MANIFEST.read_text()
)

truth = {
    str(x["session_id"]):
    tuple(map(str, x["roles"]))
    for x in manifest["systems"]
    if x.get("split") == "train"
}

belief = FrozenBeliefBundle.load(
    BELIEF_PATH
)

# ------------------------------------------------------------
# Load attacker states s_{t-1}.
# ------------------------------------------------------------

states = defaultdict(dict)

for r in ROUNDS:
    p = ROOT / f"states_train_round_{r-1:02d}.jsonl"

    for x in read_jsonl(p):
        sid = str(x["session_id"])

        if sid not in truth:
            continue

        states[sid][r] = (
            PublicBeliefState.from_public_dict(x)
        )


# ------------------------------------------------------------
# Build one reusable EvidenceFrame for every real executed query
# for each TRAIN MAS.
#
# Important:
# attacker state changes between rounds,
# but the target MAS execution of q is independent of our
# private belief/history. Therefore the same observed evidence
# can be replayed against another attacker state.
# ------------------------------------------------------------

frames = defaultdict(dict)

for r in ROUNDS:
    candidates = read_jsonl(
        ROOT / f"candidates_train_round_{r:02d}.jsonl"
    )

    responses = read_jsonl(
        ROOT / f"responses_train_round_{r:02d}.jsonl"
    )

    scored = read_jsonl(
        ROOT / f"scored_train_round_{r:02d}.jsonl"
    )

    resp_by = {
        (
            str(x["session_id"]),
            int(x["candidate_index"]),
        ): x
        for x in responses
    }

    score_by = {}

    for x in scored:
        sid = str(
            x.get(
                "system_id",
                x.get("session_id", "")
            )
        )

        idx = int(
            dict(
                x.get("action", {})
            ).get(
                "variant",
                x.get("candidate_index", 0)
            )
        )

        if sid:
            score_by[(sid, idx)] = x

    for c in candidates:
        sid = str(c["session_id"])

        if sid not in truth:
            continue

        if not c.get("valid"):
            continue

        idx = int(
            c["candidate_index"]
        )

        response = resp_by.get(
            (sid, idx)
        )

        score = score_by.get(
            (sid, idx)
        )

        if (
            response is None
            or score is None
            or response.get("execution_status")
            != "ok"
        ):
            continue

        q = str(c["query"]).strip()

        if not q:
            continue

        key = normq(q)

        # dedupe exact queries within same MAS
        if key in frames[sid]:
            continue

        frame = EvidenceFrame(
            query=q,
            answer=str(
                response.get(
                    "final_output",
                    ""
                )
            ),
            v135={
                str(k): float(v)
                for k, v
                in dict(
                    score.get(
                        "evidence",
                        {}
                    )
                ).items()
            },
            v134={
                str(k): float(v)
                for k, v
                in dict(
                    score.get(
                        "metadata",
                        {}
                    ).get(
                        "evidence_v134",
                        {}
                    )
                ).items()
            },
        )

        frames[sid][key] = {
            "query": q,
            "frame": frame,
        }


print("=" * 100)
print("V3.6 COUNTERFACTUAL STATE × QUERY SIGNAL AUDIT")
print("=" * 100)
print("NO OPENAI")
print("NO MAS EXECUTIONS")
print()


system_rows = []
all_state_query_rows = []

for sid in sorted(truth):
    if sid not in states:
        continue

    rounds = sorted(
        states[sid]
    )

    pool = list(
        frames[sid].values()
    )

    if (
        len(rounds) < 2
        or len(pool) < 2
    ):
        continue

    utility_matrix = []

    query_names = [
        x["query"]
        for x in pool
    ]

    for r in rounds:
        state = states[sid][r]

        before = safe_auc(
            dict(state.role_probs),
            truth[sid],
        )

        row = []

        for qi, item in enumerate(pool):
            _, new_roles = role_only_predict(
                belief,
                [
                    *state.history,
                    item["frame"],
                ],
            )

            after = safe_auc(
                new_roles,
                truth[sid],
            )

            delta = float(
                after - before
            )

            row.append(delta)

            all_state_query_rows.append(
                {
                    "session_id": sid,
                    "round": r,
                    "query_index": qi,
                    "query": item["query"],
                    "auc_before": before,
                    "auc_after": after,
                    "delta_auc": delta,
                }
            )

        utility_matrix.append(row)

    U = np.asarray(
        utility_matrix,
        dtype=float,
    )

    # Contextual oracle:
    # independently choose the best q for each state.
    contextual_oracle = float(
        np.mean(
            np.max(
                U,
                axis=1,
            )
        )
    )

    # Global/static oracle:
    # choose ONE q using mean utility across all states,
    # then use that same q everywhere.
    global_mean = np.mean(
        U,
        axis=0,
    )

    global_idx = int(
        np.argmax(global_mean)
    )

    global_static = float(
        np.mean(
            U[:, global_idx]
        )
    )

    adaptive_headroom = (
        contextual_oracle
        - global_static
    )

    oracle_indices = np.argmax(
        U,
        axis=1,
    )

    unique_oracles = int(
        len(
            set(
                map(
                    int,
                    oracle_indices,
                )
            )
        )
    )

    # How much does utility of SAME q vary across states?
    q_std = np.std(
        U,
        axis=0,
    )

    q_range = np.ptp(
        U,
        axis=0,
    )

    # Ranking similarity between states.
    corrs = []

    for i in range(
        len(rounds)
    ):
        for j in range(
            i + 1,
            len(rounds),
        ):
            c = spearmanr(
                U[i],
                U[j],
            ).correlation

            if (
                c is not None
                and np.isfinite(c)
            ):
                corrs.append(
                    float(c)
                )

    mean_rank_corr = (
        float(np.mean(corrs))
        if corrs
        else float("nan")
    )

    # fraction of queries for which state changes utility meaningfully
    fraction_range_005 = float(
        np.mean(
            q_range >= 0.05
        )
    )

    fraction_range_010 = float(
        np.mean(
            q_range >= 0.10
        )
    )

    row = {
        "session_id": sid,
        "states": len(rounds),
        "queries": len(pool),
        "contextual_oracle_delta_auc":
        contextual_oracle,
        "global_static_delta_auc":
        global_static,
        "adaptive_headroom":
        adaptive_headroom,
        "unique_oracle_queries":
        unique_oracles,
        "mean_same_query_utility_std":
        float(np.mean(q_std)),
        "mean_same_query_utility_range":
        float(np.mean(q_range)),
        "fraction_queries_range_ge_005":
        fraction_range_005,
        "fraction_queries_range_ge_010":
        fraction_range_010,
        "mean_cross_state_utility_rank_corr":
        mean_rank_corr,
        "global_best_query":
        query_names[global_idx],
    }

    system_rows.append(row)

    print(
        f"{sid} "
        f"states={len(rounds)} "
        f"queries={len(pool)} "
        f"ctxOracle={contextual_oracle:+.4f} "
        f"global={global_static:+.4f} "
        f"HEADROOM={adaptive_headroom:+.4f} "
        f"uniqueOracle={unique_oracles} "
        f"qRange={np.mean(q_range):.4f} "
        f"rankCorr={mean_rank_corr:.3f}"
    )


if not system_rows:
    raise RuntimeError(
        "No usable counterfactual systems"
    )


def M(k):
    vals = [
        float(x[k])
        for x in system_rows
        if np.isfinite(
            float(x[k])
        )
    ]

    return float(
        np.mean(vals)
    )


aggregate = {
    "systems":
    len(system_rows),

    "mean_queries_per_system":
    M("queries"),

    "mean_contextual_oracle_delta_auc":
    M(
        "contextual_oracle_delta_auc"
    ),

    "mean_global_static_delta_auc":
    M(
        "global_static_delta_auc"
    ),

    "mean_adaptive_headroom":
    M("adaptive_headroom"),

    "mean_unique_oracle_queries":
    M("unique_oracle_queries"),

    "mean_same_query_utility_std":
    M("mean_same_query_utility_std"),

    "mean_same_query_utility_range":
    M("mean_same_query_utility_range"),

    "mean_fraction_queries_range_ge_005":
    M(
        "fraction_queries_range_ge_005"
    ),

    "mean_fraction_queries_range_ge_010":
    M(
        "fraction_queries_range_ge_010"
    ),

    "mean_cross_state_utility_rank_corr":
    M(
        "mean_cross_state_utility_rank_corr"
    ),
}


print()
print("=" * 100)
print("AGGREGATE")
print("=" * 100)

for k, v in aggregate.items():
    if isinstance(v, float):
        print(
            f"{k}: {v:.6f}"
        )
    else:
        print(
            f"{k}: {v}"
        )


print()
print("=" * 100)
print("INTERPRETATION")
print("=" * 100)

headroom = aggregate[
    "mean_adaptive_headroom"
]

unique_oracle = aggregate[
    "mean_unique_oracle_queries"
]

rank_corr = aggregate[
    "mean_cross_state_utility_rank_corr"
]

if (
    headroom >= 0.02
    and unique_oracle >= 2.0
):
    print(
        "PASS: strong state-dependent opportunity exists."
    )

    print(
        "The same MAS requires different probes at different "
        "attacker states; adaptive ranking is learnable."
    )

elif headroom >= 0.01:
    print(
        "MIXED: some adaptive headroom exists, but it is modest."
    )

    print(
        "A state-conditioned selector may help, but gains "
        "will likely be limited."
    )

else:
    print(
        "FAIL: very little contextual oracle advantage."
    )

    print(
        "Under the current sensor/query distribution, "
        "a global query policy is nearly sufficient."
    )


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

result = {
    "experiment":
    "V3.6 counterfactual State x Query signal audit",

    "assumption":
    (
        "Target MAS execution for a given system/query is "
        "independent of the attacker's private inference history; "
        "therefore cached evidence can be replayed against multiple "
        "attacker belief states."
    ),

    "aggregate":
    aggregate,

    "systems":
    system_rows,
}

(
    OUT / "signal_summary.json"
).write_text(
    json.dumps(
        result,
        indent=2,
    )
)

with (
    OUT
    / "state_query_rows.jsonl"
).open("w") as f:
    for x in all_state_query_rows:
        f.write(
            json.dumps(x)
            + "\n"
        )


print()
print(
    "SAVED:",
    OUT / "signal_summary.json",
)
