#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


EXP = Path("artifacts/eqs_v3_2_closed_loop/pilot")
SRC = Path("artifacts/v14_adaptive_role_v1")
V31_PATH = Path("scripts/v3_1_mas_auc_semantic.py")
CKPT = Path("artifacts/eqs_v3_1_mas_auc")
OUT = Path("artifacts/eqs_v3_4_state_sensitivity")

SEEDS = (7, 17, 29)
ROUNDS = range(1, 7)


def read_jsonl(p):
    out = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def normq(q):
    return " ".join(
        str(q).strip().lower().split()
    )


def load_v31():
    spec = importlib.util.spec_from_file_location(
        "v31audit",
        V31_PATH,
    )

    mod = importlib.util.module_from_spec(spec)

    sys.modules[spec.name] = mod

    spec.loader.exec_module(mod)

    return mod


def load_models(v31):
    models = []
    emb_dim = None

    for seed in SEEDS:
        p = (
            CKPT
            / f"selector_history_belief_query_seed{seed}.pt"
        )

        ck = torch.load(
            p,
            map_location="cpu",
        )

        d = int(ck["emb_dim"])

        if emb_dim is None:
            emb_dim = d

        assert d == emb_dim

        m = v31.AUCSelector(
            d,
            "history_belief_query",
        )

        m.load_state_dict(
            ck["state_dict"],
            strict=True,
        )

        m.eval()

        models.append(m)

    return models, emb_dim


def make_groups(states, bank):
    groups = []

    for i, x in enumerate(states):
        candidates = []

        if i == 0:
            candidates = [
                SimpleNamespace(query=q)
                for q in bank
            ]

        groups.append(
            SimpleNamespace(
                sid=x["session_id"],
                history=x["history"],
                role_probs=x["role_probs"],
                candidates=candidates,
            )
        )

    return groups


def rankvec(scores):
    order = np.argsort(-scores)

    ranks = np.empty(
        len(scores),
        dtype=float,
    )

    ranks[order] = np.arange(
        len(scores),
        dtype=float,
    )

    return ranks


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        a.size < 3
        or np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return 1.0

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def mean(xs):
    return float(
        np.mean(
            np.asarray(
                list(xs),
                dtype=float,
            )
        )
    )


def top_index(scores):
    return int(
        torch.argmax(scores).item()
    )


@torch.no_grad()
def ensemble_score(
    models,
    turns,
    belief,
    qvec,
):
    return torch.stack(
        [
            m.score(
                turns,
                belief,
                qvec,
            ).float()
            for m in models
        ]
    ).mean(0)


v31 = load_v31()

models, emb_dim = load_models(v31)

bank_data = json.loads(
    (EXP / "query_bank.json").read_text()
)

bank = bank_data["queries"]

print(
    f"[LOAD] bank={len(bank)} emb_dim={emb_dim}"
)


# ------------------------------------------------------------
# Load pre-selection HBQ states:
# Round r uses states from r-1.
# ------------------------------------------------------------

states_by_round = {}

all_states = []

for r in ROUNDS:
    p = (
        EXP
        / "history_belief_query"
        / f"states_dev_round_{r-1:02d}.jsonl"
    )

    states = read_jsonl(p)

    states.sort(
        key=lambda x: x["session_id"]
    )

    states_by_round[r] = states

    all_states.extend(states)


# ------------------------------------------------------------
# Build/reuse all embeddings.
# No API calls.
# ------------------------------------------------------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

embed_dir = (
    OUT / "embeddings"
)

embed_dir.mkdir(
    parents=True,
    exist_ok=True,
)

groups = make_groups(
    all_states,
    bank,
)

cache, runtime_dim = (
    v31.load_or_build_embeddings(
        SRC,
        embed_dir,
        groups,
        device,
    )
)

assert int(runtime_dim) == emb_dim

qvec_all = torch.stack(
    [
        cache[
            v31.query_text(q)
        ].float()
        for q in bank
    ]
)

role_order = list(
    v31.TRAIN_ROLES
)


def turns_of(state):
    hist = state["history"]

    if not hist:
        return torch.zeros(
            0,
            emb_dim,
            dtype=torch.float32,
        )

    return torch.stack(
        [
            cache[
                v31.turn_text(h)
            ].float()
            for h in hist
        ]
    )


def belief_of(state):
    return torch.tensor(
        [
            float(
                state[
                    "role_probs"
                ][r]
            )
            for r in role_order
        ],
        dtype=torch.float32,
    )


def available_indices(state):
    used = {
        normq(
            h.get(
                "query",
                ""
            )
        )
        for h in state["history"]
    }

    return [
        i
        for i, q in enumerate(bank)
        if normq(q) not in used
    ]


def score_variant(
    state,
    turns,
    belief,
):
    ids = available_indices(
        state
    )

    qvec = qvec_all[ids]

    scores = ensemble_score(
        models,
        turns,
        belief,
        qvec,
    )

    return ids, scores


rows = []

round_report = []


for r in ROUNDS:
    states = states_by_round[r]

    n = len(states)

    print()
    print("=" * 100)
    print(f"ROUND {r}")
    print("=" * 100)

    true_outputs = []

    for i, state in enumerate(states):
        turns = turns_of(
            state
        )

        belief = belief_of(
            state
        )

        ids, true_scores = score_variant(
            state,
            turns,
            belief,
        )

        true_pick_local = top_index(
            true_scores
        )

        true_pick = ids[
            true_pick_local
        ]

        true_outputs.append(
            {
                "ids": ids,
                "scores": true_scores,
                "pick": true_pick,
                "turns": turns,
                "belief": belief,
            }
        )


    actual_rows = read_jsonl(
        EXP
        / "history_belief_query"
        / f"candidates_dev_round_{r:02d}.jsonl"
    )

    actual = {
        x["session_id"]:
        x["query"]
        for x in actual_rows
    }


    recompute_match = []

    same_zero_history = []
    same_uniform_belief = []
    same_length_only = []
    same_shuffled_history = []
    same_shuffled_belief = []

    corr_zero_history = []
    corr_uniform_belief = []
    corr_length_only = []
    corr_shuffled_history = []
    corr_shuffled_belief = []

    pairwise_rank_corrs = []

    state_score_vectors = []


    # --------------------------------------------------------
    # Cross-system rank similarity under TRUE states
    # --------------------------------------------------------

    for x in true_outputs:
        state_score_vectors.append(
            x["scores"].numpy()
        )

    for a, b in itertools.combinations(
        state_score_vectors,
        2,
    ):
        pairwise_rank_corrs.append(
            corr(
                rankvec(a),
                rankvec(b),
            )
        )


    for i, state in enumerate(states):
        true = true_outputs[i]

        ids = true["ids"]

        true_scores = true["scores"]

        true_pick = true["pick"]

        picked_query = bank[
            true_pick
        ]

        recompute_match.append(
            float(
                picked_query
                == actual[
                    state["session_id"]
                ]
            )
        )


        # ----------------------------------------------------
        # A: ZERO HISTORY CONTENT
        # preserves number of turns
        # ----------------------------------------------------

        zturns = torch.zeros_like(
            true["turns"]
        )

        _, zhist_scores = score_variant(
            state,
            zturns,
            true["belief"],
        )

        zhist_pick = ids[
            top_index(
                zhist_scores
            )
        ]

        same_zero_history.append(
            float(
                zhist_pick
                == true_pick
            )
        )

        corr_zero_history.append(
            corr(
                rankvec(
                    true_scores.numpy()
                ),
                rankvec(
                    zhist_scores.numpy()
                ),
            )
        )


        # ----------------------------------------------------
        # B: UNIFORM BELIEF
        # ----------------------------------------------------

        uniform_belief = torch.full(
            (
                len(role_order),
            ),
            0.5,
            dtype=torch.float32,
        )

        _, ub_scores = score_variant(
            state,
            true["turns"],
            uniform_belief,
        )

        ub_pick = ids[
            top_index(
                ub_scores
            )
        ]

        same_uniform_belief.append(
            float(
                ub_pick
                == true_pick
            )
        )

        corr_uniform_belief.append(
            corr(
                rankvec(
                    true_scores.numpy()
                ),
                rankvec(
                    ub_scores.numpy()
                ),
            )
        )


        # ----------------------------------------------------
        # C: LENGTH ONLY
        # zero-content history + uniform belief
        # ----------------------------------------------------

        _, length_scores = score_variant(
            state,
            zturns,
            uniform_belief,
        )

        length_pick = ids[
            top_index(
                length_scores
            )
        ]

        same_length_only.append(
            float(
                length_pick
                == true_pick
            )
        )

        corr_length_only.append(
            corr(
                rankvec(
                    true_scores.numpy()
                ),
                rankvec(
                    length_scores.numpy()
                ),
            )
        )


        # ----------------------------------------------------
        # D: SHUFFLED HISTORY
        # another system, same round / same turn count
        # ----------------------------------------------------

        j = (i + 1) % n

        shuffled_turns = (
            true_outputs[j]["turns"]
        )

        _, sh_scores = score_variant(
            state,
            shuffled_turns,
            true["belief"],
        )

        sh_pick = ids[
            top_index(
                sh_scores
            )
        ]

        same_shuffled_history.append(
            float(
                sh_pick
                == true_pick
            )
        )

        corr_shuffled_history.append(
            corr(
                rankvec(
                    true_scores.numpy()
                ),
                rankvec(
                    sh_scores.numpy()
                ),
            )
        )


        # ----------------------------------------------------
        # E: SHUFFLED BELIEF
        # ----------------------------------------------------

        shuffled_belief = (
            true_outputs[j]["belief"]
        )

        _, sb_scores = score_variant(
            state,
            true["turns"],
            shuffled_belief,
        )

        sb_pick = ids[
            top_index(
                sb_scores
            )
        ]

        same_shuffled_belief.append(
            float(
                sb_pick
                == true_pick
            )
        )

        corr_shuffled_belief.append(
            corr(
                rankvec(
                    true_scores.numpy()
                ),
                rankvec(
                    sb_scores.numpy()
                ),
            )
        )


        top2 = torch.topk(
            true_scores,
            k=2,
        ).values

        gap = float(
            top2[0] - top2[1]
        )


        rows.append(
            {
                "round": r,
                "session_id":
                state["session_id"],
                "actual_query":
                actual[state["session_id"]],
                "true_top1":
                bank[true_pick],
                "zero_history_top1":
                bank[zhist_pick],
                "uniform_belief_top1":
                bank[ub_pick],
                "length_only_top1":
                bank[length_pick],
                "shuffled_history_top1":
                bank[sh_pick],
                "shuffled_belief_top1":
                bank[sb_pick],
                "true_top1_gap":
                gap,
                "same_zero_history":
                float(
                    zhist_pick
                    == true_pick
                ),
                "same_uniform_belief":
                float(
                    ub_pick
                    == true_pick
                ),
                "same_length_only":
                float(
                    length_pick
                    == true_pick
                ),
                "same_shuffled_history":
                float(
                    sh_pick
                    == true_pick
                ),
                "same_shuffled_belief":
                float(
                    sb_pick
                    == true_pick
                ),
            }
        )


    M = np.stack(
        state_score_vectors,
        axis=0,
    )

    # Remove each state's global score offset.
    Mc = (
        M
        - M.mean(
            axis=1,
            keepdims=True,
        )
    )

    state_effect_std = float(
        np.mean(
            np.std(
                Mc,
                axis=0,
            )
        )
    )

    top1_gaps = []

    for scores in state_score_vectors:
        vals = np.sort(
            scores
        )

        top1_gaps.append(
            float(
                vals[-1]
                - vals[-2]
            )
        )


    true_queries = [
        bank[
            x["pick"]
        ]
        for x in true_outputs
    ]

    length_queries = []

    for i, state in enumerate(states):
        ids = true_outputs[i]["ids"]

        zturns = torch.zeros_like(
            true_outputs[i]["turns"]
        )

        uniform = torch.full(
            (
                len(role_order),
            ),
            0.5,
        )

        _, scores = score_variant(
            state,
            zturns,
            uniform,
        )

        pick = ids[
            top_index(
                scores
            )
        ]

        length_queries.append(
            bank[pick]
        )


    summary = {
        "round": r,

        "unique_true_top1":
        len(set(true_queries)),

        "unique_length_only_top1":
        len(set(length_queries)),

        "recomputed_matches_actual":
        mean(recompute_match),

        "same_top1_zero_history_content":
        mean(same_zero_history),

        "same_top1_uniform_belief":
        mean(same_uniform_belief),

        "same_top1_length_only":
        mean(same_length_only),

        "same_top1_shuffled_history":
        mean(same_shuffled_history),

        "same_top1_shuffled_belief":
        mean(same_shuffled_belief),

        "rank_corr_zero_history_content":
        mean(corr_zero_history),

        "rank_corr_uniform_belief":
        mean(corr_uniform_belief),

        "rank_corr_length_only":
        mean(corr_length_only),

        "rank_corr_shuffled_history":
        mean(corr_shuffled_history),

        "rank_corr_shuffled_belief":
        mean(corr_shuffled_belief),

        "mean_pairwise_rank_corr_across_systems":
        mean(pairwise_rank_corrs),

        "mean_state_induced_score_std":
        state_effect_std,

        "mean_true_top1_gap":
        mean(top1_gaps),

        "state_effect_to_top1_gap_ratio":
        (
            state_effect_std
            / max(
                mean(top1_gaps),
                1e-12,
            )
        ),

        "true_top1_query":
        true_queries[0],

        "length_only_top1_query":
        length_queries[0],
    }

    round_report.append(
        summary
    )


    print(
        f"unique TRUE top1          : "
        f"{summary['unique_true_top1']}"
    )

    print(
        f"actual reproduction       : "
        f"{summary['recomputed_matches_actual']:.3f}"
    )

    print(
        f"same after ZERO history   : "
        f"{summary['same_top1_zero_history_content']:.3f}"
    )

    print(
        f"same after UNIFORM belief : "
        f"{summary['same_top1_uniform_belief']:.3f}"
    )

    print(
        f"same under LENGTH ONLY    : "
        f"{summary['same_top1_length_only']:.3f}"
    )

    print(
        f"same after SHUFFLE history: "
        f"{summary['same_top1_shuffled_history']:.3f}"
    )

    print(
        f"same after SHUFFLE belief : "
        f"{summary['same_top1_shuffled_belief']:.3f}"
    )

    print(
        f"rank corr LENGTH ONLY     : "
        f"{summary['rank_corr_length_only']:.4f}"
    )

    print(
        f"rank corr across systems  : "
        f"{summary['mean_pairwise_rank_corr_across_systems']:.4f}"
    )

    print(
        f"state score variation     : "
        f"{summary['mean_state_induced_score_std']:.6f}"
    )

    print(
        f"top1-top2 score gap       : "
        f"{summary['mean_true_top1_gap']:.6f}"
    )

    print(
        f"state/gap ratio           : "
        f"{summary['state_effect_to_top1_gap_ratio']:.4f}"
    )

    print(
        "TRUE Q: "
        + summary[
            "true_top1_query"
        ]
    )

    print(
        "LENGTH-ONLY Q: "
        + summary[
            "length_only_top1_query"
        ]
    )


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

(
    OUT / "summary.json"
).write_text(
    json.dumps(
        {
            "experiment":
            "V3.4 HBQ state sensitivity audit",

            "n_systems":
            len(
                states_by_round[1]
            ),

            "query_bank":
            len(bank),

            "rounds":
            round_report,
        },
        indent=2,
    )
)

(
    OUT / "per_session.jsonl"
).write_text(
    "".join(
        json.dumps(x)
        + "\n"
        for x in rows
    )
)


print()
print("=" * 100)
print("FINAL DIAGNOSIS")
print("=" * 100)

length_same = mean(
    [
        x[
            "same_top1_length_only"
        ]
        for x in round_report
    ]
)

shuffle_hist_same = mean(
    [
        x[
            "same_top1_shuffled_history"
        ]
        for x in round_report
    ]
)

shuffle_belief_same = mean(
    [
        x[
            "same_top1_shuffled_belief"
        ]
        for x in round_report
    ]
)

rank_across = mean(
    [
        x[
            "mean_pairwise_rank_corr_across_systems"
        ]
        for x in round_report
    ]
)

print(
    f"Mean same Top1 under LENGTH ONLY : "
    f"{length_same:.3f}"
)

print(
    f"Mean same Top1 after history swap: "
    f"{shuffle_hist_same:.3f}"
)

print(
    f"Mean same Top1 after belief swap : "
    f"{shuffle_belief_same:.3f}"
)

print(
    f"Mean cross-system rank corr      : "
    f"{rank_across:.4f}"
)

print()

if (
    length_same >= 0.80
    and rank_across >= 0.95
):
    print(
        "DIAGNOSIS: STRONG POLICY COLLAPSE."
    )

    print(
        "HBQ ranking is driven primarily by "
        "global/round-level structure rather than "
        "system-specific evidence."
    )

elif (
    shuffle_hist_same >= 0.80
    and shuffle_belief_same >= 0.80
):
    print(
        "DIAGNOSIS: WEAK STATE SENSITIVITY."
    )

    print(
        "History/belief perturbations rarely change "
        "the selected probe."
    )

else:
    print(
        "DIAGNOSIS: STATE AFFECTS SCORES, but argmax "
        "may still be dominated by a global query prior."
    )

print()
print(
    "Saved:",
    OUT / "summary.json",
)
