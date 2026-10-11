#!/usr/bin/env python3

# ============================================================
# STRUCTURAL OBSERVABILITY / ACTIVE GRAPH TOMOGRAPHY — KILL TEST
#
# No API calls.
#
# Goal:
#   Given a bank of V11 probes, represent hidden MAS structure
#   explicitly, learn for each probe how output changes as the
#   structure changes, and ask whether matrix-based observability
#   selects complementary probes better than random.
#
# Two structural parameterizations:
#
# 1) precedence6:
#       [exists_A, exists_B, exists_C,
#        order_AB, order_AC, order_BC]
#
# 2) graph9:
#       flattened 3x3 matrix:
#       diagonal = agent existence
#       off-diagonal = DIRECT specialist-to-specialist edge
#
# Structural models are fitted only on:
#       count 0 / count 1 / count 2 topologies
#
# Evaluation is on unseen:
#       all six count-3 permutations
#
# So selection never uses count-3 responses.
# ============================================================

import json
import re
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf


# ============================================================
# CONFIG
# ============================================================

SEED = 20260916

CACHE = Path(
    "artifacts/execution_operator_big/api_runs.jsonl"
)

OUT = Path(
    "artifacts/structural_observability"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

N_TOTAL = 160
TRAIN_N = 100
BASE_DIM = 64

AGENTS = "ABC"
AIDX = {
    a: i
    for i, a in enumerate(AGENTS)
}

ALPHAS = [
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
]

BUDGETS = [
    1,
    2,
    3,
    5,
    10,
    20,
]

RANDOM_REPS = 500

EPS = 1e-6


# ============================================================
# LOAD CACHE
# ============================================================

print("=" * 110)
print("STRUCTURAL OBSERVABILITY / ACTIVE GRAPH TOMOGRAPHY")
print("=" * 110)

if not CACHE.exists():
    raise FileNotFoundError(
        f"Missing cache: {CACHE}"
    )

by_pos = {}

with CACHE.open(
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        r = json.loads(line)

        pos = int(
            r["position"]
        )

        by_pos[pos] = r


positions = sorted(
    by_pos
)[:N_TOTAL]

rows = [
    by_pos[p]
    for p in positions
]

if len(rows) != N_TOTAL:
    raise RuntimeError(
        f"Expected {N_TOTAL} records, got {len(rows)}"
    )

TRAIN = np.arange(
    TRAIN_N
)

DEV = np.arange(
    TRAIN_N,
    N_TOTAL
)

print("TOTAL queries:", len(rows))
print("TRAIN queries:", len(TRAIN))
print("DEV candidate probes:", len(DEV))


# ============================================================
# DISCOVER TOPOLOGY KEYS ROBUSTLY
#
# Expected family:
#
# RM
# RAM / RBM / RCM
# RABM / RBAM / ...
# RABCM / RACBM / ...
# ============================================================

common_keys = set(
    rows[0]["outputs"].keys()
)

for r in rows[1:]:

    common_keys &= set(
        r["outputs"].keys()
    )


seq_to_key = {}

for key in sorted(
    common_keys
):

    m = re.fullmatch(
        r"R([ABC]*)M",
        key
    )

    if not m:
        continue

    seq = m.group(1)

    # Only permutations with no repeated specialist.
    if (
        len(seq) <= 3
        and len(set(seq)) == len(seq)
        and all(
            x in AGENTS
            for x in seq
        )
    ):

        seq_to_key[
            seq
        ] = key


all_seqs = sorted(
    seq_to_key,
    key=lambda x: (
        len(x),
        x
    )
)

print()
print("Discovered structural topologies:")

for seq in all_seqs:

    print(
        f"  {seq_to_key[seq]:8s} "
        f"specialists={seq!r}"
    )


if len(all_seqs) != 16:

    raise RuntimeError(
        "Expected exactly 16 R...M topologies, "
        f"found {len(all_seqs)}: {all_seqs}"
    )


FIT_SEQS = [
    s
    for s in all_seqs
    if len(s) <= 2
]

HELD_SEQS = [
    s
    for s in all_seqs
    if len(s) == 3
]


print()
print(
    "Structural fitting topologies:",
    len(FIT_SEQS),
    FIT_SEQS
)

print(
    "Unseen count-3 evaluation:",
    len(HELD_SEQS),
    HELD_SEQS
)


if (
    len(FIT_SEQS) != 10
    or len(HELD_SEQS) != 6
):
    raise RuntimeError(
        "Unexpected topology split."
    )


# ============================================================
# EMBED FINAL BLACK-BOX OUTPUTS
#
# PCA/scaler are fitted ONLY on TRAIN-query outputs from the
# 10 structural-fitting topologies.
# ============================================================

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

RAW = {}

for seq in all_seqs:

    key = seq_to_key[
        seq
    ]

    print(
        "Encoding",
        key
    )

    texts = [
        r["outputs"][key]
        for r in rows
    ]

    RAW[seq] = encoder.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


pool_raw = np.vstack([
    RAW[s][TRAIN]
    for s in FIT_SEQS
])

pca = PCA(
    n_components=BASE_DIM,
    random_state=SEED
)

pool_pca = pca.fit_transform(
    pool_raw
)

scaler_y = StandardScaler()

scaler_y.fit(
    pool_pca
)


def transform_output(x):

    return scaler_y.transform(
        pca.transform(x)
    ).astype(
        np.float64
    )


Y = {
    s: transform_output(
        RAW[s]
    )
    for s in all_seqs
}


Y_FIT = np.stack(
    [
        Y[s]
        for s in FIT_SEQS
    ],
    axis=1
)

# shape:
# query x topology x embedding

Y_HELD = np.stack(
    [
        Y[s]
        for s in HELD_SEQS
    ],
    axis=1
)


print()
print(
    "Y_FIT shape:",
    Y_FIT.shape
)

print(
    "Y_HELD shape:",
    Y_HELD.shape
)


# ============================================================
# STRUCTURAL REPRESENTATIONS
# ============================================================

ORDER_PAIRS = [
    ("A", "B"),
    ("A", "C"),
    ("B", "C"),
]


def z_precedence6(seq):

    presence = np.array([
        1.0
        if a in seq
        else 0.0
        for a in AGENTS
    ])

    order = []

    pos = {
        a: i
        for i, a in enumerate(seq)
    }

    for a, b in ORDER_PAIRS:

        if (
            a not in pos
            or b not in pos
        ):

            order.append(
                0.0
            )

        elif pos[a] < pos[b]:

            order.append(
                1.0
            )

        else:

            order.append(
                -1.0
            )

    return np.concatenate([
        presence,
        np.asarray(
            order,
            dtype=np.float64
        )
    ])


def z_graph9(seq):

    M = np.zeros(
        (3, 3),
        dtype=np.float64
    )

    # Diagonal = existence.
    for a in seq:

        i = AIDX[a]

        M[i, i] = 1.0

    # Direct specialist-to-specialist communication.
    for a, b in zip(
        seq[:-1],
        seq[1:]
    ):

        i = AIDX[a]
        j = AIDX[b]

        M[i, j] = 1.0

    return M.reshape(-1)


ENCODINGS = {
    "precedence6":
        z_precedence6,

    "graph9":
        z_graph9,
}


# ============================================================
# MATRIX UTILITIES
# ============================================================

def mahalanobis2(
    diff,
    precision
):

    return np.einsum(
        "...d,df,...f->...",
        diff,
        precision,
        diff
    )


HELD_PAIRS = list(
    itertools.combinations(
        range(
            len(HELD_SEQS)
        ),
        2
    )
)


def pair_separation_vector(
    outputs_6xd,
    precision
):

    vals = []

    for i, j in HELD_PAIRS:

        d = (
            outputs_6xd[i]
            -
            outputs_6xd[j]
        )

        vals.append(
            float(
                d
                @ precision
                @ d
            )
        )

    return np.asarray(
        vals,
        dtype=np.float64
    )


# ============================================================
# RUN ONE STRUCTURAL PARAMETERIZATION
# ============================================================

def run_encoding(
    encoding_name,
    z_fn,
):

    print()
    print("=" * 110)
    print(
        "ENCODING:",
        encoding_name
    )
    print("=" * 110)

    # --------------------------------------------------------
    # Structural coordinates
    # --------------------------------------------------------

    Z_fit_raw = np.stack([
        z_fn(s)
        for s in FIT_SEQS
    ])

    Z_held_raw = np.stack([
        z_fn(s)
        for s in HELD_SEQS
    ])

    z_mu = Z_fit_raw.mean(
        axis=0
    )

    z_sd = Z_fit_raw.std(
        axis=0
    )

    if np.any(
        z_sd < 1e-12
    ):

        bad = np.where(
            z_sd < 1e-12
        )[0]

        raise RuntimeError(
            f"{encoding_name}: "
            f"constant structural dimensions: {bad}"
        )

    Z_fit = (
        Z_fit_raw
        -
        z_mu
    ) / z_sd

    Z_held = (
        Z_held_raw
        -
        z_mu
    ) / z_sd

    P = Z_fit.shape[1]

    X_fit = np.concatenate(
        [
            np.ones(
                (len(FIT_SEQS), 1)
            ),
            Z_fit,
        ],
        axis=1
    )

    X_held = np.concatenate(
        [
            np.ones(
                (len(HELD_SEQS), 1)
            ),
            Z_held,
        ],
        axis=1
    )

    print(
        "Structural dimension:",
        P
    )

    print(
        "Design rank:",
        np.linalg.matrix_rank(
            X_fit
        ),
        "/",
        X_fit.shape[1]
    )

    print(
        "Design condition number:",
        round(
            np.linalg.cond(
                X_fit
            ),
            4
        )
    )

    # Dimensions that ACTUALLY vary among the unseen
    # count-3 hypotheses.
    held_var = Z_held_raw.var(
        axis=0
    )

    active_dims = np.where(
        held_var > 1e-12
    )[0]

    print(
        "Held-out varying structural dims:",
        active_dims.tolist()
    )

    if len(active_dims) == 0:

        raise RuntimeError(
            "No structural coordinates vary "
            "among held-out topologies."
        )

    # --------------------------------------------------------
    # Pick Ridge alpha using ONLY TRAIN queries and ONLY
    # count<=2 topologies.
    #
    # Leave one surrogate topology out.
    # --------------------------------------------------------

    penalty = np.eye(
        X_fit.shape[1]
    )

    # Do not penalize intercept.
    penalty[0, 0] = 0.0

    alpha_rows = []

    for alpha in ALPHAS:

        errors = []

        for leave in range(
            len(FIT_SEQS)
        ):

            keep = [
                i
                for i in range(
                    len(FIT_SEQS)
                )
                if i != leave
            ]

            Xt = X_fit[
                keep
            ]

            Xv = X_fit[
                leave
            ]

            # mapping:
            # Y -> beta
            solve = np.linalg.solve(
                Xt.T @ Xt
                +
                alpha * penalty,

                Xt.T
            )

            # TRAIN query x parameter x embedding
            beta = np.einsum(
                "ab,nbd->nad",
                solve,
                Y_FIT[
                    TRAIN
                ][:, keep, :]
            )

            pred = np.einsum(
                "a,nad->nd",
                Xv,
                beta
            )

            true = Y_FIT[
                TRAIN,
                leave,
                :
            ]

            errors.append(
                np.mean(
                    (
                        pred
                        -
                        true
                    ) ** 2
                )
            )

        alpha_rows.append({
            "encoding":
                encoding_name,

            "alpha":
                alpha,

            "cv_mse":
                float(
                    np.mean(
                        errors
                    )
                ),
        })


    alpha_df = pd.DataFrame(
        alpha_rows
    )

    alpha_df.to_csv(
        OUT
        /
        f"{encoding_name}_alpha_cv.csv",
        index=False
    )

    best_alpha = float(
        alpha_df.sort_values(
            "cv_mse"
        ).iloc[0]["alpha"]
    )

    print(
        "Selected alpha:",
        best_alpha
    )

    # --------------------------------------------------------
    # Fit structural response Jacobian J_q for EVERY probe
    # using only count<=2 surrogate structures.
    #
    # y(q,G) ~= b_q + J_q z(G)
    # --------------------------------------------------------

    solve_full = np.linalg.solve(
        X_fit.T @ X_fit
        +
        best_alpha * penalty,

        X_fit.T
    )

    # query x parameter x embedding
    beta_all = np.einsum(
        "ab,nbd->nad",
        solve_full,
        Y_FIT
    )

    # Intercept:
    BIAS = beta_all[
        :,
        0,
        :
    ]

    # J:
    # query x embedding x structural-dim
    J = np.transpose(
        beta_all[
            :,
            1:,
            :
        ],
        (
            0,
            2,
            1
        )
    )

    # --------------------------------------------------------
    # Fit global output-noise / model-residual covariance
    # using TRAIN + count<=2 only.
    # --------------------------------------------------------

    pred_fit = np.einsum(
        "ta,nad->ntd",
        X_fit,
        beta_all
    )

    residual = (
        Y_FIT
        -
        pred_fit
    )

    residual_train = residual[
        TRAIN
    ].reshape(
        -1,
        BASE_DIM
    )

    lw = LedoitWolf()

    lw.fit(
        residual_train
    )

    precision = lw.precision_.astype(
        np.float64
    )

    print(
        "Residual covariance shrinkage:",
        round(
            float(
                lw.shrinkage_
            ),
            5
        )
    )

    # --------------------------------------------------------
    # Structural observability matrix:
    #
    # Omega_q = J_q^T Sigma^{-1} J_q
    # --------------------------------------------------------

    OMEGA = np.zeros(
        (
            N_TOTAL,
            P,
            P
        ),
        dtype=np.float64
    )

    for qid in range(
        N_TOTAL
    ):

        jq = J[
            qid
        ]

        OMEGA[
            qid
        ] = (
            jq.T
            @ precision
            @ jq
        )

        # numerical symmetry
        OMEGA[
            qid
        ] = (
            OMEGA[qid]
            +
            OMEGA[qid].T
        ) / 2.0


    # Only dimensions that distinguish the unseen hypotheses.
    OMEGA_ACTIVE = OMEGA[
        :,
        active_dims
    ][
        :,
        :,
        active_dims
    ]

    # --------------------------------------------------------
    # Predict unseen count-3 structures.
    #
    # IMPORTANT:
    # count-3 outputs are NOT used to fit beta/J.
    # --------------------------------------------------------

    # query x held-topology x embedding
    PRED_HELD = np.einsum(
        "ta,nad->ntd",
        X_held,
        beta_all
    )

    # --------------------------------------------------------
    # Probe-wise predicted and ACTUAL pair separations
    # among count-3 topologies.
    # --------------------------------------------------------

    predicted_pair_sep = {}
    actual_pair_sep = {}

    for qid in DEV:

        predicted_pair_sep[
            qid
        ] = pair_separation_vector(
            PRED_HELD[
                qid
            ],
            precision
        )

        actual_pair_sep[
            qid
        ] = pair_separation_vector(
            Y_HELD[
                qid
            ],
            precision
        )

    # --------------------------------------------------------
    # Selection methods
    # --------------------------------------------------------

    DEV_LIST = [
        int(x)
        for x in DEV
    ]

    MAX_BUDGET = max(
        BUDGETS
    )


    def greedy_dopt():

        cur = (
            EPS
            *
            np.eye(
                len(
                    active_dims
                )
            )
        )

        remaining = set(
            DEV_LIST
        )

        seq = []

        for _ in range(
            MAX_BUDGET
        ):

            base_sign, base_logdet = (
                np.linalg.slogdet(
                    cur
                )
            )

            if base_sign <= 0:

                base_logdet = -1e30

            best_q = None
            best_score = -np.inf

            for qid in remaining:

                candidate = (
                    cur
                    +
                    OMEGA_ACTIVE[
                        qid
                    ]
                )

                sign, logdet = (
                    np.linalg.slogdet(
                        candidate
                    )
                )

                if sign <= 0:
                    score = -np.inf
                else:
                    score = (
                        logdet
                        -
                        base_logdet
                    )

                if score > best_score:

                    best_score = score
                    best_q = qid

            seq.append(
                best_q
            )

            cur = (
                cur
                +
                OMEGA_ACTIVE[
                    best_q
                ]
            )

            remaining.remove(
                best_q
            )

        return seq


    def greedy_eopt():

        cur = (
            EPS
            *
            np.eye(
                len(
                    active_dims
                )
            )
        )

        remaining = set(
            DEV_LIST
        )

        seq = []

        for _ in range(
            MAX_BUDGET
        ):

            best_q = None
            best_score = -np.inf

            for qid in remaining:

                candidate = (
                    cur
                    +
                    OMEGA_ACTIVE[
                        qid
                    ]
                )

                eig = np.linalg.eigvalsh(
                    candidate
                )

                score = float(
                    eig[0]
                )

                if score > best_score:

                    best_score = score
                    best_q = qid

            seq.append(
                best_q
            )

            cur = (
                cur
                +
                OMEGA_ACTIVE[
                    best_q
                ]
            )

            remaining.remove(
                best_q
            )

        return seq


    # Strongest single-probe total information.
    trace_scores = {
        qid:
            float(
                np.trace(
                    OMEGA_ACTIVE[
                        qid
                    ]
                )
            )
        for qid in DEV_LIST
    }

    TRACE_SEQ = sorted(
        DEV_LIST,
        key=lambda q:
            trace_scores[q],
        reverse=True
    )[
        :MAX_BUDGET
    ]


    # Predicted worst-pair discrimination.
    pred_worst_scores = {
        qid:
            float(
                np.min(
                    predicted_pair_sep[
                        qid
                    ]
                )
            )
        for qid in DEV_LIST
    }

    PRED_WORST_SEQ = sorted(
        DEV_LIST,
        key=lambda q:
            pred_worst_scores[q],
        reverse=True
    )[
        :MAX_BUDGET
    ]


    DOPT_SEQ = greedy_dopt()
    EOPT_SEQ = greedy_eopt()


    # --------------------------------------------------------
    # Oracle upper bound:
    #
    # Uses REAL held-out count-3 outputs.
    # NOT a legal attack selector.
    #
    # It tells us whether the V11 candidate bank itself contains
    # complementary probes capable of resolving the hard pairs.
    # --------------------------------------------------------

    def greedy_actual_maximin():

        current = np.zeros(
            len(
                HELD_PAIRS
            ),
            dtype=np.float64
        )

        remaining = set(
            DEV_LIST
        )

        seq = []

        for _ in range(
            MAX_BUDGET
        ):

            best_q = None
            best_score = -np.inf

            for qid in remaining:

                candidate = (
                    current
                    +
                    actual_pair_sep[
                        qid
                    ]
                )

                score = float(
                    np.min(
                        candidate
                    )
                )

                if score > best_score:

                    best_score = score
                    best_q = qid

            seq.append(
                best_q
            )

            current = (
                current
                +
                actual_pair_sep[
                    best_q
                ]
            )

            remaining.remove(
                best_q
            )

        return seq


    ORACLE_SEQ = (
        greedy_actual_maximin()
    )


    SEQUENCES = {
        "TRACE_static":
            TRACE_SEQ,

        "DOPT_observability":
            DOPT_SEQ,

        "EOPT_observability":
            EOPT_SEQ,

        "PRED_worst_pair":
            PRED_WORST_SEQ,

        "ORACLE_actual_maximin":
            ORACLE_SEQ,
    }


    # --------------------------------------------------------
    # Identification evaluation
    #
    # Actual hidden count-3 response:
    #       Y_HELD[q,true]
    #
    # Candidate prediction:
    #       PRED_HELD[q,candidate]
    #
    # Accumulate Mahalanobis residual over selected probes.
    # --------------------------------------------------------

    def evaluate(
        selected
    ):

        selected = list(
            selected
        )

        nheld = len(
            HELD_SEQS
        )

        error = np.zeros(
            (
                nheld,
                nheld
            ),
            dtype=np.float64
        )

        pair_accum = np.zeros(
            len(
                HELD_PAIRS
            ),
            dtype=np.float64
        )

        for qid in selected:

            actual = Y_HELD[
                qid
            ]

            pred = PRED_HELD[
                qid
            ]

            # true x candidate x embedding
            diff = (
                actual[
                    :,
                    None,
                    :
                ]
                -
                pred[
                    None,
                    :,
                    :
                ]
            )

            error += mahalanobis2(
                diff,
                precision
            )

            pair_accum += (
                actual_pair_sep[
                    qid
                ]
            )


        pred_idx = np.argmin(
            error,
            axis=1
        )

        topo_acc = float(
            np.mean(
                pred_idx
                ==
                np.arange(
                    nheld
                )
            )
        )

        ranks = []

        margins = []

        for true_i in range(
            nheld
        ):

            order = np.argsort(
                error[
                    true_i
                ]
            )

            rank = int(
                np.where(
                    order
                    ==
                    true_i
                )[0][0]
            ) + 1

            ranks.append(
                rank
            )

            true_err = error[
                true_i,
                true_i
            ]

            wrong = np.delete(
                error[
                    true_i
                ],
                true_i
            )

            margins.append(
                float(
                    np.min(
                        wrong
                    )
                    -
                    true_err
                )
            )


        # Accuracy on the structural coordinates that actually
        # vary among the six count-3 hypotheses.
        coord_hits = []

        for true_i, guess_i in enumerate(
            pred_idx
        ):

            truth = Z_held_raw[
                true_i,
                active_dims
            ]

            guess = Z_held_raw[
                guess_i,
                active_dims
            ]

            coord_hits.extend(
                (
                    truth
                    ==
                    guess
                ).astype(
                    np.float64
                ).tolist()
            )


        return {
            "topology_accuracy":
                topo_acc,

            "mean_rank":
                float(
                    np.mean(
                        ranks
                    )
                ),

            "struct_coord_accuracy":
                float(
                    np.mean(
                        coord_hits
                    )
                ),

            "mean_true_margin":
                float(
                    np.mean(
                        margins
                    )
                ),

            "actual_min_pair_sep":
                float(
                    np.min(
                        pair_accum
                    )
                ),

            "actual_p10_pair_sep":
                float(
                    np.quantile(
                        pair_accum,
                        0.10
                    )
                ),

            "actual_mean_pair_sep":
                float(
                    np.mean(
                        pair_accum
                    )
                ),
        }


    # --------------------------------------------------------
    # Deterministic selection results
    # --------------------------------------------------------

    result_rows = []

    selection_rows = []

    for method, sequence in (
        SEQUENCES.items()
    ):

        for rank, qid in enumerate(
            sequence,
            start=1
        ):

            selection_rows.append({
                "encoding":
                    encoding_name,

                "method":
                    method,

                "selection_rank":
                    rank,

                "query_index":
                    qid,

                "source_position":
                    positions[
                        qid
                    ],

                "task":
                    rows[
                        qid
                    ][
                        "task"
                    ],
            })


        for budget in BUDGETS:

            m = evaluate(
                sequence[
                    :budget
                ]
            )

            result_rows.append({
                "encoding":
                    encoding_name,

                "method":
                    method,

                "budget":
                    budget,

                **m,
            })


    # --------------------------------------------------------
    # Random baseline distribution
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED
        +
        (
            11
            if encoding_name
            ==
            "precedence6"
            else 29
        )
    )

    random_by_budget = {
        b: []
        for b in BUDGETS
    }

    for _ in range(
        RANDOM_REPS
    ):

        perm = rng.permutation(
            DEV_LIST
        )

        for budget in BUDGETS:

            m = evaluate(
                perm[
                    :budget
                ]
            )

            random_by_budget[
                budget
            ].append(
                m
            )


    for budget in BUDGETS:

        vals = random_by_budget[
            budget
        ]

        row = {
            "encoding":
                encoding_name,

            "method":
                "RANDOM",

            "budget":
                budget,
        }

        for metric in vals[0]:

            arr = np.asarray([
                x[metric]
                for x in vals
            ])

            row[
                metric
            ] = float(
                np.mean(
                    arr
                )
            )

            row[
                metric
                +
                "_q025"
            ] = float(
                np.quantile(
                    arr,
                    .025
                )
            )

            row[
                metric
                +
                "_q975"
            ] = float(
                np.quantile(
                    arr,
                    .975
                )
            )

        result_rows.append(
            row
        )


    results = pd.DataFrame(
        result_rows
    )

    selections = pd.DataFrame(
        selection_rows
    )

    # --------------------------------------------------------
    # Ratio/delta vs random
    # --------------------------------------------------------

    for budget in BUDGETS:

        rand = results[
            (
                results["method"]
                ==
                "RANDOM"
            )
            &
            (
                results["budget"]
                ==
                budget
            )
        ].iloc[0]

        mask = (
            results["budget"]
            ==
            budget
        )

        results.loc[
            mask,
            "topology_acc_delta_vs_random"
        ] = (
            results.loc[
                mask,
                "topology_accuracy"
            ]
            -
            rand[
                "topology_accuracy"
            ]
        )

        results.loc[
            mask,
            "rank_improvement_vs_random"
        ] = (
            rand[
                "mean_rank"
            ]
            -
            results.loc[
                mask,
                "mean_rank"
            ]
        )

        denom = max(
            float(
                rand[
                    "actual_min_pair_sep"
                ]
            ),
            1e-12
        )

        results.loc[
            mask,
            "min_pair_sep_ratio_vs_random"
        ] = (
            results.loc[
                mask,
                "actual_min_pair_sep"
            ]
            /
            denom
        )


    results.to_csv(
        OUT
        /
        f"{encoding_name}_results.csv",
        index=False
    )

    selections.to_csv(
        OUT
        /
        f"{encoding_name}_selections.csv",
        index=False
    )

    # --------------------------------------------------------
    # Spectral diagnostics
    # --------------------------------------------------------

    spec_rows = []

    for qid in DEV_LIST:

        eig = np.linalg.eigvalsh(
            OMEGA_ACTIVE[
                qid
            ]
        )

        spec_rows.append({
            "encoding":
                encoding_name,

            "query_index":
                qid,

            "trace":
                float(
                    np.sum(
                        eig
                    )
                ),

            "lambda_min":
                float(
                    eig[0]
                ),

            "lambda_max":
                float(
                    eig[-1]
                ),

            "condition":
                float(
                    (
                        eig[-1]
                        +
                        EPS
                    )
                    /
                    (
                        eig[0]
                        +
                        EPS
                    )
                ),
        })


    pd.DataFrame(
        spec_rows
    ).to_csv(
        OUT
        /
        f"{encoding_name}_spectral.csv",
        index=False
    )


    print()
    print(
        "RESULTS:",
        encoding_name
    )

    display = results[
        [
            "method",
            "budget",
            "topology_accuracy",
            "mean_rank",
            "struct_coord_accuracy",
            "actual_min_pair_sep",
            "actual_p10_pair_sep",
            "topology_acc_delta_vs_random",
            "rank_improvement_vs_random",
            "min_pair_sep_ratio_vs_random",
        ]
    ].copy()

    print(
        display
        .sort_values(
            [
                "budget",
                "method"
            ]
        )
        .round(4)
        .to_string(
            index=False
        )
    )


    print()
    print(
        "KEY BUDGETS:"
    )

    key = results[
        results[
            "budget"
        ].isin(
            [
                5,
                10
            ]
        )
    ][
        [
            "method",
            "budget",
            "topology_accuracy",
            "mean_rank",
            "struct_coord_accuracy",
            "min_pair_sep_ratio_vs_random",
        ]
    ]

    print(
        key
        .sort_values(
            [
                "budget",
                "method"
            ]
        )
        .round(4)
        .to_string(
            index=False
        )
    )

    return (
        results,
        selections
    )


# ============================================================
# RUN BOTH REPRESENTATIONS
# ============================================================

all_results = []
all_selections = []

for name, fn in (
    ENCODINGS.items()
):

    r, s = run_encoding(
        name,
        fn
    )

    all_results.append(
        r
    )

    all_selections.append(
        s
    )


ALL_RESULTS = pd.concat(
    all_results,
    ignore_index=True
)

ALL_SELECTIONS = pd.concat(
    all_selections,
    ignore_index=True
)

ALL_RESULTS.to_csv(
    OUT
    /
    "all_results.csv",
    index=False
)

ALL_SELECTIONS.to_csv(
    OUT
    /
    "all_selections.csv",
    index=False
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 110)
print("FINAL MATRIX-OBSERVABILITY COMPARISON")
print("=" * 110)

final = ALL_RESULTS[
    ALL_RESULTS[
        "budget"
    ].isin(
        [
            5,
            10
        ]
    )
][
    [
        "encoding",
        "method",
        "budget",
        "topology_accuracy",
        "mean_rank",
        "struct_coord_accuracy",
        "actual_min_pair_sep",
        "topology_acc_delta_vs_random",
        "rank_improvement_vs_random",
        "min_pair_sep_ratio_vs_random",
    ]
]

print(
    final
    .sort_values(
        [
            "encoding",
            "budget",
            "method"
        ]
    )
    .round(4)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("INTERPRETATION GATES")
print("=" * 110)

print("""
This is an exploratory upper-bound test of STRUCTURAL
OBSERVABILITY using the existing V11 probe bank.

The selector is allowed to see each candidate probe on the
count<=2 SURROGATE topologies.

It NEVER uses the six count-3 outputs for fitting or selection.

Evaluation asks whether selected probes generalize to the unseen
count-3 permutations.


Most important methods:

DOPT_observability
    Greedily maximizes log det of accumulated structural
    observability.

    It rewards complementary probes rather than repeatedly
    selecting probes that expose the same structural direction.

EOPT_observability
    Maximizes the SMALLEST eigenvalue.

    It explicitly targets the currently least-observable
    structural direction.

TRACE_static
    Picks individually strong probes but ignores redundancy.

PRED_worst_pair
    Strong scalar baseline based on predicted separation.

ORACLE_actual_maximin
    Illegal diagnostic upper bound.
    Uses the unseen count-3 outputs themselves.
    It tells us whether the candidate bank contains enough
    information in principle.


PROMISING RESULT:

    DOPT / EOPT > RANDOM
    on unseen count-3 structures,

especially if:

    min_pair_sep_ratio_vs_random > 1

AND:

    mean_rank improves

AND ideally:

    topology accuracy improves

at several budgets.


VERY IMPORTANT:

The current 60 DEV probes have been inspected in previous
experiments. This remains exploratory evidence only.

If the idea works here, the next experiment must use a fresh
untouched probe set.
""")


print()
print("Saved under:")
print(OUT)

print()
print("DONE")
