#!/usr/bin/env python3

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

CACHE = Path(
    "artifacts/execution_operator_big/api_runs.jsonl"
)

OUT = Path(
    "artifacts/execution_operator_diagnosis"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_N = 100
BASE_DIM = 64
SEED = 20260916

SPECIALISTS = "ABC"

PREFIX = {
    k: [
        "".join(p)
        for p in itertools.permutations(
            SPECIALISTS,
            k
        )
    ]
    for k in (1, 2, 3)
}

TOPOLOGIES = (
    ["RM"]
    + ["R" + s + "M" for s in PREFIX[1]]
    + ["R" + s + "M" for s in PREFIX[2]]
    + ["R" + s + "M" for s in PREFIX[3]]
)

PREFIX_KEYS = (
    ["R", "A", "B", "C"]
    + [
        "R" + s
        for k in (1, 2, 3)
        for s in PREFIX[k]
    ]
)

ALL_KEYS = sorted(
    set(
        PREFIX_KEYS
        + ["AM", "BM", "CM"]
        + TOPOLOGIES
    )
)

BUDGETS = [1, 2, 3, 5, 10, 20]


# ============================================================
# HELPERS
# ============================================================

def seq_of(topology):
    return topology[1:-1]


def count_of(topology):
    return len(seq_of(topology))


def rho_safe(a, b):

    x = spearmanr(
        a,
        b
    ).statistic

    if not np.isfinite(x):
        return 0.0

    return float(x)


def top_lift(score, actual, frac=0.20):

    k = max(
        1,
        math.ceil(
            len(score) * frac
        )
    )

    ix = np.argsort(score)[-k:]

    return float(
        actual[ix].mean()
        /
        max(
            actual.mean(),
            1e-12
        )
    )


def softmax_log(logp):

    z = logp - np.max(logp)

    p = np.exp(z)

    return p / p.sum()


def count_from_posterior(
    posterior,
    topology_list
):

    pc = {
        0: 0.0,
        1: 0.0,
        2: 0.0,
        3: 0.0,
    }

    for p, h in zip(
        posterior,
        topology_list
    ):
        pc[count_of(h)] += p

    return max(
        pc,
        key=pc.get
    )


# ============================================================
# LOAD
# ============================================================

rows = []

for line in CACHE.open(
    encoding="utf-8"
):

    if line.strip():
        rows.append(
            json.loads(line)
        )

rows = {
    int(r["position"]): r
    for r in rows
}

rows = [
    rows[k]
    for k in sorted(rows)
]

if len(rows) < 160:
    raise RuntimeError(
        f"Expected 160 rows; got {len(rows)}"
    )

rows = rows[:160]

TRAIN = np.arange(TRAIN_N)
TEST = np.arange(
    TRAIN_N,
    len(rows)
)

print("=" * 78)
print("EXECUTION OPERATOR FAILURE DECOMPOSITION")
print("=" * 78)

print("TOTAL:", len(rows))
print("TRAIN:", len(TRAIN))
print("TEST:", len(TEST))
print("TOPOLOGIES:", len(TOPOLOGIES))


# ============================================================
# EMBEDDINGS
# ============================================================

enc = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tasks = [
    r["task"]
    for r in rows
]

print("\nEncoding tasks...")

q_raw = enc.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)

raw = {}

for key in ALL_KEYS:

    print("Encoding", key)

    raw[key] = enc.encode(
        [
            r["outputs"][key]
            for r in rows
        ],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


# ============================================================
# COMMON REPRESENTATION
#
# Do not fit PCA on final R...M outputs.
# ============================================================

fit_keys = (
    ["R", "A", "B", "C"]
    + [
        "R" + s
        for k in (1, 2, 3)
        for s in PREFIX[k]
    ]
    + ["AM", "BM", "CM"]
)

fit_pool = [
    q_raw[TRAIN]
]

for key in fit_keys:
    fit_pool.append(
        raw[key][TRAIN]
    )

fit_pool = np.vstack(
    fit_pool
)

pca = PCA(
    n_components=BASE_DIM,
    random_state=SEED
)

pca_train = pca.fit_transform(
    fit_pool
)

scaler = StandardScaler()

scaler.fit(
    pca_train
)


def transform(x):

    return scaler.transform(
        pca.transform(x)
    )


q = transform(q_raw)

data = {
    key: transform(raw[key])
    for key in ALL_KEYS
}


# ============================================================
# TASK-CONDITIONED SPECIALIST OPERATORS
#
# h_next = A h + B q + c mask + b
# ============================================================

def fit_r(indices):

    reg = Ridge(
        alpha=3.0
    )

    reg.fit(
        q[indices],
        data["R"][indices]
    )

    return reg


def specialist_feature(
    prev,
    qb,
    mask
):

    return np.concatenate(
        [
            prev,
            qb,
            mask,
            np.ones(
                (len(qb), 1)
            ),
        ],
        axis=1
    )


def fit_specialists(
    indices,
    max_parent_depth
):

    models = {}

    for agent in SPECIALISTS:

        X = []
        Y = []

        # --------------------------------------------
        # q -> A/B/C
        # --------------------------------------------

        prev = np.zeros(
            (
                len(indices),
                BASE_DIM
            )
        )

        mask = np.zeros(
            (
                len(indices),
                1
            )
        )

        X.append(
            specialist_feature(
                prev,
                q[indices],
                mask
            )
        )

        Y.append(
            data[agent][indices]
        )

        # --------------------------------------------
        # R... -> R...A
        # --------------------------------------------

        for depth in range(
            0,
            max_parent_depth + 1
        ):

            seqs = (
                [""]
                if depth == 0
                else PREFIX[depth]
            )

            for seq in seqs:

                if agent in seq:
                    continue

                parent = "R" + seq
                child = parent + agent

                prev = data[parent][indices]

                mask = np.ones(
                    (
                        len(indices),
                        1
                    )
                )

                X.append(
                    specialist_feature(
                        prev,
                        q[indices],
                        mask
                    )
                )

                Y.append(
                    data[child][indices]
                )

        X = np.vstack(X)
        Y = np.vstack(Y)

        reg = Ridge(
            alpha=3.0,
            fit_intercept=False
        )

        reg.fit(
            X,
            Y
        )

        models[agent] = reg

    return models


def specialist_step(
    model,
    h,
    qb
):

    mask = np.ones(
        (
            len(qb),
            1
        )
    )

    return model.predict(
        specialist_feature(
            h,
            qb,
            mask
        )
    )


def predict_prefix(
    r_model,
    s_models,
    ids,
    seq
):

    qb = q[ids]

    h = r_model.predict(
        qb
    )

    for agent in seq:

        h = specialist_step(
            s_models[agent],
            h,
            qb
        )

    return h


# ============================================================
# M OPERATOR
# ============================================================

def m_feature(
    parent,
    qb
):

    return np.concatenate(
        [
            parent,
            qb,
            np.ones(
                (
                    len(qb),
                    1
                )
            ),
        ],
        axis=1
    )


def m_parent_list(max_r_count):

    # Always include primitive anchor contexts.
    parents = [
        "R",
        "A",
        "B",
        "C",
    ]

    for depth in range(
        1,
        max_r_count + 1
    ):

        parents.extend(
            [
                "R" + s
                for s in PREFIX[depth]
            ]
        )

    return parents


def fit_m(
    indices,
    max_r_count
):

    X = []
    Y = []

    for parent in m_parent_list(
        max_r_count
    ):

        target = parent + "M"

        X.append(
            m_feature(
                data[parent][indices],
                q[indices]
            )
        )

        Y.append(
            data[target][indices]
        )

    X = np.vstack(X)
    Y = np.vstack(Y)

    reg = Ridge(
        alpha=3.0,
        fit_intercept=False
    )

    reg.fit(
        X,
        Y
    )

    return reg


def predict_m(
    m_model,
    parent,
    qb
):

    return m_model.predict(
        m_feature(
            parent,
            qb
        )
    )


# ============================================================
# FULL MODEL
# ============================================================

def fit_structural_model(
    indices,
    specialist_level,
    m_level
):

    return {
        "R": fit_r(indices),

        "S": fit_specialists(
            indices,
            specialist_level
        ),

        "M": fit_m(
            indices,
            m_level
        ),
    }


def predict_topology(
    model,
    ids,
    topology
):

    seq = seq_of(
        topology
    )

    parent = predict_prefix(
        model["R"],
        model["S"],
        ids,
        seq
    )

    return predict_m(
        model["M"],
        parent,
        q[ids]
    )


def predict_all(
    model,
    ids
):

    return {
        h: predict_topology(
            model,
            ids,
            h
        )
        for h in TOPOLOGIES
    }


# ============================================================
# PART 1:
# PREFIX COMPOSITION LADDER
# ============================================================

print("\n" + "=" * 78)
print("PART 1: PREFIX COMPOSITION LADDER")
print("=" * 78)

prefix_rows = []

r_full = fit_r(
    TRAIN
)

for level in [0, 1, 2]:

    s_model = fit_specialists(
        TRAIN,
        level
    )

    for depth in [1, 2, 3]:

        seqs = PREFIX[depth]

        pred = {}
        actual = {}

        mse = []

        for seq in seqs:

            key = "R" + seq

            pred[seq] = predict_prefix(
                r_full,
                s_model,
                TEST,
                seq
            )

            actual[seq] = data[key][TEST]

            mse.append(
                np.mean(
                    (
                        pred[seq]
                        -
                        actual[seq]
                    ) ** 2
                )
            )

        rhos = []
        lifts = []

        for a, b in itertools.combinations(
            seqs,
            2
        ):

            ps = np.sum(
                (
                    pred[a]
                    -
                    pred[b]
                ) ** 2,
                axis=1
            )

            ys = np.sum(
                (
                    actual[a]
                    -
                    actual[b]
                ) ** 2,
                axis=1
            )

            rhos.append(
                rho_safe(
                    ps,
                    ys
                )
            )

            lifts.append(
                top_lift(
                    ps,
                    ys
                )
            )

        prefix_rows.append({
            "specialist_train_parent_depth": level,
            "target_prefix_depth": depth,
            "mse": np.mean(mse),
            "contrast_rho": (
                np.mean(rhos)
                if rhos
                else np.nan
            ),
            "contrast_lift": (
                np.mean(lifts)
                if lifts
                else np.nan
            ),
        })


prefix_df = pd.DataFrame(
    prefix_rows
)

prefix_df.to_csv(
    OUT / "prefix_ladder.csv",
    index=False
)

print(
    prefix_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# PART 2:
# M DEPTH LADDER
#
# Uses ACTUAL parent states to isolate M itself.
# ============================================================

print("\n" + "=" * 78)
print("PART 2: MERGER DEPTH GENERALIZATION")
print("=" * 78)

m_rows = []

for m_level in [0, 1, 2, 3]:

    m_model = fit_m(
        TRAIN,
        m_level
    )

    for depth in [1, 2, 3]:

        seqs = PREFIX[depth]

        pred = {}
        actual = {}

        mses = []

        for seq in seqs:

            parent = "R" + seq
            final = parent + "M"

            pred[seq] = predict_m(
                m_model,
                data[parent][TEST],
                q[TEST]
            )

            actual[seq] = data[final][TEST]

            mses.append(
                np.mean(
                    (
                        pred[seq]
                        -
                        actual[seq]
                    ) ** 2
                )
            )

        rhos = []
        lifts = []

        for a, b in itertools.combinations(
            seqs,
            2
        ):

            ps = np.sum(
                (
                    pred[a]
                    -
                    pred[b]
                ) ** 2,
                axis=1
            )

            ys = np.sum(
                (
                    actual[a]
                    -
                    actual[b]
                ) ** 2,
                axis=1
            )

            rhos.append(
                rho_safe(
                    ps,
                    ys
                )
            )

            lifts.append(
                top_lift(
                    ps,
                    ys
                )
            )

        m_rows.append({
            "M_train_through_count": m_level,
            "target_count": depth,
            "mse": np.mean(mses),
            "contrast_rho": np.mean(rhos),
            "contrast_lift": np.mean(lifts),
        })


m_df = pd.DataFrame(
    m_rows
)

m_df.to_csv(
    OUT / "merger_ladder.csv",
    index=False
)

print(
    m_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# PART 3:
# OOF PREDICTIONS FOR HONEST CALIBRATION
# ============================================================

print("\n" + "=" * 78)
print("PART 3: OOF CALIBRATION")
print("=" * 78)

VARIANTS = {
    # Maximum structural extrapolation.
    "strict": {
        "specialist_level": 0,
        "m_level": 0,
    },

    # Prefix depth2 and M through count2 seen.
    # Count3 remains extrapolation.
    "ladder": {
        "specialist_level": 1,
        "m_level": 2,
    },

    # Transition-model upper bound.
    # All transition types seen on TRAIN,
    # but TEST prompts remain held out.
    "full_transition": {
        "specialist_level": 2,
        "m_level": 3,
    },
}


def fit_direct(
    train_ids
):

    models = {}

    for h in TOPOLOGIES:

        reg = Ridge(
            alpha=5.0
        )

        reg.fit(
            q[train_ids],
            data[h][train_ids]
        )

        models[h] = reg

    return models


def predict_direct(
    models,
    ids
):

    return {
        h: models[h].predict(
            q[ids]
        )
        for h in TOPOLOGIES
    }


kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

OOF = {}
TEST_PRED = {}

all_variants = list(
    VARIANTS.keys()
) + ["direct"]


for variant in all_variants:

    print("\nBuilding:", variant)

    OOF[variant] = {
        h: np.zeros(
            (
                TRAIN_N,
                BASE_DIM
            )
        )
        for h in TOPOLOGIES
    }

    for fold, (
        tr_pos,
        va_pos
    ) in enumerate(
        kf.split(TRAIN),
        start=1
    ):

        tr_ids = TRAIN[tr_pos]
        va_ids = TRAIN[va_pos]

        if variant == "direct":

            model = fit_direct(
                tr_ids
            )

            pp = predict_direct(
                model,
                va_ids
            )

        else:

            cfg = VARIANTS[variant]

            model = fit_structural_model(
                tr_ids,
                cfg["specialist_level"],
                cfg["m_level"]
            )

            pp = predict_all(
                model,
                va_ids
            )

        for h in TOPOLOGIES:
            OOF[variant][h][va_pos] = pp[h]

        print(
            f"  fold {fold}/5 complete"
        )

    # Full TRAIN -> TEST model

    if variant == "direct":

        model = fit_direct(
            TRAIN
        )

        TEST_PRED[variant] = predict_direct(
            model,
            TEST
        )

    else:

        cfg = VARIANTS[variant]

        model = fit_structural_model(
            TRAIN,
            cfg["specialist_level"],
            cfg["m_level"]
        )

        TEST_PRED[variant] = predict_all(
            model,
            TEST
        )


# ============================================================
# CALIBRATION
# ============================================================

CAL = {}


for variant in all_variants:

    # --------------------------------------------
    # Global bias
    # --------------------------------------------

    global_resid = []

    for h in TOPOLOGIES:

        global_resid.append(
            data[h][TRAIN]
            -
            OOF[variant][h]
        )

    global_resid = np.vstack(
        global_resid
    )

    global_bias = global_resid.mean(
        axis=0
    )

    global_centered = (
        global_resid
        -
        global_bias
    )

    lw_global = LedoitWolf().fit(
        global_centered
    )

    # --------------------------------------------
    # Topology-specific bias + shared covariance
    # --------------------------------------------

    topo_bias = {}
    topo_centered = []

    for h in TOPOLOGIES:

        rr = (
            data[h][TRAIN]
            -
            OOF[variant][h]
        )

        bb = rr.mean(
            axis=0
        )

        topo_bias[h] = bb

        topo_centered.append(
            rr - bb
        )

    topo_centered = np.vstack(
        topo_centered
    )

    lw_topo = LedoitWolf().fit(
        topo_centered
    )

    CAL[variant] = {
        "global_bias": global_bias,
        "global_precision":
            lw_global.precision_,

        "topo_bias": topo_bias,
        "topo_precision":
            lw_topo.precision_,
    }


# ============================================================
# PART 4:
# RAW VS CALIBRATED IDENTIFICATION
# ============================================================

def candidate_mu(
    variant,
    h,
    i,
    mode
):

    base = TEST_PRED[variant][h][i]

    if mode == "raw":
        return base

    if mode == "global":
        return (
            base
            +
            CAL[variant]["global_bias"]
        )

    if mode == "topo_bias":
        return (
            base
            +
            CAL[variant]["topo_bias"][h]
        )

    raise ValueError(mode)


def precision_for(
    variant,
    mode
):

    if mode == "raw":
        return None

    if mode == "global":
        return CAL[variant][
            "global_precision"
        ]

    return CAL[variant][
        "topo_precision"
    ]


def distance(
    y,
    mu,
    precision
):

    d = y - mu

    if precision is None:
        return float(
            d @ d
        )

    return float(
        d
        @ precision
        @ d
    )


id_rows = []

best_confusions = {}


for variant in all_variants:

    for mode in [
        "raw",
        "global",
        "topo_bias",
    ]:

        precision = precision_for(
            variant,
            mode
        )

        topo_true = []
        topo_pred = []

        count_true = []
        count_pred = []

        role_jacc = []

        for i in range(
            len(TEST)
        ):

            for true_h in TOPOLOGIES:

                y = data[true_h][TEST][i]

                dists = {}

                for cand in TOPOLOGIES:

                    mu = candidate_mu(
                        variant,
                        cand,
                        i,
                        mode
                    )

                    dists[cand] = distance(
                        y,
                        mu,
                        precision
                    )

                guess = min(
                    dists,
                    key=dists.get
                )

                topo_true.append(
                    true_h
                )

                topo_pred.append(
                    guess
                )

                count_true.append(
                    count_of(true_h)
                )

                count_pred.append(
                    count_of(guess)
                )

                a = set(
                    seq_of(true_h)
                )

                b = set(
                    seq_of(guess)
                )

                if not a and not b:
                    jj = 1.0
                else:
                    jj = (
                        len(a & b)
                        /
                        max(
                            len(a | b),
                            1
                        )
                    )

                role_jacc.append(
                    jj
                )

        topology_acc = np.mean(
            np.asarray(topo_true)
            ==
            np.asarray(topo_pred)
        )

        count_acc = np.mean(
            np.asarray(count_true)
            ==
            np.asarray(count_pred)
        )

        id_rows.append({
            "variant": variant,
            "calibration": mode,
            "topology_acc":
                topology_acc,
            "count_acc":
                count_acc,
            "role_set_jaccard":
                np.mean(role_jacc),
        })

        if mode == "topo_bias":

            cm_count = confusion_matrix(
                count_true,
                count_pred,
                labels=[0,1,2,3],
                normalize="true"
            )

            pd.DataFrame(
                cm_count,
                index=[
                    "true_0",
                    "true_1",
                    "true_2",
                    "true_3",
                ],
                columns=[
                    "pred_0",
                    "pred_1",
                    "pred_2",
                    "pred_3",
                ]
            ).to_csv(
                OUT
                /
                f"count_confusion_{variant}.csv"
            )

            cm_top = confusion_matrix(
                topo_true,
                topo_pred,
                labels=TOPOLOGIES,
                normalize="true"
            )

            pd.DataFrame(
                cm_top,
                index=TOPOLOGIES,
                columns=TOPOLOGIES
            ).to_csv(
                OUT
                /
                f"topology_confusion_{variant}.csv"
            )


id_df = pd.DataFrame(
    id_rows
)

id_df.to_csv(
    OUT / "calibrated_identification.csv",
    index=False
)

print(
    id_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# PART 5:
# POSTERIOR-ADAPTIVE ACTIVE PROBING
# ============================================================

print("\n" + "=" * 78)
print("PART 5: POSTERIOR-ADAPTIVE ACTIVE PROBING")
print("=" * 78)


def calibrated_test_means(
    variant,
    mode
):

    result = {}

    for h in TOPOLOGIES:

        x = TEST_PRED[variant][h].copy()

        if mode == "global":

            x = (
                x
                +
                CAL[variant]["global_bias"]
            )

        elif mode == "topo_bias":

            x = (
                x
                +
                CAL[variant]["topo_bias"][h]
            )

        result[h] = x

    return result


def query_score(
    means,
    precision,
    qidx,
    posterior,
    focus
):

    total = 0.0

    for a in range(
        len(TOPOLOGIES)
    ):

        for b in range(
            a + 1,
            len(TOPOLOGIES)
        ):

            ha = TOPOLOGIES[a]
            hb = TOPOLOGIES[b]

            if (
                focus == "count"
                and
                count_of(ha)
                ==
                count_of(hb)
            ):
                continue

            d = (
                means[ha][qidx]
                -
                means[hb][qidx]
            )

            md = float(
                d
                @ precision
                @ d
            )

            total += (
                posterior[a]
                *
                posterior[b]
                *
                md
            )

    return total


def update_posterior(
    posterior,
    means,
    precision,
    qidx,
    observed
):

    logp = np.log(
        posterior + 1e-300
    )

    for k, h in enumerate(
        TOPOLOGIES
    ):

        d = (
            observed
            -
            means[h][qidx]
        )

        logp[k] += (
            -0.5
            *
            float(
                d
                @ precision
                @ d
            )
        )

    return softmax_log(
        logp
    )


def predict_from_posterior(
    posterior
):

    h = TOPOLOGIES[
        int(
            np.argmax(
                posterior
            )
        )
    ]

    n = count_from_posterior(
        posterior,
        TOPOLOGIES
    )

    return h, n


def simulate_adaptive(
    variant,
    calibration,
    focus
):

    means = calibrated_test_means(
        variant,
        calibration
    )

    precision = (
        CAL[variant]["global_precision"]
        if calibration == "global"
        else CAL[variant]["topo_precision"]
    )

    results = {
        b: {
            "top": [],
            "count": [],
        }
        for b in BUDGETS
    }

    for true_h in TOPOLOGIES:

        posterior = np.ones(
            len(TOPOLOGIES)
        ) / len(TOPOLOGIES)

        unused = list(
            range(
                len(TEST)
            )
        )

        for step in range(
            1,
            max(BUDGETS) + 1
        ):

            scores = [
                query_score(
                    means,
                    precision,
                    qidx,
                    posterior,
                    focus
                )
                for qidx in unused
            ]

            best_pos = int(
                np.argmax(
                    scores
                )
            )

            qidx = unused.pop(
                best_pos
            )

            observed = data[
                true_h
            ][TEST][qidx]

            posterior = update_posterior(
                posterior,
                means,
                precision,
                qidx,
                observed
            )

            if step in BUDGETS:

                gh, gn = (
                    predict_from_posterior(
                        posterior
                    )
                )

                results[step]["top"].append(
                    gh == true_h
                )

                results[step]["count"].append(
                    gn
                    ==
                    count_of(true_h)
                )

    return results


def simulate_static(
    variant,
    calibration
):

    means = calibrated_test_means(
        variant,
        calibration
    )

    precision = (
        CAL[variant]["global_precision"]
        if calibration == "global"
        else CAL[variant]["topo_precision"]
    )

    prior = np.ones(
        len(TOPOLOGIES)
    ) / len(TOPOLOGIES)

    scores = np.asarray([
        query_score(
            means,
            precision,
            i,
            prior,
            "topology"
        )
        for i in range(
            len(TEST)
        )
    ])

    order = np.argsort(
        scores
    )[::-1]

    results = {
        b: {
            "top": [],
            "count": [],
        }
        for b in BUDGETS
    }

    for true_h in TOPOLOGIES:

        posterior = prior.copy()

        for step, qidx in enumerate(
            order[:max(BUDGETS)],
            start=1
        ):

            observed = data[
                true_h
            ][TEST][qidx]

            posterior = update_posterior(
                posterior,
                means,
                precision,
                qidx,
                observed
            )

            if step in BUDGETS:

                gh, gn = (
                    predict_from_posterior(
                        posterior
                    )
                )

                results[step]["top"].append(
                    gh == true_h
                )

                results[step]["count"].append(
                    gn
                    ==
                    count_of(true_h)
                )

    return results


def simulate_random(
    variant,
    calibration,
    reps=100
):

    means = calibrated_test_means(
        variant,
        calibration
    )

    precision = (
        CAL[variant]["global_precision"]
        if calibration == "global"
        else CAL[variant]["topo_precision"]
    )

    rng = np.random.default_rng(
        SEED
    )

    store = {
        b: {
            "top": [],
            "count": [],
        }
        for b in BUDGETS
    }

    for _ in range(reps):

        for true_h in TOPOLOGIES:

            posterior = np.ones(
                len(TOPOLOGIES)
            ) / len(TOPOLOGIES)

            order = rng.permutation(
                len(TEST)
            )

            for step, qidx in enumerate(
                order[:max(BUDGETS)],
                start=1
            ):

                observed = data[
                    true_h
                ][TEST][qidx]

                posterior = update_posterior(
                    posterior,
                    means,
                    precision,
                    qidx,
                    observed
                )

                if step in BUDGETS:

                    gh, gn = (
                        predict_from_posterior(
                            posterior
                        )
                    )

                    store[step]["top"].append(
                        gh == true_h
                    )

                    store[step]["count"].append(
                        gn
                        ==
                        count_of(true_h)
                    )

    return store


active_rows = []


for variant in all_variants:

    # Topology-specific bias is the main diagnostic.
    calibration = "topo_bias"

    print(
        "\nActive simulation:",
        variant
    )

    adaptive_top = simulate_adaptive(
        variant,
        calibration,
        "topology"
    )

    adaptive_count = simulate_adaptive(
        variant,
        calibration,
        "count"
    )

    static = simulate_static(
        variant,
        calibration
    )

    random = simulate_random(
        variant,
        calibration,
        reps=100
    )

    for b in BUDGETS:

        active_rows.append({
            "variant": variant,
            "budget": b,

            "adaptive_topology_acc":
                np.mean(
                    adaptive_top[b]["top"]
                ),

            "adaptive_topology_count_acc":
                np.mean(
                    adaptive_top[b]["count"]
                ),

            "adaptive_count_topology_acc":
                np.mean(
                    adaptive_count[b]["top"]
                ),

            "adaptive_count_acc":
                np.mean(
                    adaptive_count[b]["count"]
                ),

            "static_topology_acc":
                np.mean(
                    static[b]["top"]
                ),

            "static_count_acc":
                np.mean(
                    static[b]["count"]
                ),

            "random_topology_acc":
                np.mean(
                    random[b]["top"]
                ),

            "random_count_acc":
                np.mean(
                    random[b]["count"]
                ),
        })


active_df = pd.DataFrame(
    active_rows
)

active_df.to_csv(
    OUT / "adaptive_active_results.csv",
    index=False
)

print(
    active_df
    .round(4)
    .to_string(index=False)
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 78)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 78)

print("\nPREFIX LADDER:")
print(
    prefix_df
    .round(4)
    .to_string(index=False)
)

print("\nMERGER LADDER:")
print(
    m_df
    .round(4)
    .to_string(index=False)
)

print("\nCALIBRATED IDENTIFICATION:")
print(
    id_df
    .round(4)
    .to_string(index=False)
)

print("\nADAPTIVE ACTIVE:")
print(
    active_df
    .round(4)
    .to_string(index=False)
)

print("\nFiles saved:")

for f in [
    "prefix_ladder.csv",
    "merger_ladder.csv",
    "calibrated_identification.csv",
    "adaptive_active_results.csv",
]:

    print(
        OUT / f
    )

print("\nAlso saved per-variant:")
print("  count_confusion_<variant>.csv")
print("  topology_confusion_<variant>.csv")

print("\nINTERPRETATION GATES:")
print(
    "1) Prefix fails before M -> primitive specialist composition is the bottleneck."
)
print(
    "2) Prefix works, M ladder fails across depth -> merger/context generalization is the bottleneck."
)
print(
    "3) Structural predictions work but calibration sharply improves identification -> inference/model-bias was the bottleneck."
)
print(
    "4) Adaptive posterior > static/random -> spectral information-directed probing is doing useful sequential work."
)
print(
    "5) Count-focused adaptive > topology-focused for count -> count needs its own information objective."
)
print(
    "6) Even full-transition + calibration remains near chance -> observational overlap/noise is the likely next issue; do NOT add model capacity yet."
)
