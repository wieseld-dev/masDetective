#!/usr/bin/env python3

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score

from sentence_transformers import SentenceTransformer


CACHE = Path(
    "artifacts/execution_operator_big/api_runs.jsonl"
)

OUT = Path(
    "artifacts/execution_operator_big"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_N = 100
BASE_DIM = 64
LATENT_DIM = 32

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

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
    + [
        "R" + s + "M"
        for s in PREFIX[1]
    ]
    + [
        "R" + s + "M"
        for s in PREFIX[2]
    ]
    + [
        "R" + s + "M"
        for s in PREFIX[3]
    ]
)

ALL_PREFIX_KEYS = (
    ["R", "A", "B", "C"]
    + [
        "R" + s
        for k in (1,2,3)
        for s in PREFIX[k]
    ]
)

ALL_KEYS = sorted(
    set(
        ALL_PREFIX_KEYS
        + ["AM", "BM", "CM"]
        + TOPOLOGIES
    )
)


def seq_of(topology):
    return topology[1:-1]


def count_of(topology):
    return len(seq_of(topology))


def rho_safe(x, y):

    r = spearmanr(
        x,
        y
    ).statistic

    if not np.isfinite(r):
        return 0.0

    return float(r)


def top_lift(score, actual, frac=.20):

    k = max(
        1,
        math.ceil(
            frac * len(score)
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


###############################################################################
# LOAD
###############################################################################

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
        f"Expected 160 rows, got {len(rows)}"
    )

rows = rows[:160]

TRAIN = np.arange(TRAIN_N)
TEST = np.arange(
    TRAIN_N,
    len(rows)
)

print("=" * 72)
print("BIG EXECUTION-OPERATOR TEST")
print("=" * 72)

print("TOTAL:", len(rows))
print("TRAIN:", len(TRAIN))
print("TEST:", len(TEST))
print("DEVICE:", DEVICE)
print("TOPOLOGIES:", len(TOPOLOGIES))


###############################################################################
# EMBEDDINGS
###############################################################################

enc = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
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


###############################################################################
# FIXED BASE SPACE
#
# IMPORTANT:
# PCA does NOT see the final R...M target outputs.
###############################################################################

TRAIN_REP_KEYS = (
    ["R", "A", "B", "C"]
    + [
        "R" + s
        for s in PREFIX[1]
    ]
    + [
        "R" + s
        for s in PREFIX[2]
    ]
    + ["AM", "BM", "CM"]
)

fit_pool = [
    q_raw[TRAIN]
]

for key in TRAIN_REP_KEYS:
    fit_pool.append(
        raw[key][TRAIN]
    )

fit_pool = np.vstack(
    fit_pool
)

pca = PCA(
    n_components=BASE_DIM,
    random_state=0
)

tmp = pca.fit_transform(
    fit_pool
)

scaler = StandardScaler()
scaler.fit(tmp)


def transform(x):

    return scaler.transform(
        pca.transform(x)
    )


q = transform(q_raw)

data = {
    key: transform(raw[key])
    for key in ALL_KEYS
}


###############################################################################
# EXACT CAUSAL TRANSITION DATA
###############################################################################

def transition_specs():

    specs = []

    # Original prompt -> R
    specs.append(
        ("R", None, "R")
    )

    # Original prompt -> singleton specialist
    for a in SPECIALISTS:
        specs.append(
            (a, None, a)
        )

    # R -> RA/RB/RC
    for a in SPECIALISTS:
        specs.append(
            (
                a,
                "R",
                "R" + a
            )
        )

    # Length 1 -> length 2.
    # We deliberately DO NOT train on length2 -> length3.
    for s in PREFIX[1]:

        parent = "R" + s

        for a in SPECIALISTS:

            if a in s:
                continue

            child = parent + a

            specs.append(
                (
                    a,
                    parent,
                    child
                )
            )

    # M only learns primitive anchor contexts.
    #
    # It NEVER sees R... -> R...M during operator training.
    for a in SPECIALISTS:
        specs.append(
            (
                "M",
                a,
                a + "M"
            )
        )

    return specs


TRANSITION_SPECS = transition_specs()


###############################################################################
# FIXED TASK-CONDITIONED LINEAR OPERATORS
#
# h_next = A_i h_prev + B_i q + c_i mask + b_i
###############################################################################

def make_transition_arrays(indices):

    result = {
        a: {
            "q": [],
            "prev": [],
            "mask": [],
            "next": [],
        }
        for a in ["R","A","B","C","M"]
    }

    for i in indices:

        for agent, prev_key, next_key in TRANSITION_SPECS:

            result[agent]["q"].append(
                q[i]
            )

            if prev_key is None:

                result[agent]["prev"].append(
                    np.zeros(BASE_DIM)
                )

                result[agent]["mask"].append(
                    [0.0]
                )

            else:

                result[agent]["prev"].append(
                    data[prev_key][i]
                )

                result[agent]["mask"].append(
                    [1.0]
                )

            result[agent]["next"].append(
                data[next_key][i]
            )

    for agent in result:

        for k in result[agent]:
            result[agent][k] = np.asarray(
                result[agent][k],
                dtype=np.float32
            )

    return result


def fit_fixed(indices):

    td = make_transition_arrays(
        indices
    )

    models = {}

    for agent, d in td.items():

        X = np.concatenate(
            [
                d["prev"],
                d["q"],
                d["mask"],
                np.ones(
                    (len(d["q"]),1)
                ),
            ],
            axis=1
        )

        reg = Ridge(
            alpha=3.0,
            fit_intercept=False
        )

        reg.fit(
            X,
            d["next"]
        )

        models[agent] = reg

    return models


def fixed_step(
    models,
    agent,
    h,
    qb,
    mask
):

    X = np.concatenate(
        [
            h,
            qb,
            mask,
            np.ones(
                (len(qb),1)
            )
        ],
        axis=1
    )

    return models[agent].predict(X)


def fixed_rollout(
    models,
    qb,
    topology
):

    n = len(qb)

    h = np.zeros(
        (n, BASE_DIM)
    )

    mask = np.zeros(
        (n,1)
    )

    for role in topology:

        h = fixed_step(
            models,
            role,
            h,
            qb,
            mask
        )

        mask = np.ones(
            (n,1)
        )

    return h


###############################################################################
# LEARNED SHARED EXECUTION SPACE
###############################################################################

class LiftedOperators(nn.Module):

    def __init__(
        self,
        mode="mlp"
    ):

        super().__init__()

        if mode == "linear":

            self.encoder = nn.Linear(
                BASE_DIM,
                LATENT_DIM
            )

            self.decoder = nn.Linear(
                LATENT_DIM,
                BASE_DIM
            )

        else:

            self.encoder = nn.Sequential(
                nn.Linear(
                    BASE_DIM,
                    64
                ),
                nn.GELU(),
                nn.Linear(
                    64,
                    LATENT_DIM
                )
            )

            self.decoder = nn.Sequential(
                nn.Linear(
                    LATENT_DIM,
                    64
                ),
                nn.GELU(),
                nn.Linear(
                    64,
                    BASE_DIM
                )
            )

        self.ops = nn.ModuleDict({
            a: nn.Linear(
                2*LATENT_DIM + 2,
                LATENT_DIM
            )
            for a in ["R","A","B","C","M"]
        })

    def step(
        self,
        agent,
        h,
        zq,
        mask
    ):

        one = torch.ones(
            (len(zq),1),
            device=zq.device
        )

        x = torch.cat(
            [
                h,
                zq,
                mask,
                one
            ],
            dim=1
        )

        return self.ops[agent](x)

    def rollout(
        self,
        qb,
        topology
    ):

        zq = self.encoder(qb)

        h = torch.zeros(
            (
                len(qb),
                LATENT_DIM
            ),
            device=qb.device
        )

        mask = torch.zeros(
            (len(qb),1),
            device=qb.device
        )

        for role in topology:

            h = self.step(
                role,
                h,
                zq,
                mask
            )

            mask = torch.ones(
                (len(qb),1),
                device=qb.device
            )

        return self.decoder(h)


def train_lifted(
    mode,
    seed=42,
    steps=2200
):

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LiftedOperators(
        mode=mode
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=1e-4
    )

    td = make_transition_arrays(
        TRAIN
    )

    tdata = {}

    for agent, d in td.items():

        tdata[agent] = {
            k: torch.tensor(
                v,
                dtype=torch.float32,
                device=DEVICE
            )
            for k,v in d.items()
        }

    # Reconstruction pool excludes every R...M target.
    recon_np = [
        q[TRAIN]
    ]

    for key in TRAIN_REP_KEYS:
        recon_np.append(
            data[key][TRAIN]
        )

    recon = torch.tensor(
        np.vstack(recon_np),
        dtype=torch.float32,
        device=DEVICE
    )

    qtrain = torch.tensor(
        q[TRAIN],
        dtype=torch.float32,
        device=DEVICE
    )

    # Known compositions only.
    # No R...M final targets.
    rollout_keys = (
        ["R","A","B","C"]
        + [
            "R" + s
            for s in PREFIX[1]
        ]
        + [
            "R" + s
            for s in PREFIX[2]
        ]
        + ["AM","BM","CM"]
    )

    rollout_targets = {
        key: torch.tensor(
            data[key][TRAIN],
            dtype=torch.float32,
            device=DEVICE
        )
        for key in rollout_keys
    }

    best = float("inf")
    best_state = None

    for step in range(1, steps+1):

        opt.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------
        # Transition consistency
        # ----------------------------------------------

        trans_loss = 0.0

        for agent, d in tdata.items():

            zq = model.encoder(
                d["q"]
            )

            zprev = model.encoder(
                d["prev"]
            )

            zprev = (
                zprev
                *
                d["mask"]
            )

            pred = model.step(
                agent,
                zprev,
                zq,
                d["mask"]
            )

            target = model.encoder(
                d["next"]
            )

            trans_loss = (
                trans_loss
                +
                F.mse_loss(
                    pred,
                    target
                )
            )

        trans_loss = (
            trans_loss
            /
            len(tdata)
        )

        # ----------------------------------------------
        # Preserve observable information
        # ----------------------------------------------

        recon_pred = model.decoder(
            model.encoder(
                recon
            )
        )

        recon_loss = F.mse_loss(
            recon_pred,
            recon
        )

        # ----------------------------------------------
        # Explicit composition consistency on only
        # compositions available for training.
        # ----------------------------------------------

        rollout_loss = 0.0

        for key in rollout_keys:

            pred = model.rollout(
                qtrain,
                key
            )

            rollout_loss = (
                rollout_loss
                +
                F.mse_loss(
                    pred,
                    rollout_targets[key]
                )
            )

        rollout_loss = (
            rollout_loss
            /
            len(rollout_keys)
        )

        loss = (
            trans_loss
            +
            0.7 * rollout_loss
            +
            1.0 * recon_loss
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        opt.step()

        lv = float(
            loss.detach().cpu()
        )

        if lv < best:

            best = lv

            best_state = {
                k: v.detach().cpu().clone()
                for k,v
                in model.state_dict().items()
            }

        if step % 400 == 0:

            print(
                f"[{mode}] "
                f"step={step} "
                f"loss={lv:.5f} "
                f"trans={float(trans_loss):.5f} "
                f"roll={float(rollout_loss):.5f} "
                f"recon={float(recon_loss):.5f}"
            )

    model.load_state_dict(
        best_state
    )

    return model


###############################################################################
# FIT MODELS
###############################################################################

print("\nFitting fixed task-conditioned operators...")

fixed = fit_fixed(
    TRAIN
)

print("\nTraining learned LINEAR shared space...")

lift_linear = train_lifted(
    "linear"
)

print("\nTraining learned NONLINEAR shared space...")

lift_mlp = train_lifted(
    "mlp"
)


###############################################################################
# PREDICT TEST TOPOLOGIES
###############################################################################

qtest = q[TEST]

pred = {
    "fixed": {},
    "linear_lift": {},
    "mlp_lift": {},
    "direct": {},
}

for topology in TOPOLOGIES:

    pred["fixed"][topology] = fixed_rollout(
        fixed,
        qtest,
        topology
    )


qt = torch.tensor(
    qtest,
    dtype=torch.float32,
    device=DEVICE
)

with torch.no_grad():

    for topology in TOPOLOGIES:

        pred["linear_lift"][topology] = (
            lift_linear
            .rollout(
                qt,
                topology
            )
            .cpu()
            .numpy()
        )

        pred["mlp_lift"][topology] = (
            lift_mlp
            .rollout(
                qt,
                topology
            )
            .cpu()
            .numpy()
        )


###############################################################################
# VERY STRONG SUPERVISED BASELINE
#
# It IS allowed to see final topology outputs on TRAIN.
###############################################################################

for topology in TOPOLOGIES:

    reg = Ridge(
        alpha=5.0
    )

    reg.fit(
        q[TRAIN],
        data[topology][TRAIN]
    )

    pred["direct"][topology] = reg.predict(
        qtest
    )


actual = {
    topology: data[topology][TEST]
    for topology in TOPOLOGIES
}


###############################################################################
# CONTRAST DEFINITIONS
###############################################################################

def role_pairs():

    xs = [
        "RAM",
        "RBM",
        "RCM"
    ]

    return list(
        itertools.combinations(
            xs,
            2
        )
    )


def order2_pairs():

    return [
        ("RABM","RBAM"),
        ("RACM","RCAM"),
        ("RBCM","RCBM"),
    ]


def order3_pairs():

    xs = [
        "R" + s + "M"
        for s in PREFIX[3]
    ]

    return list(
        itertools.combinations(
            xs,
            2
        )
    )


def count_insertion_pairs():

    pairs = []

    for a in TOPOLOGIES:

        sa = seq_of(a)

        for b in TOPOLOGIES:

            sb = seq_of(b)

            if len(sb) != len(sa) + 1:
                continue

            # Controlled insertion at the end of the path.
            if sb.startswith(sa):
                pairs.append(
                    (a,b)
                )

    return pairs


CONTRAST_GROUPS = {
    "ROLE_COUNT1": role_pairs(),
    "ORDER_COUNT2": order2_pairs(),
    "ORDER_COUNT3": order3_pairs(),
    "COUNT_INSERTION": count_insertion_pairs(),
    "ALL": list(
        itertools.combinations(
            TOPOLOGIES,
            2
        )
    )
}


###############################################################################
# CONTRAST PERFORMANCE
###############################################################################

result_rows = []


for method in pred:

    for group, pairs in CONTRAST_GROUPS.items():

        rs = []
        ls = []

        for a,b in pairs:

            ps = np.sum(
                (
                    pred[method][a]
                    -
                    pred[method][b]
                )**2,
                axis=1
            )

            ys = np.sum(
                (
                    actual[a]
                    -
                    actual[b]
                )**2,
                axis=1
            )

            rs.append(
                rho_safe(
                    ps,
                    ys
                )
            )

            ls.append(
                top_lift(
                    ps,
                    ys
                )
            )

        result_rows.append({
            "method": method,
            "group": group,
            "rho": np.mean(rs),
            "lift": np.mean(ls),
            "n_contrasts": len(pairs),
        })


rdf = pd.DataFrame(
    result_rows
)

rdf.to_csv(
    OUT / "big_contrast_results.csv",
    index=False
)


###############################################################################
# SINGLE-QUERY TOPOLOGY IDENTIFICATION
###############################################################################

def identify_single(method):

    topo_ok = []
    count_ok = []
    jacc = []

    for i in range(len(TEST)):

        for true_h in TOPOLOGIES:

            y = actual[true_h][i]

            dist = {
                cand: np.sum(
                    (
                        y
                        -
                        pred[method][cand][i]
                    )**2
                )
                for cand in TOPOLOGIES
            }

            guess = min(
                dist,
                key=dist.get
            )

            topo_ok.append(
                guess == true_h
            )

            count_ok.append(
                count_of(guess)
                ==
                count_of(true_h)
            )

            s1 = set(
                seq_of(guess)
            )

            s2 = set(
                seq_of(true_h)
            )

            if not s1 and not s2:
                j = 1.0
            else:
                j = (
                    len(s1 & s2)
                    /
                    max(
                        len(s1 | s2),
                        1
                    )
                )

            jacc.append(j)

    return {
        "topology_acc":
            np.mean(topo_ok),

        "count_acc":
            np.mean(count_ok),

        "role_set_jaccard":
            np.mean(jacc),
    }


id_rows = []

for method in pred:

    r = identify_single(
        method
    )

    id_rows.append({
        "method": method,
        **r
    })

idf = pd.DataFrame(
    id_rows
)

idf.to_csv(
    OUT / "single_query_identification.csv",
    index=False
)


###############################################################################
# SEMANTIC LEAKAGE BASELINES
###############################################################################

def semantic_features(
    ids,
    y_source,
    use_q=True
):

    X = []
    yt = []
    yc = []

    for i in ids:

        for label, topo in enumerate(
            TOPOLOGIES
        ):

            yy = data[topo][i]
            qq = q[i]

            if use_q:

                feat = np.concatenate(
                    [
                        qq,
                        yy,
                        yy - qq,
                        yy * qq,
                    ]
                )

            else:

                feat = yy

            X.append(feat)
            yt.append(label)
            yc.append(
                count_of(topo)
            )

    return (
        np.asarray(X),
        np.asarray(yt),
        np.asarray(yc)
    )


semantic_rows = []

for use_q, name in [
    (False, "y_only"),
    (True, "q_plus_y"),
]:

    Xtr, ytr_top, ytr_count = (
        semantic_features(
            TRAIN,
            data,
            use_q=use_q
        )
    )

    Xte, yte_top, yte_count = (
        semantic_features(
            TEST,
            data,
            use_q=use_q
        )
    )

    clf_top = LogisticRegression(
        max_iter=5000,
        C=1.0
    )

    clf_top.fit(
        Xtr,
        ytr_top
    )

    clf_count = LogisticRegression(
        max_iter=5000,
        C=1.0
    )

    clf_count.fit(
        Xtr,
        ytr_count
    )

    semantic_rows.append({
        "baseline": name,
        "topology_acc":
            accuracy_score(
                yte_top,
                clf_top.predict(Xte)
            ),

        "count_acc":
            accuracy_score(
                yte_count,
                clf_count.predict(Xte)
            ),
    })


sdf = pd.DataFrame(
    semantic_rows
)

sdf.to_csv(
    OUT / "semantic_baselines.csv",
    index=False
)


###############################################################################
# ACTIVE MULTI-QUERY IDENTIFICATION
###############################################################################

BUDGETS = [
    1,2,3,5,10,20
]


def dispersion_score(method):

    score = []

    for i in range(len(TEST)):

        vals = [
            pred[method][h][i]
            for h in TOPOLOGIES
        ]

        ds = []

        for a,b in itertools.combinations(
            vals,
            2
        ):

            ds.append(
                np.sum(
                    (a-b)**2
                )
            )

        score.append(
            np.mean(ds)
        )

    return np.asarray(score)


def multi_identification(
    method,
    selected
):

    topo_ok = []
    count_ok = []

    for true_h in TOPOLOGIES:

        cand_dist = {}

        for cand in TOPOLOGIES:

            d = 0.0

            for i in selected:

                d += np.sum(
                    (
                        actual[true_h][i]
                        -
                        pred[method][cand][i]
                    )**2
                )

            cand_dist[cand] = d

        guess = min(
            cand_dist,
            key=cand_dist.get
        )

        topo_ok.append(
            guess == true_h
        )

        count_ok.append(
            count_of(guess)
            ==
            count_of(true_h)
        )

    return (
        np.mean(topo_ok),
        np.mean(count_ok)
    )


rng = np.random.default_rng(
    20260916
)

active_rows = []


for method in pred:

    score = dispersion_score(
        method
    )

    order = np.argsort(
        score
    )[::-1]

    for b in BUDGETS:

        chosen = order[:b]

        ta, ca = multi_identification(
            method,
            chosen
        )

        random_top = []
        random_count = []

        for _ in range(300):

            ri = rng.choice(
                len(TEST),
                size=b,
                replace=False
            )

            rt, rc = multi_identification(
                method,
                ri
            )

            random_top.append(rt)
            random_count.append(rc)

        active_rows.append({
            "method": method,
            "budget": b,

            "active_topology_acc": ta,
            "random_topology_acc":
                np.mean(random_top),

            "active_count_acc": ca,
            "random_count_acc":
                np.mean(random_count),
        })


adf = pd.DataFrame(
    active_rows
)

adf.to_csv(
    OUT / "active_identification.csv",
    index=False
)


###############################################################################
# FIXED-MODEL LEARNING CURVE
#
# Diagnostic for the old "60 examples may simply be too few" issue.
###############################################################################

curve_rows = []

for ntrain in [
    20,40,60,80,100
]:

    fm = fit_fixed(
        TRAIN[:ntrain]
    )

    fp = {
        h: fixed_rollout(
            fm,
            qtest,
            h
        )
        for h in TOPOLOGIES
    }

    rs = []

    for a,b in CONTRAST_GROUPS["ALL"]:

        ps = np.sum(
            (
                fp[a]
                -
                fp[b]
            )**2,
            axis=1
        )

        ys = np.sum(
            (
                actual[a]
                -
                actual[b]
            )**2,
            axis=1
        )

        rs.append(
            rho_safe(
                ps,
                ys
            )
        )

    curve_rows.append({
        "ntrain": ntrain,
        "macro_all_rho":
            np.mean(rs)
    })


cdf = pd.DataFrame(
    curve_rows
)

cdf.to_csv(
    OUT / "learning_curve.csv",
    index=False
)


###############################################################################
# REPORT
###############################################################################

print("\n" + "="*72)
print("1. STRUCTURAL CONTRAST GENERALIZATION")
print("="*72)

print(
    rdf.pivot(
        index="group",
        columns="method",
        values=["rho","lift"]
    ).round(3).to_string()
)


print("\n" + "="*72)
print("2. SINGLE-QUERY IDENTIFICATION")
print("="*72)

print(
    idf.round(3).to_string(
        index=False
    )
)

print()
print(
    f"Topology random chance = "
    f"{1/len(TOPOLOGIES):.3f}"
)

counts = [
    count_of(x)
    for x in TOPOLOGIES
]

largest_count_prior = max(
    counts.count(c)
    for c in set(counts)
) / len(counts)

print(
    "Largest-class count prior = "
    f"{largest_count_prior:.3f}"
)


print("\n" + "="*72)
print("3. SEMANTIC LEAKAGE")
print("="*72)

print(
    sdf.round(3).to_string(
        index=False
    )
)


print("\n" + "="*72)
print("4. ACTIVE MULTI-QUERY IDENTIFICATION")
print("="*72)

print(
    adf.round(3).to_string(
        index=False
    )
)


print("\n" + "="*72)
print("5. FIXED OPERATOR LEARNING CURVE")
print("="*72)

print(
    cdf.round(3).to_string(
        index=False
    )
)


print("\nSaved:")
for f in [
    "big_contrast_results.csv",
    "single_query_identification.csv",
    "semantic_baselines.csv",
    "active_identification.csv",
    "learning_curve.csv",
]:
    print(OUT / f)


print("\nIMPORTANT INTERPRETATION:")
print(
    "- fixed works -> task-conditioned state was the main missing piece."
)
print(
    "- only learned lift works -> evidence for a learned shared execution space."
)
print(
    "- only direct/semantic works -> structure is observable, but operator algebra is not yet justified."
)
print(
    "- nothing works -> inspect stochasticity/observational identifiability before increasing model capacity."
)
print(
    "- this is still a mechanism benchmark; a final black-box attack must train on surrogate MAS and test on an unseen target using final I/O only."
)
