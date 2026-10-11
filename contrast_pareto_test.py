#!/usr/bin/env python3

import json
import itertools
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

CACHE = Path("artifacts/execution_operator_big/api_runs.jsonl")
OUT = Path("artifacts/contrast_pareto")
OUT.mkdir(parents=True, exist_ok=True)

SEED_SPLIT = 20260916
TRAIN_N = 100

BASE_DIM = 64
LATENT = 32

STEPS = 4000
LR = 3e-4
WEIGHT_DECAY = 1e-5

# 6 lambdas x 3 seeds = 18 trainings
LAMBDAS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
SEEDS = [11, 22, 33]

L_ONE = 1.0
L_ROLL2 = 1.0
L_ROLL3 = 1.0
L_RECON = 0.25
L_VAR = 0.10
L_COV = 0.01

SPECIALISTS = "ABC"

PREFIX = {
    k: [
        "".join(p)
        for p in itertools.permutations(SPECIALISTS, k)
    ]
    for k in (1, 2, 3)
}

PREFIX_KEYS = (
    ["R"]
    + ["R" + s for k in (1, 2, 3) for s in PREFIX[k]]
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 88)
print("CONTRAST-PRESERVING COMPOSITIONAL LATENT SPACE")
print("Pareto: compositionality vs structural discriminability")
print("=" * 88)
print("DEVICE:", device)
print("LAMBDAS:", LAMBDAS)
print("SEEDS:", SEEDS)


# ============================================================
# LOAD EXACT SAME 160 PROBES
# ============================================================

records = []

with CACHE.open(encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

records = {
    int(r["position"]): r
    for r in records
}

records = [
    records[k]
    for k in sorted(records)
][:160]

assert len(records) == 160

TRAIN = np.arange(TRAIN_N)
TEST = np.arange(TRAIN_N, 160)

print("TOTAL:", len(records))
print("TRAIN:", len(TRAIN))
print("TEST:", len(TEST))


# ============================================================
# FIXED SEMANTIC SPACE
# ============================================================

st = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tasks = [r["task"] for r in records]

q_raw = st.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)

raw = {}

for key in PREFIX_KEYS:

    print("Encoding", key)

    raw[key] = st.encode(
        [r["outputs"][key] for r in records],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )

# Fit ONLY TRAIN.
pool = [q_raw[TRAIN]]

for key in PREFIX_KEYS:
    pool.append(raw[key][TRAIN])

pool = np.vstack(pool)

pca = PCA(
    n_components=BASE_DIM,
    random_state=SEED_SPLIT
)

pca.fit(pool)

scaler = StandardScaler()
scaler.fit(
    pca.transform(pool)
)


def transform(x):

    return scaler.transform(
        pca.transform(x)
    ).astype(np.float32)


q = transform(q_raw)

data = {
    k: transform(v)
    for k, v in raw.items()
}


# ============================================================
# MODEL
# ============================================================

class Model(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(BASE_DIM * 2, 128),
            nn.GELU(),
            nn.Linear(128, LATENT),
        )

        self.decoder = nn.Sequential(
            nn.Linear(LATENT + BASE_DIM, 128),
            nn.GELU(),
            nn.Linear(128, BASE_DIM),
        )

        # Critical scientific constraint:
        # agent dynamics remain LINEAR in learned execution space.
        self.ops = nn.ModuleDict({
            a: nn.Linear(
                LATENT,
                LATENT,
                bias=True
            )
            for a in SPECIALISTS
        })

    def encode(self, h, qv):

        return self.encoder(
            torch.cat([h, qv], dim=-1)
        )

    def decode(self, z, qv):

        return self.decoder(
            torch.cat([z, qv], dim=-1)
        )

    def step(self, z, a):

        return self.ops[a](z)


# ============================================================
# REGULARIZATION
# ============================================================

def variance_loss(z):

    std = torch.sqrt(
        z.var(dim=0) + 1e-4
    )

    return torch.mean(
        F.relu(1.0 - std)
    )


def covariance_loss(z):

    z = z - z.mean(
        dim=0,
        keepdim=True
    )

    cov = (
        z.T @ z
        /
        max(z.shape[0] - 1, 1)
    )

    off = cov - torch.diag(
        torch.diag(cov)
    )

    return (
        off.pow(2).sum()
        /
        z.shape[1]
    )


# ============================================================
# CONTRAST GEOMETRY
#
# For each query:
#
# actual pairwise squared distances between execution paths
# predicted pairwise squared distances between rollouts
#
# We normalize each query's distance vector before comparing.
# Thus loss preserves relative contrast geometry, not arbitrary
# cross-space scale.
# ============================================================

PAIR_CACHE = {}

for depth in (2, 3):

    seqs = PREFIX[depth]

    PAIR_CACHE[depth] = list(
        itertools.combinations(
            range(len(seqs)),
            2
        )
    )


def pairwise_distance_vector(states, depth):

    # states: list of [B,D], one per sequence
    vals = []

    for i, j in PAIR_CACHE[depth]:

        vals.append(
            (
                (states[i] - states[j]) ** 2
            ).mean(dim=-1)
        )

    # [B, num_pairs]
    return torch.stack(
        vals,
        dim=1
    )


def normalize_contrasts(d):

    # d = [num_queries, num_topology_pairs]
    #
    # Normalize EACH topology contrast ACROSS queries.
    # The target is probe discriminability:
    # for a fixed structural hypothesis pair, rank which
    # queries separate it most strongly.

    mu = d.mean(
        dim=0,
        keepdim=True
    )

    sd = d.std(
        dim=0,
        keepdim=True
    ).clamp_min(1e-4)

    return (
        d - mu
    ) / sd


# ============================================================
# HELPERS
# ============================================================

def T(x, ids):

    return torch.tensor(
        x[ids],
        dtype=torch.float32,
        device=device
    )


def safe_rho(x, y):

    r = spearmanr(
        x,
        y
    ).statistic

    if not np.isfinite(r):
        return 0.0

    return float(r)


def lift20(score, actual):

    k = max(
        1,
        int(np.ceil(
            0.20 * len(score)
        ))
    )

    idx = np.argsort(score)[-k:]

    return float(
        actual[idx].mean()
        /
        max(actual.mean(), 1e-12)
    )


# ============================================================
# TRAIN
# ============================================================

def train_model(seed, lambda_contrast):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Model().to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    qb = T(q, TRAIN)

    H = {
        key: T(data[key], TRAIN)
        for key in PREFIX_KEYS
    }

    # Actual semantic contrast geometry is a fixed target.
    actual_contrast = {}

    for depth in (2, 3):

        actual_states = [
            H["R" + seq]
            for seq in PREFIX[depth]
        ]

        actual_contrast[depth] = (
            normalize_contrasts(
                pairwise_distance_vector(
                    actual_states,
                    depth
                )
            ).detach()
        )

    for step in range(1, STEPS + 1):

        opt.zero_grad()

        Z = {
            key: model.encode(
                H[key],
                qb
            )
            for key in PREFIX_KEYS
        }

        # --------------------------------------------
        # 1-step teacher-forced
        # --------------------------------------------

        one_terms = []

        for depth in (1, 2, 3):

            for seq in PREFIX[depth]:

                parent = "R" + seq[:-1]
                child = "R" + seq
                agent = seq[-1]

                pred = model.step(
                    Z[parent],
                    agent
                )

                one_terms.append(
                    F.mse_loss(
                        pred,
                        Z[child]
                    )
                )

        one = torch.stack(
            one_terms
        ).mean()

        # --------------------------------------------
        # rollout composition + predicted states
        # --------------------------------------------

        rollout_states = {}

        roll_losses = {}

        for depth in (2, 3):

            states = []
            terms = []

            for seq in PREFIX[depth]:

                z = Z["R"]

                for a in seq:
                    z = model.step(
                        z,
                        a
                    )

                states.append(z)

                terms.append(
                    F.mse_loss(
                        z,
                        Z["R" + seq]
                    )
                )

            rollout_states[depth] = states

            roll_losses[depth] = (
                torch.stack(
                    terms
                ).mean()
            )

        # --------------------------------------------
        # contrast-preservation
        # --------------------------------------------

        contrast_terms = []

        for depth in (2, 3):

            pred_d = pairwise_distance_vector(
                rollout_states[depth],
                depth
            )

            pred_d = normalize_contrasts(
                pred_d
            )

            contrast_terms.append(
                F.mse_loss(
                    pred_d,
                    actual_contrast[depth]
                )
            )

        contrast = torch.stack(
            contrast_terms
        ).mean()

        # --------------------------------------------
        # reconstruction + anti-collapse
        # --------------------------------------------

        recon_terms = []
        all_z = []

        for key in PREFIX_KEYS:

            z = Z[key]

            rec = model.decode(
                z,
                qb
            )

            recon_terms.append(
                F.mse_loss(
                    rec,
                    H[key]
                )
            )

            all_z.append(z)

        recon = torch.stack(
            recon_terms
        ).mean()

        zz = torch.cat(
            all_z,
            dim=0
        )

        var = variance_loss(zz)
        cov = covariance_loss(zz)

        loss = (
            L_ONE * one
            + L_ROLL2 * roll_losses[2]
            + L_ROLL3 * roll_losses[3]
            + L_RECON * recon
            + L_VAR * var
            + L_COV * cov
            + lambda_contrast * contrast
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        opt.step()

        if (
            step == 1
            or step % 1000 == 0
        ):

            print(
                f"lambda={lambda_contrast:g} "
                f"seed={seed} "
                f"step={step:04d} "
                f"loss={loss.item():.5f} "
                f"one={one.item():.5f} "
                f"r2={roll_losses[2].item():.5f} "
                f"r3={roll_losses[3].item():.5f} "
                f"contrast={contrast.item():.5f} "
                f"recon={recon.item():.5f}"
            )

    return model


# ============================================================
# LOCKED TEST EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, seed, lambda_contrast):

    model.eval()

    qb = T(q, TEST)

    H = {
        key: T(data[key], TEST)
        for key in PREFIX_KEYS
    }

    Z = {
        key: model.encode(
            H[key],
            qb
        )
        for key in PREFIX_KEYS
    }

    rows = []

    # --------------------------------------------
    # Composition
    # --------------------------------------------

    predicted = {}

    for depth in (1, 2, 3):

        teacher = []
        rollout = []

        for seq in PREFIX[depth]:

            parent = "R" + seq[:-1]
            child = "R" + seq
            agent = seq[-1]

            z_teacher = model.step(
                Z[parent],
                agent
            )

            teacher.append(
                F.mse_loss(
                    z_teacher,
                    Z[child]
                ).item()
            )

            z = Z["R"]

            for a in seq:
                z = model.step(
                    z,
                    a
                )

            predicted[seq] = (
                z.cpu().numpy()
            )

            rollout.append(
                F.mse_loss(
                    z,
                    Z[child]
                ).item()
            )

        tm = float(
            np.mean(teacher)
        )

        rm = float(
            np.mean(rollout)
        )

        rows.append({
            "lambda": lambda_contrast,
            "seed": seed,
            "depth": depth,
            "teacher_mse": tm,
            "rollout_mse": rm,
            "ratio": rm / max(tm, 1e-12),
        })

    # --------------------------------------------
    # Structural ranking
    #
    # Same philosophy as earlier experiments:
    # across TEST prompts, does predicted separation
    # rank actual separation?
    # --------------------------------------------

    structural = []

    for depth in (2, 3):

        seqs = PREFIX[depth]

        for a, b in itertools.combinations(
            seqs,
            2
        ):

            pred_sep = np.mean(
                (
                    predicted[a]
                    -
                    predicted[b]
                ) ** 2,
                axis=1
            )

            actual_sep = np.mean(
                (
                    data["R" + a][TEST]
                    -
                    data["R" + b][TEST]
                ) ** 2,
                axis=1
            )

            structural.append({
                "lambda": lambda_contrast,
                "seed": seed,
                "depth": depth,
                "pair": a + "_vs_" + b,
                "rho": safe_rho(
                    pred_sep,
                    actual_sep
                ),
                "lift20": lift20(
                    pred_sep,
                    actual_sep
                ),
            })

    # --------------------------------------------
    # TEST contrast-geometry MSE itself
    # --------------------------------------------

    geom = []

    for depth in (2, 3):

        pred_states = [
            torch.tensor(
                predicted[s],
                dtype=torch.float32,
                device=device
            )
            for s in PREFIX[depth]
        ]

        actual_states = [
            H["R" + s]
            for s in PREFIX[depth]
        ]

        pdist = normalize_contrasts(
            pairwise_distance_vector(
                pred_states,
                depth
            )
        )

        adist = normalize_contrasts(
            pairwise_distance_vector(
                actual_states,
                depth
            )
        )

        geom.append({
            "lambda": lambda_contrast,
            "seed": seed,
            "depth": depth,
            "test_geometry_mse":
                F.mse_loss(
                    pdist,
                    adist
                ).item()
        })

    return rows, structural, geom


# ============================================================
# RUN PARETO SWEEP
# ============================================================

all_comp = []
all_struct = []
all_geom = []

for lam in LAMBDAS:

    print("\n" + "#" * 88)
    print("LAMBDA CONTRAST =", lam)
    print("#" * 88)

    for seed in SEEDS:

        model = train_model(
            seed,
            lam
        )

        c, s, g = evaluate(
            model,
            seed,
            lam
        )

        all_comp.extend(c)
        all_struct.extend(s)
        all_geom.extend(g)

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


comp = pd.DataFrame(all_comp)
struct = pd.DataFrame(all_struct)
geom = pd.DataFrame(all_geom)

comp.to_csv(
    OUT / "composition_all.csv",
    index=False
)

struct.to_csv(
    OUT / "structural_all.csv",
    index=False
)

geom.to_csv(
    OUT / "geometry_all.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

comp_summary = (
    comp.groupby(
        ["lambda", "depth"]
    )
    .agg(
        teacher_mean=("teacher_mse", "mean"),
        rollout_mean=("rollout_mse", "mean"),
        ratio_mean=("ratio", "mean"),
        ratio_std=("ratio", "std"),
    )
    .reset_index()
)

struct_summary = (
    struct.groupby(
        ["lambda", "depth"]
    )
    .agg(
        rho_mean=("rho", "mean"),
        rho_std=("rho", "std"),
        lift_mean=("lift20", "mean"),
        lift_std=("lift20", "std"),
    )
    .reset_index()
)

geom_summary = (
    geom.groupby(
        ["lambda", "depth"]
    )
    .agg(
        geometry_mse_mean=(
            "test_geometry_mse",
            "mean"
        ),
        geometry_mse_std=(
            "test_geometry_mse",
            "std"
        ),
    )
    .reset_index()
)

summary = (
    comp_summary
    .merge(
        struct_summary,
        on=["lambda", "depth"]
    )
    .merge(
        geom_summary,
        on=["lambda", "depth"]
    )
)

summary.to_csv(
    OUT / "pareto_summary.csv",
    index=False
)


print("\n" + "=" * 100)
print("LOCKED TEST PARETO SUMMARY")
print("=" * 100)

print(
    summary
    .round(5)
    .to_string(index=False)
)


# ============================================================
# DEPTH-3 FRONTIER
# ============================================================

d3 = (
    summary[
        summary.depth == 3
    ]
    .copy()
    .sort_values("lambda")
)

print("\n" + "=" * 100)
print("DEPTH-3 DECISION TABLE")
print("=" * 100)

print(
    d3[
        [
            "lambda",
            "ratio_mean",
            "rho_mean",
            "rho_std",
            "lift_mean",
            "lift_std",
            "geometry_mse_mean",
        ]
    ]
    .round(5)
    .to_string(index=False)
)


# Pareto nondomination:
# lower ratio is better, higher rho is better.

pareto = []

for i, row in d3.iterrows():

    dominated = False

    for j, other in d3.iterrows():

        if i == j:
            continue

        no_worse = (
            other["ratio_mean"]
            <= row["ratio_mean"]
            and
            other["rho_mean"]
            >= row["rho_mean"]
        )

        strictly_better = (
            other["ratio_mean"]
            < row["ratio_mean"]
            or
            other["rho_mean"]
            > row["rho_mean"]
        )

        if no_worse and strictly_better:
            dominated = True
            break

    if not dominated:
        pareto.append(row)

pareto_df = pd.DataFrame(pareto)

pareto_df.to_csv(
    OUT / "pareto_frontier_depth3.csv",
    index=False
)

print("\n" + "=" * 100)
print("NON-DOMINATED DEPTH-3 POINTS")
print("=" * 100)

print(
    pareto_df[
        [
            "lambda",
            "ratio_mean",
            "rho_mean",
            "lift_mean",
            "geometry_mse_mean",
        ]
    ]
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 100)
print("PRE-SPECIFIED INTERPRETATION")
print("=" * 100)

print("""
Known references:

FIXED semantic representation:
    depth3 rollout/teacher ~ 2.74x
    structural rho roughly ~ 0.16-0.19
    lift roughly ~ 1.30x

UNCONSTRAINED learned latent:
    depth3 rollout/teacher ~ 1.15x
    structural rho ~ 0.055
    lift ~ 1.026x


What would be genuinely interesting:

1) BEST CASE
   A lambda gives:
       ratio << 2.74x
   while:
       rho >= roughly fixed-space signal (~0.18)
   and useful lift.

   => evidence that compositionality and structural
      discriminability can coexist in one learned execution space.

2) TRADE-OFF
   Increasing lambda restores rho/lift only while ratio moves
   back toward ~2.74x.

   => evidence for a representation trade-off under the current
      shared 32D linear-operator formulation.

3) CONTRAST LOSS DOES NOT GENERALIZE
   TRAIN contrast becomes tiny but TEST rho stays near zero.

   => structural geometry is being memorized on TRAIN prompts.

4) RHO IMPROVES, LIFT DOES NOT
   => broad ranking improves but the highest-value active probes
      are still not identified reliably.

Do NOT choose a winner from training loss.
The result is the locked TEST Pareto curve.
""")


print("\nSaved:")
print(OUT / "composition_all.csv")
print(OUT / "structural_all.csv")
print(OUT / "geometry_all.csv")
print(OUT / "pareto_summary.csv")
print(OUT / "pareto_frontier_depth3.csv")
