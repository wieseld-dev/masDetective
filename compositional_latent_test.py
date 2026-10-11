#!/usr/bin/env python3

# ============================================================
# COMPOSITIONAL LATENT EXECUTION SPACE
#
# Question:
# Can we learn phi() such that agent execution becomes
# approximately compositional:
#
#       z_next ~= W_agent z
#
# and, critically:
#
#       z_RABC ~= W_C W_B W_A z_R
#
# on LOCKED TEST prompts?
#
# No API calls. Uses existing execution_operator_big cache.
# ============================================================

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


CACHE = Path("artifacts/execution_operator_big/api_runs.jsonl")
OUT = Path("artifacts/compositional_latent")
OUT.mkdir(parents=True, exist_ok=True)

SEED_SPLIT = 20260916
TRAIN_N = 100

BASE_DIM = 64
LATENT = 32

STEPS = 5000
LR = 3e-4
WEIGHT_DECAY = 1e-5

SEEDS = [11, 22, 33, 44, 55]

# Loss weights
LAMBDA_ONE = 1.0
LAMBDA_ROLL2 = 1.0
LAMBDA_ROLL3 = 1.0
LAMBDA_RECON = 0.25
LAMBDA_VAR = 0.10
LAMBDA_COV = 0.01

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

print("=" * 80)
print("COMPOSITIONAL LATENT EXECUTION SPACE")
print("=" * 80)
print("DEVICE:", device)


# ============================================================
# LOAD CACHE
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
# EMBEDDINGS
# ============================================================

enc = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

tasks = [r["task"] for r in records]

q_raw = enc.encode(
    tasks,
    normalize_embeddings=True,
    convert_to_numpy=True,
    batch_size=64
)

raw = {}

for key in PREFIX_KEYS:
    print("Encoding", key)

    raw[key] = enc.encode(
        [r["outputs"][key] for r in records],
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )


# Fit representation ONLY on TRAIN.

pool = [q_raw[TRAIN]]

for key in PREFIX_KEYS:
    pool.append(raw[key][TRAIN])

pool = np.vstack(pool)

pca = PCA(
    n_components=BASE_DIM,
    random_state=SEED_SPLIT
)

pca.fit(pool)

train_transformed = pca.transform(pool)

scaler = StandardScaler()
scaler.fit(train_transformed)


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
#
# state encoder gets:
#   [state_embedding, task_embedding]
#
# q is therefore available in the latent execution state.
#
# Each specialist is a LINEAR latent operator.
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

    def step(self, z, agent):

        return self.ops[agent](z)


# ============================================================
# ANTI-COLLAPSE
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
        LATENT
    )


# ============================================================
# TRAINING
# ============================================================

def T(x, ids):

    return torch.tensor(
        x[ids],
        dtype=torch.float32,
        device=device
    )


def train_model(seed):

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = Model().to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    ids = TRAIN

    qb = T(q, ids)

    H = {
        key: T(data[key], ids)
        for key in PREFIX_KEYS
    }

    for step in range(1, STEPS + 1):

        opt.zero_grad()

        # Encode every TRUE state.
        Z = {
            key: model.encode(
                H[key],
                qb
            )
            for key in PREFIX_KEYS
        }

        # ------------------------------------------
        # One-step teacher-forced transitions
        # ------------------------------------------

        one_losses = []

        for depth in (1, 2, 3):

            for seq in PREFIX[depth]:

                parent_seq = seq[:-1]

                parent = (
                    "R" + parent_seq
                )

                child = "R" + seq

                agent = seq[-1]

                pred = model.step(
                    Z[parent],
                    agent
                )

                one_losses.append(
                    F.mse_loss(
                        pred,
                        Z[child]
                    )
                )

        one_loss = torch.stack(
            one_losses
        ).mean()

        # ------------------------------------------
        # Explicit 2-step composition
        #
        # Start from TRUE encoded R.
        # No teacher forcing between operators.
        # ------------------------------------------

        roll2_losses = []

        for seq in PREFIX[2]:

            z = Z["R"]

            for agent in seq:
                z = model.step(
                    z,
                    agent
                )

            roll2_losses.append(
                F.mse_loss(
                    z,
                    Z["R" + seq]
                )
            )

        roll2 = torch.stack(
            roll2_losses
        ).mean()

        # ------------------------------------------
        # Explicit 3-step composition
        # ------------------------------------------

        roll3_losses = []

        for seq in PREFIX[3]:

            z = Z["R"]

            for agent in seq:
                z = model.step(
                    z,
                    agent
                )

            roll3_losses.append(
                F.mse_loss(
                    z,
                    Z["R" + seq]
                )
            )

        roll3 = torch.stack(
            roll3_losses
        ).mean()

        # ------------------------------------------
        # Reconstruction / information preservation
        # ------------------------------------------

        recon_losses = []

        all_z = []

        for key in PREFIX_KEYS:

            z = Z[key]

            rec = model.decode(
                z,
                qb
            )

            recon_losses.append(
                F.mse_loss(
                    rec,
                    H[key]
                )
            )

            all_z.append(z)

        recon = torch.stack(
            recon_losses
        ).mean()

        zz = torch.cat(
            all_z,
            dim=0
        )

        vloss = variance_loss(zz)
        closs = covariance_loss(zz)

        loss = (
            LAMBDA_ONE * one_loss
            + LAMBDA_ROLL2 * roll2
            + LAMBDA_ROLL3 * roll3
            + LAMBDA_RECON * recon
            + LAMBDA_VAR * vloss
            + LAMBDA_COV * closs
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0
        )

        opt.step()

        if (
            step == 1
            or step % 500 == 0
        ):

            print(
                f"seed={seed} "
                f"step={step:04d} "
                f"loss={loss.item():.5f} "
                f"one={one_loss.item():.5f} "
                f"r2={roll2.item():.5f} "
                f"r3={roll3.item():.5f} "
                f"recon={recon.item():.5f} "
                f"var={vloss.item():.5f}"
            )

    return model


# ============================================================
# TEST
# ============================================================

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


@torch.no_grad()
def evaluate(model, seed):

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

    depth_rows = []

    predicted = {}

    # Roll out from TRUE encoded R.
    for depth in (1, 2, 3):

        teacher_errors = []
        rollout_errors = []

        for seq in PREFIX[depth]:

            parent = (
                "R" + seq[:-1]
            )

            child = "R" + seq

            agent = seq[-1]

            # Teacher forced.
            teacher = model.step(
                Z[parent],
                agent
            )

            teacher_errors.append(
                F.mse_loss(
                    teacher,
                    Z[child]
                ).item()
            )

            # Full composition from R.
            z = Z["R"]

            for a in seq:
                z = model.step(
                    z,
                    a
                )

            predicted[seq] = (
                z.cpu().numpy()
            )

            rollout_errors.append(
                F.mse_loss(
                    z,
                    Z[child]
                ).item()
            )

        teacher_mean = float(
            np.mean(teacher_errors)
        )

        rollout_mean = float(
            np.mean(rollout_errors)
        )

        depth_rows.append({
            "seed": seed,
            "depth": depth,
            "teacher_mse": teacher_mean,
            "rollout_mse": rollout_mean,
            "rollout_over_teacher":
                rollout_mean
                /
                max(
                    teacher_mean,
                    1e-12
                ),
        })

    # ------------------------------------------
    # Structural contrast
    #
    # Compare predicted separation between
    # alternative sequences to actual output
    # separation in ORIGINAL semantic space.
    # ------------------------------------------

    contrast_rows = []

    for depth in (2, 3):

        seqs = PREFIX[depth]

        for a, b in itertools.combinations(
            seqs,
            2
        ):

            predicted_sep = np.sum(
                (
                    predicted[a]
                    -
                    predicted[b]
                ) ** 2,
                axis=1
            )

            actual_sep = np.sum(
                (
                    data["R" + a][TEST]
                    -
                    data["R" + b][TEST]
                ) ** 2,
                axis=1
            )

            contrast_rows.append({
                "seed": seed,
                "depth": depth,
                "pair": a + "_vs_" + b,
                "rho": safe_rho(
                    predicted_sep,
                    actual_sep
                ),
                "lift20": lift20(
                    predicted_sep,
                    actual_sep
                ),
            })

    # ------------------------------------------
    # Spectrum of latent operators
    # ------------------------------------------

    spectral_rows = []

    for agent in SPECIALISTS:

        W = (
            model.ops[agent]
            .weight
            .detach()
            .cpu()
            .numpy()
        )

        eig = np.linalg.eigvals(W)

        sv = np.linalg.svd(
            W,
            compute_uv=False
        )

        spectral_rows.append({
            "seed": seed,
            "agent": agent,
            "spectral_radius":
                np.max(np.abs(eig)),
            "sigma1": sv[0],
            "frobenius":
                np.linalg.norm(
                    W,
                    "fro"
                ),
        })

    return (
        depth_rows,
        contrast_rows,
        spectral_rows
    )


# ============================================================
# RUN SEEDS
# ============================================================

all_depth = []
all_contrast = []
all_spectral = []

for seed in SEEDS:

    print("\n" + "=" * 80)
    print("TRAIN SEED", seed)
    print("=" * 80)

    model = train_model(seed)

    d, c, s = evaluate(
        model,
        seed
    )

    all_depth.extend(d)
    all_contrast.extend(c)
    all_spectral.extend(s)


depth_df = pd.DataFrame(
    all_depth
)

contrast_df = pd.DataFrame(
    all_contrast
)

spectral_df = pd.DataFrame(
    all_spectral
)

depth_df.to_csv(
    OUT / "depth_results.csv",
    index=False
)

contrast_df.to_csv(
    OUT / "contrast_results.csv",
    index=False
)

spectral_df.to_csv(
    OUT / "spectral_results.csv",
    index=False
)


# ============================================================
# SUMMARIES
# ============================================================

depth_summary = (
    depth_df
    .groupby("depth")
    .agg(
        teacher_mean=(
            "teacher_mse",
            "mean"
        ),
        teacher_std=(
            "teacher_mse",
            "std"
        ),
        rollout_mean=(
            "rollout_mse",
            "mean"
        ),
        rollout_std=(
            "rollout_mse",
            "std"
        ),
        ratio_mean=(
            "rollout_over_teacher",
            "mean"
        ),
        ratio_std=(
            "rollout_over_teacher",
            "std"
        ),
    )
    .reset_index()
)

contrast_summary = (
    contrast_df
    .groupby("depth")
    .agg(
        rho_mean=("rho", "mean"),
        rho_std=("rho", "std"),
        lift_mean=("lift20", "mean"),
        lift_std=("lift20", "std"),
    )
    .reset_index()
)

spectral_summary = (
    spectral_df
    .groupby("agent")
    .agg(
        radius_mean=(
            "spectral_radius",
            "mean"
        ),
        radius_std=(
            "spectral_radius",
            "std"
        ),
        sigma1_mean=(
            "sigma1",
            "mean"
        ),
    )
    .reset_index()
)

depth_summary.to_csv(
    OUT / "depth_summary.csv",
    index=False
)

contrast_summary.to_csv(
    OUT / "contrast_summary.csv",
    index=False
)

spectral_summary.to_csv(
    OUT / "spectral_summary.csv",
    index=False
)


print("\n" + "=" * 80)
print("LOCKED TEST: COMPOSITION")
print("=" * 80)

print(
    depth_summary
    .round(5)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("LOCKED TEST: STRUCTURAL CONTRAST")
print("=" * 80)

print(
    contrast_summary
    .round(5)
    .to_string(index=False)
)

print("\n" + "=" * 80)
print("LATENT OPERATOR SPECTRUM")
print("=" * 80)

print(
    spectral_summary
    .round(5)
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("PRE-SPECIFIED DECISION GATES")
print("=" * 80)

d3 = depth_summary[
    depth_summary.depth == 3
].iloc[0]

c3 = contrast_summary[
    contrast_summary.depth == 3
].iloc[0]

print(
    f"depth3 rollout/teacher = "
    f"{d3.ratio_mean:.3f}x"
)

print(
    f"depth3 structural rho = "
    f"{c3.rho_mean:+.3f}"
)

print(
    f"depth3 lift20 = "
    f"{c3.lift_mean:.3f}x"
)

print("""
Reference from fixed-space drift diagnostic:
    depth1 ratio ~ 2.30x
    depth2 ratio ~ 2.55x
    depth3 ratio ~ 2.74x

Interpretation:

STRONG SUPPORT:
  TEST depth3 ratio materially below 2.74x
  AND held-out structural rho/lift improve or remain useful.

REPRESENTATION FIT ONLY:
  latent rollout error improves,
  but structural rho/lift collapse.
  => phi learned an easy predictive coordinate system,
     not useful structural execution geometry.

NO SUPPORT:
  TEST ratio remains around fixed-space baseline
  and structural contrast does not improve.
  => this learned lifting does not solve compositional drift.

IMPORTANT:
  Do not judge success from TRAIN loss.
  The claim lives or dies on LOCKED TEST composition
  + structural contrast.
""")
