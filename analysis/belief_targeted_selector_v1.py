#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer


SEED = 20261008

DATA = Path("artifacts/q38_selector_v41_data")
OUT = Path("artifacts/belief_targeted_selector_v1")

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
    "qa_testing",
    "translation",
    "financial_analysis",
    "database_design",
    "accessibility_review",
    "fact_checking",
    "requirements_analysis",
    "incident_management",
    "experiment_design",
    "risk_assessment",
    "documentation",
    "customer_support_triage",
]

ROLE_DESCRIPTIONS = {
    "code":
        "software coding programming debugging implementation algorithms",

    "writing":
        "writing prose editing rewriting composition communication",

    "math":
        "mathematics calculation equations quantitative reasoning",

    "planning":
        "planning scheduling decomposition strategy task coordination",

    "research":
        "research literature search investigation evidence synthesis",

    "data":
        "data analysis statistics datasets visualization analytics",

    "design":
        "design architecture user experience visual product design",

    "operations":
        "operations procedures workflows execution logistics",

    "qa_testing":
        "quality assurance testing validation test cases software verification",

    "translation":
        "translation multilingual language localization",

    "financial_analysis":
        "financial analysis accounting valuation budgeting finance",

    "database_design":
        "database schema SQL normalization data modeling",

    "accessibility_review":
        "accessibility review inclusive design WCAG usability",

    "fact_checking":
        "fact checking verification claims evidence accuracy",

    "requirements_analysis":
        "requirements analysis specifications constraints stakeholder needs",

    "incident_management":
        "incident response troubleshooting outage management escalation",

    "experiment_design":
        "experiment design hypothesis testing controls evaluation methodology",

    "risk_assessment":
        "risk assessment threats hazards likelihood impact mitigation",

    "documentation":
        "technical documentation manuals guides API documentation",

    "customer_support_triage":
        "customer support triage tickets diagnosis prioritization escalation",
}

N_ROLES = len(ROLES)

EPOCHS = 35
PATIENCE = 6
BATCH_GROUPS = 64

LR = 3e-4
WEIGHT_DECAY = 1e-3

LISTWISE_TEMP = 0.025
PAIR_EPS = 0.005

ENSEMBLE_SEEDS = [
    20261008,
    20261019,
    20261031,
    20261113,
    20261127,
]


# ============================================================
# IO
# ============================================================

def read_jsonl(path):
    rows = []

    with Path(path).open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


# ============================================================
# EMBEDDINGS
# ============================================================

def encode_unique(
    encoder,
    texts,
    batch_size=256,
):
    unique = list(
        dict.fromkeys(
            texts
        )
    )

    vec = encoder.encode(
        unique,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(
        np.float32
    )

    return {
        t: v
        for t, v
        in zip(
            unique,
            vec,
        )
    }


# ============================================================
# FEATURES
#
# Selector sees ONLY:
#   query
#   current belief
#   public history
#   round
#
# NO future response
# NO sensor evidence
# NO truth
# NO activation
#
# Structured features:
#
# p_r                  current belief
# uncertainty_r        4*p*(1-p)
# entropy_r            Bernoulli entropy
# query-role semantic alignment
# history-role semantic alignment
# query-history novelty
# query embedding
# round
# ============================================================

def entropy01(p):
    p = np.clip(
        p,
        1e-6,
        1 - 1e-6,
    )

    h = (
        -p * np.log(p)
        - (1-p) * np.log(1-p)
    )

    return (
        h
        / math.log(2.0)
    ).astype(
        np.float32
    )


def belief_vector(group):
    rp = group[
        "role_probs_before"
    ]

    return np.asarray(
        [
            float(
                rp[r]
            )
            for r in ROLES
        ],
        dtype=np.float32,
    )


def build_feature(
    group,
    query,
    qmap,
    hmap,
    role_matrix,
):
    p = belief_vector(
        group
    )

    uncertainty = (
        4.0
        * p
        * (1.0-p)
    ).astype(
        np.float32
    )

    ent = entropy01(
        p
    )

    qv = qmap[
        query
    ]

    hist = str(
        group.get(
            "history_text",
            "- no previous public interaction",
        )
    )

    hv = hmap[
        hist
    ]

    # Semantic relationship of candidate to each hidden-role hypothesis.
    q_role = (
        role_matrix
        @ qv
    ).astype(
        np.float32
    )

    # Which roles have already been semantically exercised in public history?
    h_role = (
        role_matrix
        @ hv
    ).astype(
        np.float32
    )

    # Candidate novelty w.r.t. public trajectory.
    q_hist = np.asarray(
        [
            float(
                np.dot(
                    qv,
                    hv,
                )
            )
        ],
        dtype=np.float32,
    )

    rnd = np.asarray(
        [
            float(
                group[
                    "round"
                ]
            )
            / 8.0
        ],
        dtype=np.float32,
    )

    # Explicit belief-targeting interactions.
    uncertain_alignment = (
        uncertainty
        * q_role
    ).astype(
        np.float32
    )

    entropy_alignment = (
        ent
        * q_role
    ).astype(
        np.float32
    )

    unexplored_alignment = (
        q_role
        - h_role
    ).astype(
        np.float32
    )

    return np.concatenate(
        [
            qv,
            p,
            uncertainty,
            ent,
            q_role,
            h_role,
            uncertain_alignment,
            entropy_alignment,
            unexplored_alignment,
            q_hist,
            rnd,
        ]
    ).astype(
        np.float32
    )


# ============================================================
# PACK GROUPS
# ============================================================

def pack_groups(
    groups,
    qmap,
    hmap,
    role_matrix,
):
    Kmax = max(
        len(
            g["queries"]
        )
        for g in groups
    )

    first = build_feature(
        groups[0],
        groups[0]["queries"][0],
        qmap,
        hmap,
        role_matrix,
    )

    D = len(
        first
    )

    X = np.zeros(
        (
            len(groups),
            Kmax,
            D,
        ),
        dtype=np.float32,
    )

    G = np.zeros(
        (
            len(groups),
            Kmax,
        ),
        dtype=np.float32,
    )

    M = np.zeros(
        (
            len(groups),
            Kmax,
        ),
        dtype=bool,
    )

    systems = []
    state_ids = []
    queries = []

    for gi, group in enumerate(
        groups
    ):
        qs = list(
            group["queries"]
        )

        gains = np.asarray(
            group["auc_gains"],
            dtype=np.float32,
        )

        if len(qs) != len(gains):
            raise RuntimeError(
                f"queries/gains mismatch: "
                f"{group.get('state_id')}"
            )

        for j, q in enumerate(
            qs
        ):
            X[
                gi,
                j,
            ] = build_feature(
                group,
                q,
                qmap,
                hmap,
                role_matrix,
            )

            G[
                gi,
                j,
            ] = gains[
                j
            ]

            M[
                gi,
                j,
            ] = True

        systems.append(
            str(
                group[
                    "system_id"
                ]
            )
        )

        state_ids.append(
            str(
                group[
                    "state_id"
                ]
            )
        )

        queries.append(
            qs
        )

    return {
        "X":
            X,

        "gains":
            G,

        "mask":
            M,

        "systems":
            systems,

        "state_ids":
            state_ids,

        "queries":
            queries,

        "dim":
            D,

        "kmax":
            Kmax,
    }


# ============================================================
# MODEL
# ============================================================

class SelectorNet(nn.Module):
    def __init__(
        self,
        dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                512,
            ),
            nn.GELU(),
            nn.LayerNorm(
                512
            ),
            nn.Dropout(
                .15
            ),

            nn.Linear(
                512,
                256,
            ),
            nn.GELU(),
            nn.Dropout(
                .10
            ),

            nn.Linear(
                256,
                64,
            ),
            nn.GELU(),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        ).squeeze(
            -1
        )


# ============================================================
# GROUP LOSSES
# ============================================================

def group_losses(
    pred,
    gain,
    mask,
):
    # --------------------------------------------------------
    # LISTWISE:
    # target probability is concentrated on genuinely
    # higher-belief-improvement queries.
    # --------------------------------------------------------

    masked_pred = pred.masked_fill(
        ~mask,
        -1e9,
    )

    masked_gain = gain.masked_fill(
        ~mask,
        -1e9,
    )

    target_prob = F.softmax(
        masked_gain
        / LISTWISE_TEMP,
        dim=1,
    )

    log_pred = F.log_softmax(
        masked_pred,
        dim=1,
    )

    list_loss = -(
        target_prob
        * log_pred
    ).sum(
        dim=1
    ).mean()

    # --------------------------------------------------------
    # PAIRWISE:
    # only meaningful realized-gain differences.
    # --------------------------------------------------------

    pair_losses = []

    B, K = pred.shape

    for i in range(K):
        for j in range(i+1, K):

            valid = (
                mask[:, i]
                & mask[:, j]
            )

            if not valid.any():
                continue

            dg = (
                gain[
                    valid,
                    i
                ]
                -
                gain[
                    valid,
                    j
                ]
            )

            keep = (
                torch.abs(
                    dg
                )
                >= PAIR_EPS
            )

            if not keep.any():
                continue

            dp = (
                pred[
                    valid,
                    i
                ]
                -
                pred[
                    valid,
                    j
                ]
            )[
                keep
            ]

            sign = torch.sign(
                dg[
                    keep
                ]
            )

            weight = torch.clamp(
                torch.abs(
                    dg[
                        keep
                    ]
                )
                / 0.05,
                min=.25,
                max=3.0,
            )

            pl = (
                F.softplus(
                    -sign
                    * dp
                )
                * weight
            ).mean()

            pair_losses.append(
                pl
            )

    if pair_losses:
        pair_loss = torch.stack(
            pair_losses
        ).mean()
    else:
        pair_loss = torch.zeros(
            (),
            device=pred.device,
        )

    # --------------------------------------------------------
    # WITHIN-STATE STANDARDIZED REGRESSION.
    #
    # We care about relative utility, not global gain scale.
    # --------------------------------------------------------

    reg_losses = []

    for b in range(
        pred.shape[0]
    ):
        v = mask[
            b
        ]

        pp = pred[
            b,
            v
        ]

        gg = gain[
            b,
            v
        ]

        if len(pp) < 2:
            continue

        ppz = (
            pp
            - pp.mean()
        ) / (
            pp.std(
                unbiased=False
            )
            + 1e-4
        )

        ggz = (
            gg
            - gg.mean()
        ) / (
            gg.std(
                unbiased=False
            )
            + 0.01
        )

        reg_losses.append(
            F.smooth_l1_loss(
                ppz,
                ggz,
            )
        )

    if reg_losses:
        reg_loss = torch.stack(
            reg_losses
        ).mean()
    else:
        reg_loss = torch.zeros(
            (),
            device=pred.device,
        )

    total = (
        list_loss
        + .55 * pair_loss
        + .15 * reg_loss
    )

    return (
        total,
        list_loss,
        pair_loss,
        reg_loss,
    )


# ============================================================
# EVALUATION
# ============================================================

def rank_eval(
    packed,
    scores,
):
    gains = packed[
        "gains"
    ]

    mask = packed[
        "mask"
    ]

    systems = packed[
        "systems"
    ]

    selected = []
    randoms = []
    oracles = []
    top1 = []
    regrets = []
    rhos = []
    pair_accs = []

    rows = []

    for i in range(
        len(gains)
    ):
        v = np.flatnonzero(
            mask[
                i
            ]
        )

        g = gains[
            i,
            v
        ]

        s = scores[
            i,
            v
        ]

        choose_local = int(
            np.argmax(
                s
            )
        )

        oracle_local = int(
            np.argmax(
                g
            )
        )

        sg = float(
            g[
                choose_local
            ]
        )

        rg = float(
            np.mean(
                g
            )
        )

        og = float(
            np.max(
                g
            )
        )

        selected.append(
            sg
        )

        randoms.append(
            rg
        )

        oracles.append(
            og
        )

        regrets.append(
            og - sg
        )

        top1.append(
            float(
                choose_local
                ==
                oracle_local
            )
        )

        if (
            len(
                np.unique(
                    s
                )
            ) > 1
            and
            len(
                np.unique(
                    g
                )
            ) > 1
        ):
            rho = spearmanr(
                s,
                g,
            ).statistic

            if np.isfinite(
                rho
            ):
                rhos.append(
                    float(
                        rho
                    )
                )

        wins = 0.0
        n = 0

        for a in range(
            len(g)
        ):
            for b in range(
                a+1,
                len(g),
            ):
                dg = (
                    g[a]
                    - g[b]
                )

                if abs(
                    float(
                        dg
                    )
                ) < PAIR_EPS:
                    continue

                ds = (
                    s[a]
                    - s[b]
                )

                wins += float(
                    np.sign(
                        dg
                    )
                    ==
                    np.sign(
                        ds
                    )
                )

                n += 1

        if n:
            pair_accs.append(
                wins
                / n
            )

        rows.append({
            "system_id":
                systems[
                    i
                ],

            "selected_gain":
                sg,

            "random_gain":
                rg,

            "oracle_gain":
                og,
        })

    mean_sel = float(
        np.mean(
            selected
        )
    )

    mean_rand = float(
        np.mean(
            randoms
        )
    )

    mean_oracle = float(
        np.mean(
            oracles
        )
    )

    denom = (
        mean_oracle
        - mean_rand
    )

    capture = (
        (
            mean_sel
            - mean_rand
        )
        / denom
        if abs(
            denom
        ) > 1e-12
        else 0.0
    )

    return {
        "states":
            len(
                selected
            ),

        "mean_selected_gain":
            mean_sel,

        "mean_random_gain":
            mean_rand,

        "mean_oracle_gain":
            mean_oracle,

        "delta_selected_minus_random":
            mean_sel
            - mean_rand,

        "mean_regret":
            float(
                np.mean(
                    regrets
                )
            ),

        "oracle_top1_hit":
            float(
                np.mean(
                    top1
                )
            ),

        "mean_spearman":
            float(
                np.mean(
                    rhos
                )
            )
            if rhos
            else None,

        "pair_accuracy":
            float(
                np.mean(
                    pair_accs
                )
            )
            if pair_accs
            else None,

        "oracle_signal_captured":
            float(
                capture
            ),

        "_rows":
            rows,
    }


def cluster_bootstrap(
    rows,
    B=10000,
):
    d = defaultdict(
        list
    )

    for r in rows:
        d[
            r[
                "system_id"
            ]
        ].append(
            r[
                "selected_gain"
            ]
            -
            r[
                "random_gain"
            ]
        )

    systems = sorted(
        d
    )

    values = np.asarray(
        [
            np.mean(
                d[s]
            )
            for s in systems
        ],
        dtype=float,
    )

    rng = np.random.default_rng(
        SEED + 991
    )

    boot = np.empty(
        B,
        dtype=float,
    )

    for i in range(
        B
    ):
        idx = rng.integers(
            0,
            len(values),
            size=len(values),
        )

        boot[
            i
        ] = np.mean(
            values[
                idx
            ]
        )

    lo, hi = np.quantile(
        boot,
        [
            .025,
            .975,
        ],
    )

    pgt = float(
        np.mean(
            boot > 0
        )
    )

    return {
        "systems":
            len(
                systems
            ),

        "delta":
            float(
                values.mean()
            ),

        "ci95":
            [
                float(
                    lo
                ),
                float(
                    hi
                ),
            ],

        "p_boot_gt_zero":
            pgt,

        "two_sided_p":
            float(
                2.0
                * min(
                    pgt,
                    1.0-pgt,
                )
            ),
    }


# ============================================================
# HEURISTIC CONTROL
#
# "Ask about uncertain roles"
#
# Not the proposed final selector; just a sanity baseline.
# ============================================================

def heuristic_scores(
    groups,
    qmap,
    role_matrix,
):
    out = []

    Kmax = max(
        len(
            g["queries"]
        )
        for g in groups
    )

    arr = np.full(
        (
            len(groups),
            Kmax,
        ),
        -1e9,
        dtype=np.float32,
    )

    for i, g in enumerate(
        groups
    ):
        p = belief_vector(
            g
        )

        u = (
            4.0
            * p
            * (
                1.0-p
            )
        )

        for j, q in enumerate(
            g[
                "queries"
            ]
        ):
            sim = (
                role_matrix
                @ qmap[
                    q
                ]
            )

            # Higher when query semantically targets
            # currently uncertain roles.
            arr[
                i,
                j
            ] = float(
                np.sum(
                    u
                    * np.maximum(
                        sim,
                        0.0,
                    )
                )
                /
                (
                    np.sum(
                        u
                    )
                    + 1e-6
                )
            )

    return arr


# ============================================================
# TRAIN ONE SEED
# ============================================================

def train_seed(
    seed,
    train_pack,
    dev_pack,
):
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )

    model = SelectorNet(
        train_pack[
            "dim"
        ]
    ).to(
        DEVICE
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    Xt = torch.tensor(
        train_pack[
            "X"
        ],
        dtype=torch.float32,
    )

    Gt = torch.tensor(
        train_pack[
            "gains"
        ],
        dtype=torch.float32,
    )

    Mt = torch.tensor(
        train_pack[
            "mask"
        ],
        dtype=torch.bool,
    )

    Xd = torch.tensor(
        dev_pack[
            "X"
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    Gd = torch.tensor(
        dev_pack[
            "gains"
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    Md = torch.tensor(
        dev_pack[
            "mask"
        ],
        dtype=torch.bool,
        device=DEVICE,
    )

    best = None
    best_selected = -1e30
    best_epoch = None
    bad = 0

    n = len(
        Xt
    )

    for epoch in range(
        1,
        EPOCHS+1,
    ):
        model.train()

        perm = torch.randperm(
            n
        )

        losses = []

        for ids in perm.split(
            BATCH_GROUPS
        ):
            xb = Xt[
                ids
            ].to(
                DEVICE
            )

            gb = Gt[
                ids
            ].to(
                DEVICE
            )

            mb = Mt[
                ids
            ].to(
                DEVICE
            )

            pred = model(
                xb
            )

            loss, ll, pl, rl = group_losses(
                pred,
                gb,
                mb,
            )

            opt.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            losses.append(
                float(
                    loss
                    .detach()
                    .cpu()
                )
            )

        model.eval()

        with torch.no_grad():
            ds = model(
                Xd
            ).detach().cpu().numpy()

        ev = rank_eval(
            dev_pack,
            ds,
        )

        selected = ev[
            "mean_selected_gain"
        ]

        print(
            f"[seed={seed}] "
            f"epoch={epoch:02d} "
            f"loss={np.mean(losses):.5f} "
            f"dev_selected={selected:.6f} "
            f"random={ev['mean_random_gain']:.6f} "
            f"oracle={ev['mean_oracle_gain']:.6f} "
            f"top1={ev['oracle_top1_hit']:.4f} "
            f"rho={ev['mean_spearman']:.4f}",
            flush=True,
        )

        if selected > (
            best_selected
            + 1e-5
        ):
            best_selected = selected
            best_epoch = epoch
            bad = 0

            best = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in model.state_dict().items()
            }

        else:
            bad += 1

        if bad >= PATIENCE:
            break

    model.load_state_dict(
        best
    )

    model.eval()

    return (
        model,
        {
            "best_epoch":
                best_epoch,

            "best_dev_selected":
                float(
                    best_selected
                ),
        },
    )


@torch.no_grad()
def score_model(
    model,
    packed,
):
    X = torch.tensor(
        packed[
            "X"
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    return model(
        X
    ).cpu().numpy()


# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "DEVICE =",
        DEVICE,
        flush=True,
    )

    print(
        "SELECTOR DECISION-TIME INPUTS = "
        "belief + public history + candidate query + round",
        flush=True,
    )

    print(
        "FORBIDDEN DECISION-TIME INPUTS = "
        "future output / sensor / truth / activation",
        flush=True,
    )

    train_groups = read_jsonl(
        DATA
        / "train_groups.jsonl"
    )

    val_groups = read_jsonl(
        DATA
        / "val_groups.jsonl"
    )

    challenge_groups = read_jsonl(
        DATA
        / "challenge_groups.jsonl"
    )

    # --------------------------------------------------------
    # Split TRAIN by SYSTEM into fit/internal-dev.
    # --------------------------------------------------------

    systems = sorted({
        str(
            g[
                "system_id"
            ]
        )
        for g in train_groups
    })

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        systems
    )

    ndev = max(
        1,
        int(
            .15
            * len(
                systems
            )
        ),
    )

    dev_systems = set(
        systems[
            :ndev
        ]
    )

    fit_systems = set(
        systems[
            ndev:
        ]
    )

    fit_groups = [
        g
        for g in train_groups
        if str(
            g[
                "system_id"
            ]
        )
        in fit_systems
    ]

    dev_groups = [
        g
        for g in train_groups
        if str(
            g[
                "system_id"
            ]
        )
        in dev_systems
    ]

    print(
        "GROUPS fit/dev/val/challenge =",
        len(
            fit_groups
        ),
        len(
            dev_groups
        ),
        len(
            val_groups
        ),
        len(
            challenge_groups
        ),
        flush=True,
    )

    print(
        "SYSTEMS fit/dev/val/challenge =",
        len(
            fit_systems
        ),
        len(
            dev_systems
        ),
        len({
            str(
                g[
                    "system_id"
                ]
            )
            for g in val_groups
        }),
        len({
            str(
                g[
                    "system_id"
                ]
            )
            for g in challenge_groups
        }),
        flush=True,
    )

    # --------------------------------------------------------
    # Encoder.
    # --------------------------------------------------------

    encoder = SentenceTransformer(
        ENCODER,
        device=DEVICE,
    )

    all_groups = (
        fit_groups
        + dev_groups
        + val_groups
        + challenge_groups
    )

    all_queries = [
        str(q)
        for g in all_groups
        for q in g[
            "queries"
        ]
    ]

    all_histories = [
        str(
            g.get(
                "history_text",
                "- no previous public interaction",
            )
        )
        for g in all_groups
    ]

    qmap = encode_unique(
        encoder,
        all_queries,
    )

    hmap = encode_unique(
        encoder,
        all_histories,
    )

    role_texts = [
        ROLE_DESCRIPTIONS[
            r
        ]
        for r in ROLES
    ]

    role_matrix = encoder.encode(
        role_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(
        np.float32
    )

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Build fixed public selector features.
    # --------------------------------------------------------

    print(
        "PACKING...",
        flush=True,
    )

    fit_pack = pack_groups(
        fit_groups,
        qmap,
        hmap,
        role_matrix,
    )

    dev_pack = pack_groups(
        dev_groups,
        qmap,
        hmap,
        role_matrix,
    )

    val_pack = pack_groups(
        val_groups,
        qmap,
        hmap,
        role_matrix,
    )

    challenge_pack = pack_groups(
        challenge_groups,
        qmap,
        hmap,
        role_matrix,
    )

    print(
        "FEATURE_DIM =",
        fit_pack[
            "dim"
        ],
        flush=True,
    )

    # --------------------------------------------------------
    # Simple semantic uncertainty baseline.
    # --------------------------------------------------------

    heuristic = {}

    for name, groups, pack in [
        (
            "val",
            val_groups,
            val_pack,
        ),
        (
            "challenge",
            challenge_groups,
            challenge_pack,
        ),
    ]:
        hs = heuristic_scores(
            groups,
            qmap,
            role_matrix,
        )

        e = rank_eval(
            pack,
            hs,
        )

        b = cluster_bootstrap(
            e[
                "_rows"
            ]
        )

        e.pop(
            "_rows"
        )

        e[
            "bootstrap"
        ] = b

        heuristic[
            name
        ] = e

    # --------------------------------------------------------
    # 5-seed learned ensemble.
    # --------------------------------------------------------

    ensemble_scores = {
        "val": [],
        "challenge": [],
    }

    seed_meta = {}

    for seed in ENSEMBLE_SEEDS:
        print()
        print(
            "=" * 100,
            flush=True,
        )

        print(
            "TRAIN SEED",
            seed,
            flush=True,
        )

        print(
            "=" * 100,
            flush=True,
        )

        model, meta = train_seed(
            seed,
            fit_pack,
            dev_pack,
        )

        seed_meta[
            str(
                seed
            )
        ] = meta

        ensemble_scores[
            "val"
        ].append(
            score_model(
                model,
                val_pack,
            )
        )

        ensemble_scores[
            "challenge"
        ].append(
            score_model(
                model,
                challenge_pack,
            )
        )

        torch.save(
            {
                "state_dict":
                    model.state_dict(),

                "input_dim":
                    fit_pack[
                        "dim"
                    ],

                "roles":
                    ROLES,

                "role_descriptions":
                    ROLE_DESCRIPTIONS,

                "encoder":
                    ENCODER,

                "seed":
                    seed,

                "decision_time_inputs":
                    [
                        "current belief",
                        "public history",
                        "candidate query",
                        "round",
                    ],
            },
            OUT
            / f"selector_seed_{seed}.pt",
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    results = {}

    for name, pack in [
        (
            "val",
            val_pack,
        ),
        (
            "challenge",
            challenge_pack,
        ),
    ]:
        scores = np.mean(
            np.stack(
                ensemble_scores[
                    name
                ],
                axis=0,
            ),
            axis=0,
        )

        ev = rank_eval(
            pack,
            scores,
        )

        boot = cluster_bootstrap(
            ev[
                "_rows"
            ]
        )

        ev.pop(
            "_rows"
        )

        ev[
            "bootstrap"
        ] = boot

        results[
            name
        ] = ev

    # --------------------------------------------------------
    # Known V4.1 references from completed permutation audit.
    # --------------------------------------------------------

    references = {
        "val_v41_9perm": {
            "selected":
                0.04108572447484074,

            "random":
                0.0255567500305335,

            "oracle":
                0.13306689815802952,
        },

        "challenge_v41_9perm": {
            "selected":
                0.03292611027735462,

            "random":
                0.029052938012214025,

            "oracle":
                0.13387890898694066,
        },
    }

    summary = {
        "experiment":
            "belief_targeted_selector_v1",

        "purpose":
            (
                "Choose the candidate query expected to improve "
                "the current role belief most."
            ),

        "decision_time_inputs":
            [
                "current role belief",
                "public interaction history",
                "candidate query text",
                "round",
            ],

        "decision_time_forbidden":
            [
                "future MAS response",
                "future sensor fingerprint",
                "truth roles",
                "activation",
                "realized gain",
            ],

        "training_label_only":
            (
                "realized downstream AUC gain produced offline "
                "after the fixed sensor and belief updater"
            ),

        "fit_systems":
            len(
                fit_systems
            ),

        "internal_dev_systems":
            len(
                dev_systems
            ),

        "ensemble_seeds":
            ENSEMBLE_SEEDS,

        "seed_training":
            seed_meta,

        "heuristic_uncertainty_semantic":
            heuristic,

        "learned_selector":
            results,

        "reference_v41":
            references,
    }

    (
        OUT
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL BELIEF-TARGETED SELECTOR"
    )

    print(
        "=" * 100
    )

    for name in [
        "val",
        "challenge",
    ]:
        e = results[
            name
        ]

        h = heuristic[
            name
        ]

        print()
        print(
            name.upper()
        )

        print(
            "HEURISTIC "
            f"selected={h['mean_selected_gain']:.6f} "
            f"random={h['mean_random_gain']:.6f} "
            f"oracle={h['mean_oracle_gain']:.6f} "
            f"capture={h['oracle_signal_captured']:.4f}"
        )

        print(
            "LEARNED   "
            f"selected={e['mean_selected_gain']:.6f} "
            f"random={e['mean_random_gain']:.6f} "
            f"oracle={e['mean_oracle_gain']:.6f} "
            f"top1={e['oracle_top1_hit']:.4f} "
            f"rho={e['mean_spearman']:.4f} "
            f"pairAcc={e['pair_accuracy']:.4f} "
            f"capture={e['oracle_signal_captured']:.4f}"
        )

        print(
            "BOOTSTRAP  ",
            e[
                "bootstrap"
            ],
        )

    print()
    print(
        "SAVED =",
        OUT
        / "summary.json"
    )


if __name__ == "__main__":
    main()
