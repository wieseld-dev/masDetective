#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, TensorDataset

from v11.role_universe import TRAIN_ROLES


# ============================================================
# CONFIG
# ============================================================

SEED = 20261008

GROUP_ROOT = Path(
    "artifacts/q38_selector_v41_data"
)

RAW_ROOT = Path(
    "artifacts/adaptive_v11_curriculum_v2"
)

OUT = Path(
    "artifacts/forward_evidence_eig_v1"
)

ENCODER_NAME = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

ROLES = list(TRAIN_ROLES)
R = len(ROLES)
D_OUT = 2 * R

BATCH = 256
EPOCHS = 40
PATIENCE = 6
LR = 2e-4

POST_PARTICLES = 64
MC_OBSERVATIONS = 12
POSTERIOR_ENSEMBLE = 3

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# IO
# ============================================================

def read_jsonl(p: Path):
    if not p.exists():
        return []

    return [
        json.loads(x)
        for x in p.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def normq(x):
    return " ".join(
        str(x or "").split()
    )


def stable_int(*parts):
    s = "||".join(
        map(str, parts)
    ).encode()

    return int.from_bytes(
        hashlib.sha256(s).digest()[:8],
        "big",
    )


def round_from_name(p: Path):
    m = re.search(
        r"round_(\d+)",
        p.name,
    )

    return (
        int(m.group(1))
        if m
        else None
    )


# ============================================================
# INDEX CURRENT V13.4 / V13.5 OBSERVATIONS
# ============================================================

def evidence_vector(sc):
    e135 = dict(
        sc.get(
            "evidence",
            {},
        )
    )

    e134 = dict(
        sc.get(
            "metadata",
            {},
        ).get(
            "evidence_v134",
            {},
        )
    )

    y135 = [
        float(
            e135.get(r, 0.0)
        )
        for r in ROLES
    ]

    y134 = [
        float(
            e134.get(r, 0.0)
        )
        for r in ROLES
    ]

    return np.asarray(
        y135 + y134,
        dtype=np.float32,
    )


def truth_vector(resp, sc):
    truth = (
        resp.get(
            "truth_roles_training_only"
        )
        or sc.get(
            "truth_roles"
        )
        or []
    )

    truth = set(
        map(str, truth)
    )

    return np.asarray(
        [
            float(r in truth)
            for r in ROLES
        ],
        dtype=np.float32,
    )


def build_observation_index():

    # key:
    # (public system/session id, round, normalized query)
    #
    # Multiple stochastic repeats are retained and later averaged.
    index = defaultdict(list)

    n_response = 0
    n_scored = 0
    n_joined = 0

    dirs = set(
        p.parent
        for p in RAW_ROOT.rglob(
            "responses_train_round_*.jsonl"
        )
    )

    print(
        "[raw] directories =",
        len(dirs),
        flush=True,
    )

    for d in sorted(dirs):

        response_files = sorted(
            d.glob(
                "responses_train_round_*.jsonl"
            )
        )

        for rp in response_files:

            rnd = round_from_name(rp)

            if rnd is None:
                continue

            sp = d / (
                f"scored_train_round_"
                f"{rnd:02d}.jsonl"
            )

            if not sp.exists():
                continue

            scores = read_jsonl(sp)

            score_idx = defaultdict(list)

            for sc in scores:
                q = normq(
                    sc.get("query")
                )

                ids = {
                    str(
                        sc.get(
                            "system_id",
                            ""
                        )
                    ),
                    str(
                        sc.get(
                            "session_id",
                            ""
                        )
                    ),
                }

                ids.discard("")

                for sid in ids:
                    score_idx[
                        (sid, q)
                    ].append(sc)

            responses = read_jsonl(rp)

            n_response += len(
                responses
            )

            n_scored += len(
                scores
            )

            for rr in responses:

                if (
                    rr.get(
                        "execution_status"
                    )
                    != "ok"
                ):
                    continue

                if not rr.get(
                    "valid",
                    True,
                ):
                    continue

                q = normq(
                    rr.get("query")
                )

                ids = []

                for k in (
                    "session_id",
                    "system_id",
                ):
                    x = str(
                        rr.get(k, "")
                    )

                    if (
                        x
                        and x not in ids
                    ):
                        ids.append(x)

                found = []

                for sid in ids:
                    found.extend(
                        score_idx.get(
                            (sid, q),
                            [],
                        )
                    )

                # Deduplicate score records.
                uniq = {}
                for sc in found:
                    sig = json.dumps(
                        {
                            "e":
                                sc.get(
                                    "evidence",
                                    {},
                                ),
                            "m":
                                sc.get(
                                    "metadata",
                                    {},
                                ).get(
                                    "evidence_v134",
                                    {},
                                ),
                        },
                        sort_keys=True,
                    )
                    uniq[sig] = sc

                found = list(
                    uniq.values()
                )

                if not found:
                    continue

                # Usually exactly one.
                # If several equivalent records exist,
                # each is a stochastic sample.
                for sc in found:

                    y = evidence_vector(
                        sc
                    )

                    z = truth_vector(
                        rr,
                        sc,
                    )

                    for sid in ids:
                        index[
                            (
                                sid,
                                rnd,
                                q,
                            )
                        ].append(
                            (
                                z,
                                y,
                            )
                        )

                    n_joined += 1

    print(
        "[raw] responses =",
        n_response,
    )

    print(
        "[raw] scored    =",
        n_scored,
    )

    print(
        "[raw] joins     =",
        n_joined,
    )

    print(
        "[raw] index keys=",
        len(index),
    )

    return index


def lookup_observation(
    index,
    sid,
    rnd,
    query,
):
    xs = index.get(
        (
            str(sid),
            int(rnd),
            normq(query),
        ),
        [],
    )

    if not xs:
        return None

    # Require truth configuration to agree.
    z0 = xs[0][0]

    same_truth = [
        (z, y)
        for z, y in xs
        if np.array_equal(
            z,
            z0,
        )
    ]

    if (
        len(same_truth)
        != len(xs)
    ):
        return None

    # Average stochastic evidence repeats.
    y = np.mean(
        np.stack(
            [
                x[1]
                for x in same_truth
            ]
        ),
        axis=0,
    ).astype(
        np.float32
    )

    return {
        "z": z0.astype(
            np.float32
        ),
        "y": y,
        "repeats": len(
            same_truth
        ),
    }


# ============================================================
# EXACT V4.1 GROUPS + CURRENT EVIDENCE TARGETS
# ============================================================

def attach_groups(
    groups,
    obs_index,
):

    rows = []

    missing = 0

    for g in groups:

        sid = str(
            g["system_id"]
        )

        rnd = int(
            g["round"]
        )

        queries = list(
            g["queries"]
        )

        gains = list(
            g["auc_gains"]
        )

        if (
            len(queries)
            != len(gains)
        ):
            raise RuntimeError(
                "query/gain size mismatch"
            )

        for j, (
            query,
            gain,
        ) in enumerate(
            zip(
                queries,
                gains,
            )
        ):

            obs = lookup_observation(
                obs_index,
                sid,
                rnd,
                query,
            )

            if obs is None:
                missing += 1
                continue

            rows.append({
                "system_id":
                    sid,

                "state_id":
                    str(
                        g["state_id"]
                    ),

                "round":
                    rnd,

                "query":
                    str(query),

                "history_text":
                    str(
                        g.get(
                            "history_text",
                            "- no previous public interaction",
                        )
                    ),

                "belief":
                    np.asarray(
                        [
                            float(
                                g[
                                    "role_probs_before"
                                ][r]
                            )
                            for r
                            in ROLES
                        ],
                        dtype=np.float32,
                    ),

                "z":
                    obs["z"],

                "y":
                    obs["y"],

                "auc_gain":
                    float(gain),

                "candidate_index":
                    int(
                        g[
                            "candidate_indices"
                        ][j]
                    )
                    if (
                        "candidate_indices"
                        in g
                    )
                    else j,

                "repeats":
                    int(
                        obs[
                            "repeats"
                        ]
                    ),
            })

    total = (
        len(rows)
        + missing
    )

    rate = (
        len(rows)
        / max(
            total,
            1,
        )
    )

    print(
        "[align] matched =",
        len(rows),
        "/",
        total,
        f"({rate:.3%})",
    )

    if rate < 0.80:
        raise RuntimeError(
            "Raw/current evidence alignment below 80%; "
            "stop instead of training on a biased subset."
        )

    return rows


# ============================================================
# EMBEDDINGS
# ============================================================

def encode_maps(
    encoder,
    rows,
):

    queries = sorted({
        x["query"]
        for x in rows
    })

    histories = sorted({
        x["history_text"]
        for x in rows
    })

    print(
        "[embed] unique queries =",
        len(queries),
    )

    print(
        "[embed] unique histories =",
        len(histories),
    )

    qv = encoder.encode(
        queries,
        batch_size=128,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(
        np.float32
    )

    hv = encoder.encode(
        histories,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(
        np.float32
    )

    return (
        {
            q: qv[i]
            for i, q
            in enumerate(
                queries
            )
        },
        {
            h: hv[i]
            for i, h
            in enumerate(
                histories
            )
        },
    )


# ============================================================
# MODEL
# ============================================================

class ForwardNet(nn.Module):

    def __init__(
        self,
        d_in,
        d_out,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                d_in,
                512,
            ),
            nn.GELU(),
            nn.LayerNorm(
                512
            ),
            nn.Dropout(
                .12
            ),

            nn.Linear(
                512,
                256,
            ),
            nn.GELU(),
            nn.Dropout(
                .08
            ),

            nn.Linear(
                256,
                d_out,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


def feature(
    row,
    qmap,
    hmap,
    use_z,
):
    parts = [
        qmap[
            row["query"]
        ],

        hmap[
            row["history_text"]
        ],

        row["belief"],

        np.asarray(
            [
                row["round"]
                / 8.0
            ],
            dtype=np.float32,
        ),
    ]

    if use_z:
        parts.append(
            row["z"]
        )

    return np.concatenate(
        parts
    ).astype(
        np.float32
    )


def build_xy(
    rows,
    qmap,
    hmap,
    use_z,
):
    X = np.stack([
        feature(
            x,
            qmap,
            hmap,
            use_z,
        )
        for x in rows
    ])

    Y = np.stack([
        x["y"]
        for x in rows
    ])

    return (
        X.astype(
            np.float32
        ),
        Y.astype(
            np.float32
        ),
    )


def train_model(
    name,
    Xtr,
    Ytr,
    Xdev,
    Ydev,
):

    ymu = Ytr.mean(
        axis=0
    )

    ysd = Ytr.std(
        axis=0
    )

    ysd = np.where(
        ysd < 1e-4,
        1.0,
        ysd,
    ).astype(
        np.float32
    )

    Ytrz = (
        (Ytr - ymu)
        / ysd
    ).astype(
        np.float32
    )

    Ydevz = (
        (Ydev - ymu)
        / ysd
    ).astype(
        np.float32
    )

    model = ForwardNet(
        Xtr.shape[1],
        Ytr.shape[1],
    ).to(
        DEVICE
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-3,
    )

    loss_fn = nn.SmoothL1Loss(
        beta=.5
    )

    ds = TensorDataset(
        torch.tensor(
            Xtr
        ),
        torch.tensor(
            Ytrz
        ),
    )

    dl = DataLoader(
        ds,
        batch_size=BATCH,
        shuffle=True,
        num_workers=0,
    )

    Xd = torch.tensor(
        Xdev,
        device=DEVICE,
    )

    Yd = torch.tensor(
        Ydevz,
        device=DEVICE,
    )

    best = None
    best_loss = float(
        "inf"
    )

    bad = 0

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        losses = []

        for xb, yb in dl:

            xb = xb.to(
                DEVICE
            )

            yb = yb.to(
                DEVICE
            )

            opt.zero_grad(
                set_to_none=True
            )

            pred = model(
                xb
            )

            loss = loss_fn(
                pred,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            opt.step()

            losses.append(
                float(
                    loss.detach()
                    .cpu()
                )
            )

        model.eval()

        with torch.no_grad():
            dp = model(
                Xd
            )

            dloss = float(
                loss_fn(
                    dp,
                    Yd,
                )
                .detach()
                .cpu()
            )

        print(
            f"[{name}] "
            f"epoch={epoch:02d} "
            f"train={np.mean(losses):.5f} "
            f"dev={dloss:.5f}",
            flush=True,
        )

        if (
            dloss
            < best_loss
            - 1e-5
        ):
            best_loss = dloss

            best = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in model
                .state_dict()
                .items()
            }

            bad = 0

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
        ymu.astype(
            np.float32
        ),
        ysd.astype(
            np.float32
        ),
        best_loss,
    )


@torch.no_grad()
def predict(
    model,
    X,
    ymu,
    ysd,
    batch=1024,
):

    out = []

    for i in range(
        0,
        len(X),
        batch,
    ):

        xb = torch.tensor(
            X[
                i:i+batch
            ],
            device=DEVICE,
        )

        z = model(
            xb
        ).cpu().numpy()

        out.append(
            z
        )

    Z = np.concatenate(
        out,
        axis=0,
    )

    return (
        Z * ysd[None, :]
        + ymu[None, :]
    )


# ============================================================
# PREDICTION METRICS
# ============================================================

def prediction_metrics(
    y,
    p,
):

    err = p - y

    rmse = float(
        np.sqrt(
            np.mean(
                err**2
            )
        )
    )

    mae = float(
        np.mean(
            np.abs(
                err
            )
        )
    )

    r2s = []
    cors = []

    for j in range(
        y.shape[1]
    ):
        yy = y[:, j]
        pp = p[:, j]

        den = float(
            np.sum(
                (
                    yy
                    - yy.mean()
                )**2
            )
        )

        if den > 1e-9:
            r2s.append(
                1.0
                - float(
                    np.sum(
                        (
                            yy
                            - pp
                        )**2
                    )
                )
                / den
            )

        if (
            np.std(yy) > 1e-8
            and
            np.std(pp) > 1e-8
        ):
            cors.append(
                float(
                    np.corrcoef(
                        yy,
                        pp,
                    )[0, 1]
                )
            )

    return {
        "rmse":
            rmse,

        "mae":
            mae,

        "mean_dim_r2":
            float(
                np.mean(
                    r2s
                )
            )
            if r2s
            else None,

        "mean_dim_corr":
            float(
                np.mean(
                    cors
                )
            )
            if cors
            else None,
    }


# ============================================================
# BAYESIAN EIG
# ============================================================

def sample_particles(
    belief,
    n,
    rng,
):

    Z = (
        rng.random(
            (
                n,
                len(
                    belief
                ),
            )
        )
        < belief[
            None,
            :
        ]
    ).astype(
        np.float32
    )

    # Avoid degenerate all-empty hypothesis.
    empty = (
        Z.sum(
            axis=1
        )
        == 0
    )

    if empty.any():
        Z[
            empty,
            int(
                np.argmax(
                    belief
                )
            ),
        ] = 1.0

    U, counts = np.unique(
        Z,
        axis=0,
        return_counts=True,
    )

    w = counts.astype(
        np.float64
    )

    w /= w.sum()

    return (
        U.astype(
            np.float32
        ),
        w,
    )


def evidence_mu_for_particles(
    model,
    ymu,
    ysd,
    qvec,
    hvec,
    belief,
    rnd,
    Z,
):

    n = len(Z)

    base = np.concatenate(
        [
            qvec,
            hvec,
            belief,
            np.asarray(
                [
                    rnd
                    / 8.0
                ],
                dtype=np.float32,
            ),
        ]
    )

    X = np.concatenate(
        [
            np.repeat(
                base[
                    None,
                    :
                ],
                n,
                axis=0,
            ),
            Z,
        ],
        axis=1,
    ).astype(
        np.float32
    )

    return predict(
        model,
        X,
        ymu,
        ysd,
        batch=1024,
    )


def entropy(
    w,
):
    w = np.asarray(
        w,
        dtype=np.float64,
    )

    w = w[
        w > 0
    ]

    return float(
        -np.sum(
            w
            * np.log(
                w
            )
        )
    )


def mc_information_gain(
    mus,
    weights,
    noise_var,
    rng,
    nsamples,
):

    if len(weights) <= 1:
        return 0.0

    noise_var = np.maximum(
        noise_var,
        1e-6,
    )

    sd = np.sqrt(
        noise_var
    )

    h0 = entropy(
        weights
    )

    vals = []

    log_prior = np.log(
        np.maximum(
            weights,
            1e-15,
        )
    )

    latent = rng.choice(
        len(weights),
        size=nsamples,
        p=weights,
    )

    for k in latent:

        y = (
            mus[k]
            + rng.normal(
                size=mus.shape[1]
            )
            * sd
        )

        diff = (
            y[
                None,
                :
            ]
            - mus
        )

        ll = (
            -0.5
            * np.sum(
                diff**2
                / noise_var[
                    None,
                    :
                ],
                axis=1,
            )
        )

        a = (
            log_prior
            + ll
        )

        a -= np.max(
            a
        )

        post = np.exp(
            a
        )

        post /= post.sum()

        vals.append(
            h0
            - entropy(
                post
            )
        )

    return float(
        np.mean(
            vals
        )
    )


# ============================================================
# SELECTION EVAL
# ============================================================

def cluster_bootstrap(
    records,
    B=20000,
):

    by = defaultdict(
        list
    )

    for x in records:
        by[
            x[
                "system_id"
            ]
        ].append(
            x[
                "selected_gain"
            ]
            - x[
                "random_gain"
            ]
        )

    systems = sorted(
        by
    )

    observed = float(
        np.mean([
            z
            for sid in systems
            for z in by[sid]
        ])
    )

    rng = np.random.default_rng(
        SEED + 99
    )

    bs = np.empty(
        B,
        dtype=float,
    )

    for b in range(B):

        samp = rng.choice(
            systems,
            size=len(
                systems
            ),
            replace=True,
        )

        vals = []

        for sid in samp:
            vals.extend(
                by[sid]
            )

        bs[b] = np.mean(
            vals
        )

    lo, hi = np.quantile(
        bs,
        [
            .025,
            .975,
        ],
    )

    return {
        "delta_selected_minus_random":
            observed,

        "cluster_ci95":
            [
                float(lo),
                float(hi),
            ],

        "p_boot_gt_zero":
            float(
                np.mean(
                    bs > 0
                )
            ),
    }


def evaluate_eig(
    val_groups,
    qmap,
    hmap,
    z_model,
    ymu,
    ysd,
    noise_var,
):

    records = []

    for gi, g in enumerate(
        val_groups,
        1,
    ):

        belief = np.asarray(
            [
                float(
                    g[
                        "role_probs_before"
                    ][r]
                )
                for r in ROLES
            ],
            dtype=np.float32,
        )

        history = str(
            g.get(
                "history_text",
                "- no previous public interaction",
            )
        )

        hvec = hmap[
            history
        ]

        rnd = int(
            g["round"]
        )

        scores = []

        for cj, query in enumerate(
            g["queries"]
        ):

            qvec = qmap[
                str(query)
            ]

            rep_scores = []

            for rep in range(
                POSTERIOR_ENSEMBLE
            ):

                rng = np.random.default_rng(
                    stable_int(
                        g["state_id"],
                        cj,
                        rep,
                        SEED,
                    )
                )

                Z, w = sample_particles(
                    belief,
                    POST_PARTICLES,
                    rng,
                )

                mus = (
                    evidence_mu_for_particles(
                        z_model,
                        ymu,
                        ysd,
                        qvec,
                        hvec,
                        belief,
                        rnd,
                        Z,
                    )
                )

                rep_scores.append(
                    mc_information_gain(
                        mus,
                        w,
                        noise_var,
                        rng,
                        MC_OBSERVATIONS,
                    )
                )

            scores.append(
                float(
                    np.mean(
                        rep_scores
                    )
                )
            )

        gains = np.asarray(
            g[
                "auc_gains"
            ],
            dtype=float,
        )

        scores = np.asarray(
            scores,
            dtype=float,
        )

        selected = int(
            np.argmax(
                scores
            )
        )

        best = float(
            np.max(
                gains
            )
        )

        random_gain = float(
            np.mean(
                gains
            )
        )

        selected_gain = float(
            gains[
                selected
            ]
        )

        records.append({
            "system_id":
                str(
                    g[
                        "system_id"
                    ]
                ),

            "state_id":
                str(
                    g[
                        "state_id"
                    ]
                ),

            "round":
                rnd,

            "selected_index":
                selected,

            "selected_query":
                str(
                    g[
                        "queries"
                    ][
                        selected
                    ]
                ),

            "selected_eig":
                float(
                    scores[
                        selected
                    ]
                ),

            "selected_gain":
                selected_gain,

            "random_gain":
                random_gain,

            "oracle_gain":
                best,

            "top1":
                int(
                    abs(
                        selected_gain
                        - best
                    )
                    < 1e-12
                ),

            "eig_scores":
                scores.tolist(),

            "auc_gains":
                gains.tolist(),
        })

        if (
            gi % 50 == 0
            or
            gi == len(
                val_groups
            )
        ):
            print(
                f"[EIG] "
                f"{gi}/"
                f"{len(val_groups)}",
                flush=True,
            )

    ms = float(
        np.mean([
            x[
                "selected_gain"
            ]
            for x in records
        ])
    )

    mr = float(
        np.mean([
            x[
                "random_gain"
            ]
            for x in records
        ])
    )

    mo = float(
        np.mean([
            x[
                "oracle_gain"
            ]
            for x in records
        ])
    )

    capture = (
        (ms - mr)
        / (
            mo - mr
        )
        if (
            mo
            > mr
            + 1e-12
        )
        else None
    )

    summary = {
        "groups":
            len(records),

        "mean_selected_gain":
            ms,

        "mean_random_gain":
            mr,

        "mean_oracle_gain":
            mo,

        "mean_regret":
            mo - ms,

        "top1":
            float(
                np.mean([
                    x[
                        "top1"
                    ]
                    for x
                    in records
                ])
            ),

        "oracle_headroom_captured":
            float(
                capture
            )
            if (
                capture
                is not None
            )
            else None,

        "bootstrap":
            cluster_bootstrap(
                records
            ),
    }

    return (
        summary,
        records,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    random.seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    torch.manual_seed(
        SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            SEED
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "DEVICE =",
        DEVICE,
    )

    print(
        "ROLES =",
        len(ROLES),
    )

    print(
        "NO API CALLS",
        flush=True,
    )

    train_groups = read_jsonl(
        GROUP_ROOT
        / "train_groups.jsonl"
    )

    val_groups = read_jsonl(
        GROUP_ROOT
        / "val_groups.jsonl"
    )

    print(
        "TRAIN GROUPS =",
        len(train_groups),
    )

    print(
        "VAL GROUPS =",
        len(val_groups),
    )

    # --------------------------------
    # Align exact current dual-sensor
    # observations.
    # --------------------------------

    obs_index = (
        build_observation_index()
    )

    train_rows = attach_groups(
        train_groups,
        obs_index,
    )

    val_rows = attach_groups(
        val_groups,
        obs_index,
    )

    # --------------------------------
    # Internal split ONLY within
    # selector TRAIN systems.
    # Final V4.1 VAL stays untouched.
    # --------------------------------

    train_sids = sorted({
        x["system_id"]
        for x in train_rows
    })

    rng = random.Random(
        SEED
    )

    rng.shuffle(
        train_sids
    )

    n_dev = max(
        1,
        int(
            .15
            * len(
                train_sids
            )
        ),
    )

    dev_sids = set(
        train_sids[
            :n_dev
        ]
    )

    fit_sids = set(
        train_sids[
            n_dev:
        ]
    )

    fit_rows = [
        x
        for x in train_rows
        if x[
            "system_id"
        ]
        in fit_sids
    ]

    dev_rows = [
        x
        for x in train_rows
        if x[
            "system_id"
        ]
        in dev_sids
    ]

    print(
        "FIT systems/rows =",
        len(fit_sids),
        len(fit_rows),
    )

    print(
        "DEV systems/rows =",
        len(dev_sids),
        len(dev_rows),
    )

    print(
        "FINAL VAL systems/rows =",
        len({
            x["system_id"]
            for x in val_rows
        }),
        len(val_rows),
    )

    # --------------------------------
    # Text representation.
    # --------------------------------

    encoder = SentenceTransformer(
        ENCODER_NAME,
        device=DEVICE,
    )

    all_rows = (
        fit_rows
        + dev_rows
        + val_rows
    )

    qmap, hmap = encode_maps(
        encoder,
        all_rows,
    )

    del encoder

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------
    # PUBLIC observation model
    # no hidden Z.
    # --------------------------------

    Xf0, Yf = build_xy(
        fit_rows,
        qmap,
        hmap,
        use_z=False,
    )

    Xd0, Yd = build_xy(
        dev_rows,
        qmap,
        hmap,
        use_z=False,
    )

    Xv0, Yv = build_xy(
        val_rows,
        qmap,
        hmap,
        use_z=False,
    )

    public_model, pmu, psd, ploss = (
        train_model(
            "PUBLIC",
            Xf0,
            Yf,
            Xd0,
            Yd,
        )
    )

    public_val_pred = predict(
        public_model,
        Xv0,
        pmu,
        psd,
    )

    public_metrics = (
        prediction_metrics(
            Yv,
            public_val_pred,
        )
    )

    print(
        "\nPUBLIC VAL:",
        json.dumps(
            public_metrics,
            indent=2,
        ),
    )

    del public_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------
    # Z-CONDITIONED observation model.
    # This model receives true Z ONLY
    # during offline supervised fitting.
    # --------------------------------

    Xfz, Yf = build_xy(
        fit_rows,
        qmap,
        hmap,
        use_z=True,
    )

    Xdz, Yd = build_xy(
        dev_rows,
        qmap,
        hmap,
        use_z=True,
    )

    Xvz, Yv = build_xy(
        val_rows,
        qmap,
        hmap,
        use_z=True,
    )

    z_model, zmu, zsd, zloss = (
        train_model(
            "Z_CONDITIONED",
            Xfz,
            Yf,
            Xdz,
            Yd,
        )
    )

    z_val_pred = predict(
        z_model,
        Xvz,
        zmu,
        zsd,
    )

    z_metrics = (
        prediction_metrics(
            Yv,
            z_val_pred,
        )
    )

    print(
        "\nZ-CONDITIONED VAL:",
        json.dumps(
            z_metrics,
            indent=2,
        ),
    )

    # --------------------------------
    # Estimate observation noise ONLY
    # from internal DEV residuals.
    # --------------------------------

    z_dev_pred = predict(
        z_model,
        Xdz,
        zmu,
        zsd,
    )

    resid = (
        Yd
        - z_dev_pred
    )

    resid_var = np.var(
        resid,
        axis=0,
    )

    target_var = np.var(
        Yf,
        axis=0,
    )

    noise_var = np.maximum(
        resid_var,
        .05
        * target_var
        + 1e-5,
    ).astype(
        np.float32
    )

    print(
        "noise var median =",
        float(
            np.median(
                noise_var
            )
        ),
    )

    # --------------------------------
    # EIG selection:
    # NO true Z.
    # NO candidate response.
    # NO target evidence.
    #
    # Inputs at decision time:
    # belief/history/query only.
    # --------------------------------

    eig_summary, eig_records = (
        evaluate_eig(
            val_groups,
            qmap,
            hmap,
            z_model,
            zmu,
            zsd,
            noise_var,
        )
    )

    print(
        "\n"
        + "="*100
    )

    print(
        "FORWARD-EVIDENCE EIG RESULT"
    )

    print(
        "="*100
    )

    print(
        json.dumps(
            eig_summary,
            indent=2,
        )
    )

    result = {
        "experiment":
            "forward_evidence_eig_v1",

        "selector_uses_truth":
            False,

        "forward_model_training_uses_truth_z":
            True,

        "decision_time_inputs":
            [
                "current role belief",
                "public history",
                "candidate query",
                "posterior-sampled Z hypotheses",
            ],

        "fit_systems":
            len(fit_sids),

        "internal_dev_systems":
            len(dev_sids),

        "final_val_systems":
            len({
                x["system_id"]
                for x in val_rows
            }),

        "aligned_train_rows":
            len(train_rows),

        "aligned_val_rows":
            len(val_rows),

        "public_forward_prediction":
            public_metrics,

        "z_conditioned_forward_prediction":
            z_metrics,

        "public_internal_best_loss":
            ploss,

        "z_internal_best_loss":
            zloss,

        "eig_selection":
            eig_summary,

        "reference_v41_original_val": {
            "selected":
                0.04231024657686006,

            "random":
                0.0255567500305335,

            "oracle":
                0.13306689815802952,
        },

        "reference_v41_perm_a": {
            "selected":
                0.03439551705116009,

            "random":
                0.0255567500305335,

            "oracle":
                0.13306689815802952,
        },
    }

    (
        OUT
        / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    with (
        OUT
        / "eig_val_records.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        for x in eig_records:
            f.write(
                json.dumps(x)
                + "\n"
            )

    torch.save(
        {
            "state_dict":
                z_model
                .state_dict(),

            "input_dim":
                Xfz.shape[1],

            "output_dim":
                D_OUT,

            "target_mean":
                zmu,

            "target_std":
                zsd,

            "noise_var":
                noise_var,

            "roles":
                ROLES,

            "encoder":
                ENCODER_NAME,
        },
        OUT
        / "z_forward_model.pt",
    )

    print(
        "\nSAVED:",
        OUT
        / "summary.json",
    )


if __name__ == "__main__":
    main()
