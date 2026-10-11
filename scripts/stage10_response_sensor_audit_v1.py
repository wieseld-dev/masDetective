#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from adaptive_v11_active.io import read_jsonl
from v11.role_universe import (
    TRAIN_ROLES,
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


# ================================================================
# CONFIG
# ================================================================

ROOT = Path(
    "artifacts/adaptive_v11_curriculum_v2/stage_10"
)

BASE = Path(
    "artifacts/stage10_response_sensor_v1"
)

OUT = Path(
    "artifacts/stage10_response_sensor_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

CACHE = OUT / "cache"
CACHE.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 20260923

EMBED_MODEL = "all-MiniLM-L6-v2"

EPOCHS = 60
PATIENCE = 8
BATCH = 256
LR = 4e-4
WD = 1e-4

BOOTSTRAPS = 2000


# ================================================================
# BASIC UTILS
# ================================================================

def stable_hash(text: str) -> int:
    d = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        d,
        "big",
    )


def sha_texts(texts):
    h = hashlib.sha256()

    for t in texts:
        h.update(
            str(t).encode(
                "utf-8"
            )
        )
        h.update(
            b"\0"
        )

    return h.hexdigest()


def text_stats(text: str):
    text = str(text)

    words = text.split()

    nchar = len(text)
    nword = len(words)

    uniq = (
        len(
            set(
                w.lower()
                for w in words
            )
        )
        / max(
            1,
            nword,
        )
    )

    digits = sum(
        c.isdigit()
        for c in text
    )

    uppers = sum(
        c.isupper()
        for c in text
    )

    punct = sum(
        c in ".,;:!?-()[]{}"
        for c in text
    )

    bullets = sum(
        line.lstrip().startswith(
            (
                "-",
                "*",
                "•",
            )
        )
        for line
        in text.splitlines()
    )

    avg_word = (
        sum(
            len(w)
            for w in words
        )
        / max(
            1,
            nword,
        )
    )

    return np.asarray([
        math.log1p(
            nchar
        ),
        math.log1p(
            nword
        ),
        math.log1p(
            len(
                text.splitlines()
            )
        ),
        uniq,
        digits
        / max(
            1,
            nchar,
        ),
        uppers
        / max(
            1,
            nchar,
        ),
        punct
        / max(
            1,
            nchar,
        ),
        math.log1p(
            text.count(":")
        ),
        math.log1p(
            bullets
        ),
        avg_word,
    ], dtype=np.float32)


# ================================================================
# DATA
# ================================================================

def load_rows():
    rows = []

    for cycle in (
        "cycle_01",
        "cycle_02",
    ):
        cdir = ROOT / cycle

        for rp in sorted(
            cdir.glob(
                "responses_train_round_*.jsonl"
            )
        ):
            for x in read_jsonl(
                rp
            ):
                if (
                    x.get(
                        "execution_status"
                    )
                    != "ok"
                ):
                    continue

                if not bool(
                    x.get(
                        "valid",
                        True,
                    )
                ):
                    continue

                q = str(
                    x.get(
                        "query",
                        "",
                    )
                ).strip()

                response = str(
                    x.get(
                        "final_output",
                        "",
                    )
                    or x.get(
                        "output",
                        "",
                    )
                    or ""
                ).strip()

                selected = set(
                    map(
                        str,
                        x.get(
                            "selected_roles_training_only",
                            [],
                        ),
                    )
                )

                truth = tuple(
                    sorted(
                        map(
                            str,
                            x.get(
                                "truth_roles_training_only",
                                [],
                            ),
                        )
                    )
                )

                if (
                    not q
                    or not response
                    or not truth
                ):
                    continue

                y = np.asarray(
                    [
                        1.0
                        if role
                        in selected
                        else 0.0
                        for role
                        in TRAIN_ROLES
                    ],
                    dtype=np.float32,
                )

                sig = (
                    "|".join(
                        truth
                    )
                )

                rows.append({
                    "cycle":
                        cycle,

                    "round":
                        int(
                            x["round"]
                        ),

                    "session_id":
                        str(
                            x[
                                "session_id"
                            ]
                        ),

                    "system_id":
                        str(
                            x.get(
                                "system_id",
                                x["session_id"],
                            )
                        ),

                    "query":
                        q,

                    "response":
                        response,

                    "mode":
                        str(
                            x.get(
                                "mode",
                                "unknown",
                            )
                        ),

                    "truth_roles":
                        truth,

                    "K":
                        len(
                            truth
                        ),

                    "mas_signature":
                        sig,

                    "y":
                        y,
                })

    if not rows:
        raise RuntimeError(
            "No usable rows"
        )

    return rows


def apply_original_split(
    rows,
):
    path = (
        BASE
        / "split_assignment.json"
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing original split: {path}"
        )

    assignment = json.loads(
        path.read_text()
    )

    missing = set()

    for x in rows:
        sig = x[
            "mas_signature"
        ]

        if sig not in assignment:
            missing.add(
                sig
            )
            continue

        x["split"] = (
            assignment[sig]
        )

    if missing:
        raise RuntimeError(
            f"{len(missing)} MAS signatures "
            "missing from original split"
        )

    return assignment


# ================================================================
# EMBEDDINGS WITH CACHE
# ================================================================

def cached_encode(
    encoder,
    texts,
    name,
):
    texts = list(
        map(
            str,
            texts,
        )
    )

    digest = sha_texts(
        texts
    )

    ep = (
        CACHE
        / f"{name}.npy"
    )

    mp = (
        CACHE
        / f"{name}.json"
    )

    if (
        ep.exists()
        and mp.exists()
    ):
        meta = json.loads(
            mp.read_text()
        )

        if (
            meta.get(
                "sha256"
            )
            == digest
            and int(
                meta.get(
                    "rows",
                    -1,
                )
            )
            == len(
                texts
            )
        ):
            print(
                f"[cache:{name}] HIT",
                flush=True,
            )

            return np.load(
                ep
            )

    print(
        f"[cache:{name}] MISS",
        flush=True,
    )

    unique = list(
        dict.fromkeys(
            texts
        )
    )

    print(
        f"[embed:{name}] "
        f"rows={len(texts)} "
        f"unique={len(unique)}",
        flush=True,
    )

    ue = encoder.encode(
        unique,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    index = {
        t: i
        for i, t
        in enumerate(
            unique
        )
    }

    emb = np.stack([
        ue[
            index[t]
        ]
        for t in texts
    ]).astype(
        np.float32
    )

    np.save(
        ep,
        emb,
    )

    mp.write_text(
        json.dumps({
            "sha256":
                digest,

            "rows":
                len(
                    texts
                ),

            "dim":
                int(
                    emb.shape[1]
                ),
        }, indent=2)
    )

    print(
        f"[cache:{name}] SAVED",
        flush=True,
    )

    return emb


# ================================================================
# FEATURES
# ================================================================

def make_features(
    rows,
    encoder,
):
    queries = [
        x["query"]
        for x in rows
    ]

    responses = [
        x["response"]
        for x in rows
    ]

    qemb = cached_encode(
        encoder,
        queries,
        "query_embeddings",
    )

    remb = cached_encode(
        encoder,
        responses,
        "response_embeddings",
    )

    role_texts = [
        (
            f"{role}. "
            f"{ROLE_DESCRIPTIONS.get(role, role)} "
            f"{anchor_probe(role)}"
        )
        for role
        in TRAIN_ROLES
    ]

    role_emb = encoder.encode(
        role_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(
        np.float32
    )

    q_role = (
        qemb
        @ role_emb.T
    ).astype(
        np.float32
    )

    r_role = (
        remb
        @ role_emb.T
    ).astype(
        np.float32
    )

    qstats = np.stack([
        text_stats(
            x["query"]
        )
        for x in rows
    ]).astype(
        np.float32
    )

    rstats = np.stack([
        text_stats(
            x["response"]
        )
        for x in rows
    ]).astype(
        np.float32
    )

    qr_cos = np.sum(
        qemb * remb,
        axis=1,
        keepdims=True,
    ).astype(
        np.float32
    )

    # Exact original baseline.
    X_query = np.concatenate([
        qemb,
        q_role,
        qstats,
    ], axis=1).astype(
        np.float32
    )

    # Response semantics only.
    X_response = np.concatenate([
        remb,
        r_role,
        rstats,
    ], axis=1).astype(
        np.float32
    )

    # Query semantics +
    # ONLY superficial response statistics.
    X_query_stats = np.concatenate([
        qemb,
        q_role,
        qstats,
        rstats,
    ], axis=1).astype(
        np.float32
    )

    # Exact original full representation.
    X_full = np.concatenate([
        qemb,
        q_role,
        qstats,

        remb,
        r_role,
        rstats,

        np.abs(
            qemb - remb
        ),

        qemb * remb,

        qr_cos,
    ], axis=1).astype(
        np.float32
    )

    Y = np.stack([
        x["y"]
        for x in rows
    ]).astype(
        np.float32
    )

    aux = {
        "qemb":
            qemb,

        "remb":
            remb,

        "q_role":
            q_role,

        "r_role":
            r_role,

        "qstats":
            qstats,

        "rstats":
            rstats,
    }

    return (
        X_query,
        X_response,
        X_query_stats,
        X_full,
        Y,
        aux,
    )


# ================================================================
# MODEL
# ================================================================

class SensorNet(nn.Module):

    def __init__(
        self,
        dim,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                256,
            ),
            nn.GELU(),
            nn.Dropout(
                0.12
            ),

            nn.Linear(
                256,
                128,
            ),
            nn.GELU(),
            nn.Dropout(
                0.06
            ),

            nn.Linear(
                128,
                len(
                    TRAIN_ROLES
                ),
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.net(
            x
        )


def idx_for(
    rows,
    split,
):
    return np.asarray([
        i
        for i, x
        in enumerate(
            rows
        )
        if x["split"]
        == split
    ], dtype=int)


def macro_auc(
    y,
    p,
):
    vals = []

    for j in range(
        y.shape[1]
    ):
        if (
            len(
                np.unique(
                    y[:, j]
                )
            )
            < 2
        ):
            continue

        vals.append(
            roc_auc_score(
                y[:, j],
                p[:, j],
            )
        )

    if not vals:
        return float(
            "nan"
        )

    return float(
        np.mean(
            vals
        )
    )


def macro_ap(
    y,
    p,
):
    vals = []

    for j in range(
        y.shape[1]
    ):
        if (
            y[:, j].sum()
            == 0
        ):
            continue

        vals.append(
            average_precision_score(
                y[:, j],
                p[:, j],
            )
        )

    if not vals:
        return float(
            "nan"
        )

    return float(
        np.mean(
            vals
        )
    )


def probabilistic_metrics(
    y,
    p,
):
    return {
        "macro_auc":
            macro_auc(
                y,
                p,
            ),

        "micro_auc":
            float(
                roc_auc_score(
                    y.ravel(),
                    p.ravel(),
                )
            ),

        "macro_ap":
            macro_ap(
                y,
                p,
            ),

        "micro_ap":
            float(
                average_precision_score(
                    y.ravel(),
                    p.ravel(),
                )
            ),
    }


def threshold_metrics(
    y,
    p,
    thresholds,
):
    pred = (
        p
        >= thresholds[
            None,
            :
        ]
    ).astype(
        np.float32
    )

    intersection = (
        pred * y
    ).sum(
        axis=1
    )

    union = (
        (
            pred + y
        )
        > 0
    ).sum(
        axis=1
    )

    jaccard = np.where(
        union > 0,
        intersection
        / np.maximum(
            union,
            1,
        ),
        1.0,
    )

    true_count = y.sum(
        axis=1
    )

    pred_count = pred.sum(
        axis=1
    )

    return {
        "macro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),

        "micro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    average="micro",
                    zero_division=0,
                )
            ),

        "exact_active_set":
            float(
                np.mean(
                    np.all(
                        pred == y,
                        axis=1,
                    )
                )
            ),

        "mean_jaccard":
            float(
                np.mean(
                    jaccard
                )
            ),

        "routing_count_mae":
            float(
                np.mean(
                    np.abs(
                        pred_count
                        - true_count
                    )
                )
            ),

        "routing_count_exact":
            float(
                np.mean(
                    pred_count
                    == true_count
                )
            ),
    }


@torch.no_grad()
def predict_raw_model(
    model,
    X,
):
    model.eval()

    out = []

    for s in range(
        0,
        len(X),
        1024,
    ):
        xb = torch.from_numpy(
            X[
                s:
                s+1024
            ]
        )

        out.append(
            torch.sigmoid(
                model(
                    xb
                )
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(
        out,
        axis=0,
    )


def load_original_model(
    name,
    X,
):
    path = (
        BASE
        / f"{name}.pt"
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing checkpoint: {path}"
        )

    ckpt = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        int(
            ckpt[
                "input_dim"
            ]
        )
        != X.shape[1]
    ):
        raise RuntimeError(
            f"{name} feature mismatch: "
            f"checkpoint={ckpt['input_dim']} "
            f"current={X.shape[1]}"
        )

    model = SensorNet(
        X.shape[1]
    )

    model.load_state_dict(
        ckpt[
            "state_dict"
        ]
    )

    mu = np.asarray(
        ckpt["mu"],
        dtype=np.float32,
    )

    sd = np.asarray(
        ckpt["sd"],
        dtype=np.float32,
    )

    thresholds = np.asarray(
        ckpt[
            "thresholds"
        ],
        dtype=np.float32,
    )

    return (
        model,
        mu,
        sd,
        thresholds,
    )


def predict_checkpoint(
    model,
    mu,
    sd,
    X,
):
    Xn = (
        X - mu
    ) / sd

    return predict_raw_model(
        model,
        Xn.astype(
            np.float32
        ),
    )


def fit_thresholds(
    y,
    p,
):
    grid = np.linspace(
        0.05,
        0.95,
        37,
    )

    thresholds = []

    for j in range(
        y.shape[1]
    ):
        best_t = 0.5
        best_f = -1.0

        for t in grid:
            pred = (
                p[:, j]
                >= t
            ).astype(
                int
            )

            f = f1_score(
                y[:, j],
                pred,
                zero_division=0,
            )

            if f > best_f:
                best_f = f
                best_t = float(
                    t
                )

        thresholds.append(
            best_t
        )

    return np.asarray(
        thresholds,
        dtype=np.float32,
    )


def train_new_model(
    name,
    X,
    Y,
    rows,
):
    # Re-seed each model so comparison isn't
    # an arbitrary RNG consequence.
    torch.manual_seed(
        SEED
    )

    np.random.seed(
        SEED
    )

    tr = idx_for(
        rows,
        "train",
    )

    va = idx_for(
        rows,
        "val",
    )

    te = idx_for(
        rows,
        "test",
    )

    mu = X[tr].mean(
        axis=0
    ).astype(
        np.float32
    )

    sd = X[tr].std(
        axis=0
    ).astype(
        np.float32
    )

    sd[
        sd < 1e-6
    ] = 1.0

    Xtr = (
        X[tr] - mu
    ) / sd

    Xva = (
        X[va] - mu
    ) / sd

    Xte = (
        X[te] - mu
    ) / sd

    Ytr = Y[tr]
    Yva = Y[va]
    Yte = Y[te]

    pos = Ytr.sum(
        axis=0
    )

    neg = (
        len(
            Ytr
        )
        - pos
    )

    pos_weight = np.clip(
        neg
        / np.maximum(
            pos,
            1.0,
        ),
        0.5,
        15.0,
    ).astype(
        np.float32
    )

    model = SensorNet(
        X.shape[1]
    )

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WD,
    )

    loss_fn = (
        nn.BCEWithLogitsLoss(
            pos_weight=torch.from_numpy(
                pos_weight
            )
        )
    )

    tx = torch.from_numpy(
        Xtr
    )

    ty = torch.from_numpy(
        Ytr
    )

    best_auc = -1e9
    best_state = None
    best_epoch = None
    bad = 0

    print()
    print(
        "=" * 100
    )
    print(
        f"TRAIN {name}"
    )
    print(
        "=" * 100
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        perm = torch.randperm(
            len(tx)
        )

        losses = []

        for start in range(
            0,
            len(tx),
            BATCH,
        ):
            ids = perm[
                start:
                start+BATCH
            ]

            logits = model(
                tx[ids]
            )

            loss = loss_fn(
                logits,
                ty[ids],
            )

            opt.zero_grad(
                set_to_none=True
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0,
            )

            opt.step()

            losses.append(
                float(
                    loss.item()
                )
            )

        pva = predict_raw_model(
            model,
            Xva,
        )

        auc = macro_auc(
            Yva,
            pva,
        )

        improved = (
            auc
            > best_auc
            + 1e-5
        )

        if improved:
            best_auc = auc
            best_epoch = epoch
            bad = 0

            best_state = {
                k:
                    v.detach()
                    .cpu()
                    .clone()
                for k, v
                in model
                .state_dict()
                .items()
            }

        else:
            bad += 1

        print(
            f"[{name}] "
            f"epoch={epoch:02d} "
            f"loss={np.mean(losses):.4f} "
            f"val_macro_auc={auc:.4f} "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if (
            bad
            >= PATIENCE
        ):
            break

    if best_state is None:
        raise RuntimeError(
            f"{name}: no checkpoint"
        )

    model.load_state_dict(
        best_state
    )

    pval = predict_raw_model(
        model,
        Xva,
    )

    ptest = predict_raw_model(
        model,
        Xte,
    )

    thresholds = fit_thresholds(
        Yva,
        pval,
    )

    metrics = {
        **probabilistic_metrics(
            Yte,
            ptest,
        ),

        **threshold_metrics(
            Yte,
            ptest,
            thresholds,
        ),
    }

    torch.save({
        "state_dict":
            model.state_dict(),

        "input_dim":
            int(
                X.shape[1]
            ),

        "mu":
            mu,

        "sd":
            sd,

        "thresholds":
            thresholds,

        "roles":
            list(
                TRAIN_ROLES
            ),

        "best_epoch":
            best_epoch,

        "internal_test_used":
            False,
    }, OUT / f"{name}.pt")

    return {
        "model":
            model,

        "mu":
            mu,

        "sd":
            sd,

        "thresholds":
            thresholds,

        "ptest":
            ptest,

        "metrics":
            metrics,

        "best_epoch":
            best_epoch,
    }


# ================================================================
# WRONG RESPONSE CONTROL
# ================================================================

def build_wrong_response_indices(
    rows,
    test_idx,
):
    rng = random.Random(
        SEED + 12345
    )

    test_idx = list(
        map(
            int,
            test_idx,
        )
    )

    donors = {}

    exact_control = 0
    relaxed_control = 0

    for i in test_idx:
        x = rows[i]

        # Strongest control:
        # preserve K + mode + round,
        # but response MUST come from a different MAS.
        cand = [
            j
            for j in test_idx
            if (
                j != i
                and rows[j][
                    "K"
                ]
                == x["K"]
                and rows[j][
                    "mode"
                ]
                == x["mode"]
                and rows[j][
                    "round"
                ]
                == x["round"]
                and rows[j][
                    "mas_signature"
                ]
                != x[
                    "mas_signature"
                ]
            )
        ]

        if cand:
            exact_control += 1

        else:
            # Relax round but still preserve K + mode.
            cand = [
                j
                for j in test_idx
                if (
                    j != i
                    and rows[j][
                        "K"
                    ]
                    == x["K"]
                    and rows[j][
                        "mode"
                    ]
                    == x["mode"]
                    and rows[j][
                        "mas_signature"
                    ]
                    != x[
                        "mas_signature"
                    ]
                )
            ]

            if cand:
                relaxed_control += 1

        if not cand:
            # Final fallback: preserve K.
            cand = [
                j
                for j in test_idx
                if (
                    j != i
                    and rows[j][
                        "K"
                    ]
                    == x["K"]
                    and rows[j][
                        "mas_signature"
                    ]
                    != x[
                        "mas_signature"
                    ]
                )
            ]

        if not cand:
            raise RuntimeError(
                f"No wrong-response donor "
                f"for test row {i}"
            )

        donors[i] = rng.choice(
            cand
        )

    print(
        "[shuffle] "
        f"strict K+mode+round={exact_control}/{len(test_idx)} "
        f"relaxed_K+mode={relaxed_control}",
        flush=True,
    )

    return donors


def make_shuffled_full_features(
    rows,
    test_idx,
    donors,
    aux,
):
    qemb = aux[
        "qemb"
    ]

    remb = aux[
        "remb"
    ]

    q_role = aux[
        "q_role"
    ]

    r_role = aux[
        "r_role"
    ]

    qstats = aux[
        "qstats"
    ]

    rstats = aux[
        "rstats"
    ]

    out = []

    same_label = 0

    for i in test_idx:
        j = donors[
            int(i)
        ]

        if np.array_equal(
            rows[i]["y"],
            rows[j]["y"],
        ):
            same_label += 1

        qe = qemb[i]
        re = remb[j]

        qr_cos = np.asarray([
            float(
                np.dot(
                    qe,
                    re,
                )
            )
        ], dtype=np.float32)

        feat = np.concatenate([
            qe,
            q_role[i],
            qstats[i],

            re,
            r_role[j],
            rstats[j],

            np.abs(
                qe - re
            ),

            qe * re,

            qr_cos,
        ]).astype(
            np.float32
        )

        out.append(
            feat
        )

    print(
        "[shuffle] "
        f"accidental exact routing-label match="
        f"{same_label}/{len(test_idx)} "
        f"({100*same_label/len(test_idx):.2f}%)",
        flush=True,
    )

    return np.stack(
        out
    ).astype(
        np.float32
    )


# ================================================================
# BREAKDOWNS
# ================================================================

def subgroup_metrics(
    rows,
    test_idx,
    Y,
    pq,
    pf,
    key,
):
    groups = defaultdict(
        list
    )

    for local_pos, global_i in enumerate(
        test_idx
    ):
        groups[
            rows[
                int(
                    global_i
                )
            ][key]
        ].append(
            local_pos
        )

    result = {}

    for value in sorted(
        groups,
        key=lambda z: str(z),
    ):
        ids = np.asarray(
            groups[value],
            dtype=int,
        )

        yy = Y[
            np.asarray(
                test_idx
            )[ids]
        ]

        q = pq[
            ids
        ]

        f = pf[
            ids
        ]

        qm = probabilistic_metrics(
            yy,
            q,
        )

        fm = probabilistic_metrics(
            yy,
            f,
        )

        result[
            str(value)
        ] = {
            "n":
                int(
                    len(ids)
                ),

            "query_macro_auc":
                qm[
                    "macro_auc"
                ],

            "full_macro_auc":
                fm[
                    "macro_auc"
                ],

            "delta_macro_auc":
                (
                    fm[
                        "macro_auc"
                    ]
                    - qm[
                        "macro_auc"
                    ]
                ),

            "query_macro_ap":
                qm[
                    "macro_ap"
                ],

            "full_macro_ap":
                fm[
                    "macro_ap"
                ],

            "delta_macro_ap":
                (
                    fm[
                        "macro_ap"
                    ]
                    - qm[
                        "macro_ap"
                    ]
                ),
        }

    return result


# ================================================================
# CLUSTER BOOTSTRAP
# ================================================================

def percentile_ci(
    values,
):
    arr = np.asarray(
        values,
        dtype=float,
    )

    arr = arr[
        np.isfinite(
            arr
        )
    ]

    if len(arr) == 0:
        return {
            "low": None,
            "median": None,
            "high": None,
        }

    return {
        "low":
            float(
                np.percentile(
                    arr,
                    2.5,
                )
            ),

        "median":
            float(
                np.percentile(
                    arr,
                    50.0,
                )
            ),

        "high":
            float(
                np.percentile(
                    arr,
                    97.5,
                )
            ),
    }


def bootstrap_by_mas(
    rows,
    test_idx,
    Y,
    pq,
    pf,
):
    rng = np.random.default_rng(
        SEED + 9001
    )

    group_to_local = defaultdict(
        list
    )

    for local_pos, global_i in enumerate(
        test_idx
    ):
        sig = rows[
            int(
                global_i
            )
        ][
            "mas_signature"
        ]

        group_to_local[
            sig
        ].append(
            local_pos
        )

    groups = sorted(
        group_to_local
    )

    if len(groups) < 2:
        raise RuntimeError(
            "Not enough bootstrap groups"
        )

    Ytest = Y[
        test_idx
    ]

    delta_auc = []
    delta_ap = []

    q_auc = []
    f_auc = []

    q_ap = []
    f_ap = []

    for _ in range(
        BOOTSTRAPS
    ):
        sampled = rng.choice(
            groups,
            size=len(groups),
            replace=True,
        )

        ids = []

        for sig in sampled:
            ids.extend(
                group_to_local[
                    sig
                ]
            )

        ids = np.asarray(
            ids,
            dtype=int,
        )

        yy = Ytest[
            ids
        ]

        qq = pq[
            ids
        ]

        ff = pf[
            ids
        ]

        qa = macro_auc(
            yy,
            qq,
        )

        fa = macro_auc(
            yy,
            ff,
        )

        qp = macro_ap(
            yy,
            qq,
        )

        fp = macro_ap(
            yy,
            ff,
        )

        if (
            np.isfinite(
                qa
            )
            and np.isfinite(
                fa
            )
        ):
            q_auc.append(
                qa
            )

            f_auc.append(
                fa
            )

            delta_auc.append(
                fa - qa
            )

        if (
            np.isfinite(
                qp
            )
            and np.isfinite(
                fp
            )
        ):
            q_ap.append(
                qp
            )

            f_ap.append(
                fp
            )

            delta_ap.append(
                fp - qp
            )

    return {
        "bootstrap_unit":
            "whole-MAS role composition",

        "n_test_mas":
            len(
                groups
            ),

        "replicates":
            BOOTSTRAPS,

        "query_macro_auc":
            percentile_ci(
                q_auc
            ),

        "full_macro_auc":
            percentile_ci(
                f_auc
            ),

        "delta_macro_auc":
            percentile_ci(
                delta_auc
            ),

        "query_macro_ap":
            percentile_ci(
                q_ap
            ),

        "full_macro_ap":
            percentile_ci(
                f_ap
            ),

        "delta_macro_ap":
            percentile_ci(
                delta_ap
            ),
    }


# ================================================================
# MAIN
# ================================================================

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

    torch.set_num_threads(
        max(
            1,
            min(
                8,
                torch.get_num_threads(),
            ),
        )
    )

    rows = load_rows()

    apply_original_split(
        rows
    )

    print(
        f"[data] rows={len(rows)}",
        flush=True,
    )

    print(
        "[K distribution]",
        {
            k: sum(
                x["K"] == k
                for x in rows
            )
            for k in sorted({
                x["K"]
                for x in rows
            })
        },
        flush=True,
    )

    for split in (
        "train",
        "val",
        "test",
    ):
        rr = [
            x
            for x in rows
            if x["split"]
            == split
        ]

        print(
            f"[split:{split}] "
            f"rows={len(rr)} "
            f"MAS={len(set(x['mas_signature'] for x in rr))}",
            flush=True,
        )

    print(
        "[encoder] loading local MiniLM...",
        flush=True,
    )

    encoder = SentenceTransformer(
        EMBED_MODEL,
        local_files_only=True,
    )

    (
        X_query,
        X_response,
        X_query_stats,
        X_full,
        Y,
        aux,
    ) = make_features(
        rows,
        encoder,
    )

    tr = idx_for(
        rows,
        "train",
    )

    va = idx_for(
        rows,
        "val",
    )

    te = idx_for(
        rows,
        "test",
    )

    print(
        "[features] "
        f"query={X_query.shape} "
        f"response={X_response.shape} "
        f"query_stats={X_query_stats.shape} "
        f"full={X_full.shape}",
        flush=True,
    )

    # ------------------------------------------------------------
    # Load ORIGINAL query-only and full models.
    # This means our main 0.7634 / 0.9129 comparison
    # is not retrained.
    # ------------------------------------------------------------

    (
        qmodel,
        qmu,
        qsd,
        qthr,
    ) = load_original_model(
        "query_only",
        X_query,
    )

    (
        fmodel,
        fmu,
        fsd,
        fthr,
    ) = load_original_model(
        "query_response",
        X_full,
    )

    p_query_test = predict_checkpoint(
        qmodel,
        qmu,
        qsd,
        X_query[
            te
        ],
    )

    p_full_test = predict_checkpoint(
        fmodel,
        fmu,
        fsd,
        X_full[
            te
        ],
    )

    Ytest = Y[
        te
    ]

    original_query_metrics = {
        **probabilistic_metrics(
            Ytest,
            p_query_test,
        ),

        **threshold_metrics(
            Ytest,
            p_query_test,
            qthr,
        ),
    }

    original_full_metrics = {
        **probabilistic_metrics(
            Ytest,
            p_full_test,
        ),

        **threshold_metrics(
            Ytest,
            p_full_test,
            fthr,
        ),
    }

    # ------------------------------------------------------------
    # Train ablations.
    # ------------------------------------------------------------

    response_only = train_new_model(
        "response_only",
        X_response,
        Y,
        rows,
    )

    query_response_stats = train_new_model(
        "query_response_stats_only",
        X_query_stats,
        Y,
        rows,
    )

    # ------------------------------------------------------------
    # Wrong-response control.
    # Use ORIGINAL full model.
    # ------------------------------------------------------------

    donors = (
        build_wrong_response_indices(
            rows,
            te,
        )
    )

    X_wrong = (
        make_shuffled_full_features(
            rows,
            te,
            donors,
            aux,
        )
    )

    p_wrong = predict_checkpoint(
        fmodel,
        fmu,
        fsd,
        X_wrong,
    )

    wrong_metrics = {
        **probabilistic_metrics(
            Ytest,
            p_wrong,
        ),

        **threshold_metrics(
            Ytest,
            p_wrong,
            fthr,
        ),
    }

    # ------------------------------------------------------------
    # Breakdowns.
    # ------------------------------------------------------------

    breakdown = {
        "K":
            subgroup_metrics(
                rows,
                te,
                Y,
                p_query_test,
                p_full_test,
                "K",
            ),

        "round":
            subgroup_metrics(
                rows,
                te,
                Y,
                p_query_test,
                p_full_test,
                "round",
            ),

        "mode":
            subgroup_metrics(
                rows,
                te,
                Y,
                p_query_test,
                p_full_test,
                "mode",
            ),
    }

    # ------------------------------------------------------------
    # Cluster bootstrap.
    # ------------------------------------------------------------

    bootstrap = bootstrap_by_mas(
        rows,
        te,
        Y,
        p_query_test,
        p_full_test,
    )

    # ------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------

    report = {
        "original_query_only":
            original_query_metrics,

        "original_query_response":
            original_full_metrics,

        "response_only":
            response_only[
                "metrics"
            ],

        "query_response_stats_only":
            query_response_stats[
                "metrics"
            ],

        "wrong_response_control":
            wrong_metrics,

        "main_deltas": {
            "full_minus_query_macro_auc":
                (
                    original_full_metrics[
                        "macro_auc"
                    ]
                    - original_query_metrics[
                        "macro_auc"
                    ]
                ),

            "full_minus_query_macro_ap":
                (
                    original_full_metrics[
                        "macro_ap"
                    ]
                    - original_query_metrics[
                        "macro_ap"
                    ]
                ),

            "full_minus_stats_macro_auc":
                (
                    original_full_metrics[
                        "macro_auc"
                    ]
                    - query_response_stats[
                        "metrics"
                    ][
                        "macro_auc"
                    ]
                ),

            "full_minus_wrong_macro_auc":
                (
                    original_full_metrics[
                        "macro_auc"
                    ]
                    - wrong_metrics[
                        "macro_auc"
                    ]
                ),

            "response_only_minus_query_macro_auc":
                (
                    response_only[
                        "metrics"
                    ][
                        "macro_auc"
                    ]
                    - original_query_metrics[
                        "macro_auc"
                    ]
                ),
        },

        "breakdown":
            breakdown,

        "bootstrap":
            bootstrap,

        "available_K":
            sorted({
                int(
                    x["K"]
                )
                for x in rows
            }),

        "label":
            "selected_roles_training_only",

        "inference_privileged_features":
            False,

        "internal_test_used":
            False,
    }

    (
        OUT
        / "REPORT.json"
    ).write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
        )
    )

    # ============================================================
    # HUMAN-READABLE SUMMARY
    # ============================================================

    print()
    print(
        "=" * 108
    )
    print(
        "STAGE-10 RESPONSE SENSOR — FULL AUDIT"
    )
    print(
        "=" * 108
    )

    def show(
        name,
        m,
    ):
        print()
        print(
            f"[{name}]"
        )

        print(
            f"macro AUC        : "
            f"{m['macro_auc']:.4f}"
        )

        print(
            f"micro AUC        : "
            f"{m['micro_auc']:.4f}"
        )

        print(
            f"macro AP         : "
            f"{m['macro_ap']:.4f}"
        )

        print(
            f"micro AP         : "
            f"{m['micro_ap']:.4f}"
        )

        print(
            f"exact active set : "
            f"{100*m['exact_active_set']:.2f}%"
        )

        print(
            f"route-count MAE  : "
            f"{m['routing_count_mae']:.4f}"
        )

        print(
            f"route-count exact: "
            f"{100*m['routing_count_exact']:.2f}%"
        )

    show(
        "QUERY ONLY — ORIGINAL",
        original_query_metrics,
    )

    show(
        "RESPONSE ONLY",
        response_only[
            "metrics"
        ],
    )

    show(
        "QUERY + RESPONSE STATS ONLY",
        query_response_stats[
            "metrics"
        ],
    )

    show(
        "QUERY + REAL RESPONSE — ORIGINAL",
        original_full_metrics,
    )

    show(
        "QUERY + WRONG RESPONSE",
        wrong_metrics,
    )

    print()
    print(
        "=" * 108
    )
    print(
        "KEY DELTAS"
    )
    print(
        "=" * 108
    )

    print(
        "Full - Query ΔAUC       : "
        f"{report['main_deltas']['full_minus_query_macro_auc']:+.4f}"
    )

    print(
        "Full - Query ΔAP        : "
        f"{report['main_deltas']['full_minus_query_macro_ap']:+.4f}"
    )

    print(
        "Full - StatsOnly ΔAUC   : "
        f"{report['main_deltas']['full_minus_stats_macro_auc']:+.4f}"
    )

    print(
        "Full - WrongResp ΔAUC   : "
        f"{report['main_deltas']['full_minus_wrong_macro_auc']:+.4f}"
    )

    print(
        "RespOnly - Query ΔAUC   : "
        f"{report['main_deltas']['response_only_minus_query_macro_auc']:+.4f}"
    )

    print()
    print(
        "=" * 108
    )
    print(
        "BREAKDOWN BY K"
    )
    print(
        "=" * 108
    )

    for k, m in (
        breakdown[
            "K"
        ].items()
    ):
        print(
            f"K={k:>2} "
            f"n={m['n']:4d} "
            f"query={m['query_macro_auc']:.4f} "
            f"full={m['full_macro_auc']:.4f} "
            f"Δ={m['delta_macro_auc']:+.4f} "
            f"ΔAP={m['delta_macro_ap']:+.4f}"
        )

    print()
    print(
        "=" * 108
    )
    print(
        "BREAKDOWN BY ROUND"
    )
    print(
        "=" * 108
    )

    for r, m in (
        breakdown[
            "round"
        ].items()
    ):
        print(
            f"round={r:>2} "
            f"n={m['n']:4d} "
            f"query={m['query_macro_auc']:.4f} "
            f"full={m['full_macro_auc']:.4f} "
            f"Δ={m['delta_macro_auc']:+.4f}"
        )

    print()
    print(
        "=" * 108
    )
    print(
        "BREAKDOWN BY MODE"
    )
    print(
        "=" * 108
    )

    for mode, m in (
        breakdown[
            "mode"
        ].items()
    ):
        print(
            f"{mode:16s} "
            f"n={m['n']:4d} "
            f"query={m['query_macro_auc']:.4f} "
            f"full={m['full_macro_auc']:.4f} "
            f"Δ={m['delta_macro_auc']:+.4f}"
        )

    b = bootstrap

    print()
    print(
        "=" * 108
    )
    print(
        "WHOLE-MAS CLUSTER BOOTSTRAP"
    )
    print(
        "=" * 108
    )

    print(
        f"test MAS groups: "
        f"{b['n_test_mas']}"
    )

    print(
        f"replicates     : "
        f"{b['replicates']}"
    )

    da = b[
        "delta_macro_auc"
    ]

    dap = b[
        "delta_macro_ap"
    ]

    print(
        "Δ Macro AUC 95% CI: "
        f"[{da['low']:+.4f}, "
        f"{da['high']:+.4f}] "
        f"median={da['median']:+.4f}"
    )

    print(
        "Δ Macro AP  95% CI: "
        f"[{dap['low']:+.4f}, "
        f"{dap['high']:+.4f}] "
        f"median={dap['median']:+.4f}"
    )

    print()
    print(
        "AVAILABLE K:",
        report[
            "available_K"
        ],
    )

    print(
        "NOTE: this dataset cannot establish "
        "generalization to missing K values."
    )

    print(
        "PRIVILEGED INPUT AT INFERENCE: NO"
    )

    print(
        "INTERNAL TEST USED: NO"
    )

    print(
        f"REPORT: {OUT / 'REPORT.json'}"
    )


if __name__ == "__main__":
    main()
