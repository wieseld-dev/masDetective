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


ROOT = Path(
    "artifacts/adaptive_v11_curriculum_v2/stage_10"
)

OUT = Path(
    "artifacts/stage10_response_sensor_v1_long"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 20260922

EMBED_MODEL = "all-MiniLM-L6-v2"

EPOCHS = 60
PATIENCE = 8
BATCH = 256
LR = 4e-4
WD = 1e-4


# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def stable_hash(text: str) -> int:
    d = hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=8,
    ).digest()

    return int.from_bytes(
        d,
        "big",
    )


def text_stats(text: str):
    text = str(text)

    words = text.split()

    nchar = len(text)
    nword = len(words)

    uniq = (
        len(set(w.lower() for w in words))
        / max(1, nword)
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
            ("-", "*", "•")
        )
        for line in text.splitlines()
    )

    avg_word = (
        sum(len(w) for w in words)
        / max(1, nword)
    )

    return np.asarray([
        math.log1p(nchar),
        math.log1p(nword),
        math.log1p(
            len(text.splitlines())
        ),
        uniq,
        digits / max(1, nchar),
        uppers / max(1, nchar),
        punct / max(1, nchar),
        math.log1p(
            text.count(":")
        ),
        math.log1p(bullets),
        avg_word,
    ], dtype=np.float32)


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

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
            for x in read_jsonl(rp):

                if (
                    x.get("execution_status")
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
                        if role in selected
                        else 0.0
                        for role
                        in TRAIN_ROLES
                    ],
                    dtype=np.float32,
                )

                # Strong whole-MAS grouping:
                # the same installed role composition can NEVER
                # appear in two different splits.
                mas_signature = (
                    "|".join(truth)
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
                            x["session_id"]
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

                    "truth_roles":
                        truth,

                    "K":
                        len(truth),

                    "mas_signature":
                        mas_signature,

                    "y":
                        y,
                })

    if not rows:
        raise RuntimeError(
            "No usable Stage-10 rows"
        )

    return rows


def build_whole_mas_split(rows):
    by_k = defaultdict(set)

    for x in rows:
        by_k[
            int(x["K"])
        ].add(
            x[
                "mas_signature"
            ]
        )

    assignment = {}

    for k in sorted(by_k):
        sigs = sorted(
            by_k[k],
            key=lambda s:
                stable_hash(
                    f"{SEED}|{k}|{s}"
                ),
        )

        n = len(sigs)

        if n < 3:
            raise RuntimeError(
                f"Too few MAS groups "
                f"for K={k}: {n}"
            )

        n_test = max(
            1,
            int(
                round(
                    0.15 * n
                )
            ),
        )

        n_val = max(
            1,
            int(
                round(
                    0.15 * n
                )
            ),
        )

        if (
            n_test
            + n_val
            >= n
        ):
            n_test = 1
            n_val = 1

        n_train = (
            n
            - n_val
            - n_test
        )

        for s in sigs[:n_train]:
            assignment[s] = "train"

        for s in sigs[
            n_train:
            n_train+n_val
        ]:
            assignment[s] = "val"

        for s in sigs[
            n_train+n_val:
        ]:
            assignment[s] = "test"

        print(
            f"[split] K={k} "
            f"MAS={n} "
            f"train={n_train} "
            f"val={n_val} "
            f"test={n_test}",
            flush=True,
        )

    for x in rows:
        x["split"] = assignment[
            x["mas_signature"]
        ]

    return assignment


# ------------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------------

def encode_unique(
    encoder,
    texts,
    *,
    name,
):
    unique = list(
        dict.fromkeys(
            map(str, texts)
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
        in enumerate(unique)
    }

    return np.stack([
        ue[
            index[str(t)]
        ]
        for t in texts
    ])


def build_features(
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

    qemb = encode_unique(
        encoder,
        queries,
        name="query",
    )

    remb = encode_unique(
        encoder,
        responses,
        name="response",
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
    ])

    rstats = np.stack([
        text_stats(
            x["response"]
        )
        for x in rows
    ])

    # --------------------------------------------------------------
    # BASELINE:
    # only information available BEFORE we see the MAS response.
    # --------------------------------------------------------------

    X_query = np.concatenate([
        qemb,
        q_role,
        qstats,
    ], axis=1).astype(
        np.float32
    )

    # --------------------------------------------------------------
    # FULL BLACK-BOX SENSOR:
    # same query features + visible final response.
    #
    # Interaction terms make it easier to learn:
    # "what changed / appeared in the response relative to the query?"
    # --------------------------------------------------------------

    qr_cos = np.sum(
        qemb * remb,
        axis=1,
        keepdims=True,
    ).astype(
        np.float32
    )

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

    return (
        X_query,
        X_full,
        Y,
        role_emb,
    )


# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------

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
        return self.net(x)


def idx_for(
    rows,
    split,
):
    return np.asarray([
        i
        for i, x
        in enumerate(rows)
        if x["split"] == split
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

    return float(
        np.mean(vals)
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

    return float(
        np.mean(vals)
    )


@torch.no_grad()
def predict(
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
                s:s+1024
            ]
        )

        out.append(
            torch.sigmoid(
                model(xb)
            )
            .cpu()
            .numpy()
        )

    return np.concatenate(
        out,
        axis=0,
    )


def fit_thresholds(
    y,
    p,
):
    thresholds = []

    grid = np.linspace(
        0.05,
        0.95,
        37,
    )

    for j in range(
        y.shape[1]
    ):
        best_t = 0.5
        best_f = -1.0

        for t in grid:
            pred = (
                p[:, j]
                >= t
            ).astype(int)

            f = f1_score(
                y[:, j],
                pred,
                zero_division=0,
            )

            if f > best_f:
                best_f = f
                best_t = float(t)

        thresholds.append(
            best_t
        )

    return np.asarray(
        thresholds,
        dtype=np.float32,
    )


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


def probabilistic_metrics(
    y,
    p,
):
    valid_micro = (
        len(
            np.unique(
                y.ravel()
            )
        )
        >= 2
    )

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
            )
            if valid_micro
            else float("nan"),

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


def train_model(
    name,
    X,
    Y,
    rows,
):
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
        len(Ytr)
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

    best_auc = -1e18
    best_epoch = 0
    best_state = None
    bad = 0

    print()
    print(
        "=" * 90
    )
    print(
        f"TRAINING {name}"
    )
    print(
        "=" * 90
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

        pv = predict(
            model,
            Xva,
        )

        va_auc = macro_auc(
            Yva,
            pv,
        )

        improved = (
            va_auc
            > best_auc
            + 1e-5
        )

        if improved:
            best_auc = va_auc
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
            f"val_macro_auc={va_auc:.4f} "
            f"{'*' if improved else ''}",
            flush=True,
        )

        if (
            bad
            >= PATIENCE
        ):
            print(
                f"[{name}] early-stop",
                flush=True,
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"{name}: no checkpoint"
        )

    model.load_state_dict(
        best_state
    )

    model.eval()

    pval = predict(
        model,
        Xva,
    )

    ptest = predict(
        model,
        Xte,
    )

    thresholds = fit_thresholds(
        Yva,
        pval,
    )

    val_metrics = {
        **probabilistic_metrics(
            Yva,
            pval,
        ),

        **threshold_metrics(
            Yva,
            pval,
            thresholds,
        ),
    }

    test_metrics = {
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

    per_role = {}

    for j, role in enumerate(
        TRAIN_ROLES
    ):
        if (
            len(
                np.unique(
                    Yte[:, j]
                )
            )
            >= 2
        ):
            auc = float(
                roc_auc_score(
                    Yte[:, j],
                    ptest[:, j],
                )
            )
        else:
            auc = None

        ap = (
            float(
                average_precision_score(
                    Yte[:, j],
                    ptest[:, j],
                )
            )
            if Yte[:, j].sum() > 0
            else None
        )

        per_role[role] = {
            "auc":
                auc,

            "ap":
                ap,

            "test_positive":
                int(
                    Yte[
                        :,
                        j
                    ].sum()
                ),

            "threshold":
                float(
                    thresholds[j]
                ),
        }

    ckpt = (
        OUT
        / f"{name}.pt"
    )

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

        "embedding_model":
            EMBED_MODEL,

        "best_epoch":
            best_epoch,

        "internal_test_used":
            False,
    }, ckpt)

    return {
        "checkpoint":
            str(
                ckpt
            ),

        "best_epoch":
            best_epoch,

        "val":
            val_metrics,

        "test":
            test_metrics,

        "per_role":
            per_role,
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

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
                4,
                torch.get_num_threads(),
            ),
        )
    )

    rows = load_rows()

    print(
        f"[data] usable rows={len(rows)}",
        flush=True,
    )

    assignment = (
        build_whole_mas_split(
            rows
        )
    )

    row_counts = {
        s: sum(
            x["split"] == s
            for x in rows
        )
        for s
        in (
            "train",
            "val",
            "test",
        )
    }

    mas_counts = {
        s: len({
            x["mas_signature"]
            for x in rows
            if x["split"] == s
        })
        for s
        in (
            "train",
            "val",
            "test",
        )
    }

    print(
        "[rows by split]",
        row_counts,
        flush=True,
    )

    print(
        "[MAS by split]",
        mas_counts,
        flush=True,
    )

    # Verify composition leakage = zero.
    split_sets = {
        s: {
            x["mas_signature"]
            for x in rows
            if x["split"] == s
        }
        for s
        in (
            "train",
            "val",
            "test",
        )
    }

    leakage = (
        len(
            split_sets["train"]
            & split_sets["val"]
        )
        + len(
            split_sets["train"]
            & split_sets["test"]
        )
        + len(
            split_sets["val"]
            & split_sets["test"]
        )
    )

    print(
        "[split leakage]",
        leakage,
        flush=True,
    )

    if leakage != 0:
        raise RuntimeError(
            "Whole-MAS split leakage"
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
        X_full,
        Y,
        role_emb,
    ) = build_features(
        rows,
        encoder,
    )

    print(
        "[features] "
        f"query_only={X_query.shape} "
        f"query_response={X_full.shape}",
        flush=True,
    )

    query_result = train_model(
        "query_only",
        X_query,
        Y,
        rows,
    )

    full_result = train_model(
        "query_response",
        X_full,
        Y,
        rows,
    )

    delta_auc = (
        full_result[
            "test"
        ][
            "macro_auc"
        ]
        - query_result[
            "test"
        ][
            "macro_auc"
        ]
    )

    delta_ap = (
        full_result[
            "test"
        ][
            "macro_ap"
        ]
        - query_result[
            "test"
        ][
            "macro_ap"
        ]
    )

    role_delta = []

    for role in TRAIN_ROLES:
        qa = (
            query_result[
                "per_role"
            ][role]["auc"]
        )

        fa = (
            full_result[
                "per_role"
            ][role]["auc"]
        )

        if (
            qa is not None
            and fa is not None
        ):
            role_delta.append({
                "role":
                    role,

                "query_auc":
                    qa,

                "full_auc":
                    fa,

                "delta":
                    fa - qa,
            })

    role_delta.sort(
        key=lambda x:
            x["delta"],
        reverse=True,
    )

    report = {
        "rows":
            len(rows),

        "row_counts":
            row_counts,

        "whole_mas_counts":
            mas_counts,

        "whole_mas_overlap":
            leakage,

        "query_only":
            query_result,

        "query_response":
            full_result,

        "response_incremental_signal": {
            "delta_test_macro_auc":
                delta_auc,

            "delta_test_macro_ap":
                delta_ap,
        },

        "per_role_auc_delta":
            role_delta,

        "label":
            "selected_roles_training_only",

        "inference_inputs":
            (
                "query + visible final response only"
            ),

        "truth_used_for":
            (
                "training labels and whole-MAS split only"
            ),

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

    (
        OUT
        / "split_assignment.json"
    ).write_text(
        json.dumps(
            assignment,
            indent=2,
            sort_keys=True,
        )
    )

    np.save(
        OUT
        / "role_embeddings.npy",
        role_emb,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "STAGE-10 RESPONSE SENSOR — FINAL"
    )
    print(
        "=" * 100
    )

    for name, result in [
        (
            "QUERY ONLY",
            query_result,
        ),
        (
            "QUERY + RESPONSE",
            full_result,
        ),
    ]:
        m = result[
            "test"
        ]

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
            f"macro F1         : "
            f"{m['macro_f1']:.4f}"
        )

        print(
            f"micro F1         : "
            f"{m['micro_f1']:.4f}"
        )

        print(
            f"exact active set : "
            f"{100*m['exact_active_set']:.2f}%"
        )

        print(
            f"mean Jaccard     : "
            f"{m['mean_jaccard']:.4f}"
        )

        print(
            f"route-count MAE  : "
            f"{m['routing_count_mae']:.4f}"
        )

        print(
            f"route-count exact: "
            f"{100*m['routing_count_exact']:.2f}%"
        )

    print()
    print(
        "RESPONSE INCREMENTAL SIGNAL"
    )

    print(
        f"Δ macro AUC = "
        f"{delta_auc:+.4f}"
    )

    print(
        f"Δ macro AP  = "
        f"{delta_ap:+.4f}"
    )

    print()
    print(
        "TOP RESPONSE-GAIN ROLES:"
    )

    for x in role_delta[:8]:
        print(
            f"{x['role']:28s} "
            f"query={x['query_auc']:.3f} "
            f"full={x['full_auc']:.3f} "
            f"delta={x['delta']:+.3f}"
        )

    print()
    print(
        "WORST RESPONSE-GAIN ROLES:"
    )

    for x in role_delta[-5:]:
        print(
            f"{x['role']:28s} "
            f"query={x['query_auc']:.3f} "
            f"full={x['full_auc']:.3f} "
            f"delta={x['delta']:+.3f}"
        )

    print()
    print(
        f"WHOLE-MAS SPLIT LEAKAGE: "
        f"{leakage}"
    )

    print(
        "PRIVILEGED INPUT AT INFERENCE: NO"
    )

    print(
        "INTERNAL TEST USED: NO"
    )

    print(
        "SENSOR TRAINING: DONE"
    )


if __name__ == "__main__":
    main()
