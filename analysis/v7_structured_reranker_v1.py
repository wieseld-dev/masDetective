#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import (
    PeftConfig,
    PeftModel,
)

import analysis.train_v13_6b_activation_judge as tr
import analysis.train_v7_benign_output_sensor_pilot_v1 as bos


ADAPTER = Path(
    "artifacts/v7_benign_output_sensor_pilot_v1/"
    "best_adapter"
)

DATA_ROOT = Path(
    "artifacts/v7_benign_final_pool_v2"
)

AUDIT = Path(
    "artifacts/v7_actual_query_manual_audit_v1/"
    "sample_300.jsonl"
)

OUT = Path(
    "artifacts/v7_structured_reranker_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926

TRAIN_LIMIT = 2048
DEV_LIMIT = 512
TEST_LIMIT = 512

MAX_LENGTH = 768
SCORE_BATCH = 8

TARGET_RECALL = 0.80


CAPABILITY_IDS = {
    28, 45, 69, 70, 82, 93, 110, 133, 147, 156,
    166, 179, 186, 233, 236, 238, 255, 267, 300,
}

DIRECTION_IDS = {
    8, 27, 39, 41, 55, 66, 80, 81, 85, 89,
    91, 100, 108, 112, 172, 182, 184, 189, 205,
    218, 220, 224, 247, 256, 259, 273, 278, 297,
}


def read_jsonl(path):
    rows = []

    with Path(path).open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def dump(path, obj):
    Path(path).write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def norm_q(x):
    return " ".join(
        str(x or "")
        .strip()
        .split()
    ).lower()


def sample_fixed(
    rows,
    n,
    seed,
):
    rows = list(rows)

    if len(rows) <= n:
        return rows

    rng = random.Random(seed)

    return rng.sample(
        rows,
        n,
    )


def load_sensor():
    cfg = PeftConfig.from_pretrained(
        str(ADAPTER)
    )

    base_name = (
        cfg.base_model_name_or_path
    )

    tok = AutoTokenizer.from_pretrained(
        str(ADAPTER)
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"
    tok.truncation_side = "right"

    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            base_name,
            num_labels=1,
            torch_dtype=torch.bfloat16,
        )
    )

    base.config.pad_token_id = (
        tok.pad_token_id
    )

    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER),
        is_trainable=False,
    )

    model.eval()
    model.to("cuda")

    return tok, model


@torch.inference_mode()
def score_rows(
    model,
    tok,
    rows,
    roles,
    role_desc,
):
    texts = []

    for row in rows:
        answer = bos.get_answer(
            row
        )

        for role in roles:
            texts.append(
                bos.sensor_text(
                    role,
                    role_desc[role],
                    answer,
                )
            )

    scores = bos.score_texts(
        model,
        tok,
        texts,
        torch.device("cuda"),
        MAX_LENGTH,
        SCORE_BATCH,
    )

    return scores.reshape(
        len(rows),
        len(roles),
    )


def labels_matrix(
    rows,
    roles,
):
    y = np.zeros(
        (
            len(rows),
            len(roles),
        ),
        dtype=np.float32,
    )

    for i, row in enumerate(rows):
        selected = set(
            bos.get_selected(row)
        )

        for j, role in enumerate(
            roles
        ):
            if role in selected:
                y[i, j] = 1.0

    return y


def binary_auc(y, s):
    return bos.binary_auc(
        np.asarray(y),
        np.asarray(s),
    )


def ranking_metrics(
    scores,
    labels,
    roles,
):
    micro = binary_auc(
        labels.reshape(-1),
        scores.reshape(-1),
    )

    role_auc = {}

    for j, role in enumerate(
        roles
    ):
        role_auc[role] = binary_auc(
            labels[:, j],
            scores[:, j],
        )

    valid_role_auc = [
        x
        for x in role_auc.values()
        if math.isfinite(x)
    ]

    within = []
    topk = []
    exact = []

    for i in range(
        len(scores)
    ):
        y = labels[i]
        s = scores[i]

        a = binary_auc(
            y,
            s,
        )

        if math.isfinite(a):
            within.append(a)

        k = int(
            y.sum()
        )

        if k <= 0:
            continue

        idx = np.argsort(
            -s
        )[:k]

        pred = set(
            idx.tolist()
        )

        truth = set(
            np.where(
                y == 1
            )[0].tolist()
        )

        topk.append(
            len(pred & truth)
            / k
        )

        exact.append(
            float(
                pred == truth
            )
        )

    return {
        "rows":
            len(scores),

        "micro_auc":
            float(micro),

        "macro_role_auc":
            float(
                np.mean(
                    valid_role_auc
                )
            ),

        "within_output_auc":
            float(
                np.mean(within)
            ),

        "topk_activation_recall":
            float(
                np.mean(topk)
            ),

        "exact_activation_set_topk":
            float(
                np.mean(exact)
            ),

        "role_auc":
            role_auc,
    }


# ============================================================
# Threshold fitting:
#
# For each role choose the HIGHEST dev threshold achieving
# recall >= TARGET_RECALL.
#
# Highest threshold -> minimum FPs under that recall constraint.
# ============================================================

def fit_thresholds(
    scores,
    labels,
    target_recall,
):
    thresholds = []

    for j in range(
        scores.shape[1]
    ):
        s = scores[:, j]
        y = labels[:, j]

        positives = np.sort(
            s[y == 1]
        )[::-1]

        if len(positives) == 0:
            thresholds.append(
                float("inf")
            )
            continue

        need = max(
            1,
            int(
                math.ceil(
                    target_recall
                    * len(positives)
                )
            ),
        )

        threshold = float(
            positives[
                min(
                    need - 1,
                    len(positives) - 1,
                )
            ]
        )

        thresholds.append(
            threshold
        )

    return np.asarray(
        thresholds,
        dtype=np.float64,
    )


def threshold_metrics(
    scores,
    labels,
    thresholds,
):
    pred = (
        scores
        >= thresholds.reshape(
            1,
            -1,
        )
    ).astype(np.int64)

    y = labels.astype(
        np.int64
    )

    tp = int(
        ((pred == 1) & (y == 1))
        .sum()
    )

    fp = int(
        ((pred == 1) & (y == 0))
        .sum()
    )

    fn = int(
        ((pred == 0) & (y == 1))
        .sum()
    )

    tn = int(
        ((pred == 0) & (y == 0))
        .sum()
    )

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    fpr = (
        fp / (fp + tn)
        if fp + tn
        else 0.0
    )

    exact = np.all(
        pred == y,
        axis=1,
    )

    return {
        "precision":
            float(precision),

        "recall":
            float(recall),

        "fpr":
            float(fpr),

        "false_positives":
            fp,

        "false_negatives":
            fn,

        "avg_fp_per_output":
            float(
                fp / len(y)
            ),

        "avg_fn_per_output":
            float(
                fn / len(y)
            ),

        "exact_activation_set_threshold":
            float(
                exact.mean()
            ),
    }


# ============================================================
# Structured residual reranker.
#
# raw score vector s in R^20
#
# x = standardized score vector
#
# z = s + x @ Delta^T + bias
#
# Delta starts at zero, so step 0 is EXACTLY V1.
# This makes catastrophic forgetting impossible unless dev
# selection explicitly chooses a worse checkpoint.
# ============================================================

class StructuredReranker(nn.Module):
    def __init__(
        self,
        n_roles,
        mean,
        std,
    ):
        super().__init__()

        self.delta = nn.Parameter(
            torch.zeros(
                n_roles,
                n_roles,
            )
        )

        self.bias = nn.Parameter(
            torch.zeros(
                n_roles
            )
        )

        self.register_buffer(
            "mean",
            torch.tensor(
                mean,
                dtype=torch.float32,
            )
        )

        self.register_buffer(
            "std",
            torch.tensor(
                std,
                dtype=torch.float32,
            )
        )

    def forward(
        self,
        scores,
    ):
        x = (
            scores
            - self.mean
        ) / self.std

        correction = (
            x
            @ self.delta.t()
        )

        return (
            scores
            + correction
            + self.bias
        )


def pairwise_loss(
    z,
    y,
):
    # diff[i,p,n] = z_p - z_n
    diff = (
        z.unsqueeze(2)
        - z.unsqueeze(1)
    )

    pos_mask = (
        y.unsqueeze(2)
        > 0.5
    )

    neg_mask = (
        y.unsqueeze(1)
        < 0.5
    )

    mask = (
        pos_mask
        & neg_mask
    )

    return F.softplus(
        -diff[mask]
    ).mean()


def hard_pairwise_loss(
    z,
    raw,
    y,
    hard_k=4,
):
    losses = []

    for i in range(
        z.shape[0]
    ):
        pos = torch.where(
            y[i] > 0.5
        )[0]

        neg = torch.where(
            y[i] < 0.5
        )[0]

        if (
            len(pos) == 0
            or len(neg) == 0
        ):
            continue

        raw_neg_scores = raw[
            i,
            neg,
        ]

        k = min(
            hard_k,
            len(neg),
        )

        hard_local = torch.topk(
            raw_neg_scores,
            k=k,
        ).indices

        hard_neg = neg[
            hard_local
        ]

        d = (
            z[i, pos]
            .unsqueeze(1)
            -
            z[i, hard_neg]
            .unsqueeze(0)
        )

        losses.append(
            F.softplus(
                -d
            ).mean()
        )

    if not losses:
        return torch.tensor(
            0.0,
            device=z.device,
        )

    return torch.stack(
        losses
    ).mean()


# ============================================================
# Role-wise affine calibration AFTER reranking.
# Fit only on dev.
# ============================================================

def fit_role_calibration(
    scores_np,
    labels_np,
    steps=1800,
    lr=0.03,
):
    scores = torch.tensor(
        scores_np,
        dtype=torch.float32,
    )

    labels = torch.tensor(
        labels_np,
        dtype=torch.float32,
    )

    n_roles = scores.shape[1]

    log_a = torch.zeros(
        n_roles,
        requires_grad=True,
    )

    bias = torch.zeros(
        n_roles,
        requires_grad=True,
    )

    opt = torch.optim.Adam(
        [log_a, bias],
        lr=lr,
    )

    best = None
    best_loss = float("inf")

    for _ in range(
        steps
    ):
        opt.zero_grad()

        scale = torch.exp(
            log_a
        )

        z = (
            scores
            * scale.unsqueeze(0)
            + bias.unsqueeze(0)
        )

        pl = pairwise_loss(
            z,
            labels,
        )

        bce = (
            F.binary_cross_entropy_with_logits(
                z,
                labels,
            )
        )

        reg = (
            0.0025
            * (log_a ** 2).mean()
            +
            0.0005
            * (bias ** 2).mean()
        )

        loss = (
            pl
            + 0.05 * bce
            + reg
        )

        loss.backward()
        opt.step()

        v = float(
            loss.detach()
        )

        if v < best_loss:
            best_loss = v

            best = (
                torch.exp(
                    log_a.detach()
                ).numpy(),

                bias.detach()
                .numpy(),
            )

    return best


def apply_affine(
    scores,
    scale,
    bias,
):
    return (
        scores
        * scale.reshape(
            1,
            -1,
        )
        + bias.reshape(
            1,
            -1,
        )
    )


def build_audited_test(
    full_test,
):
    audit = read_jsonl(
        AUDIT
    )

    audit_test = [
        x
        for x in audit
        if x["split"] == "test"
    ]

    lookup = {}

    for row in full_test:
        key = (
            str(
                row.get(
                    "session_id"
                )
            ),
            norm_q(
                row.get(
                    "query"
                )
            ),
        )

        lookup.setdefault(
            key,
            []
        ).append(
            row
        )

    rows = []
    ids = []

    for a in audit_test:
        key = (
            str(
                a.get(
                    "session_id"
                )
            ),
            norm_q(
                a.get(
                    "query"
                )
            ),
        )

        matches = lookup.get(
            key,
            []
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"audit join failure "
                f"id={a['audit_id']} "
                f"matches={len(matches)}"
            )

        rows.append(
            matches[0]
        )

        ids.append(
            int(
                a["audit_id"]
            )
        )

    strict_idx = [
        i
        for i, aid in enumerate(
            ids
        )
        if (
            aid not in CAPABILITY_IDS
            and aid not in DIRECTION_IDS
        )
    ]

    return (
        rows,
        ids,
        strict_idx,
    )


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

    role_desc = dict(
        tr.ROLE_DESCRIPTIONS
    )

    roles = list(
        role_desc.keys()
    )

    train_all = read_jsonl(
        DATA_ROOT
        / "train.jsonl"
    )

    dev_all = read_jsonl(
        DATA_ROOT
        / "dev.jsonl"
    )

    test_all = read_jsonl(
        DATA_ROOT
        / "test.jsonl"
    )

    train = sample_fixed(
        train_all,
        TRAIN_LIMIT,
        SEED + 11,
    )

    # SAME sample convention as V1/calibration.
    dev = sample_fixed(
        dev_all,
        DEV_LIMIT,
        SEED + 101,
    )

    test = sample_fixed(
        test_all,
        TEST_LIMIT,
        SEED + 202,
    )

    (
        audited,
        audited_ids,
        strict_idx,
    ) = build_audited_test(
        test_all
    )

    print(
        "===== DATA ====="
    )

    print(
        "train/dev/test/audit/strict:",
        len(train),
        len(dev),
        len(test),
        len(audited),
        len(strict_idx),
        flush=True,
    )

    # ========================================================
    # Frozen V1 feature extraction.
    # ========================================================

    print(
        "\n===== LOAD FROZEN V1 =====",
        flush=True,
    )

    tok, sensor = (
        load_sensor()
    )

    print(
        "\n===== SCORE TRAIN =====",
        flush=True,
    )

    train_s = score_rows(
        sensor,
        tok,
        train,
        roles,
        role_desc,
    )

    print(
        "\n===== SCORE DEV =====",
        flush=True,
    )

    dev_s = score_rows(
        sensor,
        tok,
        dev,
        roles,
        role_desc,
    )

    print(
        "\n===== SCORE TEST =====",
        flush=True,
    )

    test_s = score_rows(
        sensor,
        tok,
        test,
        roles,
        role_desc,
    )

    print(
        "\n===== SCORE AUDITED =====",
        flush=True,
    )

    audit_s = score_rows(
        sensor,
        tok,
        audited,
        roles,
        role_desc,
    )

    train_y = labels_matrix(
        train,
        roles,
    )

    dev_y = labels_matrix(
        dev,
        roles,
    )

    test_y = labels_matrix(
        test,
        roles,
    )

    audit_y = labels_matrix(
        audited,
        roles,
    )

    strict_s = audit_s[
        strict_idx
    ]

    strict_y = audit_y[
        strict_idx
    ]

    del sensor
    del tok

    torch.cuda.empty_cache()

    # ========================================================
    # Raw baseline metrics.
    # ========================================================

    raw = {
        "dev":
            ranking_metrics(
                dev_s,
                dev_y,
                roles,
            ),

        "test":
            ranking_metrics(
                test_s,
                test_y,
                roles,
            ),

        "audited_100":
            ranking_metrics(
                audit_s,
                audit_y,
                roles,
            ),

        "strict_82":
            ranking_metrics(
                strict_s,
                strict_y,
                roles,
            ),
    }

    print(
        "\n===== RAW BASELINE ====="
    )

    for name, m in raw.items():
        print(
            name,
            "within=",
            round(
                m[
                    "within_output_auc"
                ],
                6,
            ),
            "micro=",
            round(
                m[
                    "micro_auc"
                ],
                6,
            ),
            flush=True,
        )

    # ========================================================
    # Train tiny structured reranker on TRAIN ONLY.
    # ========================================================

    mean = train_s.mean(
        axis=0
    )

    std = train_s.std(
        axis=0
    )

    std = np.maximum(
        std,
        1e-3,
    )

    reranker = StructuredReranker(
        len(roles),
        mean,
        std,
    )

    train_s_t = torch.tensor(
        train_s,
        dtype=torch.float32,
    )

    train_y_t = torch.tensor(
        train_y,
        dtype=torch.float32,
    )

    dev_s_t = torch.tensor(
        dev_s,
        dtype=torch.float32,
    )

    opt = torch.optim.Adam(
        reranker.parameters(),
        lr=0.01,
    )

    best_state = {
        k: v.detach().clone()
        for k, v
        in reranker.state_dict().items()
    }

    # Baseline is step zero.
    best_dev_auc = raw[
        "dev"
    ][
        "within_output_auc"
    ]

    best_step = 0

    print(
        "\n===== STRUCTURED TRAINING =====",
        flush=True,
    )

    for step in range(
        1,
        801,
    ):
        reranker.train()

        opt.zero_grad()

        z = reranker(
            train_s_t
        )

        all_pair = pairwise_loss(
            z,
            train_y_t,
        )

        hard_pair = hard_pairwise_loss(
            z,
            train_s_t,
            train_y_t,
            hard_k=4,
        )

        bce = (
            F.binary_cross_entropy_with_logits(
                z,
                train_y_t,
            )
        )

        reg = (
            0.002
            * (
                reranker.delta
                ** 2
            ).mean()
            +
            0.0005
            * (
                reranker.bias
                ** 2
            ).mean()
        )

        loss = (
            all_pair
            + 0.25 * hard_pair
            + 0.05 * bce
            + reg
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            reranker.parameters(),
            1.0,
        )

        opt.step()

        if (
            step % 25 == 0
            or step == 1
        ):
            reranker.eval()

            with torch.no_grad():
                dz = reranker(
                    dev_s_t
                ).numpy()

            dm = ranking_metrics(
                dz,
                dev_y,
                roles,
            )

            auc = dm[
                "within_output_auc"
            ]

            print(
                f"[step {step:03d}] "
                f"loss={float(loss.detach()):.5f} "
                f"all={float(all_pair.detach()):.5f} "
                f"hard={float(hard_pair.detach()):.5f} "
                f"dev_within={auc:.6f}",
                flush=True,
            )

            if auc > best_dev_auc:
                best_dev_auc = auc
                best_step = step

                best_state = {
                    k:
                        v.detach().clone()

                    for k, v
                    in reranker
                    .state_dict()
                    .items()
                }

    reranker.load_state_dict(
        best_state
    )

    reranker.eval()

    def rerank(x):
        with torch.no_grad():
            return reranker(
                torch.tensor(
                    x,
                    dtype=torch.float32,
                )
            ).numpy()

    train_r = rerank(
        train_s
    )

    dev_r = rerank(
        dev_s
    )

    test_r = rerank(
        test_s
    )

    audit_r = rerank(
        audit_s
    )

    strict_r = audit_r[
        strict_idx
    ]

    structured = {
        "dev":
            ranking_metrics(
                dev_r,
                dev_y,
                roles,
            ),

        "test":
            ranking_metrics(
                test_r,
                test_y,
                roles,
            ),

        "audited_100":
            ranking_metrics(
                audit_r,
                audit_y,
                roles,
            ),

        "strict_82":
            ranking_metrics(
                strict_r,
                strict_y,
                roles,
            ),
    }

    # ========================================================
    # Refit role-wise calibration on DEV after reranking.
    # ========================================================

    print(
        "\n===== FIT POST-RERANK ROLE CALIBRATION =====",
        flush=True,
    )

    scale, bias = (
        fit_role_calibration(
            dev_r,
            dev_y,
        )
    )

    dev_c = apply_affine(
        dev_r,
        scale,
        bias,
    )

    test_c = apply_affine(
        test_r,
        scale,
        bias,
    )

    audit_c = apply_affine(
        audit_r,
        scale,
        bias,
    )

    strict_c = audit_c[
        strict_idx
    ]

    calibrated = {
        "dev":
            ranking_metrics(
                dev_c,
                dev_y,
                roles,
            ),

        "test":
            ranking_metrics(
                test_c,
                test_y,
                roles,
            ),

        "audited_100":
            ranking_metrics(
                audit_c,
                audit_y,
                roles,
            ),

        "strict_82":
            ranking_metrics(
                strict_c,
                strict_y,
                roles,
            ),
    }

    # ========================================================
    # Thresholds:
    # fit on DEV only, evaluate on test/audit/strict.
    # ========================================================

    raw_thr = fit_thresholds(
        dev_s,
        dev_y,
        TARGET_RECALL,
    )

    struct_thr = fit_thresholds(
        dev_r,
        dev_y,
        TARGET_RECALL,
    )

    cal_thr = fit_thresholds(
        dev_c,
        dev_y,
        TARGET_RECALL,
    )

    threshold_results = {
        "raw": {
            "test":
                threshold_metrics(
                    test_s,
                    test_y,
                    raw_thr,
                ),

            "strict_82":
                threshold_metrics(
                    strict_s,
                    strict_y,
                    raw_thr,
                ),
        },

        "structured": {
            "test":
                threshold_metrics(
                    test_r,
                    test_y,
                    struct_thr,
                ),

            "strict_82":
                threshold_metrics(
                    strict_r,
                    strict_y,
                    struct_thr,
                ),
        },

        "structured_calibrated": {
            "test":
                threshold_metrics(
                    test_c,
                    test_y,
                    cal_thr,
                ),

            "strict_82":
                threshold_metrics(
                    strict_c,
                    strict_y,
                    cal_thr,
                ),
        },
    }

    result = {
        "experiment":
            "v7_structured_reranker_v1",

        "query_used":
            False,

        "sensor_frozen":
            True,

        "sensor":
            str(
                ADAPTER
            ),

        "reranker_parameters":
            int(
                sum(
                    p.numel()
                    for p
                    in reranker.parameters()
                )
            ),

        "train_rows":
            len(train),

        "best_step":
            best_step,

        "best_dev_within_auc":
            best_dev_auc,

        "raw":
            raw,

        "structured":
            structured,

        "structured_calibrated":
            calibrated,

        "threshold_target_dev_recall":
            TARGET_RECALL,

        "threshold_metrics":
            threshold_results,

        "role_calibration": {
            role: {
                "scale":
                    float(
                        scale[i]
                    ),

                "bias":
                    float(
                        bias[i]
                    ),
            }
            for i, role in enumerate(
                roles
            )
        },
    }

    dump(
        OUT / "summary.json",
        result,
    )

    torch.save(
        {
            "state_dict":
                reranker.state_dict(),

            "mean":
                mean,

            "std":
                std,

            "roles":
                roles,
        },
        OUT / "reranker.pt",
    )

    print(
        "\n===== FINAL COMPARISON =====",
        flush=True,
    )

    for subset in (
        "dev",
        "test",
        "audited_100",
        "strict_82",
    ):
        print(
            f"\n{subset}",
            flush=True,
        )

        print(
            "  RAW     within="
            f"{raw[subset]['within_output_auc']:.4f} "
            "micro="
            f"{raw[subset]['micro_auc']:.4f}",
            flush=True,
        )

        print(
            "  STRUCT  within="
            f"{structured[subset]['within_output_auc']:.4f} "
            "micro="
            f"{structured[subset]['micro_auc']:.4f}",
            flush=True,
        )

        print(
            "  +CAL    within="
            f"{calibrated[subset]['within_output_auc']:.4f} "
            "micro="
            f"{calibrated[subset]['micro_auc']:.4f}",
            flush=True,
        )

    print(
        "\n===== FP / THRESHOLD COMPARISON =====",
        flush=True,
    )

    for variant, vv in (
        threshold_results.items()
    ):
        print(
            f"\n{variant}",
            flush=True,
        )

        for subset, m in vv.items():
            print(
                f"  {subset}: "
                f"precision={m['precision']:.4f} "
                f"recall={m['recall']:.4f} "
                f"FPR={m['fpr']:.4f} "
                f"FP/output={m['avg_fp_per_output']:.4f} "
                f"exact={m['exact_activation_set_threshold']:.4f}",
                flush=True,
            )

    print(
        "\nbest structured step:",
        best_step,
        flush=True,
    )

    print(
        "saved:",
        OUT / "summary.json",
        OUT / "reranker.pt",
        flush=True,
    )


if __name__ == "__main__":
    main()
