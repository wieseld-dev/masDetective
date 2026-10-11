#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
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
    "artifacts/v7_rolewise_calibration_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926

DEV_LIMIT = 512
TEST_LIMIT = 512

MAX_LENGTH = 768
BATCH_SIZE = 8


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

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    return rows


def norm_q(x):
    return " ".join(
        str(x or "").strip().split()
    ).lower()


def load_adapter(path):
    cfg = PeftConfig.from_pretrained(
        str(path)
    )

    base_name = cfg.base_model_name_or_path

    tok = AutoTokenizer.from_pretrained(
        str(path)
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

    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(path),
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
        answer = bos.get_answer(row)

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
        BATCH_SIZE,
    )

    return scores.reshape(
        len(rows),
        len(roles),
    )


def labels_matrix(
    rows,
    roles,
):
    mat = np.zeros(
        (len(rows), len(roles)),
        dtype=np.int64,
    )

    for i, row in enumerate(rows):
        selected = set(
            bos.get_selected(row)
        )

        for j, role in enumerate(roles):
            if role in selected:
                mat[i, j] = 1

    return mat


def binary_auc(
    labels,
    scores,
):
    return bos.binary_auc(
        np.asarray(labels),
        np.asarray(scores),
    )


def evaluate_matrix(
    score_matrix,
    label_matrix,
    roles,
):
    flat_y = label_matrix.reshape(-1)
    flat_s = score_matrix.reshape(-1)

    micro_auc = binary_auc(
        flat_y,
        flat_s,
    )

    role_aucs = {}

    for j, role in enumerate(roles):
        role_aucs[role] = binary_auc(
            label_matrix[:, j],
            score_matrix[:, j],
        )

    valid = [
        x
        for x in role_aucs.values()
        if math.isfinite(x)
    ]

    within = []
    topk = []
    exact = []
    margins = []

    for i in range(
        len(score_matrix)
    ):
        y = label_matrix[i]
        s = score_matrix[i]

        auc = binary_auc(
            y,
            s,
        )

        if math.isfinite(auc):
            within.append(auc)

        k = int(
            y.sum()
        )

        if k > 0:
            idx = np.argsort(-s)[:k]

            pred = set(
                int(x)
                for x in idx
            )

            truth = set(
                int(x)
                for x in np.where(y == 1)[0]
            )

            topk.append(
                len(pred & truth) / k
            )

            exact.append(
                float(pred == truth)
            )

        pos = s[y == 1]
        neg = s[y == 0]

        if len(pos) and len(neg):
            margins.append(
                float(
                    pos.mean()
                    - neg.mean()
                )
            )

    return {
        "rows":
            len(score_matrix),

        "micro_auc":
            float(micro_auc),

        "macro_role_auc":
            float(np.mean(valid)),

        "within_output_auc":
            float(np.mean(within)),

        "topk_activation_recall":
            float(np.mean(topk)),

        "exact_activation_set":
            float(np.mean(exact)),

        "mean_selected_minus_absent_logit":
            float(np.mean(margins)),

        "role_auc":
            role_aucs,
    }


def build_pair_indices(
    labels,
):
    pos_row = []
    pos_role = []
    neg_role = []

    for i in range(
        labels.shape[0]
    ):
        pos = np.where(
            labels[i] == 1
        )[0]

        neg = np.where(
            labels[i] == 0
        )[0]

        for p in pos:
            for n in neg:
                pos_row.append(i)
                pos_role.append(p)
                neg_role.append(n)

    return (
        torch.tensor(
            pos_row,
            dtype=torch.long,
        ),
        torch.tensor(
            pos_role,
            dtype=torch.long,
        ),
        torch.tensor(
            neg_role,
            dtype=torch.long,
        ),
    )


def fit_calibration(
    scores_np,
    labels_np,
    steps=2500,
    lr=0.03,
):
    device = torch.device("cpu")

    scores = torch.tensor(
        scores_np,
        dtype=torch.float32,
        device=device,
    )

    labels = torch.tensor(
        labels_np,
        dtype=torch.float32,
        device=device,
    )

    n_roles = scores.shape[1]

    # a_r = exp(log_a_r), guaranteeing positive scale.
    log_a = torch.zeros(
        n_roles,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    bias = torch.zeros(
        n_roles,
        dtype=torch.float32,
        requires_grad=True,
        device=device,
    )

    (
        pair_row,
        pair_pos,
        pair_neg,
    ) = build_pair_indices(
        labels_np
    )

    pair_row = pair_row.to(device)
    pair_pos = pair_pos.to(device)
    pair_neg = pair_neg.to(device)

    opt = torch.optim.Adam(
        [log_a, bias],
        lr=lr,
    )

    best_loss = float("inf")
    best = None

    for step in range(
        1,
        steps + 1,
    ):
        opt.zero_grad()

        scale = torch.exp(
            log_a
        )

        calibrated = (
            scores
            * scale.unsqueeze(0)
            + bias.unsqueeze(0)
        )

        pos_s = calibrated[
            pair_row,
            pair_pos,
        ]

        neg_s = calibrated[
            pair_row,
            pair_neg,
        ]

        # Pairwise logistic surrogate for within-output AUC.
        pair_loss = F.softplus(
            -(pos_s - neg_s)
        ).mean()

        # Small BCE term prevents pathological score geometry.
        bce_loss = (
            F.binary_cross_entropy_with_logits(
                calibrated.reshape(-1),
                labels.reshape(-1),
            )
        )

        # Keep calibration conservative.
        reg = (
            0.0025
            * (log_a ** 2).mean()
            +
            0.0005
            * (bias ** 2).mean()
        )

        loss = (
            pair_loss
            + 0.05 * bce_loss
            + reg
        )

        loss.backward()
        opt.step()

        x = float(
            loss.detach()
        )

        if x < best_loss:
            best_loss = x

            best = {
                "log_a":
                    log_a.detach()
                    .clone(),

                "bias":
                    bias.detach()
                    .clone(),
            }

        if (
            step % 250 == 0
            or step == 1
        ):
            print(
                f"[cal] step={step:4d} "
                f"loss={x:.6f} "
                f"pair={float(pair_loss):.6f} "
                f"bce={float(bce_loss):.6f}",
                flush=True,
            )

    scale = torch.exp(
        best["log_a"]
    ).cpu().numpy()

    bias_np = (
        best["bias"]
        .cpu()
        .numpy()
    )

    return (
        scale,
        bias_np,
        best_loss,
    )


def apply_calibration(
    matrix,
    scale,
    bias,
):
    return (
        matrix
        * scale.reshape(1, -1)
        + bias.reshape(1, -1)
    )


def sample_fixed(
    rows,
    limit,
    seed,
):
    rows = list(rows)

    if len(rows) <= limit:
        return rows

    rng = random.Random(seed)

    return rng.sample(
        rows,
        limit,
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
            str(row.get("session_id")),
            norm_q(row.get("query")),
        )

        lookup.setdefault(
            key,
            []
        ).append(row)

    rows = []
    ids = []

    for a in audit_test:
        key = (
            str(a.get("session_id")),
            norm_q(a.get("query")),
        )

        matches = lookup.get(
            key,
            []
        )

        if len(matches) != 1:
            raise RuntimeError(
                f"join failure "
                f"id={a['audit_id']} "
                f"matches={len(matches)}"
            )

        rows.append(
            matches[0]
        )

        ids.append(
            int(a["audit_id"])
        )

    strict_idx = [
        i
        for i, aid in enumerate(ids)
        if (
            aid not in CAPABILITY_IDS
            and aid not in DIRECTION_IDS
        )
    ]

    strict_rows = [
        rows[i]
        for i in strict_idx
    ]

    return (
        rows,
        ids,
        strict_rows,
    )


def main():
    role_desc = dict(
        tr.ROLE_DESCRIPTIONS
    )

    roles = list(
        role_desc.keys()
    )

    dev_all = read_jsonl(
        DATA_ROOT / "dev.jsonl"
    )

    test_all = read_jsonl(
        DATA_ROOT / "test.jsonl"
    )

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
        audited_100,
        audited_ids,
        strict_82,
    ) = build_audited_test(
        test_all
    )

    print(
        "===== DATA =====",
        flush=True,
    )

    print(
        "dev:",
        len(dev),
        "test:",
        len(test),
        "audited:",
        len(audited_100),
        "strict:",
        len(strict_82),
        flush=True,
    )

    print(
        "\n===== LOAD SENSOR =====",
        flush=True,
    )

    tok, model = load_adapter(
        ADAPTER
    )

    print(
        "\n===== SCORE DEV =====",
        flush=True,
    )

    dev_scores = score_rows(
        model,
        tok,
        dev,
        roles,
        role_desc,
    )

    dev_labels = labels_matrix(
        dev,
        roles,
    )

    print(
        "\n===== SCORE TEST =====",
        flush=True,
    )

    test_scores = score_rows(
        model,
        tok,
        test,
        roles,
        role_desc,
    )

    test_labels = labels_matrix(
        test,
        roles,
    )

    print(
        "\n===== SCORE AUDITED TEST =====",
        flush=True,
    )

    audit_scores = score_rows(
        model,
        tok,
        audited_100,
        roles,
        role_desc,
    )

    audit_labels = labels_matrix(
        audited_100,
        roles,
    )

    # Extract strict subset from the 100 scored rows.
    strict_keep = [
        i
        for i, aid in enumerate(
            audited_ids
        )
        if (
            aid not in CAPABILITY_IDS
            and aid not in DIRECTION_IDS
        )
    ]

    strict_scores = audit_scores[
        strict_keep
    ]

    strict_labels = audit_labels[
        strict_keep
    ]

    # --------------------------------------------------------
    # RAW metrics
    # --------------------------------------------------------

    raw = {
        "dev":
            evaluate_matrix(
                dev_scores,
                dev_labels,
                roles,
            ),

        "test":
            evaluate_matrix(
                test_scores,
                test_labels,
                roles,
            ),

        "audited_100":
            evaluate_matrix(
                audit_scores,
                audit_labels,
                roles,
            ),

        "strict_82":
            evaluate_matrix(
                strict_scores,
                strict_labels,
                roles,
            ),
    }

    print(
        "\n===== RAW =====",
        flush=True,
    )

    for k, m in raw.items():
        print(
            k,
            "within=",
            round(
                m["within_output_auc"],
                6,
            ),
            "micro=",
            round(
                m["micro_auc"],
                6,
            ),
            "topk=",
            round(
                m["topk_activation_recall"],
                6,
            ),
            "exact=",
            round(
                m["exact_activation_set"],
                6,
            ),
            flush=True,
        )

    # --------------------------------------------------------
    # FIT ONLY ON DEV.
    # --------------------------------------------------------

    print(
        "\n===== FIT ROLE-WISE CALIBRATION ON DEV ONLY =====",
        flush=True,
    )

    scale, bias, best_loss = (
        fit_calibration(
            dev_scores,
            dev_labels,
        )
    )

    calibration = {
        role: {
            "scale":
                float(scale[i]),

            "bias":
                float(bias[i]),
        }
        for i, role in enumerate(
            roles
        )
    }

    (
        OUT / "calibration.json"
    ).write_text(
        json.dumps(
            {
                "adapter":
                    str(ADAPTER),

                "fit_split":
                    "dev_512",

                "objective":
                    "pairwise selected-vs-absent logistic",

                "best_loss":
                    best_loss,

                "roles":
                    calibration,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # CALIBRATED metrics
    # --------------------------------------------------------

    calibrated = {
        "dev":
            evaluate_matrix(
                apply_calibration(
                    dev_scores,
                    scale,
                    bias,
                ),
                dev_labels,
                roles,
            ),

        "test":
            evaluate_matrix(
                apply_calibration(
                    test_scores,
                    scale,
                    bias,
                ),
                test_labels,
                roles,
            ),

        "audited_100":
            evaluate_matrix(
                apply_calibration(
                    audit_scores,
                    scale,
                    bias,
                ),
                audit_labels,
                roles,
            ),

        "strict_82":
            evaluate_matrix(
                apply_calibration(
                    strict_scores,
                    scale,
                    bias,
                ),
                strict_labels,
                roles,
            ),
    }

    result = {
        "sensor":
            str(ADAPTER),

        "query_used":
            False,

        "calibration_fit_on":
            "dev only",

        "raw":
            raw,

        "calibrated":
            calibrated,

        "delta": {},
    }

    for split in raw:
        result["delta"][split] = {
            metric:
                calibrated[split][metric]
                - raw[split][metric]

            for metric in (
                "micro_auc",
                "macro_role_auc",
                "within_output_auc",
                "topk_activation_recall",
                "exact_activation_set",
                "mean_selected_minus_absent_logit",
            )
        }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== CALIBRATED =====",
        flush=True,
    )

    for k, m in calibrated.items():
        print(
            k,
            "within=",
            round(
                m["within_output_auc"],
                6,
            ),
            "micro=",
            round(
                m["micro_auc"],
                6,
            ),
            "topk=",
            round(
                m["topk_activation_recall"],
                6,
            ),
            "exact=",
            round(
                m["exact_activation_set"],
                6,
            ),
            flush=True,
        )

    print(
        "\n===== DELTAS =====",
        flush=True,
    )

    for split, d in result[
        "delta"
    ].items():
        print(
            split,
            "within_delta=",
            f"{d['within_output_auc']:+.6f}",
            "micro_delta=",
            f"{d['micro_auc']:+.6f}",
            "topk_delta=",
            f"{d['topk_activation_recall']:+.6f}",
            "exact_delta=",
            f"{d['exact_activation_set']:+.6f}",
            flush=True,
        )

    print(
        "\nSaved:",
        OUT / "summary.json",
        OUT / "calibration.json",
        flush=True,
    )


if __name__ == "__main__":
    main()
