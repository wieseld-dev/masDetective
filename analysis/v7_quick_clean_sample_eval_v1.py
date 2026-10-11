#!/usr/bin/env python3

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

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


AUDIT = Path(
    "artifacts/v7_actual_query_manual_audit_v1/"
    "sample_300.jsonl"
)

TEST = Path(
    "artifacts/v7_benign_final_pool_v2/"
    "test.jsonl"
)

OLD = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/"
    "best_adapter"
)

NEW = Path(
    "artifacts/v7_benign_output_sensor_pilot_v1/"
    "best_adapter"
)

OUT = Path(
    "artifacts/v7_quick_clean_sample_eval_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 768
BATCH_SIZE = 8


# ============================================================
# IDs supplied by the independent review.
# ============================================================

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
    out = []

    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))

    return out


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


def metrics_for_indices(
    rows,
    matrix,
    indices,
    roles,
):
    labels_all = []
    scores_all = []

    role_labels = {
        r: []
        for r in roles
    }

    role_scores = {
        r: []
        for r in roles
    }

    within = []
    topk = []
    exact = []
    margins = []

    for i in indices:
        row = rows[i]

        selected = set(
            bos.get_selected(row)
        )

        y = np.asarray([
            1 if r in selected else 0
            for r in roles
        ])

        s = matrix[i]

        labels_all.extend(
            y.tolist()
        )

        scores_all.extend(
            s.tolist()
        )

        for j, r in enumerate(roles):
            role_labels[r].append(
                int(y[j])
            )

            role_scores[r].append(
                float(s[j])
            )

        auc = bos.binary_auc(
            y,
            s,
        )

        if math.isfinite(auc):
            within.append(auc)

        k = len(selected)

        if k > 0:
            pred_idx = np.argsort(
                -s
            )[:k]

            predicted = {
                roles[j]
                for j in pred_idx
            }

            topk.append(
                len(predicted & selected)
                / k
            )

            exact.append(
                float(
                    predicted == selected
                )
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

    micro_auc = bos.binary_auc(
        labels_all,
        scores_all,
    )

    role_auc = {}

    for r in roles:
        role_auc[r] = bos.binary_auc(
            role_labels[r],
            role_scores[r],
        )

    valid_role_auc = [
        v
        for v in role_auc.values()
        if math.isfinite(v)
    ]

    return {
        "rows":
            len(indices),

        "micro_auc":
            float(micro_auc),

        "macro_role_auc":
            float(
                np.mean(valid_role_auc)
            ),

        "within_output_auc":
            float(
                np.mean(within)
            ),

        "topk_activation_recall":
            float(
                np.mean(topk)
            ),

        "exact_activation_set":
            float(
                np.mean(exact)
            ),

        "mean_selected_minus_absent_logit":
            float(
                np.mean(margins)
            ),
    }


def main():
    audit = read_jsonl(
        AUDIT
    )

    test_rows_all = read_jsonl(
        TEST
    )

    # --------------------------------------------------------
    # Actual 100 test queries from the 300-query audit.
    # --------------------------------------------------------

    audit_test = [
        x
        for x in audit
        if x["split"] == "test"
    ]

    if len(audit_test) != 100:
        raise RuntimeError(
            f"expected 100 audit-test rows, got {len(audit_test)}"
        )

    # Join audit query/session back to full test execution row.
    lookup = {}

    for row in test_rows_all:
        key = (
            str(row.get("session_id")),
            norm_q(row.get("query")),
        )

        lookup.setdefault(
            key,
            []
        ).append(row)

    rows = []
    audit_ids = []

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
                f"join failure audit_id={a['audit_id']} "
                f"matches={len(matches)}"
            )

        rows.append(
            matches[0]
        )

        audit_ids.append(
            int(a["audit_id"])
        )

    print(
        "===== TEST AUDIT IDS ====="
    )

    print(
        audit_ids
    )

    cap_test = sorted(
        set(audit_ids)
        & CAPABILITY_IDS
    )

    dir_test = sorted(
        set(audit_ids)
        & DIRECTION_IDS
    )

    print(
        "\nCAPABILITY IDs actually in test:",
        cap_test,
    )

    print(
        "DIRECTION IDs actually in test:",
        dir_test,
    )

    all_idx = list(
        range(len(rows))
    )

    no_cap_idx = [
        i
        for i, aid in enumerate(audit_ids)
        if aid not in CAPABILITY_IDS
    ]

    strict_idx = [
        i
        for i, aid in enumerate(audit_ids)
        if (
            aid not in CAPABILITY_IDS
            and aid not in DIRECTION_IDS
        )
    ]

    subsets = {
        "ALL_AUDITED_TEST_100":
            all_idx,

        "NO_CAPABILITY":
            no_cap_idx,

        "STRICT_NO_CAP_OR_DIRECTION":
            strict_idx,
    }

    role_desc = dict(
        tr.ROLE_DESCRIPTIONS
    )

    roles = list(
        role_desc.keys()
    )

    result = {
        "audit_test_rows":
            len(rows),

        "capability_ids_in_test":
            cap_test,

        "direction_ids_in_test":
            dir_test,

        "subset_sizes": {
            k: len(v)
            for k, v in subsets.items()
        },

        "query_used":
            False,

        "old_adapter":
            str(OLD),

        "new_adapter":
            str(NEW),

        "metrics": {},
    }

    # ========================================================
    # OLD
    # ========================================================

    print(
        "\n===== LOAD OLD =====",
        flush=True,
    )

    tok_old, model_old = load_adapter(
        OLD
    )

    old_matrix = score_rows(
        model_old,
        tok_old,
        rows,
        roles,
        role_desc,
    )

    old_metrics = {}

    for name, idx in subsets.items():
        old_metrics[name] = (
            metrics_for_indices(
                rows,
                old_matrix,
                idx,
                roles,
            )
        )

    del model_old
    del tok_old

    torch.cuda.empty_cache()

    # ========================================================
    # NEW
    # ========================================================

    print(
        "\n===== LOAD NEW =====",
        flush=True,
    )

    tok_new, model_new = load_adapter(
        NEW
    )

    new_matrix = score_rows(
        model_new,
        tok_new,
        rows,
        roles,
        role_desc,
    )

    new_metrics = {}

    for name, idx in subsets.items():
        new_metrics[name] = (
            metrics_for_indices(
                rows,
                new_matrix,
                idx,
                roles,
            )
        )

    # ========================================================
    # Compare.
    # ========================================================

    for name in subsets:
        delta = {
            metric:
                new_metrics[name][metric]
                - old_metrics[name][metric]

            for metric in (
                "micro_auc",
                "macro_role_auc",
                "within_output_auc",
                "topk_activation_recall",
                "exact_activation_set",
                "mean_selected_minus_absent_logit",
            )
        }

        result["metrics"][name] = {
            "old":
                old_metrics[name],

            "new":
                new_metrics[name],

            "delta_new_minus_old":
                delta,
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
        "\n===== QUICK CLEAN EVAL SUMMARY ====="
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n===== COMPACT ====="
    )

    for name in subsets:
        o = old_metrics[name]
        n = new_metrics[name]

        print(
            f"\n{name} "
            f"(N={len(subsets[name])})"
        )

        print(
            "  OLD within_auc="
            f"{o['within_output_auc']:.4f} "
            "micro="
            f"{o['micro_auc']:.4f} "
            "topk="
            f"{o['topk_activation_recall']:.4f} "
            "exact="
            f"{o['exact_activation_set']:.4f}"
        )

        print(
            "  NEW within_auc="
            f"{n['within_output_auc']:.4f} "
            "micro="
            f"{n['micro_auc']:.4f} "
            "topk="
            f"{n['topk_activation_recall']:.4f} "
            "exact="
            f"{n['exact_activation_set']:.4f}"
        )

        print(
            "  DELTA within_auc="
            f"{n['within_output_auc'] - o['within_output_auc']:+.4f}"
        )


if __name__ == "__main__":
    main()
