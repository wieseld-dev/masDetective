#!/usr/bin/env python3

from __future__ import annotations

import json
import random
from pathlib import Path

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


OLD_ADAPTER = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/"
    "best_adapter"
)

NEW_ADAPTER = Path(
    "artifacts/v7_benign_output_sensor_pilot_v1/"
    "best_adapter"
)

TEST = Path(
    "artifacts/v7_benign_final_pool_v2/"
    "test.jsonl"
)

OUT = Path(
    "artifacts/v7_benign_output_sensor_baseline_compare_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SEED = 20260926 + 202
LIMIT = 512
MAX_LENGTH = 768
EVAL_BATCH_SIZE = 8


def load_adapter(path: Path):
    cfg = PeftConfig.from_pretrained(
        str(path)
    )

    base_name = (
        cfg.base_model_name_or_path
    )

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

    base.config.pad_token_id = (
        tok.pad_token_id
    )

    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(path),
        is_trainable=False,
    )

    model.eval()

    device = torch.device(
        "cuda"
    )

    model.to(
        device
    )

    return (
        tok,
        model,
        device,
        base_name,
    )


def main():
    rows = bos.read_jsonl(
        TEST
    )

    rng = random.Random(
        SEED
    )

    if len(rows) > LIMIT:
        rows = rng.sample(
            rows,
            LIMIT,
        )

    # IMPORTANT:
    # bos.evaluate() also samples if limit is passed.
    # We already fixed the exact shared subset above,
    # so pass limit=None below.
    role_desc = dict(
        tr.ROLE_DESCRIPTIONS
    )

    roles = list(
        role_desc.keys()
    )

    result = {
        "test_source":
            str(TEST),

        "rows":
            len(rows),

        "seed":
            SEED,

        "query_used":
            False,

        "input_variables": [
            "candidate_role",
            "role_description",
            "final_output",
        ],
    }

    print(
        "===== OLD V13.8b =====",
        flush=True,
    )

    tok_old, model_old, device, base_name = (
        load_adapter(
            OLD_ADAPTER
        )
    )

    old_metrics = bos.evaluate(
        model_old,
        tok_old,
        rows,
        roles,
        role_desc,
        device,
        MAX_LENGTH,
        EVAL_BATCH_SIZE,
        None,
        SEED,
    )

    result[
        "old_v13_8b"
    ] = {
        "adapter":
            str(
                OLD_ADAPTER
            ),

        "metrics":
            old_metrics,
    }

    print(
        json.dumps(
            old_metrics,
            indent=2,
        ),
        flush=True,
    )

    # Free GPU before loading second copy.
    del model_old
    del tok_old

    torch.cuda.empty_cache()

    print(
        "\n===== BENIGN OUTPUT SENSOR =====",
        flush=True,
    )

    tok_new, model_new, device, _ = (
        load_adapter(
            NEW_ADAPTER
        )
    )

    new_metrics = bos.evaluate(
        model_new,
        tok_new,
        rows,
        roles,
        role_desc,
        device,
        MAX_LENGTH,
        EVAL_BATCH_SIZE,
        None,
        SEED,
    )

    result[
        "benign_output_sensor"
    ] = {
        "adapter":
            str(
                NEW_ADAPTER
            ),

        "metrics":
            new_metrics,
    }

    print(
        json.dumps(
            new_metrics,
            indent=2,
        ),
        flush=True,
    )

    result["delta"] = {
        "micro_auc":
            new_metrics["micro_auc"]
            - old_metrics["micro_auc"],

        "macro_role_auc":
            new_metrics["macro_role_auc"]
            - old_metrics["macro_role_auc"],

        "within_output_auc":
            new_metrics["within_output_auc"]
            - old_metrics["within_output_auc"],

        "topk_activation_recall":
            new_metrics["topk_activation_recall"]
            - old_metrics["topk_activation_recall"],

        "exact_activation_set":
            new_metrics["exact_activation_set"]
            - old_metrics["exact_activation_set"],

        "mean_selected_minus_absent_logit":
            new_metrics[
                "mean_selected_minus_absent_logit"
            ]
            - old_metrics[
                "mean_selected_minus_absent_logit"
            ],
    }

    result[
        "base_model"
    ] = base_name

    (
        OUT / "comparison.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== DELTA =====",
        flush=True,
    )

    print(
        json.dumps(
            result["delta"],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
