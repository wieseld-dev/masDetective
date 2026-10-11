#!/usr/bin/env python3

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from peft import (
    PeftConfig,
    PeftModel,
)

import analysis.role_direct_approach.train_role_direct_sensor_v1 as tr


ROOT = Path(
    "artifacts/role_direct_sensor_v1"
)


def write_jsonl(path, rows):
    with Path(path).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--fold",
        type=int,
        required=True,
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    args = ap.parse_args()

    fold = int(args.fold)

    assert fold in {
        1, 2, 3, 4, 5
    }

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required"
        )

    fold_root = (
        ROOT
        / f"fold_{fold}"
    )

    adapter = (
        fold_root
        / "adapter"
    )

    assert adapter.exists(), adapter

    examples = tr.build_examples()

    train = [
        x
        for x in examples
        if int(
            x["pair_index"]
        ) != fold
    ]

    test = [
        x
        for x in examples
        if int(
            x["pair_index"]
        ) == fold
    ]

    assert len(train) == 240
    assert len(test) == 60

    print(
        "=" * 90,
        flush=True,
    )

    print(
        "ROLE DIRECT SENSOR — CALIBRATION SCORING",
        flush=True,
    )

    print(
        f"FOLD={fold}",
        flush=True,
    )

    print(
        "TRAIN matched queries =",
        len(train),
        flush=True,
    )

    print(
        "TRAIN output rows =",
        2 * len(train),
        flush=True,
    )

    print(
        "HELDOUT pair_index =",
        fold,
        flush=True,
    )

    cfg = PeftConfig.from_pretrained(
        adapter,
        local_files_only=True,
    )

    base_name = (
        cfg.base_model_name_or_path
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            adapter,
            local_files_only=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    base = (
        AutoModelForSequenceClassification
        .from_pretrained(
            base_name,
            num_labels=1,
            quantization_config=qconf,
            device_map={"": 0},
            dtype=torch.bfloat16,
            local_files_only=True,
        )
    )

    base.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        adapter,
        is_trainable=False,
        local_files_only=True,
    )

    model.eval()

    train_rows = tr.score_examples(
        model,
        tokenizer,
        train,
        batch_size=8,
        max_length=args.max_length,
    )

    assert len(train_rows) == 480

    out = (
        fold_root
        / "calibration_train_predictions.jsonl"
    )

    write_jsonl(
        out,
        train_rows,
    )

    print(
        "CALIBRATION ROWS =",
        len(train_rows),
        flush=True,
    )

    print(
        "OUTPUT =",
        out,
        flush=True,
    )

    print(
        "CALIBRATION_SCORING_PASS",
        flush=True,
    )

    del model
    del base

    gc.collect()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
