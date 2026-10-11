#!/usr/bin/env python3

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
    "artifacts/role_direct_sealed_v1"
)

ADAPTER = Path(
    "artifacts/role_direct_sensor_final_v1/adapter"
)

RAW = ROOT / "final_sensor_scores.jsonl"
OUT = ROOT / "query_baseline_scores.jsonl"


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


if OUT.exists():
    raise RuntimeError(
        f"Refusing to overwrite {OUT}"
    )

rows = read_jsonl(RAW)

assert len(rows) == 1200


# One baseline per unique role/query.
unique = {}

for x in rows:

    key = (
        str(x["target_role"]),
        str(x["panel"]),
        int(x["probe_rank"]),
        str(x["query"]),
    )

    unique[key] = {
        "target_role":
            str(x["target_role"]),

        "panel":
            str(x["panel"]),

        "probe_rank":
            int(x["probe_rank"]),

        "probe_strategy":
            str(x["probe_strategy"]),

        "query":
            str(x["query"]),
    }


items = list(unique.values())

assert len(items) == 120, len(items)

assert sum(
    x["panel"] == "calibration"
    for x in items
) == 60

assert sum(
    x["panel"] == "sealed_test"
    for x in items
) == 60


# IMPORTANT:
# same target-conditioned sensor,
# same query,
# but NO visible MAS response.
for x in items:

    x["text"] = tr.render_text(
        x["target_role"],
        x["query"],
        "",
    )


cfg = PeftConfig.from_pretrained(
    ADAPTER,
    local_files_only=True,
)

base_name = cfg.base_model_name_or_path

tok = AutoTokenizer.from_pretrained(
    ADAPTER,
    local_files_only=True,
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "right"
tok.truncation_side = "right"


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
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
)

base.config.pad_token_id = (
    tok.pad_token_id
)

base.config.use_cache = False


model = PeftModel.from_pretrained(
    base,
    ADAPTER,
    is_trainable=False,
    local_files_only=True,
)

model.eval()


scores = []

with torch.inference_mode():

    for start in range(
        0,
        len(items),
        8,
    ):

        batch = items[
            start:start + 8
        ]

        enc = tok(
            [
                x["text"]
                for x in batch
            ],
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to("cuda")

        z = (
            model(**enc)
            .logits
            .view(-1)
            .float()
            .cpu()
            .tolist()
        )

        scores.extend(z)

        print(
            f"baseline "
            f"{min(start + 8, len(items))}"
            f"/{len(items)}",
            flush=True,
        )


assert len(scores) == 120


with OUT.open(
    "w",
    encoding="utf-8",
) as f:

    for x, z in zip(
        items,
        scores,
    ):

        y = {
            k: v
            for k, v in x.items()
            if k != "text"
        }

        y["query_baseline_logit"] = float(z)

        f.write(
            json.dumps(
                y,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("QUERY_BASELINE_SCORE_PASS")
print("unique queries =", len(items))
print("output =", OUT)


del model
del base
gc.collect()
torch.cuda.empty_cache()
