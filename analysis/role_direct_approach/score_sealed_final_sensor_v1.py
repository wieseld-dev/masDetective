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

OUT = (
    ROOT
    / "final_sensor_scores.jsonl"
)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


candidates = read_jsonl(
    ROOT / "candidates_dev_round_01.jsonl"
)

responses = read_jsonl(
    ROOT / "responses_dev_round_01.jsonl"
)

assert len(candidates) == 1200
assert len(responses) == 1200

if OUT.exists():
    raise RuntimeError(
        f"Refusing to overwrite {OUT}"
    )


rmap = {
    (
        str(x["session_id"]),
        str(x["query"]),
    ): x
    for x in responses
}


items = []

for c in candidates:

    key = (
        str(c["session_id"]),
        str(c["query"]),
    )

    r = rmap[key]

    condition = str(
        c["condition"]
    )

    assert condition in {
        "TARGET_PRESENT",
        "TARGET_ABSENT",
    }

    role = str(
        c["target_role"]
    )

    answer = str(
        r["final_output"]
    )

    items.append({
        "target_role":
            role,

        "panel":
            str(c["panel"]),

        "pair_id":
            str(c["pair_id"]),

        "pair_index":
            int(c["pair_index"]),

        "condition":
            condition,

        "label":
            int(
                condition
                == "TARGET_PRESENT"
            ),

        "probe_rank":
            int(
                c["direct_probe_rank"]
            ),

        "probe_strategy":
            str(
                c["probe_strategy"]
            ),

        "query":
            str(c["query"]),

        "answer":
            answer,

        "text":
            tr.render_text(
                role,
                str(c["query"]),
                answer,
            ),
    })


assert sum(
    x["panel"] == "calibration"
    for x in items
) == 600

assert sum(
    x["panel"] == "sealed_test"
    for x in items
) == 600


cfg = PeftConfig.from_pretrained(
    ADAPTER,
    local_files_only=True,
)

base_name = (
    cfg.base_model_name_or_path
)

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
        dtype=torch.bfloat16,
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

        if (
            start % 160 == 0
            or start + 8 >= len(items)
        ):
            print(
                f"scored "
                f"{min(start + 8, len(items))}"
                f"/{len(items)}",
                flush=True,
            )


assert len(scores) == 1200


with OUT.open("w") as f:

    for x, z in zip(
        items,
        scores,
    ):

        y = {
            k: v
            for k, v in x.items()
            if k != "text"
        }

        y["logit"] = float(z)

        y["prob"] = float(
            torch.sigmoid(
                torch.tensor(z)
            ).item()
        )

        f.write(
            json.dumps(
                y,
                ensure_ascii=False,
            )
            + "\n"
        )


print()
print("FINAL_SENSOR_SCORE_PASS")
print("rows =", len(scores))
print("output =", OUT)

del model
del base
gc.collect()
torch.cuda.empty_cache()
