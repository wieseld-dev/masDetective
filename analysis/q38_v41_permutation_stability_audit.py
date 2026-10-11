from __future__ import annotations

import ast
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)

from peft import (
    PeftConfig,
    PeftModel,
)


TRAINER = Path(
    "analysis/train_q38_selector_v41.py"
)

ADAPTER = Path(
    "artifacts/q38_selector_v41/best_adapter"
)

DATA = Path(
    "artifacts/q38_selector_v41_data"
)

OUT = Path(
    "artifacts/q38_selector_v41_permutation_audit"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


# ------------------------------------------------------------------
# Load EXACT scoring functions out of the original trainer source.
# ------------------------------------------------------------------

src = TRAINER.read_text()
tree = ast.parse(src)


wanted = {
    "stable_seed",
    "apply_chat",
    "role_block",
    "render_user",
    "make_prompt",
    "restricted_scores",
    "score_group",
}


func_src=[]

for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in wanted:
        func_src.append(
            ast.get_source_segment(
                src,
                node,
            )
        )


# Exact SYSTEM constant from trainer.
SYSTEM = None

for node in tree.body:
    if not isinstance(node, ast.Assign):
        continue

    for target in node.targets:
        if isinstance(target, ast.Name) and target.id=="SYSTEM":
            SYSTEM=eval(
                compile(
                    ast.Expression(node.value),
                    str(TRAINER),
                    "eval",
                ),
                {"__builtins__":{}},
            )


if SYSTEM is None:
    raise RuntimeError(
        "Could not extract trainer SYSTEM"
    )


ENV={
    "np":np,
    "torch":torch,
    "random":random,
    "hashlib":hashlib,
    "SYSTEM":SYSTEM,
    "LETTERS":list("ABCDEFGH"),
    "MAX_LEN":1024,
    "EVAL_PERMS":3,
}


exec(
    "\n\n".join(func_src),
    ENV,
)


print(
    "EXTRACTED FUNCTIONS =",
    sorted(wanted),
)

print(
    "SYSTEM SHA256 =",
    hashlib.sha256(
        SYSTEM.encode()
    ).hexdigest(),
)


# ------------------------------------------------------------------
# Model = exact saved best adapter.
# ------------------------------------------------------------------

cfg = PeftConfig.from_pretrained(
    str(ADAPTER)
)

base_name = cfg.base_model_name_or_path

print(
    "BASE =",
    base_name,
)

processor = AutoProcessor.from_pretrained(
    base_name
)

tokenizer = getattr(
    processor,
    "tokenizer",
    processor,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token


label_ids={}

for letter in ENV["LETTERS"]:
    ids=tokenizer.encode(
        letter,
        add_special_tokens=False,
    )

    if len(ids)!=1:
        raise RuntimeError(
            f"{letter}: tokenization={ids}"
        )

    label_ids[letter]=int(ids[0])


print(
    "LABEL IDS =",
    label_ids,
)


t0=time.time()

base=AutoModelForMultimodalLM.from_pretrained(
    base_name,
    dtype=torch.bfloat16,
    device_map={"":0},
    low_cpu_mem_usage=True,
)

model=PeftModel.from_pretrained(
    base,
    str(ADAPTER),
    is_trainable=False,
)

model.eval()

print(
    "LOAD_SECONDS =",
    round(time.time()-t0,1),
)


ENV["tokenizer"]=tokenizer
ENV["label_ids"]=label_ids
ENV["model"]=model

score_group=ENV["score_group"]


def read(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def metrics(groups, score_rows):

    selected=[]
    random_gain=[]
    oracle=[]
    top1=[]

    records=[]

    for g,scores in zip(
        groups,
        score_rows,
    ):

        gains=np.asarray(
            g["auc_gains"],
            dtype=float,
        )

        scores=np.asarray(
            scores,
            dtype=float,
        )

        pick=int(
            np.argmax(scores)
        )

        best=float(
            np.max(gains)
        )

        selected.append(
            float(gains[pick])
        )

        random_gain.append(
            float(np.mean(gains))
        )

        oracle.append(
            best
        )

        bestset={
            i
            for i,z in enumerate(gains)
            if abs(float(z)-best)<1e-12
        }

        top1.append(
            float(pick in bestset)
        )

        records.append({
            "system_id":
                g["system_id"],
            "state_id":
                g["state_id"],
            "round":
                g["round"],
            "selected_index":
                pick,
            "selected_gain":
                float(gains[pick]),
            "random_gain":
                float(np.mean(gains)),
            "oracle_gain":
                best,
            "scores":
                scores.tolist(),
            "auc_gains":
                gains.tolist(),
        })

    ms=float(np.mean(selected))
    mr=float(np.mean(random_gain))
    mo=float(np.mean(oracle))

    return {
        "states":
            len(groups),
        "mean_selected_gain":
            ms,
        "mean_random_gain":
            mr,
        "mean_oracle_gain":
            mo,
        "mean_regret":
            mo-ms,
        "oracle_top1_hit":
            float(np.mean(top1)),
        "oracle_signal_captured":
            (
                (ms-mr)/(mo-mr)
                if mo>mr+1e-12
                else None
            ),
    },records


def evaluate_tags(
    split,
    tags,
):

    groups=read(
        DATA/f"{split}_groups.jsonl"
    )

    by_tag={}
    records_by_tag={}

    score_bank=[]

    for tag in tags:

        print()
        print(
            "="*100
        )
        print(
            split,
            tag,
        )
        print(
            "="*100
        )

        scores=[]

        for i,g in enumerate(
            groups,
            1,
        ):

            with torch.inference_mode():

                s=score_group(
                    g,
                    tag,
                )

            scores.append(
                np.asarray(
                    s,
                    dtype=float,
                )
            )

            if (
                i%50==0
                or i==len(groups)
            ):
                print(
                    f"{split}/{tag}: "
                    f"{i}/{len(groups)}",
                    flush=True,
                )

        m,recs=metrics(
            groups,
            scores,
        )

        by_tag[tag]=m
        records_by_tag[tag]=recs

        score_bank.append(
            scores
        )

        print(
            json.dumps(
                m,
                indent=2,
            )
        )

    # Aggregate all tags:
    # 3 tags x 3 permutations = effective 9-permutation ensemble.
    aggregate=[]

    for gi in range(
        len(groups)
    ):

        aggregate.append(
            np.mean(
                [
                    score_bank[t][gi]
                    for t
                    in range(
                        len(score_bank)
                    )
                ],
                axis=0,
            )
        )

    am,arecs=metrics(
        groups,
        aggregate,
    )

    by_tag[
        "AGGREGATE_9_PERM"
    ]=am

    records_by_tag[
        "AGGREGATE_9_PERM"
    ]=arecs

    print()
    print(
        split,
        "AGGREGATE_9_PERM",
        json.dumps(
            am,
            indent=2,
        ),
    )

    return by_tag,records_by_tag


summary={}
all_records={}


val_m,val_r=evaluate_tags(
    "val",
    [
        "VAL_E1",
        "PERM_A",
        "PERM_B",
    ],
)

summary["val"]=val_m
all_records["val"]=val_r


challenge_m,challenge_r=evaluate_tags(
    "challenge",
    [
        "CHALLENGE",
        "PERM_A",
        "PERM_B",
    ],
)

summary["challenge"]=challenge_m
all_records["challenge"]=challenge_r


(OUT/"summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

with (
    OUT/"records.jsonl"
).open(
    "w"
) as f:

    for split,d in all_records.items():
        for tag,rows in d.items():
            for x in rows:
                y={
                    "split":split,
                    "tag":tag,
                    **x,
                }

                f.write(
                    json.dumps(y)
                    +"\n"
                )


print()
print(
    "="*100
)
print(
    "FINAL SUMMARY"
)
print(
    "="*100
)

print(
    json.dumps(
        summary,
        indent=2,
    )
)
