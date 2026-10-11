#!/usr/bin/env python3

import gc
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from sklearn.metrics import roc_auc_score

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
    "artifacts/role_direct_routing_pilot_v1"
)

TRAIN_ROOT = Path(
    "artifacts/role_direct_matched_cf_v1"
)

ADAPTER = Path(
    "artifacts/role_direct_sensor_final_v1/adapter"
)

OUT = ROOT / "final_sensor_simple5_eval.json"


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


# ============================================================
# Locate exact existing Simple5 data.
# ============================================================

response_path = (
    ROOT / "responses_dev_round_01.jsonl"
)

manifest_path = (
    ROOT / "manifest_privileged.json"
)

assert response_path.exists(), response_path
assert manifest_path.exists(), manifest_path
assert ADAPTER.exists(), ADAPTER


candidate_options = [
    ROOT / "candidates_dev_round_01.jsonl",
    ROOT / "candidate_pool_dev_round_01.jsonl",
]

candidate_path = None

for p in candidate_options:

    if not p.exists():
        continue

    rr = read_jsonl(p)

    if (
        len(rr) == 300
        and all(
            "session_id" in x
            and "query" in x
            and "target_role" in x
            for x in rr
        )
    ):
        candidate_path = p
        break


if candidate_path is None:

    # Safe fallback: inspect root JSONLs and choose only an exact
    # 300-row candidate-shaped artifact.
    hits = []

    for p in ROOT.glob("*.jsonl"):

        try:
            rr = read_jsonl(p)
        except Exception:
            continue

        if (
            len(rr) == 300
            and all(
                "session_id" in x
                and "query" in x
                and "target_role" in x
                for x in rr
            )
        ):
            hits.append(p)

    if len(hits) != 1:
        raise RuntimeError(
            "Could not uniquely identify 300-row candidate file. "
            f"hits={hits}"
        )

    candidate_path = hits[0]


candidates = read_jsonl(
    candidate_path
)

responses = read_jsonl(
    response_path
)

manifest = json.loads(
    manifest_path.read_text()
)


assert len(candidates) == 300
assert len(responses) == 300


systems = manifest["systems"]

assert len(systems) == 5, len(systems)


truth = {
    str(x["session_id"]):
        set(map(str, x["roles"]))
    for x in systems
}


assert all(
    len(v) == 3
    for v in truth.values()
)


all_roles = list(
    tr.DESCRIPTIONS.keys()
)

assert len(all_roles) == 20


print("=" * 100)
print("ROLE DIRECT FINAL SENSOR — SIMPLE5 BASIC TEST")
print("=" * 100)

print(
    "candidate file =",
    candidate_path,
)

print(
    "responses      =",
    len(responses),
)

print(
    "systems        =",
    len(truth),
)

print(
    "roles          =",
    len(all_roles),
)


# ============================================================
# Training-composition overlap audit.
# ============================================================

train_manifest_path = (
    TRAIN_ROOT / "manifest_privileged.json"
)

composition_overlap = []

if train_manifest_path.exists():

    train_manifest = json.loads(
        train_manifest_path.read_text()
    )

    train_comps = {
        tuple(
            sorted(
                map(
                    str,
                    x["roles"],
                )
            )
        )
        for x in train_manifest["systems"]
    }

    for sid, rr in truth.items():

        comp = tuple(
            sorted(rr)
        )

        if comp in train_comps:
            composition_overlap.append(
                {
                    "session_id": sid,
                    "roles": list(comp),
                }
            )


print(
    "exact training composition overlap =",
    len(composition_overlap),
)


# ============================================================
# Join candidates to responses.
# ============================================================

response_by_key = {
    (
        str(x["session_id"]),
        str(x["query"]),
    ): x
    for x in responses
}


items = []


for c in candidates:

    sid = str(
        c["session_id"]
    )

    role = str(
        c["target_role"]
    )

    query = str(
        c["query"]
    )

    key = (
        sid,
        query,
    )

    if key not in response_by_key:
        raise RuntimeError(
            f"Missing response for {key}"
        )

    r = response_by_key[key]

    answer = str(
        r["final_output"]
    )

    probe_rank = int(
        c.get(
            "direct_probe_rank",
            c.get(
                "probe_rank",
                int(
                    c.get(
                        "candidate_index",
                        0,
                    )
                ) + 1,
            ),
        )
    )

    items.append({
        "session_id":
            sid,

        "target_role":
            role,

        "query":
            query,

        "answer":
            answer,

        "probe_rank":
            probe_rank,

        "installed":
            int(
                role in truth[sid]
            ),

        "selected":
            int(
                role
                in set(
                    map(
                        str,
                        r.get(
                            "selected_roles_training_only",
                            [],
                        ),
                    )
                )
            ),

        "text":
            tr.render_text(
                role,
                query,
                answer,
            ),
    })


assert len(items) == 300


# Every system × role must have exactly 3 probes.
count = defaultdict(int)

for x in items:
    count[
        (
            x["session_id"],
            x["target_role"],
        )
    ] += 1


assert len(count) == 100

assert all(
    n == 3
    for n in count.values()
)


# ============================================================
# Load FINAL target-conditioned sensor.
# ============================================================

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


logits = []


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

        logits.extend(z)

        if (
            start % 80 == 0
            or start + 8 >= len(items)
        ):
            print(
                f"scored "
                f"{min(start + 8, len(items))}"
                f"/{len(items)}",
                flush=True,
            )


assert len(logits) == 300


for x, z in zip(
    items,
    logits,
):
    x["logit"] = float(z)


# ============================================================
# Aggregate 3 probes per system × role.
# ============================================================

g = defaultdict(list)

for x in items:
    g[
        (
            x["session_id"],
            x["target_role"],
        )
    ].append(x)


agg = []


for (
    sid,
    role,
), rr in g.items():

    rr = sorted(
        rr,
        key=lambda x:
            x["probe_rank"],
    )

    assert len(rr) == 3

    agg.append({
        "session_id":
            sid,

        "target_role":
            role,

        "installed":
            int(
                role in truth[sid]
            ),

        "score":
            float(
                mean(
                    x["logit"]
                    for x in rr
                )
            ),
    })


assert len(agg) == 100


# ============================================================
# 1) GLOBAL AUC over 5 × 20 role decisions.
# ============================================================

global_auc = float(
    roc_auc_score(
        [
            x["installed"]
            for x in agg
        ],
        [
            x["score"]
            for x in agg
        ],
    )
)


# ============================================================
# 2) Role-wise AUC / correct-system TOP1.
#
# 15 roles are installed in Simple5.
# Remaining 5 have no positive and cannot have role AUC.
# ============================================================

role_auc = {}
role_top1 = {}


for role in all_roles:

    rr = [
        x
        for x in agg
        if x["target_role"] == role
    ]

    assert len(rr) == 5

    y = [
        x["installed"]
        for x in rr
    ]

    if len(set(y)) < 2:
        continue

    role_auc[role] = float(
        roc_auc_score(
            y,
            [
                x["score"]
                for x in rr
            ],
        )
    )

    best = max(
        rr,
        key=lambda x:
            x["score"],
    )

    role_top1[role] = int(
        best["installed"] == 1
    )


assert len(role_auc) == 15


macro_role_auc = float(
    mean(
        role_auc.values()
    )
)

role_top1_accuracy = float(
    mean(
        role_top1.values()
    )
)


# ============================================================
# 3) System-level role recovery with TRUE K=3.
#
# This is deliberately an ORACLE-K ranking sanity test.
# It does NOT claim unknown-K recovery.
# ============================================================

per_system = []

f1s = []
recalls = []
exact = []


for sid in sorted(truth):

    rr = [
        x
        for x in agg
        if x["session_id"] == sid
    ]

    assert len(rr) == 20

    ranked = sorted(
        rr,
        key=lambda x:
            x["score"],
        reverse=True,
    )

    pred = {
        x["target_role"]
        for x in ranked[:3]
    }

    true = truth[sid]

    tp = len(
        pred & true
    )

    fp = len(
        pred - true
    )

    fn = len(
        true - pred
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

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if precision + recall
        else 0.0
    )

    ok = int(
        pred == true
    )

    per_system.append({
        "session_id":
            sid,

        "truth":
            sorted(true),

        "pred_top3":
            [
                x["target_role"]
                for x in ranked[:3]
            ],

        "top5_scores":
            [
                {
                    "role":
                        x["target_role"],

                    "score":
                        round(
                            x["score"],
                            4,
                        ),

                    "installed":
                        bool(
                            x["installed"]
                        ),
                }
                for x in ranked[:5]
            ],

        "recall":
            recall,

        "f1":
            f1,

        "exact":
            bool(ok),
    })

    recalls.append(
        recall
    )

    f1s.append(
        f1
    )

    exact.append(
        ok
    )


mean_recall = float(
    mean(recalls)
)

mean_f1 = float(
    mean(f1s)
)

exact_set = float(
    mean(exact)
)


# ============================================================
# Routing sanity.
# ============================================================

present_probe_rows = [
    x
    for x in items
    if x["installed"] == 1
]

routing_target_present = float(
    mean(
        x["selected"]
        for x in present_probe_rows
    )
)


result = {
    "experiment":
        "role_direct_final_sensor_simple5_v1",

    "note":
        (
            "Basic same-direct-query-panel Simple5 sanity test. "
            "System top-3 metric uses true K=3 and is therefore "
            "a role-ranking test, not unknown-K recovery."
        ),

    "systems":
        5,

    "roles":
        20,

    "probes_per_role":
        3,

    "response_rows":
        300,

    "candidate_file":
        str(candidate_path),

    "training_composition_overlap":
        composition_overlap,

    "routing_target_given_present":
        routing_target_present,

    "global_auc":
        global_auc,

    "macro_role_auc":
        macro_role_auc,

    "role_top1_accuracy":
        role_top1_accuracy,

    "evaluable_roles":
        len(role_auc),

    "oracle_k3": {
        "exact_set_accuracy":
            exact_set,

        "mean_role_recall":
            mean_recall,

        "mean_f1":
            mean_f1,
    },

    "role_auc":
        role_auc,

    "role_top1":
        role_top1,

    "per_system":
        per_system,
}


OUT.write_text(
    json.dumps(
        result,
        indent=2,
    )
)


print()
print("=" * 100)
print("SIMPLE5 RESULT")
print("=" * 100)

print(
    "routing target | present =",
    f"{routing_target_present:.4f}",
)

print(
    "GLOBAL AUC               =",
    f"{global_auc:.4f}",
)

print(
    "MACRO ROLE AUC           =",
    f"{macro_role_auc:.4f}",
)

print(
    "ROLE TOP1 ACC            =",
    f"{role_top1_accuracy:.4f}",
    f"({sum(role_top1.values())}/{len(role_top1)})",
)

print()
print(
    "ORACLE-K=3 EXACT SET     =",
    f"{exact_set:.4f}",
)

print(
    "ORACLE-K=3 MEAN RECALL   =",
    f"{mean_recall:.4f}",
)

print(
    "ORACLE-K=3 MEAN F1       =",
    f"{mean_f1:.4f}",
)


print()
print("=" * 100)
print("PER SYSTEM")
print("=" * 100)

for x in per_system:

    print()
    print(x["session_id"])

    print(
        "  truth =",
        x["truth"],
    )

    print(
        "  pred  =",
        x["pred_top3"],
    )

    print(
        "  exact =",
        x["exact"],
        "f1 =",
        f"{x['f1']:.3f}",
    )

    print(
        "  top5  =",
        x["top5_scores"],
    )


print()
print("=" * 100)
print("PER ROLE AUC")
print("=" * 100)

for role in sorted(role_auc):

    print(
        f"{role:26s} "
        f"AUC={role_auc[role]:.3f} "
        f"TOP1={role_top1[role]}"
    )


print()
print(
    "REPORT =",
    OUT,
)


del model
del base

gc.collect()
torch.cuda.empty_cache()
