#!/usr/bin/env python3

import gc
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

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


PROJECT = Path(".")
S5_ROOT = Path("artifacts/role_direct_routing_pilot_v1")
MATCHED_ROOT = Path("artifacts/role_direct_matched_cf_v1")
SEALED_ROOT = Path("artifacts/role_direct_sealed_v1")
ADAPTER = Path("artifacts/role_direct_sensor_final_v1/adapter")

CAL_SCORE_PATH = SEALED_ROOT / "final_sensor_scores.jsonl"
FROZEN_CAL_PATH = SEALED_ROOT / "frozen_calibration.json"

OUT = S5_ROOT / "final_sensor_simple5_calibrated_eval.json"


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def set_metrics(pred, truth):
    pred = set(pred)
    truth = set(truth)

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return {
        "exact": pred == truth,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# =====================================================================
# Inputs
# =====================================================================

for p in [
    S5_ROOT / "responses_dev_round_01.jsonl",
    S5_ROOT / "manifest_privileged.json",
    CAL_SCORE_PATH,
    FROZEN_CAL_PATH,
    ADAPTER,
]:
    assert p.exists(), p


# Find exact Simple5 candidate panel.
candidate_path = S5_ROOT / "candidates_dev_round_01.jsonl"
assert candidate_path.exists(), candidate_path

candidates = read_jsonl(candidate_path)
responses = read_jsonl(
    S5_ROOT / "responses_dev_round_01.jsonl"
)

assert len(candidates) == 300
assert len(responses) == 300


s5_manifest = json.loads(
    (S5_ROOT / "manifest_privileged.json").read_text()
)

systems = s5_manifest["systems"]

assert len(systems) == 5


truth = {
    str(x["session_id"]): set(map(str, x["roles"]))
    for x in systems
}

assert all(len(x) == 3 for x in truth.values())


roles = sorted(tr.DESCRIPTIONS)

assert len(roles) == 20


# =====================================================================
# Audit exact composition overlap with final-sensor training systems
# =====================================================================

train_manifest = json.loads(
    (MATCHED_ROOT / "manifest_privileged.json").read_text()
)

train_compositions = {
    tuple(sorted(map(str, x["roles"])))
    for x in train_manifest["systems"]
}

overlap_systems = set()

for sid, rr in truth.items():
    if tuple(sorted(rr)) in train_compositions:
        overlap_systems.add(sid)


print("=" * 110)
print("ROLE DIRECT — SIMPLE5 ROLE-CALIBRATED RECOVERY")
print("=" * 110)
print("Simple5 systems                 =", len(truth))
print("roles                           =", len(roles))
print("existing MAS responses          =", len(responses))
print("exact training comp overlap     =", len(overlap_systems))

for sid in sorted(overlap_systems):
    print(
        "  OVERLAP:",
        sid,
        sorted(truth[sid]),
    )


# =====================================================================
# Build CAL-only role calibration statistics
#
# IMPORTANT:
# Uses ONLY panel == calibration.
# Simple5 is not used to fit anything.
# =====================================================================

cal_rows = [
    x
    for x in read_jsonl(CAL_SCORE_PATH)
    if x["panel"] == "calibration"
]

assert len(cal_rows) == 600


# Aggregate 3 CAL probes -> one score per role/pair/label.
cal_group = defaultdict(list)

for x in cal_rows:
    cal_group[
        (
            str(x["target_role"]),
            str(x["pair_id"]),
            int(x["label"]),
        )
    ].append(x)


cal_agg = []

for (role, pair_id, label), rr in cal_group.items():

    rr = sorted(
        rr,
        key=lambda x: int(x["probe_rank"]),
    )

    assert len(rr) == 3

    cal_agg.append({
        "role": role,
        "pair_id": pair_id,
        "label": label,
        "score": mean(
            float(x["logit"])
            for x in rr
        ),
    })


assert len(cal_agg) == 200


frozen = json.loads(
    FROZEN_CAL_PATH.read_text()
)

thresholds = {
    str(k): float(v)
    for k, v
    in frozen["k"]["3"]["role_thresholds"].items()
}

assert set(thresholds) == set(roles)


cal_stats = {}

for role in roles:

    rr = [
        x
        for x in cal_agg
        if x["role"] == role
    ]

    pos = [
        x["score"]
        for x in rr
        if x["label"] == 1
    ]

    neg = [
        x["score"]
        for x in rr
        if x["label"] == 0
    ]

    assert len(pos) == 5
    assert len(neg) == 5

    all_scores = pos + neg

    mu_pos = mean(pos)
    mu_neg = mean(neg)

    # Overall scale for frozen-threshold margin.
    overall_sd = max(
        pstdev(all_scores),
        1e-6,
    )

    # Within-class pooled scale.
    pos_var = mean(
        (x - mu_pos) ** 2
        for x in pos
    )

    neg_var = mean(
        (x - mu_neg) ** 2
        for x in neg
    )

    pooled_sd = max(
        math.sqrt(
            0.5 * (pos_var + neg_var)
        ),
        1e-6,
    )

    midpoint = 0.5 * (
        mu_pos + mu_neg
    )

    cal_stats[role] = {
        "threshold": thresholds[role],
        "mu_pos": mu_pos,
        "mu_neg": mu_neg,
        "overall_sd": overall_sd,
        "pooled_sd": pooled_sd,
        "midpoint": midpoint,
    }


# =====================================================================
# Join existing Simple5 queries + responses
# =====================================================================

response_by_key = {
    (
        str(x["session_id"]),
        str(x["query"]),
    ): x
    for x in responses
}


items = []

for c in candidates:

    sid = str(c["session_id"])
    role = str(c["target_role"])
    query = str(c["query"])

    key = (sid, query)

    assert key in response_by_key, key

    r = response_by_key[key]

    answer = str(r["final_output"])

    probe_rank = int(
        c.get(
            "direct_probe_rank",
            c.get(
                "probe_rank",
                int(c.get("candidate_index", 0)) + 1,
            ),
        )
    )

    items.append({
        "session_id": sid,
        "target_role": role,
        "query": query,
        "answer": answer,
        "probe_rank": probe_rank,
        "installed": int(
            role in truth[sid]
        ),
        "text": tr.render_text(
            role,
            query,
            answer,
        ),
    })


assert len(items) == 300


count = defaultdict(int)

for x in items:
    count[
        (
            x["session_id"],
            x["target_role"],
        )
    ] += 1

assert len(count) == 100
assert all(v == 3 for v in count.values())


# =====================================================================
# Score Simple5 using FINAL target-conditioned sensor
# =====================================================================

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

base.config.pad_token_id = tok.pad_token_id
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

        zz = (
            model(**enc)
            .logits
            .view(-1)
            .float()
            .cpu()
            .tolist()
        )

        logits.extend(zz)

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

for x, z in zip(items, logits):
    x["logit"] = float(z)


# =====================================================================
# Average 3 probes per system × role
# =====================================================================

g = defaultdict(list)

for x in items:
    g[
        (
            x["session_id"],
            x["target_role"],
        )
    ].append(x)


agg = []

for (sid, role), rr in g.items():

    rr = sorted(
        rr,
        key=lambda x: x["probe_rank"],
    )

    assert len(rr) == 3

    raw = mean(
        x["logit"]
        for x in rr
    )

    st = cal_stats[role]

    # Decoder A: original raw logit.
    raw_score = raw

    # Decoder B:
    # align each role's frozen CAL decision boundary to zero.
    cal_margin = (
        raw - st["threshold"]
    )

    # Decoder C:
    # same margin, but normalize role-specific scale.
    cal_std_margin = (
        cal_margin
        / st["overall_sd"]
    )

    # Decoder D:
    # standardized evidence relative to CAL POS/NEG midpoint.
    #
    # Direction remains fixed: larger sensor score = more PRESENT.
    # We deliberately DO NOT sign-flip roles.
    cal_effect_z = (
        raw - st["midpoint"]
    ) / st["pooled_sd"]

    agg.append({
        "session_id": sid,
        "role": role,
        "installed": int(
            role in truth[sid]
        ),
        "raw": float(raw_score),
        "cal_margin": float(cal_margin),
        "cal_std_margin": float(cal_std_margin),
        "cal_effect_z": float(cal_effect_z),
    })


assert len(agg) == 100


# =====================================================================
# Evaluate top-3 recovery
# =====================================================================

METHODS = [
    "raw",
    "cal_margin",
    "cal_std_margin",
    "cal_effect_z",
]


def evaluate(method, allowed_systems):

    per_system = []

    for sid in sorted(allowed_systems):

        rr = [
            x
            for x in agg
            if x["session_id"] == sid
        ]

        assert len(rr) == 20

        ranked = sorted(
            rr,
            key=lambda x: x[method],
            reverse=True,
        )

        pred = [
            x["role"]
            for x in ranked[:3]
        ]

        m = set_metrics(
            pred,
            truth[sid],
        )

        per_system.append({
            "system_id": sid,
            "truth": sorted(truth[sid]),
            "pred_top3": pred,
            "exact": m["exact"],
            "recall": m["recall"],
            "f1": m["f1"],
            "top5": [
                {
                    "role": x["role"],
                    "score": round(
                        x[method],
                        4,
                    ),
                    "installed": bool(
                        x["installed"]
                    ),
                }
                for x in ranked[:5]
            ],
        })

    return {
        "n_systems": len(per_system),
        "exact_set_accuracy": mean(
            int(x["exact"])
            for x in per_system
        ),
        "mean_recall": mean(
            x["recall"]
            for x in per_system
        ),
        "mean_f1": mean(
            x["f1"]
            for x in per_system
        ),
        "per_system": per_system,
    }


all_systems = set(truth)

nonoverlap_systems = (
    all_systems
    - overlap_systems
)

assert len(nonoverlap_systems) == 4


results = {}

for method in METHODS:

    results[method] = {
        "all5": evaluate(
            method,
            all_systems,
        ),
        "nonoverlap4": evaluate(
            method,
            nonoverlap_systems,
        ),
    }


# =====================================================================
# Absolute CAL-threshold presence classification
# =====================================================================

tp = tn = fp = fn = 0

for x in agg:

    pred = int(
        x["cal_margin"] >= 0
    )

    y = x["installed"]

    if y == 1 and pred == 1:
        tp += 1
    elif y == 0 and pred == 0:
        tn += 1
    elif y == 0 and pred == 1:
        fp += 1
    elif y == 1 and pred == 0:
        fn += 1


tpr = tp / (tp + fn)
tnr = tn / (tn + fp)
fpr = fp / (fp + tn)
ba = 0.5 * (tpr + tnr)


report = {
    "experiment":
        "role_direct_simple5_role_calibrated_v1",

    "important_note":
        (
            "Exploratory Simple5 sanity diagnostic. "
            "All calibration parameters come from the separate "
            "CAL panel; Simple5 itself fits no thresholds or scales."
        ),

    "systems": 5,
    "roles": 20,
    "responses_reused": 300,

    "overlap_systems":
        sorted(overlap_systems),

    "absolute_presence_using_frozen_cal_threshold": {
        "balanced_accuracy": ba,
        "tpr": tpr,
        "fpr": fpr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    },

    "calibration_stats":
        cal_stats,

    "results":
        results,
}


OUT.write_text(
    json.dumps(
        report,
        indent=2,
    )
)


print()
print("=" * 110)
print("SIMPLE5 TOP-3 RECOVERY — CALIBRATION COMPARISON")
print("=" * 110)

for method in METHODS:

    a = results[
        method
    ]["all5"]

    b = results[
        method
    ]["nonoverlap4"]

    print()
    print(method.upper())

    print(
        "  ALL 5       "
        f"EXACT={a['exact_set_accuracy']:.3f} "
        f"RECALL={a['mean_recall']:.3f} "
        f"F1={a['mean_f1']:.3f}"
    )

    print(
        "  NONOVERLAP4 "
        f"EXACT={b['exact_set_accuracy']:.3f} "
        f"RECALL={b['mean_recall']:.3f} "
        f"F1={b['mean_f1']:.3f}"
    )


print()
print("=" * 110)
print("FROZEN CAL THRESHOLD — ABSOLUTE PRESENCE")
print("=" * 110)

print(
    f"BA={ba:.3f} "
    f"TPR={tpr:.3f} "
    f"FPR={fpr:.3f} "
    f"TP={tp} TN={tn} FP={fp} FN={fn}"
)


print()
print("=" * 110)
print("BEST VIEW: PER SYSTEM FOR EACH DECODER")
print("=" * 110)

for method in METHODS:

    print()
    print("###", method.upper())

    for x in results[
        method
    ]["all5"]["per_system"]:

        print()
        print(x["system_id"])

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
            x["top5"],
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
