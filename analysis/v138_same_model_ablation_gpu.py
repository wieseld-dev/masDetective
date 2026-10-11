#!/usr/bin/env python3

import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from peft import PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import analysis.train_v13_6b_activation_judge as tr
import analysis.v137_calibration_audit as ca
import analysis.v138_same_model_ablation as audit


ROOT = Path("artifacts/v13_7_calibration_audit_v1")
OUT = Path("artifacts/v138_same_model_ablation")
OUT.mkdir(parents=True, exist_ok=True)

PRED = np.load(ROOT / "predictions.npz", allow_pickle=True)


# ============================================================
# Helpers
# ============================================================

def mean_map(rows, logits):
    d = defaultdict(list)

    for row, z in zip(rows, logits):
        d[row.text].append(float(z))

    return {
        k: float(np.mean(v))
        for k, v in d.items()
    }


def candidate_role(text):
    marker = "[CANDIDATE ROLE]\n"
    assert marker in text

    return (
        text.split(marker, 1)[1]
        .split("\n", 1)[0]
        .strip()
    )


def infer_unique(model, tok, texts, device, batch_size=4):
    unique = list(dict.fromkeys(texts))
    scores = {}

    print(
        f"[infer] unique texts={len(unique)} "
        f"batch={batch_size}",
        flush=True,
    )

    model.eval()

    with torch.no_grad():
        for start in range(0, len(unique), batch_size):
            batch = unique[start:start + batch_size]

            x = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=ca.MAX_LENGTH,
                return_tensors="pt",
            )

            x = {
                k: v.to(device)
                for k, v in x.items()
            }

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):
                z = model(**x).logits[:, 0].float()

            vals = z.detach().cpu().numpy()

            for t, s in zip(batch, vals):
                scores[t] = float(s)

            if (
                start == 0
                or start + batch_size >= len(unique)
                or (start // batch_size) % 50 == 0
            ):
                print(
                    f"[infer] {min(start+batch_size,len(unique))}"
                    f"/{len(unique)}",
                    flush=True,
                )

    return scores


def make_pair_rows(
    pairs,
    full_score,
    no_score,
    answer_score,
):
    rows = []

    for p in pairs:
        assert p.pos_text in full_score
        assert p.neg_text in full_score

        np_text = audit.no_answer_text(p.pos_text)
        nn_text = audit.no_answer_text(p.neg_text)

        ap_text = audit.answer_only_text(p.pos_text)
        an_text = audit.answer_only_text(p.neg_text)

        assert np_text in no_score
        assert nn_text in no_score
        assert ap_text in answer_score
        assert an_text in answer_score

        rows.append({
            "type":
                audit.base_type(p.pair_type),

            "full_type":
                p.pair_type,

            "pos_role":
                candidate_role(p.pos_text),

            "neg_role":
                candidate_role(p.neg_text),

            "full_margin":
                full_score[p.pos_text]
                - full_score[p.neg_text],

            "noanswer_margin":
                no_score[np_text]
                - no_score[nn_text],

            "answeronly_margin":
                answer_score[ap_text]
                - answer_score[an_text],
        })

    return rows


def margins(rows, mode, alpha=0.0):
    if mode == "full":
        return np.asarray([
            x["full_margin"]
            for x in rows
        ])

    if mode == "noanswer":
        return np.asarray([
            x["noanswer_margin"]
            for x in rows
        ])

    if mode == "answeronly":
        return np.asarray([
            x["answeronly_margin"]
            for x in rows
        ])

    if mode == "residual":
        return np.asarray([
            x["full_margin"]
            - alpha * x["noanswer_margin"]
            for x in rows
        ])

    raise KeyError(mode)


def metric(rows, mode, alpha=0.0):
    m = margins(rows, mode, alpha)

    return {
        "n": int(len(m)),
        "accuracy":
            float(np.mean(m > 0)),
        "mean_margin":
            float(np.mean(m)),
        "median_margin":
            float(np.median(m)),
    }


# ============================================================
# Reconstruct exact CAL / TEST data
# ============================================================

data = audit.reconstruct()

rd = data["rd"]

cal_cases = data["cal_cases"]
test_cases = data["test_cases"]

cal_pairs = data["cal_pairs"]
test_pairs = data["test_pairs"]

cal_rows, _ = tr.build_presence(
    cal_cases, rd
)

test_rows, _ = tr.build_presence(
    test_cases, rd
)

assert len(cal_rows) == len(PRED["cal_logits"])
assert len(test_rows) == len(PRED["test_logits"])

cal_full_score = mean_map(
    cal_rows,
    np.asarray(
        PRED["cal_logits"],
        dtype=float,
    ),
)

test_full_score = mean_map(
    test_rows,
    np.asarray(
        PRED["test_logits"],
        dtype=float,
    ),
)

print("===== RECONSTRUCTION =====")
print("CAL cases :", len(cal_cases))
print("TEST cases:", len(test_cases))
print("CAL pairs :", len(cal_pairs))
print("TEST pairs:", len(test_pairs))


# ============================================================
# Build transformed input sets
# ============================================================

def endpoint_texts(pairs):
    out = []

    for p in pairs:
        out.append(p.pos_text)
        out.append(p.neg_text)

    return out


cal_endpoints = endpoint_texts(cal_pairs)
test_endpoints = endpoint_texts(test_pairs)

all_endpoints = cal_endpoints + test_endpoints

noanswer_texts = [
    audit.no_answer_text(x)
    for x in all_endpoints
]

answeronly_texts = [
    audit.answer_only_text(x)
    for x in all_endpoints
]

print("\n===== INFERENCE LOAD =====")
print(
    "unique NOANSWER  :",
    len(set(noanswer_texts)),
)
print(
    "unique ANSWERONLY:",
    len(set(answeronly_texts)),
)


# ============================================================
# Load exact V13.7 epoch-1 model
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA required. Submit via Slurm GPU."
    )

device = torch.device("cuda:0")

print(
    "\nGPU:",
    torch.cuda.get_device_name(0),
    flush=True,
)

print(
    "adapter:",
    ca.ADAPTER,
    flush=True,
)

print(
    "base model:",
    ca.BASE_MODEL,
    flush=True,
)

tok = AutoTokenizer.from_pretrained(
    ca.ADAPTER
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

model = AutoModelForSequenceClassification.from_pretrained(
    ca.BASE_MODEL,
    num_labels=1,
    torch_dtype=torch.float16,
)

model.config.pad_token_id = tok.pad_token_id

model = PeftModel.from_pretrained(
    model,
    ca.ADAPTER,
)

model.to(device)
model.eval()


# ============================================================
# New inference ONLY for the ablations
# ============================================================

print("\n===== NO-ANSWER INFERENCE =====")
no_score = infer_unique(
    model,
    tok,
    noanswer_texts,
    device,
    batch_size=4,
)

print("\n===== ANSWER-ONLY INFERENCE =====")
answer_score = infer_unique(
    model,
    tok,
    answeronly_texts,
    device,
    batch_size=4,
)


# ============================================================
# Construct pair metrics
# ============================================================

CAL = make_pair_rows(
    cal_pairs,
    cal_full_score,
    no_score,
    answer_score,
)

TEST = make_pair_rows(
    test_pairs,
    test_full_score,
    no_score,
    answer_score,
)

TARGET = "within_answer_probe_target_absent"
HARD = "within_answer_hard_neighbor_absent"


# ============================================================
# Tune alpha strictly on CAL
# ============================================================

cal_target = [
    x for x in CAL
    if x["type"] == TARGET
]

cal_hard = [
    x for x in CAL
    if x["type"] == HARD
]

grid = np.linspace(
    0.0,
    3.0,
    121,
)

best = None

for alpha in grid:
    mt = metric(
        cal_target,
        "residual",
        alpha,
    )

    mh = metric(
        cal_hard,
        "residual",
        alpha,
    )

    objective = (
        0.5 * mt["accuracy"]
        + 0.5 * mh["accuracy"]
    )

    candidate = (
        objective,
        -float(alpha),
        float(alpha),
        mt["accuracy"],
        mh["accuracy"],
    )

    if best is None or candidate > best:
        best = candidate


(
    cal_objective,
    _,
    alpha,
    cal_target_acc,
    cal_hard_acc,
) = best

print("\n===== CAL ALPHA SELECTION =====")
print("alpha              :", alpha)
print("target absent acc  :", cal_target_acc)
print("hard neighbor acc  :", cal_hard_acc)
print("balanced objective :", cal_objective)


# ============================================================
# Untouched TEST
# ============================================================

types = sorted(
    set(x["type"] for x in TEST)
)

test_by_type = {}

print(
    "\n===== UNTOUCHED TEST "
    "FULL / NOANSWER / ANSWERONLY / RESIDUAL ====="
)

for typ in types:
    rr = [
        x for x in TEST
        if x["type"] == typ
    ]

    full = metric(rr, "full")
    no = metric(rr, "noanswer")
    ans = metric(rr, "answeronly")
    res = metric(
        rr,
        "residual",
        alpha,
    )

    test_by_type[typ] = {
        "full": full,
        "noanswer": no,
        "answeronly": ans,
        "residual": res,
    }

    print(
        f"{typ:42s} "
        f"n={len(rr):3d} | "
        f"FULL={full['accuracy']:.4f} "
        f"NOANS={no['accuracy']:.4f} "
        f"ANSONLY={ans['accuracy']:.4f} "
        f"RESID={res['accuracy']:.4f}"
    )


# ============================================================
# Probe-target absent by ABSENT target role
# ============================================================

print(
    "\n===== PROBE-TARGET-ABSENT "
    "BY ABSENT TARGET ====="
)

target_by_role = {}

for role in rd:
    rr = [
        x for x in TEST
        if x["type"] == TARGET
        and x["neg_role"] == role
    ]

    if not rr:
        continue

    full = metric(rr, "full")
    no = metric(rr, "noanswer")
    ans = metric(rr, "answeronly")
    res = metric(
        rr,
        "residual",
        alpha,
    )

    target_by_role[role] = {
        "full": full,
        "noanswer": no,
        "answeronly": ans,
        "residual": res,
    }

    print(
        f"{role:26s} "
        f"n={len(rr):3d} | "
        f"FULL={full['accuracy']:.3f} "
        f"NOANS={no['accuracy']:.3f} "
        f"ANSONLY={ans['accuracy']:.3f} "
        f"RESID={res['accuracy']:.3f}"
    )


# ============================================================
# SAME-ROLE activation CF per role
# ============================================================

cf = [
    x for x in TEST
    if x["type"].startswith(
        "activation_cf_"
    )
]

print(
    "\n===== SAME-ROLE ACTIVATION "
    "FULL vs ANSWERONLY ====="
)

cf_by_role = {}

for role in rd:
    rr = [
        x for x in cf
        if x["pos_role"] == role
    ]

    if len(rr) < 3:
        continue

    full = metric(rr, "full")
    ans = metric(rr, "answeronly")
    no = metric(rr, "noanswer")
    res = metric(
        rr,
        "residual",
        alpha,
    )

    cf_by_role[role] = {
        "full": full,
        "answeronly": ans,
        "noanswer": no,
        "residual": res,
    }

    print(
        f"{role:26s} "
        f"n={len(rr):3d} | "
        f"FULL={full['accuracy']:.3f} "
        f"ANSONLY={ans['accuracy']:.3f} "
        f"NOANS={no['accuracy']:.3f} "
        f"RESID={res['accuracy']:.3f}"
    )


# ============================================================
# Critical invariance
# ============================================================

cf_no = margins(
    cf,
    "noanswer",
)

cf_full = margins(
    cf,
    "full",
)

cf_res = margins(
    cf,
    "residual",
    alpha,
)

print("\n===== CAUSAL SANITY =====")
print(
    "max |NOANSWER same-role margin|:",
    float(np.max(np.abs(cf_no))),
)
print(
    "max |FULL-RESID same-role margin|:",
    float(np.max(np.abs(cf_full-cf_res))),
)

assert np.max(
    np.abs(cf_no)
) < 1e-6

assert np.max(
    np.abs(cf_full-cf_res)
) < 1e-6


# ============================================================
# Save
# ============================================================

result = {
    "protocol":
        "v13.8_same_v137_model_input_ablation",

    "training": False,
    "api_calls": 0,
    "mas_calls": 0,

    "adapter":
        str(ca.ADAPTER),

    "base_model":
        str(ca.BASE_MODEL),

    "cal_cases":
        len(cal_cases),

    "test_cases":
        len(test_cases),

    "cal_pairs":
        len(CAL),

    "test_pairs":
        len(TEST),

    "alpha_selected_on_cal":
        float(alpha),

    "cal_balanced_objective":
        float(cal_objective),

    "test_by_type":
        test_by_type,

    "target_absent_by_role":
        target_by_role,

    "same_role_cf_by_role":
        cf_by_role,
}

(OUT / "summary.json").write_text(
    json.dumps(
        result,
        indent=2,
        sort_keys=True,
    )
)

np.savez_compressed(
    OUT / "ablation_scores.npz",

    noanswer_text=np.asarray(
        list(no_score.keys()),
        dtype=object,
    ),

    noanswer_logit=np.asarray(
        list(no_score.values()),
        dtype=np.float32,
    ),

    answeronly_text=np.asarray(
        list(answer_score.keys()),
        dtype=object,
    ),

    answeronly_logit=np.asarray(
        list(answer_score.values()),
        dtype=np.float32,
    ),
)

print("\nDONE")
print("wrote:", OUT / "summary.json")
print("wrote:", OUT / "ablation_scores.npz")
