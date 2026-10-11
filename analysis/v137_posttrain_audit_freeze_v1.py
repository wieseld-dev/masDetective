#!/usr/bin/env python3

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

ROOT = Path("artifacts/v137_presence_v1")
MODEL = ROOT / "model_v1"
DATA = ROOT / "dataset_v1"
OUT = ROOT / "frozen_decoder_v1"

OUT.mkdir(parents=True, exist_ok=True)


def readj(path):
    with Path(path).open() as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def poisson_binomial(probs):
    d = np.array([1.0], dtype=float)

    for p in probs:
        nxt = np.zeros(len(d) + 1)
        nxt[:-1] += d * (1.0 - p)
        nxt[1:] += d * p
        d = nxt

    return d


metrics_path = MODEL / "DEV_METRICS.json"
pred_path = MODEL / "dev_predictions.jsonl"
adapter_dir = MODEL / "best_adapter"
dataset_meta_path = DATA / "meta.json"

required = [
    metrics_path,
    pred_path,
    dataset_meta_path,
    adapter_dir / "adapter_config.json",
]

for p in required:
    if not p.exists():
        raise RuntimeError(
            f"MISSING REQUIRED ARTIFACT: {p}"
        )

metrics = json.loads(metrics_path.read_text())
dataset_meta = json.loads(dataset_meta_path.read_text())
pred = readj(pred_path)

if metrics.get("internal_test_used") is not False:
    raise RuntimeError(
        "DEV_METRICS does not assert internal_test_used=false"
    )

if dataset_meta.get("internal_test_used") is not False:
    raise RuntimeError(
        "dataset meta does not assert internal_test_used=false"
    )

if len(pred) != 760:
    raise RuntimeError(
        f"Expected 760 DEV predictions, got {len(pred)}"
    )

ids = [x["example_id"] for x in pred]

if len(ids) != len(set(ids)):
    raise RuntimeError(
        "Duplicate DEV example IDs"
    )

systems = defaultdict(list)

for x in pred:
    systems[str(x["system_id"])].append(x)

if len(systems) != 38:
    raise RuntimeError(
        f"Expected 38 DEV systems, got {len(systems)}"
    )

for sid, rows in systems.items():
    if len(rows) != 20:
        raise RuntimeError(
            f"{sid}: expected 20 role probes, got {len(rows)}"
        )

k_counts = Counter()

for rows in systems.values():
    ks = {
        int(x["K_eval_only"])
        for x in rows
    }

    if len(ks) != 1:
        raise RuntimeError(
            "Inconsistent K inside DEV system"
        )

    k_counts[next(iter(ks))] += 1

if k_counts != Counter({3: 24, 4: 14}):
    raise RuntimeError(
        f"Unexpected DEV K composition: {dict(k_counts)}"
    )


# ---------------------------------------------------------
# Verify saved classification head.
# ---------------------------------------------------------

from safetensors import safe_open

score_keys = []

for sf in adapter_dir.glob("*.safetensors"):
    with safe_open(
        str(sf),
        framework="pt",
        device="cpu",
    ) as f:
        for k in f.keys():
            if "score" in k:
                score_keys.append(k)

if not score_keys:
    raise RuntimeError(
        "Saved adapter does not contain score head"
    )


# ---------------------------------------------------------
# Raw presence metrics.
# ---------------------------------------------------------

y = np.asarray(
    [int(x["true_label"]) for x in pred],
    dtype=int,
)

logits = np.asarray(
    [float(x["logit"]) for x in pred],
    dtype=float,
)

raw_p = 1.0 / (
    1.0 + np.exp(-logits)
)

raw_auc = float(
    roc_auc_score(y, raw_p)
)

raw_pr = float(
    average_precision_score(y, raw_p)
)

raw_bal = float(
    balanced_accuracy_score(
        y,
        (raw_p >= 0.5).astype(int),
    )
)


# ---------------------------------------------------------
# Freeze global Platt calibration on DEV.
# ---------------------------------------------------------

cal = LogisticRegression(
    solver="lbfgs",
    C=1e6,
    max_iter=10000,
)

cal.fit(
    logits.reshape(-1, 1),
    y,
)

A = float(cal.coef_[0, 0])
B = float(cal.intercept_[0])

cal_p = cal.predict_proba(
    logits.reshape(-1, 1)
)[:, 1]

cal_auc = float(
    roc_auc_score(y, cal_p)
)

cal_pr = float(
    average_precision_score(y, cal_p)
)

best_threshold = 0.5
best_bal = -1.0

for t in np.linspace(
    0.01,
    0.99,
    197,
):
    bal = float(
        balanced_accuracy_score(
            y,
            (cal_p >= t).astype(int),
        )
    )

    if bal > best_bal:
        best_bal = bal
        best_threshold = float(t)


# ---------------------------------------------------------
# Frozen Poisson-binomial K decoder.
# ---------------------------------------------------------

p_by_id = {
    row["example_id"]: float(p)
    for row, p in zip(pred, cal_p)
}

correct = 0
correct3 = 0
correct4 = 0

n3 = 0
n4 = 0

abs_errors = []
exact_sets = 0

system_results = []

for sid, rows in sorted(systems.items()):
    true_k = int(
        rows[0]["K_eval_only"]
    )

    role_prob = {
        row["target_role"]:
            p_by_id[row["example_id"]]
        for row in rows
    }

    true_roles = {
        row["target_role"]
        for row in rows
        if int(row["true_label"]) == 1
    }

    ordered_roles = sorted(role_prob)

    posterior = poisson_binomial(
        [
            role_prob[r]
            for r in ordered_roles
        ]
    )

    pred_k = max(
        range(3, 9),
        key=lambda k:
            posterior[k],
    )

    ok = int(
        pred_k == true_k
    )

    correct += ok

    abs_errors.append(
        abs(pred_k - true_k)
    )

    if true_k == 3:
        n3 += 1
        correct3 += ok

    if true_k == 4:
        n4 += 1
        correct4 += ok

    ranked = sorted(
        ordered_roles,
        key=lambda r:
            role_prob[r],
        reverse=True,
    )

    pred_roles = set(
        ranked[:pred_k]
    )

    exact_sets += int(
        pred_roles == true_roles
    )

    system_results.append({
        "system_id": sid,
        "true_K": true_k,
        "pred_K": pred_k,
        "posterior_K": [
            float(z)
            for z in posterior
        ],
        "role_probabilities":
            role_prob,
        "true_roles":
            sorted(true_roles),
        "pred_roles":
            sorted(pred_roles),
    })


decoder = {
    "version":
        "v137_frozen_decoder_v1",

    "presence_calibration": {
        "type": "global_platt",
        "formula":
            "sigmoid(A * raw_logit + B)",
        "A": A,
        "B": B,
    },

    "presence_threshold":
        best_threshold,

    "K_decoder": {
        "type":
            "poisson_binomial_argmax",
        "allowed_K":
            [3, 4, 5, 6, 7, 8],
    },

    "adapter_path":
        str(adapter_dir),

    "dev_predictions_sha256":
        sha(pred_path),

    "DEV_METRICS_sha256":
        sha(metrics_path),

    "score_head_keys":
        sorted(score_keys),

    "internal_test_used":
        False,
}

decoder_path = (
    OUT / "FROZEN_DECODER.json"
)

decoder_path.write_text(
    json.dumps(
        decoder,
        indent=2,
        sort_keys=True,
    )
)

decoder_sha = sha(decoder_path)

(
    OUT / "FROZEN_DECODER.sha256"
).write_text(
    decoder_sha
    + "  FROZEN_DECODER.json\n"
)

with (
    OUT /
    "dev_system_results_calibrated.jsonl"
).open("w") as f:
    for x in system_results:
        f.write(
            json.dumps(x)
            + "\n"
        )


report = {
    "presence_raw": {
        "roc_auc": raw_auc,
        "pr_auc": raw_pr,
        "balanced_accuracy_0_5":
            raw_bal,
    },

    "presence_calibrated": {
        "roc_auc": cal_auc,
        "pr_auc": cal_pr,
        "best_balanced_accuracy":
            best_bal,
        "selected_threshold":
            best_threshold,
        "platt_A": A,
        "platt_B": B,
    },

    "K_DEV": {
        "systems":
            len(systems),

        "exact_K":
            correct / len(systems),

        "K3_exact":
            correct3 / n3,

        "K4_exact":
            correct4 / n4,

        "K_MAE":
            float(
                np.mean(abs_errors)
            ),

        "exact_role_set":
            exact_sets
            / len(systems),
    },

    "internal_test_used":
        False,

    "frozen_decoder_sha256":
        decoder_sha,
}

(
    OUT / "DEV_FREEZE_REPORT.json"
).write_text(
    json.dumps(
        report,
        indent=2,
        sort_keys=True,
    )
)

print("=" * 90)
print("V13.7 POST-TRAIN AUDIT + FREEZE")
print("=" * 90)

print(
    "DEV predictions:",
    len(pred)
)

print(
    "DEV systems    :",
    len(systems)
)

print(
    "K composition  :",
    dict(k_counts)
)

print()
print(
    "RAW ROC-AUC    :",
    round(raw_auc, 4)
)

print(
    "RAW PR-AUC     :",
    round(raw_pr, 4)
)

print(
    "RAW BalAcc     :",
    round(raw_bal, 4)
)

print()
print(
    "CAL BalAcc     :",
    round(best_bal, 4)
)

print(
    "threshold      :",
    round(best_threshold, 4)
)

print()
print(
    "EXACT K        :",
    f"{correct / len(systems):.2%}"
)

print(
    "K=3 exact      :",
    f"{correct3 / n3:.2%}"
)

print(
    "K=4 exact      :",
    f"{correct4 / n4:.2%}"
)

print(
    "K MAE          :",
    round(
        float(np.mean(abs_errors)),
        4,
    )
)

print(
    "EXACT ROLE SET :",
    f"{exact_sets / len(systems):.2%}"
)

print()
print("SCORE HEAD SAVE CHECK: PASS")
print("INTERNAL TEST USED: NO")
print("DECODER FROZEN: PASS")
print("DECODER SHA:", decoder_sha)

done = ROOT / "V137_DEV_FROZEN_DONE.txt"

done.write_text(
    "V13.7 training completed, DEV evaluated, "
    "decoder frozen, internal test untouched.\n"
    f"decoder_sha256={decoder_sha}\n"
)

print()
print("V13.7 DEV PIPELINE: DONE")
