from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_squared_error
from scipy.stats import spearmanr

from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()

from targets.hierarchical_mas import ROLE_PROMPTS
from v11.role_universe import TRAIN_ROLES


TRAIN_ROOT = Path("artifacts/activation_reward_sweep_v1")
DEV_ROOT = Path("artifacts/v14a_ckpt_eval_r07_v1")
OUT = Path("artifacts/activation_external_dev_v1")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

LAYERS = [16, 24]
MAX_LEN = 512
BATCH_PAIRS = 8

ALPHA_ROLE = 10.0
ALPHA_ACT = 10.0

NEUTRAL_SYSTEM = (
    "You are a helpful assistant. "
    "Answer the user's request normally."
)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def safe_spearman(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    if len(a) < 2:
        return float("nan")

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")

    return float(spearmanr(a, b).statistic)


# ============================================================
# 1. LOAD TRAIN ACTIVATION FEATURES
# ============================================================

train_meta = read_jsonl(
    TRAIN_ROOT / "metadata.jsonl"
)

train_delta_all = np.load(
    TRAIN_ROOT / "role_delta_features.npy"
).astype(np.float32)

train_summary = json.loads(
    (TRAIN_ROOT / "summary.json").read_text()
)

tested_layers = list(
    map(int, train_summary["tested_layers"])
)

layer_to_idx = {
    l: i for i, l in enumerate(tested_layers)
}

for l in LAYERS:
    if l not in layer_to_idx:
        raise RuntimeError(
            f"Missing layer {l} in saved TRAIN features"
        )

y_train = np.asarray(
    [float(x["reward"]) for x in train_meta],
    dtype=np.float64,
)

role_to_i = {
    r: i for i, r in enumerate(TRAIN_ROLES)
}


def role_features(rows):
    X = np.zeros(
        (len(rows), len(TRAIN_ROLES) + 1),
        dtype=np.float32,
    )

    for i, x in enumerate(rows):
        roles = [
            str(r)
            for r in x["selected_roles"]
            if str(r) in role_to_i
        ]

        for r in roles:
            X[i, role_to_i[r]] = 1.0

        X[i, -1] = len(roles)

    return X


Xrole_train = role_features(train_meta)

X16_train = train_delta_all[
    :, layer_to_idx[16], :
]

X24_train = train_delta_all[
    :, layer_to_idx[24], :
]

X1624_train = np.concatenate(
    [X16_train, X24_train],
    axis=1,
)


# ============================================================
# 2. LOAD 48 HELD-OUT DEV EXECUTIONS
# ============================================================

dev_rows = []

for rnd in range(1, 7):

    responses = read_jsonl(
        DEV_ROOT / f"responses_dev_round_{rnd:02d}.jsonl"
    )

    metrics = read_jsonl(
        DEV_ROOT / f"metrics_dev_round_{rnd:02d}.jsonl"
    )

    reward_by_sid = {
        str(x["session_id"]):
            float(x["chosen_reward_training_only"])
        for x in metrics
    }

    for r in responses:

        if str(r.get("execution_status", "")) != "ok":
            continue

        sid = str(r["session_id"])

        if sid not in reward_by_sid:
            raise RuntimeError(
                f"Missing reward for {sid} round {rnd}"
            )

        selected = [
            str(x)
            for x in r.get(
                "selected_roles_training_only",
                []
            )
            if str(x) in ROLE_PROMPTS
        ]

        if not selected:
            continue

        dev_rows.append({
            "session_id": sid,
            "round": rnd,
            "candidate_index":
                int(r["candidate_index"]),
            "query": str(r["query"]),
            "selected_roles": selected,
            "reward":
                reward_by_sid[sid],
        })


print("=" * 90)
print("EXTERNAL DEV ACTIVATION VALIDATION")
print("=" * 90)

print("TRAIN rows =", len(train_meta))
print("DEV rows   =", len(dev_rows))
print(
    "DEV systems =",
    len(set(x["session_id"] for x in dev_rows))
)

if len(dev_rows) != 48:
    print(
        "WARNING: expected 48 DEV rows, got",
        len(dev_rows),
    )


# ============================================================
# 3. EXTRACT DEV WHITE-BOX ACTIVATIONS
# ============================================================

device = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

model.eval()

if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

hidden_size = int(model.config.hidden_size)


def make_text(system_prompt, query):
    return tok.apply_chat_template(
        [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": query,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


pairs = []

for ei, x in enumerate(dev_rows):
    for role in x["selected_roles"]:
        pairs.append(
            (ei, role, x["query"])
        )


delta_sum = {
    l: np.zeros(
        (len(dev_rows), hidden_size),
        dtype=np.float32,
    )
    for l in LAYERS
}

counts = np.zeros(
    len(dev_rows),
    dtype=np.int32,
)


with torch.inference_mode():

    for start in range(
        0,
        len(pairs),
        BATCH_PAIRS,
    ):

        batch = pairs[
            start:start + BATCH_PAIRS
        ]

        role_texts = [
            make_text(
                ROLE_PROMPTS[role],
                query,
            )
            for _, role, query in batch
        ]

        neutral_texts = [
            make_text(
                NEUTRAL_SYSTEM,
                query,
            )
            for _, _, query in batch
        ]

        texts = role_texts + neutral_texts

        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
        )

        enc = {
            k: v.to(device)
            for k, v in enc.items()
        }

        last = (
            enc["attention_mask"].sum(dim=1)
            - 1
        )

        out = model(
            **enc,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

        B = len(batch)

        ar = torch.arange(
            2 * B,
            device=device,
        )

        for layer in LAYERS:

            h = out.hidden_states[layer]

            v = h[
                ar,
                last,
                :
            ].float().cpu().numpy()

            vr = v[:B]
            vn = v[B:]

            for bi, (
                example_i,
                _role,
                _query,
            ) in enumerate(batch):

                delta_sum[layer][example_i] += (
                    vr[bi] - vn[bi]
                )

        for example_i, _, _ in batch:
            counts[example_i] += 1

        print(
            f"[activation] "
            f"{min(start+len(batch), len(pairs))}"
            f"/{len(pairs)}",
            flush=True,
        )


if np.any(counts <= 0):
    raise RuntimeError(
        "DEV example with zero selected-role count"
    )


X16_dev = (
    delta_sum[16]
    / counts[:, None]
)

X24_dev = (
    delta_sum[24]
    / counts[:, None]
)

X1624_dev = np.concatenate(
    [X16_dev, X24_dev],
    axis=1,
)

Xrole_dev = role_features(dev_rows)

y_dev = np.asarray(
    [float(x["reward"]) for x in dev_rows],
    dtype=np.float64,
)

groups_dev = np.asarray(
    [str(x["session_id"]) for x in dev_rows],
    dtype=object,
)


# ============================================================
# 4. TRAIN ONLY ON TRAIN SYSTEMS
#    THEN FREEZE AND PREDICT DEV
# ============================================================

sr = StandardScaler()

XR_tr = sr.fit_transform(
    Xrole_train
)

XR_dev = sr.transform(
    Xrole_dev
)

role_model = Ridge(
    alpha=ALPHA_ROLE,
    solver="lsqr",
)

role_model.fit(
    XR_tr,
    y_train,
)

pred_role_train = role_model.predict(
    XR_tr
)

pred_role_dev = role_model.predict(
    XR_dev
)

resid_train = (
    y_train - pred_role_train
)


def fit_activation_residual(
    X_train,
    X_dev,
):
    sa = StandardScaler()

    XA_tr = sa.fit_transform(
        X_train
    )

    XA_dev = sa.transform(
        X_dev
    )

    model = Ridge(
        alpha=ALPHA_ACT,
        solver="lsqr",
    )

    model.fit(
        XA_tr,
        resid_train,
    )

    return (
        pred_role_dev
        + model.predict(XA_dev)
    )


pred16 = fit_activation_residual(
    X16_train,
    X16_dev,
)

pred24 = fit_activation_residual(
    X24_train,
    X24_dev,
)

pred1624 = fit_activation_residual(
    X1624_train,
    X1624_dev,
)


# ============================================================
# 5. METRICS
# ============================================================

def evaluate(name, pred):

    pos = (
        y_dev > 0
    ).astype(int)

    if len(np.unique(pos)) == 2:
        auc = float(
            roc_auc_score(
                pos,
                pred,
            )
        )
    else:
        auc = float("nan")

    per_system = []

    for sid in sorted(
        set(groups_dev.tolist())
    ):
        ix = np.where(
            groups_dev == sid
        )[0]

        per_system.append({
            "system_id": sid,
            "spearman":
                safe_spearman(
                    y_dev[ix],
                    pred[ix],
                ),
            "mse":
                float(
                    mean_squared_error(
                        y_dev[ix],
                        pred[ix],
                    )
                ),
        })

    valid_spear = [
        x["spearman"]
        for x in per_system
        if not math.isnan(x["spearman"])
    ]

    return {
        "name": name,

        "positive_auc":
            auc,

        "global_spearman":
            safe_spearman(
                y_dev,
                pred,
            ),

        "mean_within_system_spearman":
            float(
                np.mean(valid_spear)
            ),

        "mse":
            float(
                mean_squared_error(
                    y_dev,
                    pred,
                )
            ),

        "pearson":
            float(
                np.corrcoef(
                    y_dev,
                    pred,
                )[0, 1]
            ),

        "predicted_positive_accuracy":
            float(
                np.mean(
                    (pred > 0)
                    ==
                    (y_dev > 0)
                )
            ),

        "per_system":
            per_system,
    }


results = [
    evaluate(
        "role_only",
        pred_role_dev,
    ),
    evaluate(
        "role_plus_resid_L16",
        pred16,
    ),
    evaluate(
        "role_plus_resid_L24",
        pred24,
    ),
    evaluate(
        "role_plus_resid_L16_L24",
        pred1624,
    ),
]


# ============================================================
# 6. SYSTEM-CLUSTER BOOTSTRAP:
#    compare activation models to role-only
# ============================================================

rng = np.random.default_rng(
    20261007
)

systems = sorted(
    set(groups_dev.tolist())
)

base_sqerr = (
    pred_role_dev - y_dev
) ** 2

pred_map = {
    "role_plus_resid_L16":
        pred16,
    "role_plus_resid_L24":
        pred24,
    "role_plus_resid_L16_L24":
        pred1624,
}

for r in results[1:]:

    pred = pred_map[r["name"]]

    act_sqerr = (
        pred - y_dev
    ) ** 2

    # positive value = activation reduced MSE
    system_gain = []

    for sid in systems:
        ix = np.where(
            groups_dev == sid
        )[0]

        system_gain.append(
            float(
                np.mean(
                    base_sqerr[ix]
                    - act_sqerr[ix]
                )
            )
        )

    system_gain = np.asarray(
        system_gain,
        dtype=np.float64,
    )

    boot = []

    for _ in range(20000):
        z = rng.choice(
            system_gain,
            size=len(system_gain),
            replace=True,
        )

        boot.append(
            float(np.mean(z))
        )

    r["incremental_vs_role_only"] = {
        "mean_mse_reduction":
            float(
                np.mean(system_gain)
            ),

        "systems_improved":
            int(
                np.sum(system_gain > 0)
            ),

        "systems_worse":
            int(
                np.sum(system_gain < 0)
            ),

        "bootstrap_95_ci": [
            float(
                np.quantile(
                    boot,
                    0.025,
                )
            ),
            float(
                np.quantile(
                    boot,
                    0.975,
                )
            ),
        ],
    }


# ============================================================
# 7. SAVE + REPORT
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(
    OUT / "dev_delta_L16.npy",
    X16_dev.astype(np.float16),
)

np.save(
    OUT / "dev_delta_L24.npy",
    X24_dev.astype(np.float16),
)

with (
    OUT / "dev_metadata.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for x in dev_rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


clean = []

for r in results:
    x = dict(r)
    x.pop("per_system", None)
    clean.append(x)


summary_out = {
    "train_rows":
        len(train_meta),

    "dev_rows":
        len(dev_rows),

    "dev_systems":
        len(systems),

    "results":
        clean,
}


(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary_out,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 120)
print("EXTERNAL HELD-OUT DEV — ACTIVATION TEACHER TRANSFER")
print("=" * 120)

print(
    f"{'METHOD':30s} "
    f"{'AUC+':>7s} "
    f"{'G-SPR':>8s} "
    f"{'W-SPR':>8s} "
    f"{'PEAR':>8s} "
    f"{'MSE':>10s} "
    f"{'SIGN':>8s}"
)

for r in clean:

    print(
        f"{r['name']:30s} "
        f"{r['positive_auc']:7.3f} "
        f"{r['global_spearman']:8.3f} "
        f"{r['mean_within_system_spearman']:8.3f} "
        f"{r['pearson']:8.3f} "
        f"{r['mse']:10.4f} "
        f"{r['predicted_positive_accuracy']:8.3f}"
    )


print()

for r in clean[1:]:

    d = r[
        "incremental_vs_role_only"
    ]

    print(
        r["name"],
        "MSE_GAIN=",
        f"{d['mean_mse_reduction']:+.5f}",
        "CI95=",
        [
            round(x, 5)
            for x in d["bootstrap_95_ci"]
        ],
        "systems improved/worse=",
        (
            d["systems_improved"],
            d["systems_worse"],
        ),
    )


print()
print(
    "SUMMARY =",
    OUT / "summary.json",
)
