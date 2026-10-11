from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from transformers import AutoTokenizer, AutoModelForCausalLM

from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()

from targets.hierarchical_mas import ROLE_PROMPTS
from v11.role_universe import TRAIN_ROLES


# ============================================================
# PATHS
# ============================================================

QROOT = Path("artifacts/activation_reward_sweep_v1")
DROOT = Path("artifacts/activation_external_dev_v1")
OUT = Path("artifacts/generic_activation_teacher_v1")

MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

# Qwen L24 was selected before external DEV validation.
QWEN_LAYER = 24

# 8 roughly depth-spaced Mistral layers.
MISTRAL_LAYERS = [1, 5, 10, 14, 19, 23, 28, 32]

MAX_LEN = 512
BATCH_PAIRS = 4
RIDGE_ALPHA = 10.0

NEUTRAL_SYSTEM = (
    "You are a helpful assistant. "
    "Answer the user's request normally."
)


# ============================================================
# HELPERS
# ============================================================

def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def safe_spearman(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2:
        return float("nan")

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")

    return float(spearmanr(a, b).statistic)


def role_features(rows):
    role_to_i = {
        r: i for i, r in enumerate(TRAIN_ROLES)
    }

    X = np.zeros(
        (len(rows), len(TRAIN_ROLES) + 1),
        dtype=np.float32,
    )

    for i, row in enumerate(rows):
        roles = [
            str(r)
            for r in row["selected_roles"]
            if str(r) in role_to_i
        ]

        for r in roles:
            X[i, role_to_i[r]] = 1.0

        X[i, -1] = len(roles)

    return X


def percentile_ranks_by_group(pred, groups):
    """
    Convert arbitrary model score scales into [0,1] within-system ranks.
    Highest predicted score -> rank near 1.
    """
    out = np.zeros(len(pred), dtype=np.float64)

    for sid in sorted(set(groups.tolist())):
        ix = np.where(groups == sid)[0]

        r = rankdata(
            pred[ix],
            method="average",
        )

        if len(ix) == 1:
            out[ix] = 0.5
        else:
            out[ix] = (
                (r - 1.0)
                / (len(ix) - 1.0)
            )

    return out


def evaluate_rank(name, score, y, groups):
    per_system_spearman = []
    selected_reward = []
    random_reward = []
    oracle_reward = []
    positive_selected = []

    for sid in sorted(set(groups.tolist())):
        ix = np.where(groups == sid)[0]

        per_system_spearman.append(
            safe_spearman(
                y[ix],
                score[ix],
            )
        )

        j = ix[np.argmax(score[ix])]

        selected_reward.append(
            float(y[j])
        )

        random_reward.append(
            float(np.mean(y[ix]))
        )

        oracle_reward.append(
            float(np.max(y[ix]))
        )

        positive_selected.append(
            float(y[j] > 0)
        )

    valid = [
        x for x in per_system_spearman
        if not math.isnan(x)
    ]

    sel = float(np.mean(selected_reward))
    rnd = float(np.mean(random_reward))
    ora = float(np.mean(oracle_reward))

    denom = ora - rnd

    return {
        "name": name,
        "global_spearman":
            safe_spearman(y, score),

        "mean_within_system_spearman":
            float(np.mean(valid)),

        "selected_mean_reward":
            sel,

        "random_mean_reward":
            rnd,

        "oracle_mean_reward":
            ora,

        "headroom_capture":
            (
                float((sel-rnd)/denom)
                if abs(denom) > 1e-12
                else float("nan")
            ),

        "selected_positive_fraction":
            float(np.mean(positive_selected)),
    }


def crossfit_residual_prediction(
    Xrole,
    Xact,
    y,
    groups,
):
    """
    Role identity explains the easy privileged component first.
    Activation model is trained only on the residual.

    Returned value = activation residual prediction only.
    This isolates model-specific internal signal.
    """
    pred_resid = np.zeros(
        len(y),
        dtype=np.float64,
    )

    gkf = GroupKFold(
        n_splits=min(
            4,
            len(set(groups.tolist())),
        )
    )

    for tr, te in gkf.split(
        Xact,
        y,
        groups=groups,
    ):
        sr = StandardScaler()

        XRtr = sr.fit_transform(
            Xrole[tr]
        )

        role_model = Ridge(
            alpha=RIDGE_ALPHA,
            solver="lsqr",
        )

        role_model.fit(
            XRtr,
            y[tr],
        )

        role_train_pred = role_model.predict(
            XRtr
        )

        residual_train = (
            y[tr]
            - role_train_pred
        )

        sa = StandardScaler()

        XAtr = sa.fit_transform(
            Xact[tr]
        )

        XAte = sa.transform(
            Xact[te]
        )

        act_model = Ridge(
            alpha=RIDGE_ALPHA,
            solver="lsqr",
        )

        act_model.fit(
            XAtr,
            residual_train,
        )

        pred_resid[te] = (
            act_model.predict(XAte)
        )

    return pred_resid


def fit_full_residual_predictor(
    Xrole_train,
    Xact_train,
    y_train,
    Xact_test,
):
    sr = StandardScaler()

    XRtr = sr.fit_transform(
        Xrole_train
    )

    role_model = Ridge(
        alpha=RIDGE_ALPHA,
        solver="lsqr",
    )

    role_model.fit(
        XRtr,
        y_train,
    )

    residual_train = (
        y_train
        - role_model.predict(XRtr)
    )

    sa = StandardScaler()

    XAtr = sa.fit_transform(
        Xact_train
    )

    XAtest = sa.transform(
        Xact_test
    )

    act_model = Ridge(
        alpha=RIDGE_ALPHA,
        solver="lsqr",
    )

    act_model.fit(
        XAtr,
        residual_train,
    )

    return act_model.predict(
        XAtest
    )


# ============================================================
# LOAD EXISTING QWEN FEATURES
# ============================================================

train_meta = read_jsonl(
    QROOT / "metadata.jsonl"
)

dev_meta = read_jsonl(
    DROOT / "dev_metadata.jsonl"
)

qsummary = json.loads(
    (QROOT / "summary.json").read_text()
)

q_layers = list(
    map(int, qsummary["tested_layers"])
)

q_layer_to_i = {
    l: i
    for i, l in enumerate(q_layers)
}

if QWEN_LAYER not in q_layer_to_i:
    raise RuntimeError(
        f"Qwen L{QWEN_LAYER} unavailable"
    )

qtrain_all = np.load(
    QROOT / "role_delta_features.npy"
).astype(np.float32)

qtrain = qtrain_all[
    :,
    q_layer_to_i[QWEN_LAYER],
    :
]

qdev = np.load(
    DROOT / "dev_delta_L24.npy"
).astype(np.float32)

if len(train_meta) != len(qtrain):
    raise RuntimeError("Qwen TRAIN mismatch")

if len(dev_meta) != len(qdev):
    raise RuntimeError("Qwen DEV mismatch")


y_train = np.asarray(
    [float(x["reward"]) for x in train_meta],
    dtype=np.float64,
)

g_train = np.asarray(
    [str(x["session_id"]) for x in train_meta],
    dtype=object,
)

y_dev = np.asarray(
    [float(x["reward"]) for x in dev_meta],
    dtype=np.float64,
)

g_dev = np.asarray(
    [str(x["session_id"]) for x in dev_meta],
    dtype=object,
)

Xrole_train = role_features(train_meta)


print("=" * 100)
print("GENERIC ACTIVATION TEACHER V1")
print("=" * 100)

print("TRAIN rows   =", len(train_meta))
print("TRAIN systems=", len(set(g_train.tolist())))
print("DEV rows     =", len(dev_meta))
print("DEV systems  =", len(set(g_dev.tolist())))


# ============================================================
# MISTRAL ACTIVATION EXTRACTION
# ============================================================

tok = AutoTokenizer.from_pretrained(
    MODEL,
    local_files_only=True,
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    local_files_only=True,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

model.eval()
model.config.use_cache = False

hidden_size = int(
    model.config.hidden_size
)

num_layers = int(
    model.config.num_hidden_layers
)

print("Mistral layers =", num_layers)
print("Mistral hidden =", hidden_size)
print("sweep layers   =", MISTRAL_LAYERS)


def make_text(system_prompt, query):
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": query,
        },
    ]

    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    except Exception:
        # Some Mistral templates reject explicit system role.
        merged = (
            "Instructions:\n"
            + system_prompt
            + "\n\nUser request:\n"
            + query
        )

        return tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": merged,
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )


def extract(rows, label):
    pairs = []

    for ei, row in enumerate(rows):
        for role in row["selected_roles"]:
            if role in ROLE_PROMPTS:
                pairs.append(
                    (
                        ei,
                        role,
                        row["query"],
                    )
                )

    sums = {
        layer: np.zeros(
            (len(rows), hidden_size),
            dtype=np.float32,
        )
        for layer in MISTRAL_LAYERS
    }

    counts = np.zeros(
        len(rows),
        dtype=np.int32,
    )

    with torch.inference_mode():

        for start in range(
            0,
            len(pairs),
            BATCH_PAIRS,
        ):
            batch = pairs[
                start:
                start + BATCH_PAIRS
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

            texts = (
                role_texts
                + neutral_texts
            )

            enc = tok(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
            )

            enc = {
                k: v.cuda()
                for k, v in enc.items()
            }

            last = (
                enc["attention_mask"]
                .sum(dim=1)
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
                device="cuda",
            )

            for layer in MISTRAL_LAYERS:
                h = out.hidden_states[layer]

                v = h[
                    ar,
                    last,
                    :
                ].float().cpu().numpy()

                vr = v[:B]
                vn = v[B:]

                for bi, (
                    ei,
                    _role,
                    _query,
                ) in enumerate(batch):

                    sums[layer][ei] += (
                        vr[bi]
                        - vn[bi]
                    )

            for ei, _, _ in batch:
                counts[ei] += 1

            if (
                start == 0
                or
                (start // BATCH_PAIRS) % 25 == 0
            ):
                print(
                    f"[{label}] "
                    f"{min(start+len(batch),len(pairs))}"
                    f"/{len(pairs)} "
                    f"VRAM="
                    f"{torch.cuda.memory_allocated()/1024**3:.2f}GiB",
                    flush=True,
                )

            del out
            del enc

    if np.any(counts <= 0):
        raise RuntimeError(
            f"{label}: zero-count row"
        )

    return {
        layer:
            sums[layer]
            / counts[:, None]
        for layer in MISTRAL_LAYERS
    }


mtrain = extract(
    train_meta,
    "MISTRAL-TRAIN",
)

mdev = extract(
    dev_meta,
    "MISTRAL-DEV",
)


# ============================================================
# TRAIN-CV: QWEN TEACHER
# ============================================================

qwen_cv_raw = crossfit_residual_prediction(
    Xrole_train,
    qtrain,
    y_train,
    g_train,
)

qwen_cv_rank = percentile_ranks_by_group(
    qwen_cv_raw,
    g_train,
)

qwen_train_result = evaluate_rank(
    "qwen_L24",
    qwen_cv_rank,
    y_train,
    g_train,
)


# ============================================================
# TRAIN-CV: MISTRAL LAYER SWEEP
# ============================================================

mistral_train_results = []
mistral_cv_predictions = {}

for layer in MISTRAL_LAYERS:
    print(
        f"[CV] Mistral layer {layer}",
        flush=True,
    )

    raw = crossfit_residual_prediction(
        Xrole_train,
        mtrain[layer],
        y_train,
        g_train,
    )

    rank = percentile_ranks_by_group(
        raw,
        g_train,
    )

    mistral_cv_predictions[layer] = rank

    result = evaluate_rank(
        f"mistral_L{layer:02d}",
        rank,
        y_train,
        g_train,
    )

    result["layer"] = layer

    mistral_train_results.append(
        result
    )


# Pick layer ONLY from TRAIN CV.
best_m = max(
    mistral_train_results,
    key=lambda x:
        (
            x["mean_within_system_spearman"],
            x["selected_mean_reward"],
        )
)

BEST_M_LAYER = int(
    best_m["layer"]
)


# ============================================================
# TRAIN-CV CONSENSUS
# ============================================================

m_best_train_rank = (
    mistral_cv_predictions[
        BEST_M_LAYER
    ]
)

consensus_train = (
    qwen_cv_rank
    + m_best_train_rank
) / 2.0

# Conservative score:
# high only if both families agree.
agreement_train = (
    consensus_train
    - 0.25
    * np.abs(
        qwen_cv_rank
        - m_best_train_rank
    )
)

consensus_train_result = evaluate_rank(
    "qwen_mistral_mean_rank",
    consensus_train,
    y_train,
    g_train,
)

agreement_train_result = evaluate_rank(
    "qwen_mistral_agreement_rank",
    agreement_train,
    y_train,
    g_train,
)


# ============================================================
# FIT ON ALL TRAIN, FREEZE, EXTERNAL DEV
# ============================================================

qwen_dev_raw = fit_full_residual_predictor(
    Xrole_train,
    qtrain,
    y_train,
    qdev,
)

mistral_dev_raw = fit_full_residual_predictor(
    Xrole_train,
    mtrain[BEST_M_LAYER],
    y_train,
    mdev[BEST_M_LAYER],
)

qwen_dev_rank = percentile_ranks_by_group(
    qwen_dev_raw,
    g_dev,
)

mistral_dev_rank = percentile_ranks_by_group(
    mistral_dev_raw,
    g_dev,
)

consensus_dev = (
    qwen_dev_rank
    + mistral_dev_rank
) / 2.0

agreement_dev = (
    consensus_dev
    - 0.25
    * np.abs(
        qwen_dev_rank
        - mistral_dev_rank
    )
)

dev_results = [
    evaluate_rank(
        "qwen_L24",
        qwen_dev_rank,
        y_dev,
        g_dev,
    ),

    evaluate_rank(
        f"mistral_L{BEST_M_LAYER:02d}",
        mistral_dev_rank,
        y_dev,
        g_dev,
    ),

    evaluate_rank(
        "qwen_mistral_mean_rank",
        consensus_dev,
        y_dev,
        g_dev,
    ),

    evaluate_rank(
        "qwen_mistral_agreement_rank",
        agreement_dev,
        y_dev,
        g_dev,
    ),
]


# ============================================================
# SAVE FEATURES + SCORES
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

for layer in MISTRAL_LAYERS:
    np.save(
        OUT / f"mistral_train_L{layer:02d}.npy",
        mtrain[layer].astype(np.float16),
    )

    np.save(
        OUT / f"mistral_dev_L{layer:02d}.npy",
        mdev[layer].astype(np.float16),
    )


train_scores = []

for i, row in enumerate(train_meta):
    train_scores.append({
        "session_id":
            row["session_id"],

        "candidate_index":
            row["candidate_index"],

        "reward":
            float(y_train[i]),

        "qwen_rank":
            float(qwen_cv_rank[i]),

        "mistral_rank":
            float(m_best_train_rank[i]),

        "consensus_rank":
            float(consensus_train[i]),

        "agreement_rank":
            float(agreement_train[i]),
    })


with (
    OUT / "train_teacher_scores.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for row in train_scores:
        f.write(
            json.dumps(row)
            + "\n"
        )


dev_scores = []

for i, row in enumerate(dev_meta):
    dev_scores.append({
        "session_id":
            row["session_id"],

        "round":
            row["round"],

        "reward":
            float(y_dev[i]),

        "qwen_rank":
            float(qwen_dev_rank[i]),

        "mistral_rank":
            float(mistral_dev_rank[i]),

        "consensus_rank":
            float(consensus_dev[i]),

        "agreement_rank":
            float(agreement_dev[i]),
    })


with (
    OUT / "dev_teacher_scores.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for row in dev_scores:
        f.write(
            json.dumps(row)
            + "\n"
        )


summary = {
    "qwen_model":
        "Qwen/Qwen2.5-1.5B-Instruct",

    "qwen_layer":
        QWEN_LAYER,

    "mistral_model":
        MODEL,

    "mistral_layers_tested":
        MISTRAL_LAYERS,

    "best_mistral_layer_train_only":
        BEST_M_LAYER,

    "train": {
        "qwen":
            qwen_train_result,

        "mistral_layers":
            mistral_train_results,

        "consensus":
            consensus_train_result,

        "agreement":
            agreement_train_result,
    },

    "external_dev": {
        "rows":
            len(dev_meta),

        "systems":
            len(set(g_dev.tolist())),

        "results":
            dev_results,
    },
}


(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 120)
print("TRAIN GROUP-CV")
print("=" * 120)

all_train = [
    qwen_train_result,
    best_m,
    consensus_train_result,
    agreement_train_result,
]

print(
    f"{'METHOD':32s} "
    f"{'G-SPR':>8s} "
    f"{'W-SPR':>8s} "
    f"{'SELECT':>9s} "
    f"{'POS%':>7s} "
    f"{'CAPTURE':>9s}"
)

for r in all_train:
    print(
        f"{r['name']:32s} "
        f"{r['global_spearman']:8.3f} "
        f"{r['mean_within_system_spearman']:8.3f} "
        f"{r['selected_mean_reward']:+9.4f} "
        f"{100*r['selected_positive_fraction']:6.1f}% "
        f"{r['headroom_capture']:9.3f}"
    )


print()
print(
    "BEST MISTRAL LAYER (TRAIN ONLY) =",
    BEST_M_LAYER,
)


print()
print("=" * 120)
print("EXTERNAL 8-SYSTEM DEV")
print("=" * 120)

print(
    f"{'METHOD':32s} "
    f"{'G-SPR':>8s} "
    f"{'W-SPR':>8s} "
    f"{'SELECT':>9s} "
    f"{'POS%':>7s} "
    f"{'CAPTURE':>9s}"
)

for r in dev_results:
    print(
        f"{r['name']:32s} "
        f"{r['global_spearman']:8.3f} "
        f"{r['mean_within_system_spearman']:8.3f} "
        f"{r['selected_mean_reward']:+9.4f} "
        f"{100*r['selected_positive_fraction']:6.1f}% "
        f"{r['headroom_capture']:9.3f}"
    )


print()
print(
    "SUMMARY =",
    OUT / "summary.json",
)

print(
    "TRAIN TEACHER =",
    OUT / "train_teacher_scores.jsonl",
)
