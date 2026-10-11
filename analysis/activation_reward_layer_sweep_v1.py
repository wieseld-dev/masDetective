from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from scipy.stats import spearmanr

from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()

from targets.hierarchical_mas import ROLE_PROMPTS
from v11.role_universe import TRAIN_ROLES


MODEL = os.environ.get(
    "ACT_MODEL",
    "Qwen/Qwen2.5-1.5B-Instruct",
)

SRC = Path(
    "artifacts/v14_adaptive_g50_fresh_v3"
)

OUT = Path(
    "artifacts/activation_reward_sweep_v1"
)

GRPO = SRC / "grpo_train_round_01.jsonl"
RESP = SRC / "responses_train_round_01.jsonl"

BATCH_PAIRS = int(
    os.environ.get("ACT_BATCH_PAIRS", "8")
)

MAX_LEN = int(
    os.environ.get("ACT_MAX_LEN", "512")
)

NEUTRAL_SYSTEM = (
    "You are a helpful assistant. "
    "Answer the user's request normally."
)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def safe_spearman(a, b):
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-12:
        return float("nan")
    if np.std(b) < 1e-12:
        return float("nan")
    return float(
        spearmanr(a, b).statistic
    )


def crossfit_predict(X, y, groups):
    unique = sorted(set(groups.tolist()))

    n_splits = min(4, len(unique))

    if n_splits < 2:
        raise RuntimeError(
            "Need at least 2 systems"
        )

    gkf = GroupKFold(
        n_splits=n_splits
    )

    pred = np.zeros(
        len(y),
        dtype=np.float64,
    )

    for tr, te in gkf.split(
        X,
        y,
        groups=groups,
    ):
        scaler = StandardScaler()

        Xtr = scaler.fit_transform(
            X[tr]
        )
        Xte = scaler.transform(
            X[te]
        )

        reg = Ridge(
            alpha=10.0,
            solver="lsqr",
        )

        reg.fit(
            Xtr,
            y[tr],
        )

        pred[te] = reg.predict(
            Xte
        )

    return pred


def evaluate_predictions(
    name,
    pred,
    y,
    groups,
):
    pos = (y > 0).astype(int)

    if len(np.unique(pos)) == 2:
        auc = float(
            roc_auc_score(pos, pred)
        )
    else:
        auc = float("nan")

    pearson = float(
        np.corrcoef(y, pred)[0, 1]
    )

    global_spear = safe_spearman(
        y,
        pred,
    )

    per_system_spear = []
    selected = []
    random_expected = []
    oracle = []

    for sid in sorted(
        set(groups.tolist())
    ):
        ix = np.where(
            groups == sid
        )[0]

        per_system_spear.append(
            safe_spearman(
                y[ix],
                pred[ix],
            )
        )

        selected.append(
            float(
                y[
                    ix[
                        np.argmax(
                            pred[ix]
                        )
                    ]
                ]
            )
        )

        random_expected.append(
            float(
                np.mean(y[ix])
            )
        )

        oracle.append(
            float(
                np.max(y[ix])
            )
        )

    valid_spear = [
        x
        for x in per_system_spear
        if not math.isnan(x)
    ]

    sel = float(np.mean(selected))
    rnd = float(
        np.mean(random_expected)
    )
    ora = float(np.mean(oracle))

    denom = ora - rnd

    capture = (
        (sel - rnd) / denom
        if abs(denom) > 1e-12
        else float("nan")
    )

    return {
        "name": name,
        "positive_auc": auc,
        "pearson_reward": pearson,
        "global_spearman": global_spear,
        "mean_within_system_spearman":
            float(np.mean(valid_spear)),
        "selected_mean_reward": sel,
        "random_mean_reward": rnd,
        "oracle_mean_reward": ora,
        "headroom_capture": capture,
        "selected_positive_fraction":
            float(
                np.mean(
                    np.asarray(selected) > 0
                )
            ),
        "systems":
            len(set(groups.tolist())),
        "rows": len(y),
    }


# ============================================================
# LOAD EXISTING G50 EXPERIENCE
# ============================================================

if not GRPO.exists():
    raise FileNotFoundError(GRPO)

if not RESP.exists():
    raise FileNotFoundError(RESP)

grpo_rows = read_jsonl(GRPO)
resp_rows = read_jsonl(RESP)

resp_by = {
    (
        str(x["session_id"]),
        int(x["candidate_index"]),
    ): x
    for x in resp_rows
}

examples = []

for x in grpo_rows:
    sid = str(x["session_id"])
    idx = int(x["candidate_index"])

    r = resp_by.get(
        (sid, idx)
    )

    if r is None:
        continue

    if (
        str(
            r.get(
                "execution_status",
                ""
            )
        )
        != "ok"
    ):
        continue

    selected = [
        str(z)
        for z in r.get(
            "selected_roles_training_only",
            []
        )
    ]

    selected = [
        z for z in selected
        if z in ROLE_PROMPTS
    ]

    if not selected:
        continue

    q = str(
        x.get(
            "query",
            x.get(
                "completion",
                "",
            ),
        )
    ).strip()

    if not q:
        continue

    examples.append(
        {
            "session_id": sid,
            "candidate_index": idx,
            "query": q,
            "reward":
                float(x["reward"]),
            "selected_roles":
                selected,
        }
    )


print("=" * 90)
print("ACTIVATION REWARD SWEEP V1")
print("=" * 90)
print(
    "source GRPO rows   =",
    len(grpo_rows),
)
print(
    "usable executions =",
    len(examples),
)
print(
    "systems           =",
    len(
        set(
            x["session_id"]
            for x in examples
        )
    ),
)

if len(examples) < 100:
    raise RuntimeError(
        "Too few usable examples"
    )


# ============================================================
# BUILD ROLE-QUERY PAIRS
# ============================================================

pairs = []

for ei, x in enumerate(examples):
    for role in x["selected_roles"]:
        pairs.append(
            (
                ei,
                role,
                x["query"],
            )
        )

print(
    "selected-role/query pairs =",
    len(pairs),
)


# ============================================================
# LOAD WHITE-BOX PROXY
# ============================================================

device = "cuda"

tok = AutoTokenizer.from_pretrained(
    MODEL
)

if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token

tok.padding_side = "right"

print(
    "loading model:",
    MODEL,
    flush=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

model.eval()

if hasattr(
    model.config,
    "use_cache",
):
    model.config.use_cache = False

n_layers = int(
    model.config.num_hidden_layers
)

hidden_size = int(
    model.config.hidden_size
)

layer_ids = sorted(
    set(
        int(round(x))
        for x in np.linspace(
            1,
            n_layers,
            8,
        )
    )
)

print(
    "num layers =",
    n_layers,
)
print(
    "hidden size =",
    hidden_size,
)
print(
    "tested hidden-state indices =",
    layer_ids,
)


# ============================================================
# STORAGE
#
# query_feat:
#   neutral assistant representation
#
# delta_feat:
#   role-conditioned - neutral
#
# We average over the roles that ACTUALLY ROUTED
# in the GPT MAS execution.
# ============================================================

N = len(examples)
L = len(layer_ids)

query_sum = np.zeros(
    (N, L, hidden_size),
    dtype=np.float32,
)

delta_sum = np.zeros(
    (N, L, hidden_size),
    dtype=np.float32,
)

counts = np.zeros(
    N,
    dtype=np.int32,
)


def make_text(
    system_prompt,
    query,
):
    return tok.apply_chat_template(
        [
            {
                "role": "system",
                "content":
                    system_prompt,
            },
            {
                "role": "user",
                "content":
                    query,
            },
        ],
        tokenize=False,
        add_generation_prompt=True,
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
            for _, role, query
            in batch
        ]

        neutral_texts = [
            make_text(
                NEUTRAL_SYSTEM,
                query,
            )
            for _, _, query
            in batch
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
            k: v.to(device)
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
            device=device,
        )

        for lj, layer in enumerate(
            layer_ids
        ):
            h = out.hidden_states[
                layer
            ]

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

                query_sum[
                    example_i,
                    lj,
                ] += vn[bi]

                delta_sum[
                    example_i,
                    lj,
                ] += (
                    vr[bi]
                    - vn[bi]
                )

        for example_i, _, _ in batch:
            counts[example_i] += 1

        if (
            start == 0
            or
            (start // BATCH_PAIRS)
            % 20 == 0
        ):
            print(
                f"[activation] "
                f"{min(start+len(batch),len(pairs))}"
                f"/{len(pairs)} "
                f"alloc="
                f"{torch.cuda.memory_allocated()/1024**3:.2f}GiB",
                flush=True,
            )

        del out
        del enc


if np.any(counts <= 0):
    raise RuntimeError(
        "Found example with zero activation count"
    )

query_feat = (
    query_sum
    / counts[:, None, None]
)

delta_feat = (
    delta_sum
    / counts[:, None, None]
)


# ============================================================
# METADATA + BASELINES
# ============================================================

y = np.asarray(
    [
        x["reward"]
        for x in examples
    ],
    dtype=np.float64,
)

groups = np.asarray(
    [
        x["session_id"]
        for x in examples
    ],
    dtype=object,
)

role_to_i = {
    r: i
    for i, r in enumerate(
        TRAIN_ROLES
    )
}

role_only = np.zeros(
    (
        N,
        len(TRAIN_ROLES) + 1,
    ),
    dtype=np.float32,
)

for i, x in enumerate(examples):
    for r in x["selected_roles"]:
        if r in role_to_i:
            role_only[
                i,
                role_to_i[r],
            ] = 1.0

    role_only[
        i,
        -1,
    ] = len(
        x["selected_roles"]
    )


# ============================================================
# HELD-OUT-BY-SYSTEM ANALYSIS
# ============================================================

results = []

role_pred = crossfit_predict(
    role_only,
    y,
    groups,
)

results.append(
    evaluate_predictions(
        "selected_role_identity_only",
        role_pred,
        y,
        groups,
    )
)


for lj, layer in enumerate(
    layer_ids
):
    print(
        f"[analysis] layer={layer}",
        flush=True,
    )

    Xq = query_feat[
        :,
        lj,
        :,
    ].astype(
        np.float32
    )

    Xd = delta_feat[
        :,
        lj,
        :,
    ].astype(
        np.float32
    )

    pq = crossfit_predict(
        Xq,
        y,
        groups,
    )

    pd = crossfit_predict(
        Xd,
        y,
        groups,
    )

    rq = evaluate_predictions(
        f"query_only_L{layer:02d}",
        pq,
        y,
        groups,
    )

    rq["layer"] = layer
    rq["feature"] = "query_only"

    rd = evaluate_predictions(
        f"role_delta_L{layer:02d}",
        pd,
        y,
        groups,
    )

    rd["layer"] = layer
    rd["feature"] = "role_delta"

    results.extend(
        [rq, rd]
    )


# ============================================================
# SAVE
# ============================================================

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

np.save(
    OUT / "query_features.npy",
    query_feat.astype(
        np.float16
    ),
)

np.save(
    OUT / "role_delta_features.npy",
    delta_feat.astype(
        np.float16
    ),
)

with (
    OUT / "metadata.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:
    for x in examples:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )

best_delta = max(
    [
        x for x in results
        if x.get(
            "feature"
        ) == "role_delta"
    ],
    key=lambda x:
        x[
            "selected_mean_reward"
        ],
)

summary = {
    "model": MODEL,
    "source": str(SRC),
    "rows": N,
    "systems":
        len(set(groups.tolist())),
    "selected_role_query_pairs":
        len(pairs),
    "num_model_layers":
        n_layers,
    "tested_layers":
        layer_ids,
    "best_role_delta_by_selected_reward":
        best_delta,
    "results":
        results,
}

(
    OUT / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 130)
print("HELD-OUT SYSTEM RESULTS")
print("=" * 130)
print(
    f"{'FEATURE':28s} "
    f"{'AUC+':>7s} "
    f"{'G-SPR':>8s} "
    f"{'W-SPR':>8s} "
    f"{'SELECT':>9s} "
    f"{'RANDOM':>9s} "
    f"{'ORACLE':>9s} "
    f"{'CAPTURE':>9s}"
)

for x in results:
    print(
        f"{x['name'][:28]:28s} "
        f"{x['positive_auc']:7.3f} "
        f"{x['global_spearman']:8.3f} "
        f"{x['mean_within_system_spearman']:8.3f} "
        f"{x['selected_mean_reward']:+9.4f} "
        f"{x['random_mean_reward']:+9.4f} "
        f"{x['oracle_mean_reward']:+9.4f} "
        f"{x['headroom_capture']:9.3f}"
    )

print()
print("=" * 90)
print("BEST ROLE-CONDITIONED ACTIVATION LAYER")
print("=" * 90)
print(
    json.dumps(
        best_delta,
        indent=2,
    )
)

print()
print(
    "SUMMARY =",
    OUT / "summary.json",
)
