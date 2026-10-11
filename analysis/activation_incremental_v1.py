from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr, binomtest

from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/activation_reward_sweep_v1")
OUT = Path("artifacts/activation_incremental_v1")

META = ROOT / "metadata.jsonl"
DELTA = ROOT / "role_delta_features.npy"
SUMMARY = ROOT / "summary.json"

ALPHA_ROLE = 10.0
ALPHA_ACT = 10.0
SEED = 20261007
N_BOOT = 20000


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


def safe_spearman(a, b):
    if len(a) < 2:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(spearmanr(a, b).statistic)


meta = read_jsonl(META)
delta = np.load(DELTA).astype(np.float32)
summary = json.loads(SUMMARY.read_text())

layers = list(map(int, summary["tested_layers"]))
layer_to_idx = {l: i for i, l in enumerate(layers)}

assert len(meta) == delta.shape[0]

y = np.asarray(
    [float(x["reward"]) for x in meta],
    dtype=np.float64,
)

groups = np.asarray(
    [str(x["session_id"]) for x in meta],
    dtype=object,
)

role_to_i = {
    r: i for i, r in enumerate(TRAIN_ROLES)
}

Xrole = np.zeros(
    (len(meta), len(TRAIN_ROLES) + 1),
    dtype=np.float32,
)

for i, x in enumerate(meta):
    roles = [
        str(r)
        for r in x["selected_roles"]
        if str(r) in role_to_i
    ]

    for r in roles:
        Xrole[i, role_to_i[r]] = 1.0

    Xrole[i, -1] = len(roles)


def get_act(layer_set):
    mats = [
        delta[:, layer_to_idx[l], :]
        for l in layer_set
    ]
    return np.concatenate(mats, axis=1)


def crossfit_role_only():
    pred = np.zeros(len(y), dtype=np.float64)

    gkf = GroupKFold(
        n_splits=min(4, len(set(groups.tolist())))
    )

    for tr, te in gkf.split(
        Xrole, y, groups=groups
    ):
        sr = StandardScaler()
        xr_tr = sr.fit_transform(Xrole[tr])
        xr_te = sr.transform(Xrole[te])

        m = Ridge(
            alpha=ALPHA_ROLE,
            solver="lsqr",
        )
        m.fit(xr_tr, y[tr])

        pred[te] = m.predict(xr_te)

    return pred


def crossfit_role_plus_residual(layer_set):
    """
    Stage 1:
        role identity -> reward

    Stage 2:
        activation -> residual reward not explained by role identity

    Test prediction:
        role prediction + activation residual prediction
    """
    Xact = get_act(layer_set)

    pred = np.zeros(len(y), dtype=np.float64)

    gkf = GroupKFold(
        n_splits=min(4, len(set(groups.tolist())))
    )

    for tr, te in gkf.split(
        Xact, y, groups=groups
    ):
        # ----- role baseline -----
        sr = StandardScaler()
        xr_tr = sr.fit_transform(Xrole[tr])
        xr_te = sr.transform(Xrole[te])

        role_model = Ridge(
            alpha=ALPHA_ROLE,
            solver="lsqr",
        )
        role_model.fit(xr_tr, y[tr])

        role_tr = role_model.predict(xr_tr)
        role_te = role_model.predict(xr_te)

        residual_tr = y[tr] - role_tr

        # ----- activation explains residual -----
        sa = StandardScaler()

        xa_tr = sa.fit_transform(
            Xact[tr]
        )
        xa_te = sa.transform(
            Xact[te]
        )

        act_model = Ridge(
            alpha=ALPHA_ACT,
            solver="lsqr",
        )

        act_model.fit(
            xa_tr,
            residual_tr,
        )

        residual_te = act_model.predict(
            xa_te
        )

        pred[te] = (
            role_te
            + residual_te
        )

    return pred


def metrics(name, pred):
    pos = (y > 0).astype(int)

    auc = float(
        roc_auc_score(pos, pred)
    )

    system_rows = []

    for sid in sorted(set(groups.tolist())):
        ix = np.where(groups == sid)[0]

        selected_i = ix[
            np.argmax(pred[ix])
        ]

        system_rows.append({
            "system_id": sid,
            "selected_reward":
                float(y[selected_i]),
            "random_reward":
                float(np.mean(y[ix])),
            "oracle_reward":
                float(np.max(y[ix])),
            "within_spearman":
                safe_spearman(
                    y[ix],
                    pred[ix],
                ),
        })

    selected = np.asarray(
        [
            x["selected_reward"]
            for x in system_rows
        ]
    )

    random_r = np.asarray(
        [
            x["random_reward"]
            for x in system_rows
        ]
    )

    oracle = np.asarray(
        [
            x["oracle_reward"]
            for x in system_rows
        ]
    )

    denom = (
        float(np.mean(oracle))
        - float(np.mean(random_r))
    )

    return {
        "name": name,
        "positive_auc": auc,
        "global_spearman":
            safe_spearman(y, pred),
        "mean_within_system_spearman":
            float(np.nanmean([
                x["within_spearman"]
                for x in system_rows
            ])),
        "selected_mean_reward":
            float(np.mean(selected)),
        "random_mean_reward":
            float(np.mean(random_r)),
        "oracle_mean_reward":
            float(np.mean(oracle)),
        "headroom_capture":
            float(
                (
                    np.mean(selected)
                    - np.mean(random_r)
                )
                / denom
            ),
        "selected_positive_fraction":
            float(np.mean(selected > 0)),
        "system_rows": system_rows,
    }


pred_role = crossfit_role_only()

pred_16 = crossfit_role_plus_residual([16])
pred_24 = crossfit_role_plus_residual([24])
pred_1624 = crossfit_role_plus_residual([16, 24])

results = [
    metrics("role_only", pred_role),
    metrics("role_plus_resid_L16", pred_16),
    metrics("role_plus_resid_L24", pred_24),
    metrics("role_plus_resid_L16_L24", pred_1624),
]


# ============================================================
# PAIRED SYSTEM-LEVEL SIGNIFICANCE VS ROLE-ONLY
# ============================================================

base = {
    x["system_id"]: x["selected_reward"]
    for x in results[0]["system_rows"]
}

rng = np.random.default_rng(SEED)

for r in results[1:]:
    current = {
        x["system_id"]: x["selected_reward"]
        for x in r["system_rows"]
    }

    sids = sorted(base)

    diff = np.asarray(
        [
            current[s] - base[s]
            for s in sids
        ],
        dtype=np.float64,
    )

    boot = np.empty(
        N_BOOT,
        dtype=np.float64,
    )

    for b in range(N_BOOT):
        sample = rng.choice(
            diff,
            size=len(diff),
            replace=True,
        )
        boot[b] = np.mean(sample)

    nz = diff[np.abs(diff) > 1e-12]

    if len(nz):
        signs = int(np.sum(nz > 0))
        sign_p = float(
            binomtest(
                signs,
                len(nz),
                p=0.5,
                alternative="two-sided",
            ).pvalue
        )
    else:
        sign_p = 1.0

    r["incremental_vs_role_only"] = {
        "mean_selected_reward_gain":
            float(np.mean(diff)),
        "median_gain":
            float(np.median(diff)),
        "systems_improved":
            int(np.sum(diff > 0)),
        "systems_worse":
            int(np.sum(diff < 0)),
        "systems_tied":
            int(np.sum(diff == 0)),
        "bootstrap_95_ci": [
            float(np.quantile(boot, 0.025)),
            float(np.quantile(boot, 0.975)),
        ],
        "exact_sign_test_p":
            sign_p,
    }


OUT.mkdir(
    parents=True,
    exist_ok=True,
)

clean_results = []

for r in results:
    c = dict(r)
    c.pop("system_rows", None)
    clean_results.append(c)

final = {
    "rows": len(meta),
    "systems":
        len(set(groups.tolist())),
    "tested_layers": layers,
    "methods": clean_results,
}

(
    OUT / "summary.json"
).write_text(
    json.dumps(
        final,
        indent=2,
    )
    + "\n"
)


print()
print("=" * 125)
print("ACTIVATION INCREMENTAL SIGNAL BEYOND ROLE IDENTITY")
print("=" * 125)

print(
    f"{'METHOD':30s} "
    f"{'AUC+':>7s} "
    f"{'G-SPR':>8s} "
    f"{'W-SPR':>8s} "
    f"{'SELECT':>9s} "
    f"{'POS%':>7s} "
    f"{'CAPTURE':>9s}"
)

for r in clean_results:
    print(
        f"{r['name']:30s} "
        f"{r['positive_auc']:7.3f} "
        f"{r['global_spearman']:8.3f} "
        f"{r['mean_within_system_spearman']:8.3f} "
        f"{r['selected_mean_reward']:+9.4f} "
        f"{100*r['selected_positive_fraction']:6.1f}% "
        f"{r['headroom_capture']:9.3f}"
    )

print()

for r in clean_results[1:]:
    d = r["incremental_vs_role_only"]

    print(
        r["name"],
        "gain=",
        f"{d['mean_selected_reward_gain']:+.4f}",
        "CI95=",
        [
            round(x, 4)
            for x in d["bootstrap_95_ci"]
        ],
        "improved/worse/tied=",
        (
            d["systems_improved"],
            d["systems_worse"],
            d["systems_tied"],
        ),
        "sign_p=",
        round(
            d["exact_sign_test_p"],
            6,
        ),
    )

print()
print(
    "SUMMARY =",
    OUT / "summary.json",
)
