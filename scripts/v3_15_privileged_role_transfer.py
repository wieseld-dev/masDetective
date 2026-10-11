from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import ExtraTreesRegressor

from adaptive_v11_active.io import read_jsonl
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/eqs_v3_11e_final_shared_panel")
V14 = Path("artifacts/v14_adaptive_role_v1")
OUT = Path("artifacts/eqs_v3_15_privileged_role_transfer")
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA
# ============================================================

rows = read_jsonl(ROOT / "state_query_rows.jsonl")
assert len(rows) == 2880

systems = sorted({str(x["session_id"]) for x in rows})
assert len(systems) == 24

sid_idx = {sid:i for i,sid in enumerate(systems)}

U = np.full((24,5,24), np.nan)

for x in rows:
    i = sid_idx[str(x["session_id"])]
    s = int(x["state_round"]) - 1
    q = int(x["query_index"])
    U[i,s,q] = float(x["delta_auc"])

assert np.isfinite(U).all()


# ============================================================
# TRUE ROLE SETS — PRIVILEGED DIAGNOSTIC ONLY
# ============================================================

manifest = json.loads(
    (V14 / "manifest_privileged.json").read_text()
)

truth = {}

for x in manifest["systems"]:
    sid = str(x["session_id"])
    if sid in sid_idx:
        truth[sid] = set(map(str, x["roles"]))

assert len(truth) == 24


def role_vec(sid):
    rs = truth[sid]
    return np.asarray(
        [float(r in rs) for r in TRAIN_ROLES],
        dtype=float,
    )


R = np.stack([
    role_vec(sid)
    for sid in systems
])


# ============================================================
# SYSTEM STATIC UTILITY PROFILE
# ============================================================

SU = U.mean(axis=1)  # [system, query]


# ============================================================
# 1. DOES ROLE-SET SIMILARITY EXPLAIN UTILITY SIMILARITY?
# ============================================================

role_jaccards = []
utility_corrs = []

for i in range(24):
    for j in range(i+1,24):

        a = truth[systems[i]]
        b = truth[systems[j]]

        jac = (
            len(a & b) / len(a | b)
            if (a | b)
            else 1.0
        )

        rho = spearmanr(
            SU[i],
            SU[j],
        ).statistic

        if np.isfinite(rho):
            role_jaccards.append(float(jac))
            utility_corrs.append(float(rho))


jac_utility_pearson = float(
    pearsonr(
        role_jaccards,
        utility_corrs,
    ).statistic
)

jac_utility_spearman = float(
    spearmanr(
        role_jaccards,
        utility_corrs,
    ).statistic
)


# ============================================================
# 2. TRUE-ROLE KNN
# ============================================================

def jaccard_distance(a,b):
    sa = set(np.flatnonzero(a))
    sb = set(np.flatnonzero(b))

    if not (sa | sb):
        return 0.0

    return 1.0 - len(sa & sb) / len(sa | sb)


knn = {}

for K in [1,3,5,10]:

    selected = []

    for test_i in range(24):

        neighbors = []

        for j in range(24):
            if j == test_i:
                continue

            d = jaccard_distance(
                R[test_i],
                R[j],
            )

            neighbors.append(
                (d, SU[j])
            )

        neighbors.sort(
            key=lambda z:z[0]
        )

        pred = np.mean(
            [
                z[1]
                for z in neighbors[:K]
            ],
            axis=0,
        )

        q = int(np.argmax(pred))

        selected.extend(
            U[test_i,:,q].tolist()
        )

    knn[str(K)] = float(
        np.mean(selected)
    )


# ============================================================
# 3. STRICT LOSO MODEL: TRUE ROLES + QUERY ID
# ============================================================

selected = []
rank_corrs = []
unique_by_sid = defaultdict(list)

for test_i, test_sid in enumerate(systems):

    Xtr = []
    ytr = []

    for i, sid in enumerate(systems):

        if i == test_i:
            continue

        rv = R[i]

        for sr in range(5):
            for q in range(24):

                qoh = np.zeros(24)
                qoh[q] = 1.0

                # Include state round only as a generic temporal variable.
                round_feat = np.zeros(5)
                round_feat[sr] = 1.0

                Xtr.append(
                    np.concatenate([
                        rv,
                        qoh,
                        round_feat,
                    ])
                )

                ytr.append(
                    U[i,sr,q]
                )

    Xtr = np.asarray(Xtr)
    ytr = np.asarray(ytr)

    model = ExtraTreesRegressor(
        n_estimators=800,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=20261007,
        n_jobs=-1,
    )

    model.fit(
        Xtr,
        ytr,
    )

    for sr in range(5):

        Xq = []

        for q in range(24):

            qoh = np.zeros(24)
            qoh[q] = 1.0

            round_feat = np.zeros(5)
            round_feat[sr] = 1.0

            Xq.append(
                np.concatenate([
                    R[test_i],
                    qoh,
                    round_feat,
                ])
            )

        pred = model.predict(
            np.asarray(Xq)
        )

        true = U[test_i,sr]

        pick = int(
            np.argmax(pred)
        )

        selected.append(
            float(true[pick])
        )

        unique_by_sid[
            test_sid
        ].append(pick)

        rho = spearmanr(
            pred,
            true,
        ).statistic

        if np.isfinite(rho):
            rank_corrs.append(
                float(rho)
            )


true_role_model_delta = float(
    np.mean(selected)
)

true_role_model_rho = float(
    np.mean(rank_corrs)
)

true_role_unique = float(
    np.mean([
        len(set(v))
        for v in unique_by_sid.values()
    ])
)


# ============================================================
# 4. ROLE-ONLY STATIC PROFILE MODEL
# Predict each MAS's 24-query mean utility vector
# from TRUE role vector.
# ============================================================

static_selected = []
static_corrs = []

for test_i in range(24):

    Xtrain = np.delete(
        R,
        test_i,
        axis=0,
    )

    Ytrain = np.delete(
        SU,
        test_i,
        axis=0,
    )

    model = ExtraTreesRegressor(
        n_estimators=1000,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=314159,
        n_jobs=-1,
    )

    model.fit(
        Xtrain,
        Ytrain,
    )

    pred = model.predict(
        R[test_i].reshape(1,-1)
    )[0]

    q = int(
        np.argmax(pred)
    )

    static_selected.extend(
        U[test_i,:,q].tolist()
    )

    rho = spearmanr(
        pred,
        SU[test_i],
    ).statistic

    if np.isfinite(rho):
        static_corrs.append(
            float(rho)
        )


role_static_delta = float(
    np.mean(static_selected)
)

role_static_rho = float(
    np.mean(static_corrs)
)


# ============================================================
# BASELINES
# ============================================================

random_expected = float(
    np.mean(U)
)

per_system_static_oracle = float(
    np.mean([
        np.max(
            SU[i]
        )
        for i in range(24)
    ])
)

contextual_oracle = float(
    np.mean(
        np.max(
            U,
            axis=2,
        )
    )
)


summary = {
    "systems": 24,
    "queries": 24,
    "states_per_system": 5,

    "PRIVILEGED_DIAGNOSTIC_ONLY": True,

    "role_jaccard_vs_utility_rank_pearson":
        jac_utility_pearson,

    "role_jaccard_vs_utility_rank_spearman":
        jac_utility_spearman,

    "true_role_knn_delta_auc":
        knn,

    "true_role_query_model_delta_auc":
        true_role_model_delta,

    "true_role_query_model_spearman":
        true_role_model_rho,

    "true_role_query_model_unique_queries_per_system":
        true_role_unique,

    "true_role_static_profile_delta_auc":
        role_static_delta,

    "true_role_static_profile_spearman":
        role_static_rho,

    "random_expected_delta_auc":
        random_expected,

    "per_system_static_oracle_delta_auc":
        per_system_static_oracle,

    "contextual_oracle_delta_auc":
        contextual_oracle,
}


(OUT / "summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


print("=" * 100)
print("V3.15 PRIVILEGED TRUE-ROLE TRANSFER AUDIT")
print("=" * 100)

for k,v in summary.items():
    print(f"{k}: {v}")

print()
print("DECISION GUIDE")
print(
    "TRUE-ROLE model >> random => roles contain transferable utility signal; public state/belief is the bottleneck."
)
print(
    "TRUE-ROLE model ~ random => query utility depends on latent MAS properties beyond installed roles."
)
print(
    "Jaccard-vs-utility correlation > 0 => similar role sets tend to need similar probes."
)
