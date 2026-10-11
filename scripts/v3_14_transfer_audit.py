from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/eqs_v3_11e_final_shared_panel")
V14 = Path("artifacts/v14_adaptive_role_v1")
OUT = Path("artifacts/eqs_v3_14_transfer_audit")
OUT.mkdir(parents=True, exist_ok=True)

rows = read_jsonl(ROOT / "state_query_rows.jsonl")

assert len(rows) == 2880

systems = sorted({str(x["session_id"]) for x in rows})
assert len(systems) == 24


# ============================================================
# LOAD PUBLIC STATES
# ============================================================

states = {}

for sr in range(1, 6):
    p = V14 / f"states_train_round_{sr:02d}.jsonl"
    xs = read_jsonl(p)

    assert len(xs) == 24

    for x in xs:
        sid = str(x["session_id"])
        states[(sid, sr)] = PublicBeliefState.from_public_dict(x)

assert len(states) == 120


# ============================================================
# UTILITY TENSOR: system x state x query
# ============================================================

sid_idx = {sid:i for i,sid in enumerate(systems)}

U = np.full((24, 5, 24), np.nan)

for x in rows:
    i = sid_idx[str(x["session_id"])]
    s = int(x["state_round"]) - 1
    q = int(x["query_index"])
    U[i,s,q] = float(x["delta_auc"])

assert np.isfinite(U).all()


# ============================================================
# 1. CROSS-SYSTEM QUERY-RANK CONSISTENCY
# ============================================================

system_mean_utility = U.mean(axis=1)

pair_corr = []

for i in range(24):
    for j in range(i+1,24):
        r = spearmanr(
            system_mean_utility[i],
            system_mean_utility[j]
        ).statistic

        if np.isfinite(r):
            pair_corr.append(float(r))

mean_cross_system_rank_corr = float(np.mean(pair_corr))
median_cross_system_rank_corr = float(np.median(pair_corr))


# ============================================================
# 2. LEAVE-ONE-SYSTEM-OUT GLOBAL TRANSFER
# ============================================================

loo_global = []
random_mean = []

for i in range(24):
    train = np.delete(U, i, axis=0)

    qmean = train.mean(axis=(0,1))
    q = int(np.argmax(qmean))

    loo_global.extend(U[i,:,q].tolist())
    random_mean.extend(U[i].mean(axis=1).tolist())

loo_global = float(np.mean(loo_global))
random_mean = float(np.mean(random_mean))


# ============================================================
# STATE FEATURES
# exact public belief + judge evidence history
# ============================================================

def state_features(st):
    belief = np.array([
        float(st.role_probs[r])
        for r in TRAIN_ROLES
    ])

    hist = list(st.history)

    if not hist:
        last135 = np.zeros(20)
        last134 = np.zeros(20)
        mean135 = np.zeros(20)
        mean134 = np.zeros(20)
    else:
        v135 = np.array([
            [
                float(h.v135.get(r,0.0))
                for r in TRAIN_ROLES
            ]
            for h in hist
        ])

        v134 = np.array([
            [
                float(h.v134.get(r,0.0))
                for r in TRAIN_ROLES
            ]
            for h in hist
        ])

        last135 = v135[-1]
        last134 = v134[-1]
        mean135 = v135.mean(axis=0)
        mean134 = v134.mean(axis=0)

    return np.concatenate([
        belief,
        last135,
        last134,
        mean135,
        mean134,
        [len(hist)],
    ])


SF = {}

for k, st in states.items():
    SF[k] = state_features(st)


# ============================================================
# 3. BELIEF-NEAREST-NEIGHBOR TRANSFER
# ============================================================

knn_results = {}

for K in [1,3,5,10]:

    vals = []

    for test_i, sid in enumerate(systems):

        train_sids = [
            x for x in systems
            if x != sid
        ]

        for sr in range(1,6):

            x = SF[(sid,sr)]

            candidates = []

            for tsid in train_sids:
                for tsr in range(1,6):
                    y = SF[(tsid,tsr)]

                    d = float(
                        np.linalg.norm(x-y)
                    )

                    ti = sid_idx[tsid]

                    candidates.append(
                        (d, U[ti,tsr-1])
                    )

            candidates.sort(
                key=lambda z:z[0]
            )

            pred = np.mean(
                [
                    z[1]
                    for z in candidates[:K]
                ],
                axis=0,
            )

            pick = int(np.argmax(pred))

            vals.append(
                U[test_i,sr-1,pick]
            )

    knn_results[str(K)] = float(
        np.mean(vals)
    )


# ============================================================
# 4. STRICT LOSO TABULAR MODEL
# state features + query one-hot
# ============================================================

tabular_selected = []
tabular_spearman = []
tabular_unique = defaultdict(list)

for test_i, test_sid in enumerate(systems):

    Xtr = []
    ytr = []

    for i, sid in enumerate(systems):
        if sid == test_sid:
            continue

        for sr in range(1,6):
            sf = SF[(sid,sr)]

            for q in range(24):
                onehot = np.zeros(24)
                onehot[q] = 1.0

                Xtr.append(
                    np.concatenate([
                        sf,
                        onehot,
                    ])
                )

                ytr.append(
                    U[i,sr-1,q]
                )

    Xtr = np.asarray(Xtr)
    ytr = np.asarray(ytr)

    model = ExtraTreesRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        max_features=0.7,
        random_state=20261007,
        n_jobs=-1,
    )

    model.fit(Xtr,ytr)

    for sr in range(1,6):
        sf = SF[(test_sid,sr)]

        Xq = []

        for q in range(24):
            onehot = np.zeros(24)
            onehot[q] = 1.0

            Xq.append(
                np.concatenate([
                    sf,
                    onehot,
                ])
            )

        pred = model.predict(
            np.asarray(Xq)
        )

        truth = U[test_i,sr-1]

        pick = int(np.argmax(pred))

        tabular_selected.append(
            float(truth[pick])
        )

        tabular_unique[test_sid].append(
            pick
        )

        rho = spearmanr(
            pred,
            truth
        ).statistic

        if np.isfinite(rho):
            tabular_spearman.append(
                float(rho)
            )


tabular_delta = float(
    np.mean(tabular_selected)
)

tabular_rho = float(
    np.mean(tabular_spearman)
)

tabular_unique_mean = float(
    np.mean([
        len(set(v))
        for v in tabular_unique.values()
    ])
)


summary = {
    "systems": 24,
    "states_per_system": 5,
    "queries": 24,

    "mean_cross_system_query_rank_corr":
        mean_cross_system_rank_corr,

    "median_cross_system_query_rank_corr":
        median_cross_system_rank_corr,

    "random_expected_delta_auc":
        random_mean,

    "loo_global_static_delta_auc":
        loo_global,

    "knn_state_selector_delta_auc":
        knn_results,

    "tabular_state_query_delta_auc":
        tabular_delta,

    "tabular_state_query_spearman":
        tabular_rho,

    "tabular_unique_queries_per_system":
        tabular_unique_mean,

    "contextual_oracle_delta_auc":
        float(
            np.mean(
                np.max(U,axis=2)
            )
        ),

    "per_system_static_oracle_delta_auc":
        float(
            np.mean([
                np.max(
                    U[i].mean(axis=0)
                )
                for i in range(24)
            ])
        ),
}


(OUT/"summary.json").write_text(
    json.dumps(
        summary,
        indent=2
    )
)


print("="*100)
print("V3.14 TRANSFER / FEATURE-SUFFICIENCY AUDIT")
print("="*100)

for k,v in summary.items():
    print(f"{k}: {v}")

print()
print("INTERPRETATION GUIDE")
print(
    "cross-system corr ~0 => query utility is system-specific"
)
print(
    "KNN > random => public state geometry contains transferable signal"
)
print(
    "tabular > random/global => state contains signal but V3.13 representation/training failed"
)
print(
    "tabular ~ random => current public state is insufficient for unseen-system transfer"
)
