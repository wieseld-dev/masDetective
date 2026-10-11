from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/eqs_v3_11e_final_shared_panel")
V14 = Path("artifacts/v14_adaptive_role_v1")
OUT = Path("artifacts/eqs_v3_16_seen_system_state_transfer")
OUT.mkdir(parents=True, exist_ok=True)

rows = read_jsonl(ROOT / "state_query_rows.jsonl")
assert len(rows) == 2880

systems = sorted({str(x["session_id"]) for x in rows})
assert len(systems) == 24

sid_idx = {sid:i for i,sid in enumerate(systems)}

U = np.full((24,5,24), np.nan)

for x in rows:
    i = sid_idx[str(x["session_id"])]
    sr = int(x["state_round"]) - 1
    q = int(x["query_index"])
    U[i,sr,q] = float(x["delta_auc"])

assert np.isfinite(U).all()


# ============================================================
# PUBLIC SENSOR / BELIEF STATE
# ============================================================

states = {}

for sr in range(1,6):
    xs = read_jsonl(
        V14 / f"states_train_round_{sr:02d}.jsonl"
    )

    assert len(xs) == 24

    for x in xs:
        sid = str(x["session_id"])
        states[(sid,sr)] = \
            PublicBeliefState.from_public_dict(x)

assert len(states) == 120


def state_features(st):
    belief = np.asarray(
        [
            float(st.role_probs[r])
            for r in TRAIN_ROLES
        ],
        dtype=float,
    )

    hist = list(st.history)

    if hist:
        v135 = np.asarray([
            [
                float(h.v135.get(r,0.0))
                for r in TRAIN_ROLES
            ]
            for h in hist
        ])

        v134 = np.asarray([
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
        max135 = v135.max(axis=0)
        max134 = v134.max(axis=0)
    else:
        last135 = np.zeros(20)
        last134 = np.zeros(20)
        mean135 = np.zeros(20)
        mean134 = np.zeros(20)
        max135 = np.zeros(20)
        max134 = np.zeros(20)

    return np.concatenate([
        belief,
        last135,
        last134,
        mean135,
        mean134,
        max135,
        max134,
        [len(hist)],
    ])


SF = {
    k: state_features(st)
    for k,st in states.items()
}


# ============================================================
# FEATURE BUILDERS
# ============================================================

def query_onehot(q):
    x = np.zeros(24)
    x[q] = 1.0
    return x


def system_onehot(i):
    x = np.zeros(24)
    x[i] = 1.0
    return x


def round_onehot(sr):
    x = np.zeros(5)
    x[sr] = 1.0
    return x


# ============================================================
# LEAVE-ONE-STATE-ROUND-OUT
#
# Every MAS appears in train and test.
# Only one of its five attacker states is held out.
#
# Diagnostic only.
# ============================================================

variants = {
    # Can the observable sensor fingerprint identify useful probes?
    "public_fingerprint":
        lambda i,sid,sr,q:
            np.concatenate([
                SF[(sid,sr+1)],
                query_onehot(q),
            ]),

    # Privileged: how much does knowing system identity help?
    "system_id":
        lambda i,sid,sr,q:
            np.concatenate([
                system_onehot(i),
                query_onehot(q),
            ]),

    # Privileged ceiling for system identity + observable fingerprint.
    "system_id_plus_fingerprint":
        lambda i,sid,sr,q:
            np.concatenate([
                system_onehot(i),
                SF[(sid,sr+1)],
                query_onehot(q),
            ]),
}


results = {k: [] for k in variants}
corrs = {k: [] for k in variants}
picks = {
    k: defaultdict(list)
    for k in variants
}


for heldout_sr in range(5):

    print("="*100)
    print(
        f"HELD-OUT STATE ROUND {heldout_sr+1}"
    )
    print("="*100)

    for name, feature_fn in variants.items():

        Xtr = []
        ytr = []

        for i,sid in enumerate(systems):
            for sr in range(5):

                if sr == heldout_sr:
                    continue

                for q in range(24):
                    Xtr.append(
                        feature_fn(
                            i,sid,sr,q
                        )
                    )

                    ytr.append(
                        U[i,sr,q]
                    )

        Xtr = np.asarray(Xtr)
        ytr = np.asarray(ytr)

        model = ExtraTreesRegressor(
            n_estimators=700,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=20261007 + heldout_sr,
            n_jobs=-1,
        )

        model.fit(
            Xtr,
            ytr,
        )

        for i,sid in enumerate(systems):

            Xq = np.asarray([
                feature_fn(
                    i,
                    sid,
                    heldout_sr,
                    q,
                )
                for q in range(24)
            ])

            pred = model.predict(Xq)

            truth = U[i,heldout_sr]

            q = int(np.argmax(pred))

            results[name].append(
                float(truth[q])
            )

            picks[name][sid].append(q)

            rho = spearmanr(
                pred,
                truth,
            ).statistic

            if np.isfinite(rho):
                corrs[name].append(
                    float(rho)
                )


# ============================================================
# SIMPLE SAME-SYSTEM STATIC BASELINE
#
# For heldout state, choose query with highest average utility
# over the OTHER FOUR states of that same system.
# This is privileged diagnostic.
# ============================================================

same_system_static = []

for i in range(24):
    for test_sr in range(5):

        train_srs = [
            sr for sr in range(5)
            if sr != test_sr
        ]

        profile = U[i,train_srs].mean(
            axis=0
        )

        q = int(
            np.argmax(profile)
        )

        same_system_static.append(
            float(U[i,test_sr,q])
        )


summary = {
    "experiment":
        "V3.16 seen-system unseen-state diagnostic",

    "diagnostic_only": True,

    "systems": 24,
    "states_per_system": 5,
    "queries": 24,

    "random_expected_delta_auc":
        float(U.mean()),

    "same_system_four_state_static_delta_auc":
        float(np.mean(
            same_system_static
        )),

    "contextual_oracle_delta_auc":
        float(np.mean(
            np.max(U,axis=2)
        )),

    "variants": {},
}


for name in variants:

    unique_q = [
        len(set(v))
        for v in picks[name].values()
    ]

    summary["variants"][name] = {
        "mean_selected_delta_auc":
            float(np.mean(
                results[name]
            )),

        "mean_state_query_spearman":
            float(np.mean(
                corrs[name]
            )),

        "positive_delta_rate":
            float(np.mean(
                np.asarray(
                    results[name]
                ) > 0
            )),

        "mean_unique_queries_per_system":
            float(np.mean(unique_q)),
    }


(OUT/"summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
    )
)


print()
print("="*100)
print("V3.16 FINAL")
print("="*100)

print(
    "random_expected_delta_auc:",
    f"{summary['random_expected_delta_auc']:+.6f}"
)

print(
    "same_system_four_state_static_delta_auc:",
    f"{summary['same_system_four_state_static_delta_auc']:+.6f}"
)

print(
    "contextual_oracle_delta_auc:",
    f"{summary['contextual_oracle_delta_auc']:+.6f}"
)

for name,x in summary["variants"].items():

    print()
    print(name.upper())

    for k,v in x.items():
        print(
            f"  {k:40s}: {v:.6f}"
        )


print()
print("DECISION")

print(
    "public_fingerprint >> random but V3.14 ~ random"
    " => fingerprint is useful within a MAS but does not transfer"
    " across MAS; increase system diversity."
)

print(
    "system_id >> public_fingerprint"
    " => utility is strongly system-specific and current fingerprint"
    " does not identify that system profile well."
)

print(
    "all variants ~ random"
    " => labels/state relationship itself is poorly learnable;"
    " reconsider utility target or action panel."
)
