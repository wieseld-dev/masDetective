from __future__ import annotations

import json
import random
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
OUT = Path("artifacts/eqs_v3_17_system_learning_curve")
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
# PUBLIC SENSOR FINGERPRINT
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
    belief = np.asarray([
        float(st.role_probs[r])
        for r in TRAIN_ROLES
    ])

    hist = list(st.history)

    if hist:
        v135 = np.asarray([
            [float(h.v135.get(r,0.0)) for r in TRAIN_ROLES]
            for h in hist
        ])

        v134 = np.asarray([
            [float(h.v134.get(r,0.0)) for r in TRAIN_ROLES]
            for h in hist
        ])

        last135 = v135[-1]
        last134 = v134[-1]

        mean135 = v135.mean(axis=0)
        mean134 = v134.mean(axis=0)

        max135 = v135.max(axis=0)
        max134 = v134.max(axis=0)

    else:
        z = np.zeros(20)

        last135 = z.copy()
        last134 = z.copy()

        mean135 = z.copy()
        mean134 = z.copy()

        max135 = z.copy()
        max134 = z.copy()

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
    k:state_features(v)
    for k,v in states.items()
}


def q_onehot(q):
    x = np.zeros(24)
    x[q] = 1.0
    return x


def feat(sid,sr,q):
    return np.concatenate([
        SF[(sid,sr+1)],
        q_onehot(q),
    ])


# ============================================================
# USE SAME 4 HELD-OUT FOLDS AS V3.13
# ============================================================

v313 = json.loads(
    Path(
        "artifacts/eqs_v3_13_shared_selector_cv/"
        "summary.json"
    ).read_text()
)

fold_systems = v313["fold_systems"]

assert len(fold_systems) == 4
assert all(len(x) == 6 for x in fold_systems)


TRAIN_SIZES = [4,8,12,18]
REPEATS = 5

records = []


for fold,test_list in enumerate(fold_systems):

    test_sids = set(test_list)

    pool = [
        s for s in systems
        if s not in test_sids
    ]

    assert len(pool) == 18

    for n in TRAIN_SIZES:

        for rep in range(REPEATS):

            rng = random.Random(
                20261007
                + fold*10000
                + n*100
                + rep
            )

            if n == 18:
                train_sids = pool[:]
            else:
                train_sids = rng.sample(
                    pool,
                    n,
                )

            Xtr = []
            ytr = []

            for sid in train_sids:
                i = sid_idx[sid]

                for sr in range(5):
                    for q in range(24):

                        Xtr.append(
                            feat(
                                sid,
                                sr,
                                q,
                            )
                        )

                        ytr.append(
                            U[i,sr,q]
                        )

            Xtr = np.asarray(Xtr)
            ytr = np.asarray(ytr)

            model = ExtraTreesRegressor(
                n_estimators=600,
                min_samples_leaf=3,
                max_features=0.7,
                random_state=(
                    314159
                    + fold*10000
                    + n*100
                    + rep
                ),
                n_jobs=-1,
            )

            model.fit(
                Xtr,
                ytr,
            )

            selected = []
            rhos = []
            unique = defaultdict(list)

            for sid in sorted(test_sids):

                i = sid_idx[sid]

                for sr in range(5):

                    Xq = np.asarray([
                        feat(
                            sid,
                            sr,
                            q,
                        )
                        for q in range(24)
                    ])

                    pred = model.predict(Xq)

                    truth = U[i,sr]

                    pick = int(
                        np.argmax(pred)
                    )

                    selected.append(
                        float(
                            truth[pick]
                        )
                    )

                    unique[sid].append(
                        pick
                    )

                    rho = spearmanr(
                        pred,
                        truth,
                    ).statistic

                    if np.isfinite(rho):
                        rhos.append(
                            float(rho)
                        )

            rec = {
                "fold": fold,
                "train_systems": n,
                "repeat": rep,
                "mean_selected_delta_auc":
                    float(np.mean(selected)),
                "mean_state_query_spearman":
                    float(np.mean(rhos)),
                "positive_delta_rate":
                    float(
                        np.mean(
                            np.asarray(selected) > 0
                        )
                    ),
                "mean_unique_queries_per_system":
                    float(
                        np.mean([
                            len(set(v))
                            for v in unique.values()
                        ])
                    ),
            }

            records.append(rec)

            print(
                f"[run] fold={fold} "
                f"n={n:02d} "
                f"rep={rep} "
                f"delta={rec['mean_selected_delta_auc']:+.5f} "
                f"rho={rec['mean_state_query_spearman']:+.4f}",
                flush=True,
            )


# ============================================================
# AGGREGATE
# ============================================================

summary = {
    "experiment":
        "V3.17 strict unseen-system sample-complexity curve",

    "random_expected_delta_auc":
        float(U.mean()),

    "contextual_oracle_delta_auc":
        float(
            np.mean(
                np.max(U,axis=2)
            )
        ),

    "train_sizes": {},
}


for n in TRAIN_SIZES:

    xs = [
        r for r in records
        if r["train_systems"] == n
    ]

    vals = np.asarray([
        r["mean_selected_delta_auc"]
        for r in xs
    ])

    rhos = np.asarray([
        r["mean_state_query_spearman"]
        for r in xs
    ])

    pos = np.asarray([
        r["positive_delta_rate"]
        for r in xs
    ])

    uniq = np.asarray([
        r["mean_unique_queries_per_system"]
        for r in xs
    ])

    summary["train_sizes"][str(n)] = {
        "runs": len(xs),

        "mean_selected_delta_auc":
            float(vals.mean()),

        "std_selected_delta_auc":
            float(vals.std()),

        "mean_state_query_spearman":
            float(rhos.mean()),

        "mean_positive_delta_rate":
            float(pos.mean()),

        "mean_unique_queries_per_system":
            float(uniq.mean()),
    }


(OUT/"summary.json").write_text(
    json.dumps(
        summary,
        indent=2,
    )
)

with (OUT/"runs.jsonl").open("w") as f:
    for r in records:
        f.write(
            json.dumps(r)
            + "\n"
        )


print()
print("="*100)
print("V3.17 LEARNING CURVE")
print("="*100)

print(
    "random_expected_delta_auc:",
    f"{summary['random_expected_delta_auc']:+.6f}"
)

print(
    "contextual_oracle_delta_auc:",
    f"{summary['contextual_oracle_delta_auc']:+.6f}"
)

for n,x in summary["train_sizes"].items():

    print()
    print(f"TRAIN SYSTEMS = {n}")

    for k,v in x.items():
        if isinstance(v,float):
            print(
                f"  {k:38s}: {v:+.6f}"
            )
        else:
            print(
                f"  {k:38s}: {v}"
            )

print()
print("DECISION GUIDE")
print(
    "Clear upward curve from 4 -> 18:"
    " collect substantially more independent MAS."
)
print(
    "Flat near random:"
    " more systems alone is unlikely to fix transfer;"
    " revisit fingerprint/action representation."
)
