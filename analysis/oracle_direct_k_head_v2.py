from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OUT = Path("artifacts/oracle_direct_k_head_v2")
OUT.mkdir(parents=True, exist_ok=True)

ROLES = list(TRAIN_ROLES)
ROLE_TO_I = {r:i for i,r in enumerate(ROLES)}

LABELS = [3,4,5,6]
BUDGETS = [6,12]
SEED = 1337


def load_systems():
    groups = defaultdict(lambda: defaultdict(list))
    truths = {}

    files = sorted(
        ROOT.rglob("responses_train_round_*.jsonl")
    )

    for p in files:
        parent = str(p.parent)

        m = re.search(r"round_(\d+)", p.name)
        if not m:
            continue
        rnd = int(m.group(1))

        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)

                sid = (
                    x.get("system_id")
                    or x.get("session_id")
                )

                truth = tuple(sorted(
                    x.get("truth_roles_training_only")
                    or []
                ))

                if (
                    not sid
                    or len(truth) not in LABELS
                ):
                    continue

                selected = list(
                    x.get("selected_roles_training_only")
                    or []
                )

                if any(r not in ROLE_TO_I for r in selected):
                    continue

                key = (parent, sid)

                if key in truths and truths[key] != truth:
                    raise RuntimeError(
                        f"truth mismatch {key}"
                    )

                truths[key] = truth

                ci = x.get("candidate_index")
                try:
                    ci = int(ci)
                except Exception:
                    ci = len(groups[key][rnd])

                v = np.zeros(len(ROLES), dtype=np.float32)

                for r in selected:
                    v[ROLE_TO_I[r]] = 1.0

                groups[key][rnd].append({
                    "candidate_index": ci,
                    "v": v,
                })

    systems = []

    for key, rm in groups.items():
        rounds = sorted(rm)

        if len(rounds) < 2:
            continue

        obs = []
        valid = True

        for rnd in rounds[:2]:
            rows = sorted(
                rm[rnd],
                key=lambda x:x["candidate_index"],
            )

            if len(rows) < 6:
                valid = False
                break

            obs.extend(
                x["v"] for x in rows[:6]
            )

        if not valid:
            continue

        B = np.stack(obs)

        if B.shape != (12, len(ROLES)):
            continue

        systems.append({
            "key": key,
            "k": len(truths[key]),
            "B": B,
        })

    return systems


def oracle_features(B):
    """
    Identity-free features of the TRUE activation matrix.
    No semantic role names are exposed to the classifier.
    """

    n = B.shape[0]

    # How often each hidden role was activated.
    counts = B.sum(axis=0)
    counts_sorted = np.sort(counts)[::-1]

    # Fanout per query.
    fanout = B.sum(axis=1)

    # Cumulative number of unique discovered workers.
    cumulative = []
    novel = []
    seen = np.zeros(B.shape[1], dtype=bool)

    for row in B:
        active = row > 0

        new = active & (~seen)
        novel.append(float(new.sum()))

        seen |= active
        cumulative.append(float(seen.sum()))

    unique_seen = float(seen.sum())

    # Frequency-of-frequency occupancy statistics.
    freq = np.asarray(
        [
            np.sum(counts == j)
            for j in range(n + 1)
        ],
        dtype=np.float32,
    )

    f1 = float(np.sum(counts == 1))
    f2 = float(np.sum(counts == 2))

    # Chao1-style unseen-species estimator.
    if f2 > 0:
        chao = unique_seen + (f1*f1)/(2.0*f2)
    else:
        chao = unique_seen + f1*(f1-1.0)/2.0

    # Pairwise overlap between probe activation sets.
    jaccards = []

    for i in range(n):
        ai = B[i] > 0

        for j in range(i+1, n):
            aj = B[j] > 0

            union = np.sum(ai | aj)

            if union == 0:
                jac = 0.0
            else:
                jac = (
                    np.sum(ai & aj) / union
                )

            jaccards.append(jac)

    jaccards = np.asarray(jaccards)

    jq = np.quantile(
        jaccards,
        [0, .1, .25, .5, .75, .9, 1.0],
    )

    # Role co-activation spectrum.
    C = B.T @ B
    eig = np.linalg.eigvalsh(C)
    eig = np.sort(eig)[::-1]

    # Singular values of centered activation matrix.
    BC = B - B.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(
        BC,
        compute_uv=False,
    )

    svpad = np.zeros(len(ROLES), dtype=np.float32)
    svpad[:len(sv)] = sv

    # Effective rank / spectral entropy.
    e = np.clip(eig, 0, None)
    if e.sum() > 0:
        p = e / e.sum()
        spectral_entropy = float(
            -np.sum(
                p[p > 0] * np.log(p[p > 0])
            )
        )
        effective_rank = float(
            np.exp(spectral_entropy)
        )
    else:
        spectral_entropy = 0.0
        effective_rank = 0.0

    fanout_stats = np.asarray([
        fanout.mean(),
        fanout.std(),
        fanout.min(),
        fanout.max(),
        np.sum(fanout == 1),
        np.sum(fanout == 2),
        np.sum(fanout == 3),
        np.sum(fanout >= 4),
    ], dtype=np.float32)

    occupancy = np.asarray([
        unique_seen,
        f1,
        f2,
        chao,
        float(counts.sum()),
        float(np.mean(counts[counts > 0]))
            if np.any(counts > 0) else 0.0,
        float(np.max(counts)),
        spectral_entropy,
        effective_rank,
    ], dtype=np.float32)

    return np.concatenate([
        counts_sorted,
        np.asarray(cumulative),
        np.asarray(novel),
        freq,
        fanout_stats,
        jq,
        eig,
        svpad,
        occupancy,
    ]).astype(np.float32)


def make_X(systems, idx, budget):
    X = []
    y = []

    for i in idx:
        B = systems[i]["B"][:budget]

        X.append(
            oracle_features(B)
        )
        y.append(
            systems[i]["k"]
        )

    return np.stack(X), np.asarray(y)


def simple_union_pred(systems, idx, budget):
    y = []
    pred = []

    for i in idx:
        B = systems[i]["B"][:budget]

        k_true = systems[i]["k"]
        k_hat = int(
            np.sum(B.sum(axis=0) > 0)
        )

        y.append(k_true)
        pred.append(k_hat)

    return {
        "accuracy": float(
            np.mean(
                np.asarray(y) == np.asarray(pred)
            )
        ),
        "mae": float(
            mean_absolute_error(y,pred)
        ),
    }


def models():
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=5000,
                C=0.25,
                class_weight="balanced",
                random_state=SEED,
            )),
        ]),

        "extra": ExtraTreesClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),

        "rf": RandomForestClassifier(
            n_estimators=1000,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),

        "hist": HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=SEED,
        ),
    }


def evaluate(model,X,y):
    p = model.predict(X)

    out = {
        "accuracy":
            float(accuracy_score(y,p)),
        "mae":
            float(mean_absolute_error(y,p)),
        "confusion":
            confusion_matrix(
                y,p,labels=LABELS
            ).tolist(),
    }

    if hasattr(model,"predict_proba"):
        prob = model.predict_proba(X)
        cls = list(model.classes_)

        truep = [
            prob[i,cls.index(int(y[i]))]
            for i in range(len(y))
        ]

        conf = prob.max(axis=1)
        correct = p == y

        out["mean_p_true"] = float(
            np.mean(truep)
        )

        for t in [0.6,0.7,0.8,0.9]:
            m = conf >= t

            out[f"cov_{t}"] = float(m.mean())

            out[f"acc_{t}"] = (
                float(correct[m].mean())
                if m.any()
                else None
            )

    return out


def main():
    systems = load_systems()

    print("="*100)
    print("ORACLE DIRECT K HEAD V2")
    print("="*100)

    print(f"systems={len(systems)}")
    print(
        "K distribution=",
        dict(sorted(
            Counter(
                x["k"] for x in systems
            ).items()
        ))
    )

    ids = np.arange(len(systems))
    yall = np.asarray([
        x["k"] for x in systems
    ])

    trv, test = train_test_split(
        ids,
        test_size=.20,
        stratify=yall,
        random_state=SEED,
    )

    train, val = train_test_split(
        trv,
        test_size=.25,
        stratify=yall[trv],
        random_state=SEED+1,
    )

    print(
        f"train={len(train)} "
        f"val={len(val)} "
        f"test={len(test)}"
    )

    report = {}

    for budget in [6,12]:
        print()
        print("="*100)
        print(
            f"BUDGET={budget} ORACLE ACTIVATIONS"
        )
        print("="*100)

        ub = simple_union_pred(
            systems,test,budget
        )

        print(
            f"UNION BASELINE: "
            f"Kacc={ub['accuracy']:.4f} "
            f"MAE={ub['mae']:.4f}"
        )

        Xtr,ytr = make_X(
            systems,train,budget
        )
        Xv,yv = make_X(
            systems,val,budget
        )

        cand = []

        for name,m in models().items():
            m.fit(Xtr,ytr)

            trm = evaluate(m,Xtr,ytr)
            vm = evaluate(m,Xv,yv)

            print(
                f"{name:10s} "
                f"train={trm['accuracy']:.3f} "
                f"val={vm['accuracy']:.3f} "
                f"valMAE={vm['mae']:.3f}"
            )

            cand.append((
                vm["accuracy"],
                -vm["mae"],
                name,
            ))

        cand.sort(reverse=True)
        best_name = cand[0][2]

        print(
            f"\nSELECTED={best_name}"
        )

        fit = np.concatenate([
            train,val
        ])

        Xfit,yfit = make_X(
            systems,fit,budget
        )
        Xt,yt = make_X(
            systems,test,budget
        )

        model = models()[best_name]
        model.fit(Xfit,yfit)

        tm = evaluate(model,Xt,yt)

        print()
        print("HELD-OUT TEST")
        print(
            f"Exact K accuracy = "
            f"{tm['accuracy']:.4f}"
        )
        print(
            f"K MAE            = "
            f"{tm['mae']:.4f}"
        )
        print(
            f"Mean P(true K)   = "
            f"{tm.get('mean_p_true')}"
        )

        print(
            "Confusion [3,4,5,6]:"
        )
        for row in tm["confusion"]:
            print(row)

        if "mean_p_true" in tm:
            for t in [0.6,0.7,0.8,0.9]:
                print(
                    f"conf>={t}: "
                    f"coverage={tm[f'cov_{t}']:.3f} "
                    f"accuracy={tm[f'acc_{t}']}"
                )

        report[str(budget)] = {
            "union_baseline": ub,
            "selected_model": best_name,
            "test": tm,
        }

        joblib.dump(
            model,
            OUT/f"k_oracle_budget_{budget}.joblib",
        )

    (OUT/"summary.json").write_text(
        json.dumps(report,indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
