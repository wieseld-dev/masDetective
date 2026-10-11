from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
OUT = Path("artifacts/direct_k_head_v1")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 1337
ROLES = list(TRAIN_ROLES)

# Main deployment-oriented budgets:
# 6 = one complete candidate round
# 12 = two complete candidate rounds
BUDGETS = [6, 12]

LABELS = [3, 4, 5, 6]


def get_stage(parent: str):
    m = re.search(r"stage_(\d+)", parent)
    return int(m.group(1)) if m else -1


def load_systems():
    groups = defaultdict(lambda: defaultdict(list))
    truths = {}
    seq_counter = defaultdict(int)

    files = sorted(
        ROOT.rglob("scored_train_round_*.jsonl")
    )

    print(f"[load] scored files={len(files)}")

    for p in files:
        m = re.search(r"round_(\d+)", p.name)
        if not m:
            continue

        rnd = int(m.group(1))
        parent = str(p.parent)

        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                x = json.loads(line)

                sid = (
                    x.get("system_id")
                    or x.get("session_id")
                )

                if not sid:
                    continue

                truth = (
                    x.get("truth_roles")
                    or x.get("truth_roles_training_only")
                    or []
                )

                truth = tuple(sorted(truth))

                if len(truth) not in LABELS:
                    continue

                ev = x.get("evidence")

                if not isinstance(ev, dict):
                    continue

                if not all(r in ev for r in ROLES):
                    continue

                try:
                    z = np.asarray(
                        [float(ev[r]) for r in ROLES],
                        dtype=np.float32,
                    )
                except Exception:
                    continue

                if not np.all(np.isfinite(z)):
                    continue

                key = (parent, sid)

                if key in truths and truths[key] != truth:
                    raise RuntimeError(
                        f"truth mismatch for {key}: "
                        f"{truths[key]} vs {truth}"
                    )

                truths[key] = truth

                ci = x.get("candidate_index")

                if ci is None and isinstance(x.get("action"), dict):
                    ci = x["action"].get("candidate_index")

                fallback = seq_counter[(key, rnd)]
                seq_counter[(key, rnd)] += 1

                try:
                    ci = int(ci)
                except Exception:
                    ci = fallback

                groups[key][rnd].append({
                    "candidate_index": ci,
                    "fallback": fallback,
                    "z": z,
                })

    systems = []

    for key, round_map in groups.items():
        rounds = sorted(round_map)

        # Every K has >=2 rounds according to our audit.
        if len(rounds) < 2:
            continue

        observations = []
        valid = True

        for rnd in rounds[:2]:
            rows = sorted(
                round_map[rnd],
                key=lambda x: (
                    x["candidate_index"],
                    x["fallback"],
                ),
            )

            # Normalize candidate budget:
            # EVERY system contributes exactly 6.
            if len(rows) < 6:
                valid = False
                break

            observations.extend(
                x["z"] for x in rows[:6]
            )

        if not valid:
            continue

        Z = np.stack(observations, axis=0)

        if Z.shape != (12, len(ROLES)):
            continue

        parent, sid = key
        truth = truths[key]

        systems.append({
            "key": key,
            "parent": parent,
            "stage": get_stage(parent),
            "sid": sid,
            "truth": truth,
            "k": len(truth),
            "Z": Z,
        })

    return systems


def topk_mean(Z, k):
    k = min(k, Z.shape[0])
    zz = np.sort(Z, axis=0)
    return zz[-k:].mean(axis=0)


def shape_features(Z):
    """
    Identity-free cardinality features.

    No knowledge of which semantic role owns which score.
    Only score-distribution geometry.
    """
    mx = Z.max(axis=0)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    t2 = topk_mean(Z, 2)
    t3 = topk_mean(Z, 3)

    parts = []

    # Sort destroys role identity deliberately.
    for v in (mx, mu, sd, t2, t3):
        parts.append(
            np.sort(v)[::-1]
        )

    # How many candidate roles look plausibly active?
    thresholds = [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

    parts.append(np.asarray(
        [np.sum(mx > t) for t in thresholds],
        dtype=np.float32,
    ))

    parts.append(np.asarray(
        [np.sum(t2 > t) for t in thresholds],
        dtype=np.float32,
    ))

    # Gaps in ranked role scores.
    smx = np.sort(mx)[::-1]
    st2 = np.sort(t2)[::-1]

    parts.append(np.diff(smx[:12]) * -1.0)
    parts.append(np.diff(st2[:12]) * -1.0)

    # Per-observation geometry.
    row_sorted = np.sort(Z, axis=1)[:, ::-1]

    row_top1 = row_sorted[:, 0]
    row_top2 = row_sorted[:, 1]
    row_gap = row_top1 - row_top2

    row_pos0 = (Z > 0).sum(axis=1)
    row_pos1 = (Z > 1).sum(axis=1)
    row_pos2 = (Z > 2).sum(axis=1)

    for v in (
        row_top1,
        row_top2,
        row_gap,
        row_pos0,
        row_pos1,
        row_pos2,
    ):
        parts.append(np.asarray([
            np.mean(v),
            np.std(v),
            np.min(v),
            np.max(v),
        ], dtype=np.float32))

    return np.concatenate(parts)


def full_features(Z):
    """
    Same shape features + role-aware statistics.
    """
    mx = Z.max(axis=0)
    mu = Z.mean(axis=0)
    sd = Z.std(axis=0)
    t2 = topk_mean(Z, 2)
    t3 = topk_mean(Z, 3)

    q75 = np.quantile(Z, .75, axis=0)

    p0 = (Z > 0).mean(axis=0)
    p1 = (Z > 1).mean(axis=0)
    p2 = (Z > 2).mean(axis=0)

    role_specific = np.concatenate([
        mx,
        mu,
        sd,
        t2,
        t3,
        q75,
        p0,
        p1,
        p2,
    ])

    return np.concatenate([
        role_specific,
        shape_features(Z),
    ])


def make_X(systems, indices, budget, mode):
    feats = []
    ys = []

    fn = (
        shape_features
        if mode == "shape"
        else full_features
    )

    for i in indices:
        s = systems[i]

        # EXACT same observation budget for every K.
        Z = s["Z"][:budget]

        feats.append(fn(Z))
        ys.append(s["k"])

    return np.stack(feats), np.asarray(ys)


def metrics(model, X, y):
    pred = model.predict(X)

    out = {
        "accuracy": float(
            accuracy_score(y, pred)
        ),
        "mae": float(
            mean_absolute_error(y, pred)
        ),
        "confusion": confusion_matrix(
            y,
            pred,
            labels=LABELS,
        ).tolist(),
    }

    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        classes = list(model.classes_)

        true_ps = []
        conf = []
        correct = []

        for yi, pi, pr in zip(y, p, pred):
            true_ps.append(
                float(pi[classes.index(yi)])
            )
            conf.append(float(np.max(pi)))
            correct.append(int(pr == yi))

        out["mean_true_k_probability"] = float(
            np.mean(true_ps)
        )

        for thr in (0.60, 0.70, 0.80, 0.90):
            mask = np.asarray(conf) >= thr

            if mask.any():
                out[f"confidence_{thr:.2f}_coverage"] = float(
                    mask.mean()
                )
                out[f"confidence_{thr:.2f}_accuracy"] = float(
                    np.mean(np.asarray(correct)[mask])
                )
            else:
                out[f"confidence_{thr:.2f}_coverage"] = 0.0
                out[f"confidence_{thr:.2f}_accuracy"] = None

    return out


def stage_baseline(systems, train_idx, test_idx):
    by_stage = defaultdict(list)

    for i in train_idx:
        by_stage[systems[i]["stage"]].append(
            systems[i]["k"]
        )

    global_majority = Counter(
        systems[i]["k"] for i in train_idx
    ).most_common(1)[0][0]

    mapping = {
        stage: Counter(ks).most_common(1)[0][0]
        for stage, ks in by_stage.items()
    }

    truth = []
    pred = []

    for i in test_idx:
        s = systems[i]
        truth.append(s["k"])
        pred.append(
            mapping.get(
                s["stage"],
                global_majority,
            )
        )

    return {
        "accuracy": float(
            accuracy_score(truth, pred)
        ),
        "mapping": mapping,
    }


def make_models():
    return {
        "logreg": Pipeline([
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ]),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=800,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def main():
    systems = load_systems()

    print("=" * 100)
    print("DIRECT EXACT-K HEAD V1")
    print("=" * 100)

    print(f"usable systems={len(systems)}")
    print(
        "K distribution=",
        dict(sorted(
            Counter(s["k"] for s in systems).items()
        ))
    )

    print()
    print("STAGE -> K DISTRIBUTION")

    stage_dist = defaultdict(Counter)

    for s in systems:
        stage_dist[s["stage"]][s["k"]] += 1

    for stage in sorted(stage_dist):
        print(
            f"stage={stage:02d}: "
            f"{dict(sorted(stage_dist[stage].items()))}"
        )

    indices = np.arange(len(systems))
    y = np.asarray([s["k"] for s in systems])

    # 60 / 20 / 20, stratified by K.
    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.25,
        random_state=SEED + 1,
        stratify=y[trainval_idx],
    )

    print()
    print(
        f"split train={len(train_idx)} "
        f"val={len(val_idx)} "
        f"test={len(test_idx)}"
    )

    stage_val = stage_baseline(
        systems,
        train_idx,
        val_idx,
    )

    stage_test = stage_baseline(
        systems,
        train_idx,
        test_idx,
    )

    print()
    print(
        "STAGE-ONLY LEAKAGE BASELINE "
        f"val={stage_val['accuracy']:.3f} "
        f"test={stage_test['accuracy']:.3f}"
    )

    majority_k = Counter(
        y[train_idx]
    ).most_common(1)[0][0]

    majority_acc = float(
        np.mean(y[test_idx] == majority_k)
    )

    print(
        f"MAJORITY baseline: "
        f"K={majority_k}, "
        f"test_acc={majority_acc:.3f}"
    )

    summary = {
        "n_systems": len(systems),
        "k_distribution": dict(Counter(y.tolist())),
        "stage_distribution": {
            str(k): dict(v)
            for k, v in stage_dist.items()
        },
        "stage_only_val_accuracy":
            stage_val["accuracy"],
        "stage_only_test_accuracy":
            stage_test["accuracy"],
        "majority_k": int(majority_k),
        "majority_test_accuracy":
            majority_acc,
        "budgets": {},
    }

    for budget in BUDGETS:
        print()
        print("=" * 100)
        print(
            f"BUDGET={budget} OBSERVATIONS "
            f"({'1 round' if budget == 6 else '2 rounds'})"
        )
        print("=" * 100)

        candidates = []

        for mode in ("shape", "full"):
            Xtr, ytr = make_X(
                systems,
                train_idx,
                budget,
                mode,
            )

            Xva, yva = make_X(
                systems,
                val_idx,
                budget,
                mode,
            )

            for model_name, model in make_models().items():
                model.fit(Xtr, ytr)

                va = metrics(
                    model,
                    Xva,
                    yva,
                )

                tr = metrics(
                    model,
                    Xtr,
                    ytr,
                )

                name = f"{mode}_{model_name}"

                print(
                    f"{name:28s} "
                    f"train={tr['accuracy']:.3f} "
                    f"val={va['accuracy']:.3f} "
                    f"valMAE={va['mae']:.3f}"
                )

                candidates.append({
                    "name": name,
                    "mode": mode,
                    "model_name": model_name,
                    "val": va,
                    "train": tr,
                })

        # Select ONLY using validation.
        candidates.sort(
            key=lambda x: (
                x["val"]["accuracy"],
                -x["val"]["mae"],
            ),
            reverse=True,
        )

        best = candidates[0]

        print()
        print(
            f"SELECTED BY VALIDATION: "
            f"{best['name']}"
        )

        # Refit selected architecture on train + val.
        fit_idx = np.concatenate([
            train_idx,
            val_idx,
        ])

        Xfit, yfit = make_X(
            systems,
            fit_idx,
            budget,
            best["mode"],
        )

        Xte, yte = make_X(
            systems,
            test_idx,
            budget,
            best["mode"],
        )

        final_model = make_models()[
            best["model_name"]
        ]

        final_model.fit(
            Xfit,
            yfit,
        )

        test = metrics(
            final_model,
            Xte,
            yte,
        )

        seen = metrics(
            final_model,
            Xfit,
            yfit,
        )

        print()
        print("FINAL HELD-OUT TEST")
        print(
            f"Exact K accuracy = "
            f"{test['accuracy']:.4f}"
        )
        print(
            f"K MAE            = "
            f"{test['mae']:.4f}"
        )
        print(
            f"Seen fit accuracy= "
            f"{seen['accuracy']:.4f}"
        )

        print()
        print(
            "CONFUSION rows=true K, cols=pred K "
            "[3,4,5,6]"
        )

        for row in test["confusion"]:
            print(row)

        if "mean_true_k_probability" in test:
            print()
            print(
                "Mean P(true K) = "
                f"{test['mean_true_k_probability']:.4f}"
            )

            for thr in (
                0.60,
                0.70,
                0.80,
                0.90,
            ):
                cov = test[
                    f"confidence_{thr:.2f}_coverage"
                ]

                acc = test[
                    f"confidence_{thr:.2f}_accuracy"
                ]

                print(
                    f"conf>={thr:.2f}: "
                    f"coverage={cov:.3f}, "
                    f"accuracy={acc}"
                )

        model_path = (
            OUT /
            f"best_budget_{budget}.joblib"
        )

        joblib.dump({
            "model": final_model,
            "mode": best["mode"],
            "budget": budget,
            "roles": ROLES,
            "labels": LABELS,
        }, model_path)

        summary["budgets"][str(budget)] = {
            "selected_model": best["name"],
            "validation": best["val"],
            "test": test,
            "seen_fit": seen,
            "all_validation_candidates": candidates,
            "model_path": str(model_path),
        }

    (OUT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("SAVED")
    print("=" * 100)
    print(OUT / "summary.json")


if __name__ == "__main__":
    main()
