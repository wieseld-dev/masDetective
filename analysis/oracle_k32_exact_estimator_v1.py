from pathlib import Path
from collections import defaultdict, Counter
import json
import math
import warnings

import numpy as np

from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score, confusion_matrix

from v11.role_universe import TRAIN_ROLES

warnings.filterwarnings("ignore")

SEED = 20260922

ROOT = Path("artifacts/k32_generator_diagnostic")
RESP = ROOT / "responses_train_round_01.jsonl"

SRC_RESP = Path(
    "artifacts/adaptive_v11_curriculum_v2/"
    "stage_10/cycle_02/responses_train_round_01.jsonl"
)

ALL_ROOT = Path("artifacts/adaptive_v11_curriculum_v2")

OUT = Path("artifacts/oracle_k32_exact_estimator_v1")
OUT.mkdir(parents=True, exist_ok=True)


def read_jsonl(p):
    out = []
    with p.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def sid_of(r):
    return r.get("system_id") or r.get("session_id")


def truth_of(r):
    return tuple(sorted(r.get("truth_roles_training_only") or []))


def selected_of(r):
    return set(r.get("selected_roles_training_only") or [])


def group_rows(rows):
    by = defaultdict(list)
    for r in rows:
        sid = sid_of(r)
        if sid:
            by[sid].append(r)
    return by


# =============================================================================
# DATA INTEGRITY
# =============================================================================

rows = read_jsonl(RESP)
by = group_rows(rows)

print("=" * 110)
print("ORACLE K32 EXACT-K ESTIMATOR V1")
print("=" * 110)
print()
print("DIAGNOSTIC DATA INTEGRITY")
print("-------------------------")
print("file:", RESP)
print("rows:", len(rows))
print("systems:", len(by))

ks = []
candidate_counts = Counter()
mode_counts = Counter()
truth_inconsistency = 0
selected_not_truth = 0
candidate_index_bad = 0

for sid, rr in sorted(by.items()):
    rr = sorted(rr, key=lambda x: int(x.get("candidate_index", 0)))

    truths = {truth_of(x) for x in rr}
    if len(truths) != 1:
        truth_inconsistency += 1
        continue

    truth = set(next(iter(truths)))
    k = len(truth)
    ks.append(k)

    candidate_counts[len(rr)] += 1

    modes = {x.get("mode") for x in rr}
    for m in modes:
        mode_counts[m] += 1

    idx = [int(x.get("candidate_index", -1)) for x in rr]
    if len(set(idx)) != len(rr):
        candidate_index_bad += 1

    for x in rr:
        if not selected_of(x).issubset(truth):
            selected_not_truth += 1

print("K distribution:", dict(sorted(Counter(ks).items())))
print("candidate-count distribution:", dict(sorted(candidate_counts.items())))
print("mode/system counts:", dict(mode_counts))
print("truth inconsistencies:", truth_inconsistency)
print("selected outside truth:", selected_not_truth)
print("duplicate candidate-index systems:", candidate_index_bad)

assert truth_inconsistency == 0
assert selected_not_truth == 0
assert all(n == 32 for n in candidate_counts)
assert len(by) == 48
assert len(rows) == 1536

# Compare directly against original Stage-10 source.
if SRC_RESP.exists():
    src = group_rows(read_jsonl(SRC_RESP))

    src_k = {}
    for sid, rr in src.items():
        ts = {truth_of(x) for x in rr}
        if len(ts) == 1:
            src_k[sid] = len(next(iter(ts)))

    overlap = sorted(set(by) & set(src))
    mismatch = []

    for sid in overlap:
        new_truth = {
            truth_of(x) for x in by[sid]
        }
        old_truth = {
            truth_of(x) for x in src[sid]
        }

        if new_truth != old_truth:
            mismatch.append(sid)

    print()
    print("ORIGINAL STAGE-10 CROSS-CHECK")
    print("-----------------------------")
    print("source systems:", len(src))
    print(
        "source K distribution:",
        dict(sorted(Counter(src_k.values()).items()))
    )
    print("system overlap:", len(overlap))
    print("truth mismatches:", len(mismatch))

# Global scan to resolve the earlier K=3..6 vs K=6..8 contradiction.
global_truth = {}

for p in sorted(ALL_ROOT.glob("stage_*/cycle_*/responses_train_round_*.jsonl")):
    try:
        for r in read_jsonl(p):
            sid = sid_of(r)
            t = truth_of(r)
            if not sid or not t:
                continue

            if sid in global_truth and global_truth[sid] != t:
                raise RuntimeError(
                    f"truth changed for {sid}: "
                    f"{global_truth[sid]} vs {t}"
                )

            global_truth[sid] = t
    except Exception as e:
        print("[scan warning]", p, e)

print()
print("GLOBAL CURRICULUM TRUTH AUDIT")
print("-----------------------------")
print("unique systems:", len(global_truth))
print(
    "global K distribution:",
    dict(sorted(Counter(len(t) for t in global_truth.values()).items()))
)


# =============================================================================
# IDENTITY-FREE ORACLE FEATURES
# =============================================================================

ROLE_INDEX = {r: i for i, r in enumerate(TRAIN_ROLES)}
R = len(TRAIN_ROLES)


def entropy_effective(counts):
    total = counts.sum()

    if total <= 0:
        return 0.0, 0.0

    p = counts[counts > 0] / total
    h = float(-(p * np.log(p + 1e-12)).sum())

    return h, float(np.exp(h))


def chao1_bc(counts):
    sobs = int(np.sum(counts > 0))
    f1 = int(np.sum(counts == 1))
    f2 = int(np.sum(counts == 2))

    # Bias-corrected Chao1.
    return float(
        sobs +
        (f1 * (f1 - 1)) /
        (2.0 * (f2 + 1))
    )


def jaccard_summary(B):
    vals = []

    for i in range(len(B)):
        ai = B[i].astype(bool)

        for j in range(i + 1, len(B)):
            bj = B[j].astype(bool)

            u = np.logical_or(ai, bj).sum()
            inter = np.logical_and(ai, bj).sum()

            vals.append(
                1.0 if u == 0 else inter / u
            )

    if not vals:
        return np.zeros(7, dtype=float)

    a = np.asarray(vals, dtype=float)

    return np.asarray([
        a.mean(),
        a.std(),
        np.min(a),
        np.quantile(a, .25),
        np.median(a),
        np.quantile(a, .75),
        np.max(a),
    ])


def features_for(rr):
    rr = sorted(
        rr,
        key=lambda x: int(x.get("candidate_index", 0))
    )

    B = np.zeros((len(rr), R), dtype=float)

    for i, x in enumerate(rr):
        for role in selected_of(x):
            if role in ROLE_INDEX:
                B[i, ROLE_INDEX[role]] = 1.0

    n = len(B)

    counts = B.sum(axis=0)
    fanout = B.sum(axis=1)

    sobs = int(np.sum(counts > 0))

    f = {
        j: int(np.sum(counts == j))
        for j in range(1, 9)
    }

    f_gt8 = int(np.sum(counts > 8))

    total_acts = float(counts.sum())

    ent, effective = entropy_effective(counts)

    chao = chao1_bc(counts)

    # Identity removal: sort frequencies instead of retaining role names.
    sorted_counts = np.sort(counts)[::-1] / max(1, n)

    # Fanout histogram.
    fan_hist = np.asarray([
        np.mean(fanout == j)
        for j in range(0, 9)
    ])

    # Discovery curve in candidate order.
    seen = set()
    discovery = []
    new_per_probe = []

    for x in rr:
        before = len(seen)
        seen |= selected_of(x)

        discovery.append(len(seen) / max(1, R))
        new_per_probe.append(len(seen) - before)

    discovery = np.asarray(discovery, dtype=float)

    # Slopes / saturation.
    d8 = discovery[7] if n >= 8 else discovery[-1]
    d16 = discovery[15] if n >= 16 else discovery[-1]
    d24 = discovery[23] if n >= 24 else discovery[-1]
    d32 = discovery[-1]

    late_new_8 = float(np.sum(new_per_probe[-8:]))
    late_new_16 = float(np.sum(new_per_probe[-16:]))

    # Coactivation spectrum: invariant to permutation of role identities.
    G = B.T @ B

    eig = np.linalg.eigvalsh(G)
    eig = np.sort(np.maximum(eig, 0))[::-1]

    if eig.sum() > 0:
        eig = eig / eig.sum()

    # Row-pattern richness.
    unique_patterns = len({
        tuple(row.astype(int).tolist())
        for row in B
    })

    # Compact occupancy / unseen-species features.
    occ = np.asarray([
        sobs,
        f[1], f[2], f[3], f[4],
        f[5], f[6], f[7], f[8],
        f_gt8,
        total_acts,
        fanout.mean(),
        fanout.std(),
        fanout.min(),
        fanout.max(),
        ent,
        effective,
        chao,
        unique_patterns,
        d8,
        d16,
        d24,
        d32,
        late_new_8,
        late_new_16,
    ], dtype=float)

    full = np.concatenate([
        occ,
        sorted_counts,
        fan_hist,
        discovery,
        jaccard_summary(B),
        eig,
    ])

    return occ, full, {
        "sobs": sobs,
        "chao1": chao,
        "coverage": sobs,
        "total_activations": total_acts,
    }


systems = sorted(by)

X_occ = []
X_full = []
Y = []
meta = []

for sid in systems:
    rr = by[sid]

    t = {
        truth_of(x) for x in rr
    }

    assert len(t) == 1

    truth = next(iter(t))
    k = len(truth)

    xo, xf, m = features_for(rr)

    X_occ.append(xo)
    X_full.append(xf)
    Y.append(k)
    meta.append({
        "system_id": sid,
        "K": k,
        **m,
    })

X_occ = np.asarray(X_occ)
X_full = np.asarray(X_full)
Y = np.asarray(Y)

classes = np.array(sorted(set(Y)))

print()
print("=" * 110)
print("FEATURE DATA")
print("=" * 110)
print("X_occ :", X_occ.shape)
print("X_full:", X_full.shape)
print("Y     :", Counter(Y))


# =============================================================================
# SIMPLE BASELINES
# =============================================================================

union_pred = np.asarray([
    m["sobs"] for m in meta
])

# Evaluate union exactly as the old rule: Khat = number of unique agents observed.
union_acc = accuracy_score(Y, union_pred)
union_mae = mean_absolute_error(Y, union_pred)

# Chao1 -> nearest allowed K class in this fixed diagnostic population.
def nearest_class(v):
    return int(classes[np.argmin(np.abs(classes - v))])

chao_pred = np.asarray([
    nearest_class(m["chao1"])
    for m in meta
])

print()
print("=" * 110)
print("NON-LEARNED BASELINES")
print("=" * 110)
print(
    f"MAJORITY: K={Counter(Y).most_common(1)[0][0]} "
    f"accuracy={Counter(Y).most_common(1)[0][1] / len(Y):.4f}"
)
print(
    f"UNION COUNT: accuracy={union_acc:.4f} "
    f"MAE={union_mae:.4f}"
)
print(
    f"CHAO1 rounded-to-class: "
    f"accuracy={accuracy_score(Y, chao_pred):.4f} "
    f"MAE={mean_absolute_error(Y, chao_pred):.4f}"
)


# =============================================================================
# REPEATED STRATIFIED CV
# No role identity, no stage, no system ID, no trajectory length.
# =============================================================================

models = {
    "occupancy_logreg": (
        X_occ,
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.5,
                max_iter=10000,
                class_weight="balanced",
                random_state=SEED,
            )
        ),
    ),

    "full_logreg": (
        X_full,
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.25,
                max_iter=10000,
                class_weight="balanced",
                random_state=SEED,
            )
        ),
    ),

    "full_extra": (
        X_full,
        ExtraTreesClassifier(
            n_estimators=600,
            max_depth=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=-1,
        ),
    ),

    "full_rf": (
        X_full,
        RandomForestClassifier(
            n_estimators=600,
            max_depth=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    ),
}

cv = RepeatedStratifiedKFold(
    n_splits=4,
    n_repeats=25,
    random_state=SEED,
)

all_results = {}

print()
print("=" * 110)
print("REPEATED STRATIFIED 4-FOLD CV x25")
print("=" * 110)
print("Each system is held out; no response rows from a held-out system enter training.")
print()

for name, (X, base_model) in models.items():
    fold_acc = []
    fold_mae = []
    fold_f1 = []

    all_true = []
    all_pred = []

    for tr, te in cv.split(X, Y):
        model = clone(base_model)
        model.fit(X[tr], Y[tr])

        pred = model.predict(X[te])

        fold_acc.append(
            accuracy_score(Y[te], pred)
        )
        fold_mae.append(
            mean_absolute_error(Y[te], pred)
        )
        fold_f1.append(
            f1_score(
                Y[te],
                pred,
                average="macro",
                labels=classes,
                zero_division=0,
            )
        )

        all_true.extend(Y[te].tolist())
        all_pred.extend(pred.tolist())

    all_true = np.asarray(all_true)
    all_pred = np.asarray(all_pred)

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=classes,
    )

    per_k = {}

    for k in classes:
        mask = all_true == k
        per_k[int(k)] = float(
            np.mean(all_pred[mask] == k)
        )

    result = {
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc)),
        "mae_mean": float(np.mean(fold_mae)),
        "mae_std": float(np.std(fold_mae)),
        "macro_f1_mean": float(np.mean(fold_f1)),
        "macro_f1_std": float(np.std(fold_f1)),
        "per_k_accuracy": per_k,
        "confusion": cm.tolist(),
    }

    all_results[name] = result

    print(name)
    print(
        f"  accuracy = "
        f"{result['accuracy_mean']:.4f} "
        f"+/- {result['accuracy_std']:.4f}"
    )
    print(
        f"  MAE      = "
        f"{result['mae_mean']:.4f} "
        f"+/- {result['mae_std']:.4f}"
    )
    print(
        f"  macro-F1 = "
        f"{result['macro_f1_mean']:.4f} "
        f"+/- {result['macro_f1_std']:.4f}"
    )
    print(
        "  per-K    =",
        {
            k: round(v, 4)
            for k, v in per_k.items()
        }
    )
    print("  confusion rows true", classes.tolist())
    for row in cm:
        print("   ", row.tolist())
    print()


# =============================================================================
# PERMUTATION SANITY TEST
# Fixed full-logreg; confirms whether CV signal exceeds shuffled K labels.
# =============================================================================

print("=" * 110)
print("PERMUTATION SANITY — FULL LOGREG")
print("=" * 110)

real_model = models["full_logreg"][1]

cv_perm = StratifiedKFold(
    n_splits=4,
    shuffle=True,
    random_state=SEED,
)

real_scores = []

for tr, te in cv_perm.split(X_full, Y):
    m = clone(real_model)
    m.fit(X_full[tr], Y[tr])
    real_scores.append(
        accuracy_score(
            Y[te],
            m.predict(X_full[te])
        )
    )

real_score = float(np.mean(real_scores))

rng = np.random.default_rng(SEED)
perm_scores = []

for _ in range(100):
    yp = rng.permutation(Y)

    scores = []

    for tr, te in cv_perm.split(X_full, yp):
        m = clone(real_model)
        m.fit(X_full[tr], yp[tr])
        scores.append(
            accuracy_score(
                yp[te],
                m.predict(X_full[te])
            )
        )

    perm_scores.append(float(np.mean(scores)))

perm_scores = np.asarray(perm_scores)

pvalue = float(
    (1 + np.sum(perm_scores >= real_score)) /
    (1 + len(perm_scores))
)

print("real 4-fold accuracy :", real_score)
print("perm mean accuracy  :", float(perm_scores.mean()))
print("perm max accuracy   :", float(perm_scores.max()))
print("empirical p-value   :", pvalue)


# =============================================================================
# SAVE
# =============================================================================

summary = {
    "systems": len(systems),
    "rows": len(rows),
    "k_distribution": dict(
        sorted(
            (int(k), int(v))
            for k, v in Counter(Y).items()
        )
    ),
    "union_baseline": {
        "accuracy": float(union_acc),
        "mae": float(union_mae),
    },
    "chao1_baseline": {
        "accuracy": float(
            accuracy_score(Y, chao_pred)
        ),
        "mae": float(
            mean_absolute_error(Y, chao_pred)
        ),
    },
    "models": all_results,
    "permutation": {
        "real_accuracy": real_score,
        "perm_mean": float(perm_scores.mean()),
        "perm_max": float(perm_scores.max()),
        "pvalue": pvalue,
    },
}

with (OUT / "summary.json").open("w") as f:
    json.dump(summary, f, indent=2)

with (OUT / "per_system.jsonl").open("w") as f:
    for x in meta:
        f.write(json.dumps(x) + "\n")

print()
print("=" * 110)
print("DONE")
print("=" * 110)
print("summary:", OUT / "summary.json")
