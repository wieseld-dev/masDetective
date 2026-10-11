from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import analysis.train_sensor_rankfirst_v2 as V2


ROOT = Path("artifacts/v137_presence_v1/dataset_v1")
TRAIN_PATH = ROOT / "train_v137.jsonl"
VAL_PATH = ROOT / "dev_v137.jsonl"

V2_OUT = Path("artifacts/sensor_rankfirst_v2")
OUT = Path("artifacts/sensor_rankfirst_v2_diagnostic")

N_FOLDS = 6
LINEAR_C = 0.05


def dump(path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def fit_rank_linear(X, rows, train_idx):
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    Z = scaler.transform(X)

    by_system = defaultdict(list)
    for i in train_idx:
        by_system[str(rows[i]["system_id"])].append(i)

    diffs = []

    for sid, idxs in by_system.items():
        pos = [i for i in idxs if int(rows[i]["label"]) == 1]
        neg = [i for i in idxs if int(rows[i]["label"]) == 0]

        if not pos or not neg:
            continue

        for pi in pos:
            for ni in neg:
                diffs.append(Z[pi] - Z[ni])

    if not diffs:
        raise RuntimeError("No ranking differences constructed")

    Dpos = np.stack(diffs)

    D = np.vstack([Dpos, -Dpos])
    y = np.concatenate([
        np.ones(len(Dpos), dtype=int),
        np.zeros(len(Dpos), dtype=int),
    ])

    clf = LogisticRegression(
        C=LINEAR_C,
        fit_intercept=False,
        class_weight="balanced",
        max_iter=5000,
        solver="lbfgs",
        random_state=0,
    )
    clf.fit(D, y)

    return scaler, clf


def score_linear(scaler, clf, X):
    return clf.decision_function(
        scaler.transform(X)
    )


def rolewise_train_normalization(train_rows, train_scores, eval_rows, eval_scores):
    """
    TRAIN-label-only calibration:
    For each role, use the score distribution of ABSENT TRAIN systems
    as that role's null reference.

      evidence_z = (score - mean_absent_role) / sd_absent_role

    Nothing from evaluation labels/systems enters calibration.
    """
    stats = {}

    all_neg = [
        float(s)
        for r, s in zip(train_rows, train_scores)
        if int(r["label"]) == 0
    ]

    global_mu = float(np.mean(all_neg))
    global_sd = float(np.std(all_neg))
    global_sd = max(global_sd, 1e-6)

    roles = sorted({
        str(x["target_role"])
        for x in train_rows
    })

    for role in roles:
        vals = [
            float(s)
            for r, s in zip(train_rows, train_scores)
            if str(r["target_role"]) == role
            and int(r["label"]) == 0
        ]

        if len(vals) >= 3:
            mu = float(np.mean(vals))
            sd = max(float(np.std(vals)), 1e-6)
        else:
            mu, sd = global_mu, global_sd

        stats[role] = {
            "negative_n": len(vals),
            "negative_mean": mu,
            "negative_sd": sd,
        }

    z = []

    for row, s in zip(eval_rows, eval_scores):
        role = str(row["target_role"])
        st = stats.get(role)

        if st is None:
            mu, sd = global_mu, global_sd
        else:
            mu = st["negative_mean"]
            sd = st["negative_sd"]

        z.append((float(s) - mu) / sd)

    return np.asarray(z, dtype=np.float64), stats


def matched_pair_accuracy(rows, scores):
    groups = defaultdict(list)

    for i, row in enumerate(rows):
        pid = row.get("pair_id")
        if pid is not None:
            groups[str(pid)].append(i)

    wins = []
    margins = []

    for pid, idxs in groups.items():
        if len(idxs) != 2:
            continue

        pos = [i for i in idxs if int(rows[i]["label"]) == 1]
        neg = [i for i in idxs if int(rows[i]["label"]) == 0]

        if len(pos) != 1 or len(neg) != 1:
            continue

        m = float(scores[pos[0]] - scores[neg[0]])
        margins.append(m)

        if m > 0:
            wins.append(1.0)
        elif m < 0:
            wins.append(0.0)
        else:
            wins.append(0.5)

    return {
        "pairs": len(wins),
        "pair_accuracy": float(np.mean(wins)),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
    }


def grouped_t0_oof(rows, X):
    groups = np.asarray([
        str(x["system_id"]) for x in rows
    ])

    unique_systems = np.unique(groups)

    if len(unique_systems) != 36:
        raise RuntimeError(
            f"Expected 36 T0 systems, got {len(unique_systems)}"
        )

    splitter = GroupKFold(n_splits=N_FOLDS)

    dummy_y = np.asarray(
        [int(x["label"]) for x in rows]
    )

    oof = np.full(len(rows), np.nan, dtype=np.float64)
    fold_reports = []

    for fold, (tr, va) in enumerate(
        splitter.split(X, dummy_y, groups),
        1,
    ):
        scaler, clf = fit_rank_linear(
            X, rows, tr
        )

        scores = score_linear(
            scaler, clf, X[va]
        )

        oof[va] = scores

        fold_metric = V2.evaluate(
            [rows[i] for i in va],
            scores,
        )

        fold_reports.append({
            "fold": fold,
            "train_systems": len(set(groups[tr])),
            "heldout_systems": len(set(groups[va])),
            "metrics": fold_metric,
        })

        print(
            f"fold={fold}/{N_FOLDS} "
            f"sysAUC={fold_metric['mean_within_system_auc']:.4f} "
            f"roleAUC={fold_metric['mean_role_auc']:.4f} "
            f"perfect={fold_metric['perfect_order_rate']:.4f}",
            flush=True,
        )

    if np.isnan(oof).any():
        raise RuntimeError("OOF scores incomplete")

    return oof, fold_reports


def load_mlp_ensemble(prefix, X):
    paths = sorted(
        V2_OUT.glob(f"{prefix}_seed*.pt")
    )

    if not paths:
        raise RuntimeError(
            f"No checkpoints for {prefix}"
        )

    all_scores = []

    for p in paths:
        ck = torch.load(
            p,
            map_location="cpu",
            weights_only=False,
        )

        mean = np.asarray(
            ck["scaler_mean"],
            dtype=np.float32,
        )

        scale = np.asarray(
            ck["scaler_scale"],
            dtype=np.float32,
        )

        Z = (
            (X - mean)
            / np.maximum(scale, 1e-12)
        ).astype(np.float32)

        model = V2.RankMLP(X.shape[1])
        model.load_state_dict(ck["state_dict"])
        model.eval()

        with torch.no_grad():
            s = model(
                torch.tensor(Z, dtype=torch.float32)
            ).cpu().numpy()

        all_scores.append(s)

    return np.mean(
        np.stack(all_scores),
        axis=0,
    )


def family_feature_shift(t0_rows, X0, d_rows, XD):
    """
    Descriptive only.
    Measures T0-vs-D mean feature displacement by role,
    standardized using pooled T0 feature SD.
    """
    pooled_sd = np.std(X0, axis=0)
    pooled_sd = np.maximum(pooled_sd, 1e-6)

    shifts = {}

    roles = sorted({
        str(x["target_role"])
        for x in t0_rows
    })

    for role in roles:
        a = np.asarray([
            X0[i]
            for i, x in enumerate(t0_rows)
            if str(x["target_role"]) == role
        ])

        b = np.asarray([
            XD[i]
            for i, x in enumerate(d_rows)
            if str(x["target_role"]) == role
        ])

        if len(a) == 0 or len(b) == 0:
            continue

        delta = (
            np.mean(b, axis=0)
            - np.mean(a, axis=0)
        ) / pooled_sd

        shifts[role] = {
            "standardized_l2_shift":
                float(np.linalg.norm(delta)),
            "mean_abs_standardized_shift":
                float(np.mean(np.abs(delta))),
        }

    return shifts


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(
            f"{OUT} is non-empty; refusing overwrite."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    train = V2.read_jsonl(TRAIN_PATH)
    val = V2.read_jsonl(VAL_PATH)

    t0 = [
        x for x in train
        if str(x["probe_family"]) == "T0"
    ]

    t1 = [
        x for x in train
        if str(x["probe_family"]) == "T1"
    ]

    if len(t0) != 720:
        raise RuntimeError(
            f"Expected T0=720, got {len(t0)}"
        )

    if len(t1) != 190:
        raise RuntimeError(
            f"Expected T1=190, got {len(t1)}"
        )

    if len(val) != 760:
        raise RuntimeError(
            f"Expected VAL=760, got {len(val)}"
        )

    print("=== SENSOR V2 DOMAIN-SHIFT DIAGNOSTIC ===")
    print("T0 :", len(t0))
    print("T1 :", len(t1))
    print("VAL:", len(val))
    print("Internal TEST not loaded.")
    print()

    print("Building geometry...", flush=True)

    X0 = V2.make_matrix(t0, "full")
    X1 = V2.make_matrix(t1, "full")
    XD = V2.make_matrix(val, "full")

    X0_q = V2.make_matrix(t0, "query_only")
    XD_q = V2.make_matrix(val, "query_only")

    print(
        "embedding cache entries:",
        len(V2.embedding_cache),
        flush=True,
    )

    # -------------------------------------------------------
    # A. T0 -> T0 grouped OOF
    # -------------------------------------------------------

    print("\n=== A. T0 -> T0 GROUPED OOF ===")

    t0_oof, folds = grouped_t0_oof(
        t0, X0
    )

    t0_oof_metrics = V2.evaluate(
        t0,
        t0_oof,
    )

    print(json.dumps(
        t0_oof_metrics,
        indent=2,
    ))

    # Query-only same-family control.
    print("\n=== A2. T0 QUERY-ONLY GROUPED OOF ===")

    t0_q_oof, q_folds = grouped_t0_oof(
        t0, X0_q
    )

    t0_q_oof_metrics = V2.evaluate(
        t0,
        t0_q_oof,
    )

    print(json.dumps(
        t0_q_oof_metrics,
        indent=2,
    ))

    # -------------------------------------------------------
    # B. Fit all T0 and transfer to T1 + D
    # -------------------------------------------------------

    print("\n=== B. FIT ALL T0 -> T1 / D ===")

    idx_all = np.arange(len(t0))

    scaler, clf = fit_rank_linear(
        X0,
        t0,
        idx_all,
    )

    t0_fit_scores = score_linear(
        scaler, clf, X0
    )

    t1_scores = score_linear(
        scaler, clf, X1
    )

    d_scores = score_linear(
        scaler, clf, XD
    )

    t0_in_sample = V2.evaluate(
        t0,
        t0_fit_scores,
    )

    t1_matched = matched_pair_accuracy(
        t1,
        t1_scores,
    )

    d_raw = V2.evaluate(
        val,
        d_scores,
    )

    print("T0 IN-SAMPLE:")
    print(json.dumps(t0_in_sample, indent=2))

    print("\nT1 MATCHED:")
    print(json.dumps(t1_matched, indent=2))

    print("\nD RAW:")
    print(json.dumps(d_raw, indent=2))

    # -------------------------------------------------------
    # C. Rolewise null-score normalization
    # -------------------------------------------------------

    print("\n=== C. ROLEWISE TRAIN-ONLY NULL NORMALIZATION ===")

    d_norm_scores, role_stats = rolewise_train_normalization(
        t0,
        t0_fit_scores,
        val,
        d_scores,
    )

    d_normalized = V2.evaluate(
        val,
        d_norm_scores,
    )

    print(json.dumps(
        d_normalized,
        indent=2,
    ))

    # Also test on OOF T0 with fold-independent-ish role normalization
    # only as descriptive reference using full T0 null statistics.
    t0_oof_norm, _ = rolewise_train_normalization(
        t0,
        t0_fit_scores,
        t0,
        t0_oof,
    )

    t0_oof_normalized = V2.evaluate(
        t0,
        t0_oof_norm,
    )

    # -------------------------------------------------------
    # D. Previously trained MLP: TRAIN vs D
    # -------------------------------------------------------

    print("\n=== D. EXISTING MLP FULL: T0 TRAIN vs D VAL ===")

    mlp_t0 = load_mlp_ensemble(
        "mlp_full",
        X0,
    )

    mlp_d = load_mlp_ensemble(
        "mlp_full",
        XD,
    )

    mlp_train_metrics = V2.evaluate(
        t0,
        mlp_t0,
    )

    mlp_val_metrics = V2.evaluate(
        val,
        mlp_d,
    )

    print("MLP T0 TRAIN:")
    print(json.dumps(
        mlp_train_metrics,
        indent=2,
    ))

    print("\nMLP D VAL:")
    print(json.dumps(
        mlp_val_metrics,
        indent=2,
    ))

    # -------------------------------------------------------
    # E. Feature-family shift
    # -------------------------------------------------------

    print("\n=== E. T0 -> D FEATURE SHIFT BY ROLE ===")

    shifts = family_feature_shift(
        t0, X0, val, XD
    )

    for role, x in sorted(
        shifts.items(),
        key=lambda kv:
            kv[1]["standardized_l2_shift"],
        reverse=True,
    ):
        print(
            f"{role:<28} "
            f"L2={x['standardized_l2_shift']:.3f} "
            f"meanAbs={x['mean_abs_standardized_shift']:.3f}"
        )

    # -------------------------------------------------------
    # F. Query-only T0-fit -> D
    # -------------------------------------------------------

    print("\n=== F. QUERY-ONLY T0 -> D ===")

    q_scaler, q_clf = fit_rank_linear(
        X0_q,
        t0,
        idx_all,
    )

    q_d_scores = score_linear(
        q_scaler,
        q_clf,
        XD_q,
    )

    q_d_metrics = V2.evaluate(
        val,
        q_d_scores,
    )

    print(json.dumps(
        q_d_metrics,
        indent=2,
    ))

    summary = {
        "version": "sensor_rankfirst_v2_diagnostic",
        "internal_test_used": False,

        "t0_grouped_oof": t0_oof_metrics,
        "t0_query_only_grouped_oof": t0_q_oof_metrics,

        "t0_in_sample": t0_in_sample,
        "t1_matched_transfer": t1_matched,

        "d_raw_transfer": d_raw,
        "d_rolewise_null_normalized": d_normalized,

        "t0_oof_rolewise_null_normalized":
            t0_oof_normalized,

        "existing_mlp_full_t0_train":
            mlp_train_metrics,

        "existing_mlp_full_d_val":
            mlp_val_metrics,

        "query_only_t0_to_d":
            q_d_metrics,

        "t0_to_d_feature_shift_by_role":
            shifts,

        "rolewise_null_stats":
            role_stats,

        "fold_reports":
            folds,

        "query_only_fold_reports":
            q_folds,
    }

    dump(
        OUT / "summary.json",
        summary,
    )

    np.savez_compressed(
        OUT / "scores.npz",
        t0_oof=t0_oof,
        t0_query_oof=t0_q_oof,
        t0_fit=t0_fit_scores,
        t1=t1_scores,
        d_raw=d_scores,
        d_norm=d_norm_scores,
        mlp_t0=mlp_t0,
        mlp_d=mlp_d,
        query_d=q_d_scores,
    )

    print("\n" + "=" * 95)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 95)

    def line(name, m):
        print(
            f"{name:<34} "
            f"sysAUC={m['mean_within_system_auc']:.4f} "
            f"roleAUC={m['mean_role_auc']:.4f} "
            f"perfect={m['perfect_order_rate']:.4f} "
            f"margin={m['mean_full_separation_margin']:+.4f}"
        )

    line("T0 grouped OOF", t0_oof_metrics)
    line("T0 query-only grouped OOF", t0_q_oof_metrics)
    line("T0 in-sample", t0_in_sample)
    line("D raw transfer", d_raw)
    line("D role-normalized", d_normalized)
    line("MLP T0 train", mlp_train_metrics)
    line("MLP D val", mlp_val_metrics)
    line("Query-only T0 -> D", q_d_metrics)

    print(
        f"{'T1 matched transfer':<34} "
        f"pairAcc={t1_matched['pair_accuracy']:.4f} "
        f"meanMargin={t1_matched['mean_margin']:+.4f}"
    )

    print()
    print("Internal TEST untouched.")
    print("Summary:", OUT / "summary.json")


if __name__ == "__main__":
    main()
