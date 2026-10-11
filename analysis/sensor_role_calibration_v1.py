from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

import analysis.train_sensor_rankfirst_v2 as V2


ROOT = Path("artifacts/v137_presence_v1/dataset_v1")
TRAIN_PATH = ROOT / "train_v137.jsonl"
VAL_PATH = ROOT / "dev_v137.jsonl"

OUT = Path("artifacts/sensor_role_calibration_v1")

N_FOLDS = 6
LINEAR_C = 0.05

# Prevent the old exploding-z problem.
MIN_GLOBAL_SCALE_FRAC = 0.25
ABS_MIN_SCALE = 1e-3


def dump(path: Path, obj):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def auc_safe(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)

    if len(np.unique(y)) < 2:
        return None

    return float(roc_auc_score(y, s))


def fit_rank_linear(X, rows, train_idx):
    scaler = StandardScaler()
    scaler.fit(X[train_idx])
    Z = scaler.transform(X)

    by_system = defaultdict(list)
    for i in train_idx:
        by_system[str(rows[i]["system_id"])].append(i)

    diffs = []

    for _, idxs in by_system.items():
        pos = [i for i in idxs if int(rows[i]["label"]) == 1]
        neg = [i for i in idxs if int(rows[i]["label"]) == 0]

        if not pos or not neg:
            continue

        for pi in pos:
            for ni in neg:
                diffs.append(Z[pi] - Z[ni])

    if not diffs:
        raise RuntimeError("No ranking differences constructed.")

    Dpos = np.stack(diffs)

    D = np.vstack([
        Dpos,
        -Dpos,
    ])

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


def learn_role_calibration(rows, scores):
    """
    Uses TRAIN LABELS ONLY.

    For each role:
      direction = sign(mean_positive - mean_negative)

    Two transformations:
      signed_score = direction * raw_score

      robust_evidence =
          direction * (score - mean_negative_role)
          / robust_role_scale

    robust_role_scale uses ALL training scores for that role,
    with a global SD floor. This prevents the previous
    near-zero-negative-SD numerical explosion.
    """

    scores = np.asarray(scores, dtype=float)

    global_sd = float(np.std(scores))
    global_sd = max(global_sd, ABS_MIN_SCALE)

    roles = sorted({
        str(x["target_role"])
        for x in rows
    })

    stats = {}

    for role in roles:
        idx = [
            i for i, x in enumerate(rows)
            if str(x["target_role"]) == role
        ]

        pos = [
            float(scores[i])
            for i in idx
            if int(rows[i]["label"]) == 1
        ]

        neg = [
            float(scores[i])
            for i in idx
            if int(rows[i]["label"]) == 0
        ]

        if not pos or not neg:
            raise RuntimeError(
                f"Role {role} lacks positive/negative TRAIN examples."
            )

        pos_mean = float(np.mean(pos))
        neg_mean = float(np.mean(neg))

        delta = pos_mean - neg_mean
        direction = 1.0 if delta >= 0 else -1.0

        role_vals = np.asarray(
            [float(scores[i]) for i in idx],
            dtype=float,
        )

        role_sd = float(np.std(role_vals))

        robust_scale = max(
            role_sd,
            MIN_GLOBAL_SCALE_FRAC * global_sd,
            ABS_MIN_SCALE,
        )

        stats[role] = {
            "positive_n": len(pos),
            "negative_n": len(neg),

            "positive_mean": pos_mean,
            "negative_mean": neg_mean,

            "train_delta": float(delta),
            "direction": int(direction),

            "role_sd": role_sd,
            "global_sd": global_sd,
            "robust_scale": robust_scale,

            "train_auc_raw": auc_safe(
                [1] * len(pos) + [0] * len(neg),
                pos + neg,
            ),
        }

    return stats


def apply_role_calibration(rows, scores, stats, mode):
    out = []

    for row, score in zip(rows, scores):
        role = str(row["target_role"])

        if role not in stats:
            raise RuntimeError(
                f"Missing calibration for role={role}"
            )

        st = stats[role]

        direction = float(st["direction"])

        if mode == "sign":
            v = direction * float(score)

        elif mode == "robust":
            v = (
                direction
                * (float(score) - float(st["negative_mean"]))
                / float(st["robust_scale"])
            )

        else:
            raise ValueError(mode)

        out.append(v)

    return np.asarray(out, dtype=np.float64)


def per_role_auc(rows, scores):
    by_role = defaultdict(list)

    for row, score in zip(rows, scores):
        by_role[str(row["target_role"])].append(
            (int(row["label"]), float(score))
        )

    out = {}

    for role, vals in sorted(by_role.items()):
        y = [x[0] for x in vals]
        s = [x[1] for x in vals]

        a = auc_safe(y, s)

        out[role] = {
            "n": len(vals),
            "auc": a,
            "orientation_free_auc":
                None if a is None else max(a, 1.0 - a),
            "observed_direction":
                None if a is None else (
                    1 if a >= 0.5 else -1
                ),
        }

    return out


def matched_pair_report(rows, scores):
    groups = defaultdict(list)

    for i, row in enumerate(rows):
        pid = row.get("pair_id")

        if pid is not None:
            groups[str(pid)].append(i)

    all_wins = []
    all_margins = []

    role_wins = defaultdict(list)
    role_margins = defaultdict(list)

    for pid, idxs in groups.items():
        if len(idxs) != 2:
            continue

        pos = [
            i for i in idxs
            if int(rows[i]["label"]) == 1
        ]

        neg = [
            i for i in idxs
            if int(rows[i]["label"]) == 0
        ]

        if len(pos) != 1 or len(neg) != 1:
            continue

        pi = pos[0]
        ni = neg[0]

        margin = float(scores[pi] - scores[ni])
        role = str(rows[pi]["target_role"])

        if margin > 0:
            win = 1.0
        elif margin < 0:
            win = 0.0
        else:
            win = 0.5

        all_wins.append(win)
        all_margins.append(margin)

        role_wins[role].append(win)
        role_margins[role].append(margin)

    per_role = {}

    for role in sorted(role_wins):
        wins = role_wins[role]
        margins = role_margins[role]

        acc = float(np.mean(wins))
        mean_margin = float(np.mean(margins))

        per_role[role] = {
            "pairs": len(wins),
            "pair_accuracy": acc,
            "mean_margin": mean_margin,
            "observed_direction":
                1 if mean_margin >= 0 else -1,
        }

    return {
        "pairs": len(all_wins),
        "pair_accuracy": float(np.mean(all_wins)),
        "mean_margin": float(np.mean(all_margins)),
        "median_margin": float(np.median(all_margins)),
        "per_role": per_role,
    }


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

    assert len(t0) == 720, len(t0)
    assert len(t1) == 190, len(t1)
    assert len(val) == 760, len(val)

    print("=== ROLE / PROBE CONDITIONING DIAGNOSTIC V1 ===")
    print("T0 :", len(t0))
    print("T1 :", len(t1))
    print("VAL:", len(val))
    print("Internal TEST NOT loaded.")
    print()

    print("Building 20D geometry...", flush=True)

    X0 = V2.make_matrix(t0, "full")
    X1 = V2.make_matrix(t1, "full")
    XD = V2.make_matrix(val, "full")

    print(
        "embedding cache entries:",
        len(V2.embedding_cache),
        flush=True,
    )

    # ============================================================
    # A. STRICT T0 GROUPED OOF WITH TRAIN-FOLD-ONLY CALIBRATION
    # ============================================================

    print("\n=== A. STRICT T0 GROUPED OOF CALIBRATION ===")

    groups = np.asarray([
        str(x["system_id"])
        for x in t0
    ])

    dummy_y = np.asarray([
        int(x["label"])
        for x in t0
    ])

    splitter = GroupKFold(n_splits=N_FOLDS)

    oof_raw = np.full(len(t0), np.nan)
    oof_sign = np.full(len(t0), np.nan)
    oof_robust = np.full(len(t0), np.nan)

    fold_reports = []
    fold_role_directions = defaultdict(list)

    for fold, (tr, va) in enumerate(
        splitter.split(X0, dummy_y, groups),
        1,
    ):
        scaler, clf = fit_rank_linear(
            X0,
            t0,
            tr,
        )

        tr_scores = score_linear(
            scaler,
            clf,
            X0[tr],
        )

        va_scores = score_linear(
            scaler,
            clf,
            X0[va],
        )

        tr_rows = [t0[i] for i in tr]
        va_rows = [t0[i] for i in va]

        cal = learn_role_calibration(
            tr_rows,
            tr_scores,
        )

        va_sign = apply_role_calibration(
            va_rows,
            va_scores,
            cal,
            "sign",
        )

        va_robust = apply_role_calibration(
            va_rows,
            va_scores,
            cal,
            "robust",
        )

        oof_raw[va] = va_scores
        oof_sign[va] = va_sign
        oof_robust[va] = va_robust

        for role, st in cal.items():
            fold_role_directions[role].append(
                int(st["direction"])
            )

        raw_m = V2.evaluate(
            va_rows,
            va_scores,
        )

        sign_m = V2.evaluate(
            va_rows,
            va_sign,
        )

        robust_m = V2.evaluate(
            va_rows,
            va_robust,
        )

        fold_reports.append({
            "fold": fold,
            "train_systems":
                len(set(groups[tr])),
            "heldout_systems":
                len(set(groups[va])),

            "raw": raw_m,
            "sign": sign_m,
            "robust": robust_m,
        })

        print(
            f"fold={fold}/{N_FOLDS} "
            f"raw={raw_m['mean_within_system_auc']:.4f} "
            f"sign={sign_m['mean_within_system_auc']:.4f} "
            f"robust={robust_m['mean_within_system_auc']:.4f}",
            flush=True,
        )

    if (
        np.isnan(oof_raw).any()
        or np.isnan(oof_sign).any()
        or np.isnan(oof_robust).any()
    ):
        raise RuntimeError("OOF scores incomplete.")

    t0_raw_metrics = V2.evaluate(
        t0,
        oof_raw,
    )

    t0_sign_metrics = V2.evaluate(
        t0,
        oof_sign,
    )

    t0_robust_metrics = V2.evaluate(
        t0,
        oof_robust,
    )

    direction_stability = {}

    for role, signs in sorted(
        fold_role_directions.items()
    ):
        plus = sum(x > 0 for x in signs)
        minus = sum(x < 0 for x in signs)

        direction_stability[role] = {
            "fold_directions": signs,
            "positive_folds": plus,
            "negative_folds": minus,
            "consistency":
                max(plus, minus) / len(signs),
            "majority_direction":
                1 if plus >= minus else -1,
        }

    # ============================================================
    # B. FIT ALL T0 + LEARN CALIBRATION ON T0 ONLY
    # ============================================================

    print("\n=== B. FIT ALL T0 / FREEZE CALIBRATION ===")

    idx_all = np.arange(len(t0))

    scaler, clf = fit_rank_linear(
        X0,
        t0,
        idx_all,
    )

    t0_fit_scores = score_linear(
        scaler,
        clf,
        X0,
    )

    t1_raw = score_linear(
        scaler,
        clf,
        X1,
    )

    d_raw = score_linear(
        scaler,
        clf,
        XD,
    )

    full_cal = learn_role_calibration(
        t0,
        t0_fit_scores,
    )

    t1_sign = apply_role_calibration(
        t1,
        t1_raw,
        full_cal,
        "sign",
    )

    t1_robust = apply_role_calibration(
        t1,
        t1_raw,
        full_cal,
        "robust",
    )

    d_sign = apply_role_calibration(
        val,
        d_raw,
        full_cal,
        "sign",
    )

    d_robust = apply_role_calibration(
        val,
        d_raw,
        full_cal,
        "robust",
    )

    d_raw_metrics = V2.evaluate(
        val,
        d_raw,
    )

    d_sign_metrics = V2.evaluate(
        val,
        d_sign,
    )

    d_robust_metrics = V2.evaluate(
        val,
        d_robust,
    )

    # ============================================================
    # C. T1 MATCHED PAIRS
    # ============================================================

    print("\n=== C. T1 MATCHED PAIRS ===")

    t1_raw_report = matched_pair_report(
        t1,
        t1_raw,
    )

    t1_sign_report = matched_pair_report(
        t1,
        t1_sign,
    )

    t1_robust_report = matched_pair_report(
        t1,
        t1_robust,
    )

    # ============================================================
    # D. DIRECTION TRANSFER DIAGNOSTIC
    # Eval labels are used ONLY for diagnosis below.
    # They do NOT affect calibration/predictions.
    # ============================================================

    print("\n=== D. DIRECTION TRANSFER DIAGNOSTIC ===")

    t0_oof_role_auc = per_role_auc(
        t0,
        oof_raw,
    )

    d_role_auc = per_role_auc(
        val,
        d_raw,
    )

    direction_transfer = {}

    for role in sorted(full_cal):
        train_direction = int(
            full_cal[role]["direction"]
        )

        t0_oof_direction = (
            t0_oof_role_auc
            .get(role, {})
            .get("observed_direction")
        )

        d_direction = (
            d_role_auc
            .get(role, {})
            .get("observed_direction")
        )

        t1_direction = (
            t1_raw_report["per_role"]
            .get(role, {})
            .get("observed_direction")
        )

        direction_transfer[role] = {
            "t0_train_learned_direction":
                train_direction,

            "t0_oof_observed_direction":
                t0_oof_direction,

            "d_observed_direction":
                d_direction,

            "t1_observed_direction":
                t1_direction,

            "train_matches_t0_oof":
                None if t0_oof_direction is None
                else train_direction == t0_oof_direction,

            "train_matches_d":
                None if d_direction is None
                else train_direction == d_direction,

            "train_matches_t1":
                None if t1_direction is None
                else train_direction == t1_direction,
        }

    def agreement(field):
        vals = [
            x[field]
            for x in direction_transfer.values()
            if x[field] is not None
        ]

        if not vals:
            return None

        return float(np.mean(vals))

    agreement_summary = {
        "train_vs_t0_oof":
            agreement("train_matches_t0_oof"),

        "train_vs_d":
            agreement("train_matches_d"),

        "train_vs_t1":
            agreement("train_matches_t1"),
    }

    # ============================================================
    # SUMMARY
    # ============================================================

    summary = {
        "version":
            "sensor_role_calibration_v1",

        "internal_test_used":
            False,

        "data": {
            "t0_rows": len(t0),
            "t0_systems":
                len(set(x["system_id"] for x in t0)),
            "t1_rows": len(t1),
            "t1_pairs": len({
                x.get("pair_id")
                for x in t1
                if x.get("pair_id") is not None
            }),
            "val_rows": len(val),
            "val_systems":
                len(set(x["system_id"] for x in val)),
        },

        "calibration": {
            "direction_source":
                "TRAIN labels only",

            "direction_rule":
                "sign(mean_positive_score - mean_negative_score)",

            "robust_scale":
                "max(role_all_score_sd, "
                "0.25*global_score_sd, 1e-3)",
        },

        "t0_grouped_oof": {
            "raw": t0_raw_metrics,
            "sign_calibrated":
                t0_sign_metrics,
            "robust_role_calibrated":
                t0_robust_metrics,
        },

        "direction_stability_across_folds":
            direction_stability,

        "t0_full_calibration_stats":
            full_cal,

        "d_transfer": {
            "raw":
                d_raw_metrics,
            "t0_sign_calibrated":
                d_sign_metrics,
            "t0_robust_calibrated":
                d_robust_metrics,
        },

        "t1_matched_transfer": {
            "raw":
                t1_raw_report,
            "t0_sign_calibrated":
                t1_sign_report,
            "t0_robust_calibrated":
                t1_robust_report,
        },

        "per_role_auc": {
            "t0_oof_raw":
                t0_oof_role_auc,
            "d_raw":
                d_role_auc,
        },

        "direction_transfer":
            direction_transfer,

        "direction_agreement_summary":
            agreement_summary,

        "fold_reports":
            fold_reports,
    }

    dump(
        OUT / "summary.json",
        summary,
    )

    np.savez_compressed(
        OUT / "scores.npz",

        t0_oof_raw=oof_raw,
        t0_oof_sign=oof_sign,
        t0_oof_robust=oof_robust,

        t1_raw=t1_raw,
        t1_sign=t1_sign,
        t1_robust=t1_robust,

        d_raw=d_raw,
        d_sign=d_sign,
        d_robust=d_robust,
    )

    print("\n" + "=" * 100)
    print("FINAL ROLE-CONDITIONING SUMMARY")
    print("=" * 100)

    def line(name, m):
        print(
            f"{name:<34} "
            f"sysAUC={m['mean_within_system_auc']:.4f} "
            f"roleAUC={m['mean_role_auc']:.4f} "
            f"perfect={m['perfect_order_rate']:.4f} "
            f"topK*={m['exact_topK_oracleK_rate']:.4f} "
            f"margin={m['mean_full_separation_margin']:+.4f}"
        )

    line(
        "T0 OOF raw",
        t0_raw_metrics,
    )

    line(
        "T0 OOF sign-calibrated",
        t0_sign_metrics,
    )

    line(
        "T0 OOF robust-calibrated",
        t0_robust_metrics,
    )

    print()

    line(
        "D raw",
        d_raw_metrics,
    )

    line(
        "D T0-sign-calibrated",
        d_sign_metrics,
    )

    line(
        "D T0-robust-calibrated",
        d_robust_metrics,
    )

    print()

    print(
        f"{'T1 raw':<34} "
        f"pairAcc={t1_raw_report['pair_accuracy']:.4f} "
        f"margin={t1_raw_report['mean_margin']:+.4f}"
    )

    print(
        f"{'T1 T0-sign-calibrated':<34} "
        f"pairAcc={t1_sign_report['pair_accuracy']:.4f} "
        f"margin={t1_sign_report['mean_margin']:+.4f}"
    )

    print(
        f"{'T1 T0-robust-calibrated':<34} "
        f"pairAcc={t1_robust_report['pair_accuracy']:.4f} "
        f"margin={t1_robust_report['mean_margin']:+.4f}"
    )

    print()

    print("Direction agreement:")
    for k, v in agreement_summary.items():
        print(f"  {k:<24} {v}")

    print("\nFold direction stability:")
    for role, st in sorted(
        direction_stability.items(),
        key=lambda kv: (
            kv[1]["consistency"],
            kv[0],
        ),
    ):
        print(
            f"  {role:<28} "
            f"consistency={st['consistency']:.3f} "
            f"dirs={st['fold_directions']}"
        )

    print("\nT1 per-role RAW -> T0-SIGNED:")
    roles = sorted(
        set(t1_raw_report["per_role"])
        | set(t1_sign_report["per_role"])
    )

    for role in roles:
        a = t1_raw_report["per_role"].get(role, {})
        b = t1_sign_report["per_role"].get(role, {})

        print(
            f"  {role:<28} "
            f"pairs={a.get('pairs')} "
            f"raw={a.get('pair_accuracy')} "
            f"signed={b.get('pair_accuracy')}"
        )

    print()
    print("* topK uses TRUE K and is ranking evaluation only.")
    print("Internal TEST untouched.")
    print("Summary:", OUT / "summary.json")


if __name__ == "__main__":
    main()
