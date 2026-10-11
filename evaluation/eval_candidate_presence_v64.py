from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold

from estimator.candidate_presence_contrastive_v64 import (
    CandidateRecord,
    ContrastiveCandidateEstimatorV64,
    build_candidate_records,
    load_jsonl,
    load_profile,
)


def _safe_auc(y: Sequence[int], p: Sequence[float]) -> float:
    return float(roc_auc_score(y, p)) if len(set(map(int, y))) == 2 else float("nan")


def _binary_metrics(y: Sequence[int], p: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "auc": _safe_auc(y, p),
        "brier": float(brier_score_loss(y, p)),
    }


def _k_metrics(records: Sequence[CandidateRecord], probs: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    by_system: Dict[str, List[Tuple[CandidateRecord, float]]] = defaultdict(list)
    for r, p in zip(records, probs):
        by_system[r.system_id].append((r, float(p)))
    hard_abs: List[float] = []
    soft_abs: List[float] = []
    hard_exact: List[int] = []
    rounded_soft_exact: List[int] = []
    within1: List[int] = []
    for rows in by_system.values():
        k_true = rows[0][0].k_true
        hard = sum(int(p >= threshold) for _, p in rows)
        soft = sum(p for _, p in rows)
        hard_abs.append(abs(hard - k_true))
        soft_abs.append(abs(soft - k_true))
        hard_exact.append(int(hard == k_true))
        rounded_soft_exact.append(int(round(soft) == k_true))
        within1.append(int(abs(hard - k_true) <= 1))
    return {
        "hard_K_MAE": float(np.mean(hard_abs)),
        "soft_K_MAE": float(np.mean(soft_abs)),
        "exact_K_accuracy": float(np.mean(hard_exact)),
        "rounded_soft_K_accuracy": float(np.mean(rounded_soft_exact)),
        "within_1_K_accuracy": float(np.mean(within1)),
    }


def _baseline_k_from_train(train: Sequence[CandidateRecord], val: Sequence[CandidateRecord]) -> Dict[str, float]:
    train_sys: Dict[str, int] = {}
    val_sys: Dict[str, int] = {}
    for r in train:
        train_sys[r.system_id] = r.k_true
    for r in val:
        val_sys[r.system_id] = r.k_true
    guess = float(np.mean(list(train_sys.values())))
    mae = float(np.mean([abs(guess - k) for k in val_sys.values()]))
    rounded = int(round(guess))
    exact = float(np.mean([int(rounded == k) for k in val_sys.values()]))
    return {"constant_K_guess": guess, "constant_K_MAE": mae, "constant_K_exact": exact}


def _composition_cv(records: Sequence[CandidateRecord], profile: Dict, seed: int):
    groups = np.asarray([r.composition_key for r in records])
    unique = sorted(set(groups.tolist()))
    n_splits = min(4, len(unique))
    if n_splits < 2:
        raise ValueError("need at least two distinct compositions for composition-level CV")
    idx = np.arange(len(records))
    gkf = GroupKFold(n_splits=n_splits)
    all_pred: List[Dict] = []
    fold_summaries: List[Dict] = []
    for fi, (tr, va) in enumerate(gkf.split(idx, groups=groups), 1):
        train = [records[i] for i in tr]
        val = [records[i] for i in va]
        est = ContrastiveCandidateEstimatorV64(
            profile["candidate_roles"], profile.get("role_descriptions", {}), random_state=seed + fi
        ).fit(train)
        p = est.predict_proba_records(val)
        y = [r.label for r in val]
        summary = {
            "fold": fi,
            "heldout_compositions": sorted(set(r.composition_key for r in val)),
            **_binary_metrics(y, p),
            **_k_metrics(val, p),
            **_baseline_k_from_train(train, val),
        }
        fold_summaries.append(summary)
        print(
            f"composition CV fold {fi}: "
            f"AUC={summary['auc']:.3f} bal-acc={summary['balanced_accuracy']:.3f} "
            f"F1={summary['f1']:.3f} brier={summary['brier']:.3f} "
            f"K-MAE={summary['hard_K_MAE']:.3f} exact-K={summary['exact_K_accuracy']:.3f} "
            f"baseline-K-MAE={summary['constant_K_MAE']:.3f}"
        )
        for r, prob in zip(val, p):
            all_pred.append({
                "cv": "composition",
                "fold": fi,
                "system_id": r.system_id,
                "mode": r.mode,
                "role": r.role,
                "label": r.label,
                "prob": float(prob),
                "k_true": r.k_true,
                "composition": r.composition_key,
            })
    return fold_summaries, all_pred


def _mode_holdout(records: Sequence[CandidateRecord], profile: Dict, seed: int):
    modes = sorted(set(r.mode for r in records))
    predictions: List[Dict] = []
    summaries: List[Dict] = []
    for mi, mode in enumerate(modes, 1):
        train = [r for r in records if r.mode != mode]
        val = [r for r in records if r.mode == mode]
        est = ContrastiveCandidateEstimatorV64(
            profile["candidate_roles"], profile.get("role_descriptions", {}), random_state=seed + 100 + mi
        ).fit(train)
        p = est.predict_proba_records(val)
        y = [r.label for r in val]
        s = {
            "heldout_mode": mode,
            **_binary_metrics(y, p),
            **_k_metrics(val, p),
            **_baseline_k_from_train(train, val),
        }
        summaries.append(s)
        print(
            f"mode holdout {mode:<7}: AUC={s['auc']:.3f} bal-acc={s['balanced_accuracy']:.3f} "
            f"F1={s['f1']:.3f} brier={s['brier']:.3f} "
            f"K-MAE={s['hard_K_MAE']:.3f} exact-K={s['exact_K_accuracy']:.3f}"
        )
        for r, prob in zip(val, p):
            predictions.append({
                "cv": "mode",
                "fold": mode,
                "system_id": r.system_id,
                "mode": r.mode,
                "role": r.role,
                "label": r.label,
                "prob": float(prob),
                "k_true": r.k_true,
                "composition": r.composition_key,
            })
    return summaries, predictions


def _aggregate_predictions(rows: Sequence[Dict]) -> Dict:
    y = [int(r["label"]) for r in rows]
    p = [float(r["prob"]) for r in rows]
    pseudo = [
        CandidateRecord(
            system_id=str(r["system_id"]), role=str(r["role"]), mode=str(r["mode"]),
            true_roles=tuple(str(r["composition"]).split("|")), anchor_output="", target_output=""
        ) for r in rows
    ]
    result = {**_binary_metrics(y, p), **_k_metrics(pseudo, p)}
    per_mode = {}
    for mode in sorted(set(str(r["mode"]) for r in rows)):
        mr = [r for r in rows if str(r["mode"]) == mode]
        my = [int(r["label"]) for r in mr]
        mp = [float(r["prob"]) for r in mr]
        pseudo_m = [
            CandidateRecord(
                system_id=str(r["system_id"]), role=str(r["role"]), mode=mode,
                true_roles=tuple(str(r["composition"]).split("|")), anchor_output="", target_output=""
            ) for r in mr
        ]
        per_mode[mode] = {**_binary_metrics(my, mp), **_k_metrics(pseudo_m, mp)}
    result["per_mode"] = per_mode
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline candidate-by-candidate contrastive evaluation for V6.3 fan-out data")
    ap.add_argument("--public-log", default="artifacts/role_bootstrap_v63_public.jsonl")
    ap.add_argument("--truth-log", default="artifacts/role_bootstrap_v63_truth.jsonl")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--output-dir", default="artifacts/v64_candidate_presence")
    ap.add_argument("--estimator-out", default="artifacts/candidate_presence_v64.joblib")
    ap.add_argument("--seed", type=int, default=73)
    args = ap.parse_args()

    public = load_jsonl(args.public_log)
    truth = load_jsonl(args.truth_log)
    profile = load_profile(args.profile)
    records = build_candidate_records(public, truth)

    print("=== V6.4 CONTRASTIVE CANDIDATE-PRESENCE EVALUATION ===")
    print("API calls: 0 (reuses existing V6.3 outputs)")
    print("systems:", len(set(r.system_id for r in records)))
    print("candidate decisions:", len(records))
    print("roles:", profile["candidate_roles"])
    print("modes:", sorted(set(r.mode for r in records)))
    print("feature boundary: candidate role + anchor output + targeted external output ONLY")
    print()

    comp_folds, comp_pred = _composition_cv(records, profile, args.seed)
    print()
    mode_folds, mode_pred = _mode_holdout(records, profile, args.seed)

    comp_summary = _aggregate_predictions(comp_pred)
    mode_summary = _aggregate_predictions(mode_pred)

    print("\nSTRICT COMPOSITION-HOLDOUT OOF SUMMARY")
    print(
        f"role AUC={comp_summary['auc']:.3f} bal-acc={comp_summary['balanced_accuracy']:.3f} "
        f"F1={comp_summary['f1']:.3f} brier={comp_summary['brier']:.3f}"
    )
    print(
        f"K hard-MAE={comp_summary['hard_K_MAE']:.3f} soft-MAE={comp_summary['soft_K_MAE']:.3f} "
        f"exact-K={comp_summary['exact_K_accuracy']:.3f} within-1={comp_summary['within_1_K_accuracy']:.3f}"
    )
    print("per-mode under composition holdout:")
    for mode, s in comp_summary["per_mode"].items():
        print(
            f"  {mode:<7} AUC={s['auc']:.3f} bal-acc={s['balanced_accuracy']:.3f} "
            f"K-MAE={s['hard_K_MAE']:.3f} exact-K={s['exact_K_accuracy']:.3f}"
        )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "candidate_task": "For each hypothesized role, predict PRESENT/ABSENT from anchor + one fixed targeted probe output.",
            "attacker_observation": ["candidate_role", "anchor_external_output", "targeted_external_output"],
            "whitebox_use": "labels/evaluation only",
            "api_calls": 0,
        },
        "composition_cv_folds": comp_folds,
        "composition_oof": comp_summary,
        "mode_holdout_folds": mode_folds,
        "mode_holdout_oof": mode_summary,
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fieldnames = ["cv", "fold", "system_id", "mode", "role", "label", "prob", "k_true", "composition"]
    with (outdir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows([*comp_pred, *mode_pred])

    # Full-fit artifact for the next RL stage. Scientific claims should use the OOF metrics above,
    # not this full-fit model.
    final_est = ContrastiveCandidateEstimatorV64(
        profile["candidate_roles"], profile.get("role_descriptions", {}), random_state=args.seed
    ).fit(records)
    final_est.save(args.estimator_out)
    print("\nsaved summary:", outdir / "summary.json")
    print("saved predictions:", outdir / "predictions.csv")
    print("saved full-fit estimator for later RL:", args.estimator_out)


if __name__ == "__main__":
    main()
