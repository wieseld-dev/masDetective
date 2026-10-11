from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, f1_score, roc_auc_score

from estimator.candidate_presence_contrastive_v64 import CandidateRecord, build_candidate_records, load_jsonl, load_profile
from estimator.candidate_presence_prototype_v65 import CandidatePresencePrototypeV65
from targets.single_router_zoo_v65 import V63_TRAIN_COMPOSITIONS


def _auc(y, p):
    return float(roc_auc_score(y, p)) if len(set(y)) == 2 else float("nan")


def _metrics(records: Sequence[CandidateRecord], p: Sequence[float], est: CandidatePresencePrototypeV65) -> Dict:
    y = np.asarray([r.label for r in records], dtype=int)
    p = np.asarray(p, dtype=float)
    pred = est.predict_present_records(records, p)
    out = {
        "auc": _auc(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "brier": float(brier_score_loss(y, p)),
    }
    by_sys = defaultdict(list)
    for r, prob, pr in zip(records, p, pred):
        by_sys[r.system_id].append((r, float(prob), bool(pr)))
    hard_err=[]; soft_err=[]; exact=[]; within1=[]
    for rows in by_sys.values():
        k = rows[0][0].k_true
        kh = sum(int(pr) for _, _, pr in rows)
        ks = sum(prob for _, prob, _ in rows)
        hard_err.append(abs(kh-k)); soft_err.append(abs(ks-k)); exact.append(kh==k); within1.append(abs(kh-k)<=1)
    out.update({
        "hard_K_MAE": float(np.mean(hard_err)),
        "soft_K_MAE": float(np.mean(soft_err)),
        "exact_K_accuracy": float(np.mean(exact)),
        "within_1_K_accuracy": float(np.mean(within1)),
    })
    return out


def _constant_k_baseline(train: Sequence[CandidateRecord], val: Sequence[CandidateRecord]) -> float:
    train_k = [rows[0].k_true for sid in sorted({r.system_id for r in train})
               if (rows := [r for r in train if r.system_id == sid])]
    guess = int(round(float(np.median(train_k))))
    val_k = [rows[0].k_true for sid in sorted({r.system_id for r in val})
             if (rows := [r for r in val if r.system_id == sid])]
    return float(np.mean([abs(guess-k) for k in val_k]))


def _composition_set(records):
    return {r.composition_key for r in records}


def main() -> None:
    ap = argparse.ArgumentParser(description="V6.5 strict unseen-composition candidate-presence evaluation")
    ap.add_argument("--profile", default="config/application_profile.json")
    ap.add_argument("--v63-public", default="artifacts/role_bootstrap_v63_public.jsonl")
    ap.add_argument("--v63-truth", default="artifacts/role_bootstrap_v63_truth.jsonl")
    ap.add_argument("--train-public", default="artifacts/v65_train_public.jsonl")
    ap.add_argument("--train-truth", default="artifacts/v65_train_truth.jsonl")
    ap.add_argument("--val-public", default="artifacts/v65_val_public.jsonl")
    ap.add_argument("--val-truth", default="artifacts/v65_val_truth.jsonl")
    ap.add_argument("--estimator-out", default="artifacts/candidate_presence_v65.joblib")
    ap.add_argument("--output-dir", default="artifacts/v65_candidate_presence")
    ap.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--seed", type=int, default=73)
    ap.add_argument("--min-val-auc", type=float, default=0.75)
    ap.add_argument("--min-val-balacc", type=float, default=0.68)
    ap.add_argument("--max-val-brier", type=float, default=0.24)
    ap.add_argument("--max-val-k-mae", type=float, default=1.25)
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    profile = load_profile(args.profile)
    old = build_candidate_records(load_jsonl(args.v63_public), load_jsonl(args.v63_truth))
    new_train = build_candidate_records(load_jsonl(args.train_public), load_jsonl(args.train_truth))
    val = build_candidate_records(load_jsonl(args.val_public), load_jsonl(args.val_truth))
    train = [*old, *new_train]

    train_comp = _composition_set(train)
    val_comp = _composition_set(val)
    overlap = train_comp & val_comp
    if overlap:
        raise SystemExit(f"COMPOSITION LEAKAGE: {sorted(overlap)}")

    print("=== V6.5 STRICT UNSEEN-COMPOSITION EVALUATION ===")
    print("attacker features: candidate role description + anchor output + ONE targeted external output")
    print("privileged router/selected_roles/mode/K are NOT estimator features")
    print("old V6.3 train systems reused:", len({r.system_id for r in old}))
    print("new diverse train systems      :", len({r.system_id for r in new_train}))
    print("total train unique compositions:", len(train_comp))
    print("validation systems             :", len({r.system_id for r in val}))
    print("validation unique compositions :", len(val_comp))
    print("train/validation composition overlap:", len(overlap))
    print("embedding model:", args.embedding_model)

    est = CandidatePresencePrototypeV65(
        profile["candidate_roles"], profile.get("role_descriptions", {}),
        embedding_model=args.embedding_model, random_state=args.seed,
    ).fit(train)
    p = est.predict_proba_records(val)
    total = _metrics(val, p, est)
    baseline = _constant_k_baseline(train, val)

    print("\nPRIMARY HELD-OUT RESULT")
    print(
        f"role AUC={total['auc']:.3f} bal-acc={total['balanced_accuracy']:.3f} "
        f"F1={total['f1']:.3f} brier={total['brier']:.3f}"
    )
    print(
        f"K hard-MAE={total['hard_K_MAE']:.3f} soft-MAE={total['soft_K_MAE']:.3f} "
        f"exact-K={total['exact_K_accuracy']:.3f} within-1={total['within_1_K_accuracy']:.3f} "
        f"constant-K-baseline-MAE={baseline:.3f}"
    )

    print("\nPER MODE")
    per_mode = {}
    for mode in sorted({r.mode for r in val}):
        idx = [i for i,r in enumerate(val) if r.mode == mode]
        rr = [val[i] for i in idx]; pp = p[idx]
        m = _metrics(rr, pp, est)
        per_mode[mode] = m
        print(
            f"{mode:<7} AUC={m['auc']:.3f} bal-acc={m['balanced_accuracy']:.3f} "
            f"K-MAE={m['hard_K_MAE']:.3f} exact-K={m['exact_K_accuracy']:.3f}"
        )

    print("\nPER ROLE")
    per_role = {}
    for role in profile["candidate_roles"]:
        idx = [i for i,r in enumerate(val) if r.role == role]
        rr=[val[i] for i in idx]; pp=p[idx]
        m=_metrics(rr,pp,est)
        per_role[role]=m
        print(f"{role:<11} AUC={m['auc']:.3f} bal-acc={m['balanced_accuracy']:.3f} F1={m['f1']:.3f}")

    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "protocol": {
            "primary": "train on V6.3 + V6.5 diverse train compositions; evaluate once on completely unseen V6.5 validation compositions",
            "attacker_features": ["candidate_role_description", "anchor_external_output", "targeted_external_output"],
            "forbidden_features": ["selected_roles", "router_choice", "K_true", "execution_mode", "internal_trace"],
            "train_validation_composition_overlap": 0,
        },
        "train_systems": len({r.system_id for r in train}),
        "train_unique_compositions": len(train_comp),
        "validation_systems": len({r.system_id for r in val}),
        "validation_unique_compositions": len(val_comp),
        "overall": total,
        "per_mode": per_mode,
        "per_role": per_role,
        "constant_K_baseline_MAE": baseline,
    }
    (outdir/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with (outdir/"predictions.csv").open("w", newline="", encoding="utf-8") as f:
        fields=["system_id","mode","role","label","prob","threshold","pred","k_true","composition"]
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        pred=est.predict_present_records(val,p)
        for r,prob,pr in zip(val,p,pred):
            w.writerow({
                "system_id":r.system_id,"mode":r.mode,"role":r.role,"label":r.label,
                "prob":float(prob),"threshold":est.threshold_for(r.role),"pred":int(pr),
                "k_true":r.k_true,"composition":r.composition_key,
            })

    passed = (
        total["auc"] >= args.min_val_auc and
        total["balanced_accuracy"] >= args.min_val_balacc and
        total["brier"] <= args.max_val_brier and
        total["hard_K_MAE"] <= args.max_val_k_mae
    )
    print("\nQUALITY GATE:", "PASS" if passed else "FAIL")
    print(
        f"required: AUC>={args.min_val_auc} bal-acc>={args.min_val_balacc} "
        f"brier<={args.max_val_brier} K-MAE<={args.max_val_k_mae}"
    )

    # Save only after evaluation. This artifact may be used by a later RL stage; the held-out
    # validation metrics above remain the scientific result.
    est.save(args.estimator_out)
    print("saved estimator:", args.estimator_out)
    print("saved summary  :", outdir/"summary.json")
    if not passed and not args.no_gate:
        raise SystemExit("V6.5 quality gate FAILED. Do not start RL; inspect per-role/per-mode metrics first.")


if __name__ == "__main__":
    main()
