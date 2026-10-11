#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from adaptive_v1.bank import load_bank
from adaptive_v1.schema import EpisodeState
from adaptive_v1.kfirst import (
    STOP,
    EmpiricalKOutcomeModel,
    KBoundaryController,
    KFirstBeliefModel,
    KFirstVOIController,
    MaxUncertaintyController,
    RandomController,
    set_metrics,
)
from v11.role_universe import TRAIN_ROLES


def _candidate_banks() -> list[Path]:
    out = []

    preferred = Path("artifacts/adaptive_v1")
    if preferred.exists():
        out.extend(preferred.rglob("*.jsonl"))

    fallback = Path("artifacts")
    if fallback.exists():
        out.extend(p for p in fallback.rglob("*.jsonl") if p not in out)

    return out


def find_bank(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"bank not found: {p}")
        return p

    candidates = []
    for p in _candidate_banks():
        try:
            first = next(
                (line for line in p.read_text(encoding="utf-8").splitlines() if line.strip()),
                "",
            )
            if not first:
                continue

            obj = json.loads(first)
            if not {"system_id", "action", "evidence"} <= set(obj):
                continue

            md = obj.get("metadata") or {}
            dual = int("evidence_v134" in md)
            preferred = int("adaptive_v1" in str(p))
            candidates.append((preferred, dual, p.stat().st_mtime_ns, p.stat().st_size, p))
        except Exception:
            continue

    if not candidates:
        raise SystemExit(
            "Could not auto-detect a dual-Judge adaptive replay bank. "
            "Pass --bank PATH explicitly."
        )

    candidates.sort(reverse=True)
    return candidates[0][-1]


def split_systems(systems):
    train, dev = [], []
    for s in systems:
        name = str(s.split).lower()
        if "dev" in name or "valid" in name or "val" in name:
            dev.append(s)
        elif "train" in name:
            train.append(s)

    if not train or not dev:
        raise RuntimeError(
            f"Could not infer train/dev split: train={len(train)} dev={len(dev)}"
        )
    return train, dev


def validate_bank(systems, roles):
    role_set = set(roles)
    bad = []

    for s in systems:
        for o in s.observations:
            if set(o.evidence.keys()) != role_set:
                bad.append((s.system_id, o.action.key, "v135_roles"))
            ev134 = dict((o.metadata or {}).get("evidence_v134", {}))
            if set(ev134.keys()) != role_set:
                bad.append((s.system_id, o.action.key, "v134_roles"))

    if bad:
        raise RuntimeError(f"dual-Judge replay bank validation failed; examples={bad[:5]}")


def action_map(system):
    out = {}
    for obs in system.observations:
        out.setdefault(obs.action.key, obs)
    return out


def run_episode(system, model, controller, *, budget: int):
    amap = action_map(system)
    history = []
    stop_reason = None

    for _ in range(int(budget)):
        belief = model.predict_structured(history)
        used = {x.action.key for x in history}
        available_obs = [o for k, o in amap.items() if k not in used]
        available_actions = [o.action for o in available_obs]

        state = EpisodeState(
            roles=tuple(model.roles),
            history=list(history),
            remaining_budget=max(0, int(budget) - len(history)),
            belief=dict(belief.role_probs),
        )

        action = controller.choose(state, available_actions, belief)
        if action == STOP:
            stop_reason = "controller_stop"
            break
        if action.key not in amap:
            stop_reason = f"missing_action:{action.key}"
            break

        history.append(amap[action.key])

    belief = model.predict_structured(history)
    row = set_metrics(belief, system.truth_roles, model.roles)
    row["queries_used"] = len(history)
    row["stop_reason"] = stop_reason
    row["system_id"] = system.system_id
    return row


def run_oracle_episode(system, model, *, budget: int):
    """
    Privileged upper bound.

    At each step it inspects the actual held-out outcome of every unused action and
    uses the hidden dev truth only to choose the best next action.
    """
    amap = action_map(system)
    history = []
    truth = set(system.truth_roles)
    true_k = len(truth)

    def supervised_loss(belief):
        k_loss = abs(belief.k_map - true_k) / max(1, len(model.roles))
        pred = set(belief.predicted_roles)
        tp = len(pred & truth)
        denom = len(pred) + len(truth)
        f1 = 1.0 if denom == 0 else (2.0 * tp / denom)
        return 0.72 * k_loss + 0.28 * (1.0 - f1)

    for _ in range(int(budget)):
        used = {x.action.key for x in history}
        candidates = [o for k, o in amap.items() if k not in used]
        if not candidates:
            break

        scored = []
        for obs in candidates:
            b1 = model.predict_structured(history + [obs])
            scored.append((supervised_loss(b1), obs.action.key, obs))

        scored.sort(key=lambda x: (x[0], x[1]))
        history.append(scored[0][2])

    belief = model.predict_structured(history)
    row = set_metrics(belief, system.truth_roles, model.roles)
    row["queries_used"] = len(history)
    row["stop_reason"] = None
    row["system_id"] = system.system_id
    return row


def aggregate(rows):
    scalar_keys = [
        "k_exact",
        "k_exact_direct",
        "k_exact_roles",
        "k_abs_error",
        "expected_k_abs_error",
        "set_precision",
        "set_recall",
        "set_f1",
        "set_jaccard",
        "exact_set",
        "k_confidence",
        "membership_margin",
        "k_head_js",
        "queries_used",
    ]

    out = {}
    for key in scalar_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        out[key] = float(np.mean(vals))
        out[key + "_std"] = float(np.std(vals))

    y = np.asarray([v for r in rows for v in r["role_truth"]], dtype=np.int64)
    p = np.asarray([v for r in rows for v in r["role_probs"]], dtype=np.float64)

    out["role_micro_auc"] = (
        float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    )
    out["n_systems"] = len(rows)
    return out


def evaluate_method(dev, model, make_controller, budgets, *, repeats: int):
    result = {}

    for budget in budgets:
        summaries = []

        for rep in range(int(repeats)):
            rows = []
            for si, system in enumerate(dev):
                controller = make_controller(rep, si)
                rows.append(run_episode(system, model, controller, budget=budget))
            summaries.append(aggregate(rows))

        keys = [k for k in summaries[0] if k != "n_systems"]
        merged = {}

        for key in keys:
            vals = np.asarray([float(x[key]) for x in summaries], dtype=np.float64)
            finite = vals[np.isfinite(vals)]
            merged[key] = float(np.mean(finite)) if len(finite) else float("nan")
            merged[key + "_across_repeat_std"] = (
                float(np.std(finite)) if len(finite) else float("nan")
            )

        merged["repeats"] = int(repeats)
        merged["n_dev_systems"] = len(dev)
        result[str(budget)] = merged

    return result


def evaluate_oracle(dev, model, budgets):
    out = {}
    for budget in budgets:
        out[str(budget)] = aggregate(
            [run_oracle_episode(s, model, budget=budget) for s in dev]
        )
    return out


def main():
    ap = argparse.ArgumentParser(description="K-first adaptive MAS composition evaluation v2")
    ap.add_argument("--bank", default=None)
    ap.add_argument("--output-dir", default="artifacts/adaptive_kfirst_v2")
    ap.add_argument("--budgets", default="1,3,5,8,12,20")
    ap.add_argument("--random-repeats", type=int, default=20)
    ap.add_argument("--mc-samples", type=int, default=16)
    ap.add_argument("--max-candidate-actions", type=int, default=12)
    ap.add_argument("--seed", type=int, default=76123)
    args = ap.parse_args()

    roles = tuple(TRAIN_ROLES)
    bank_path = find_bank(args.bank)
    systems = load_bank(bank_path, roles=roles)
    validate_bank(systems, roles)
    train, dev = split_systems(systems)

    print(f"[bank] {bank_path}", flush=True)
    print(
        f"[systems] train={len(train)} dev={len(dev)} roles={len(roles)}",
        flush=True,
    )

    # The locked adaptive design is expected to be 24 train / 8 dev.
    if len(train) != 24 or len(dev) != 8:
        print(
            f"[warning] expected locked 24/8 split, got {len(train)}/{len(dev)}",
            flush=True,
        )

    model = KFirstBeliefModel(
        roles,
        base_C=0.30,
        direct_C=0.20,
        n_folds=4,
        seed=args.seed,
    )
    fit_diag = model.fit(
        train,
        permutations_per_system=4,
        max_prefix=20,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(out_dir / "model")

    outcome = EmpiricalKOutcomeModel(
        train,
        roles,
        composition_temperature=4.0,
        k_weight=1.5,
        seed=args.seed + 5000,
    )

    budgets = sorted(
        {int(x) for x in args.budgets.split(",") if x.strip() and int(x) > 0}
    )

    report = {
        "version": "adaptive_kfirst_eval_v2",
        "primary_goal": "infer K first; recover role set consistently with K",
        "bank": str(bank_path),
        "train_systems": len(train),
        "dev_systems": len(dev),
        "roles": list(roles),
        "budgets": budgets,
        "fit_diagnostics": fit_diag,
        "methods": {},
    }

    report["methods"]["random"] = evaluate_method(
        dev,
        model,
        lambda rep, si: RandomController(seed=args.seed + rep * 1009 + si),
        budgets,
        repeats=max(1, args.random_repeats),
    )

    report["methods"]["max_uncertainty"] = evaluate_method(
        dev,
        model,
        lambda rep, si: MaxUncertaintyController(),
        budgets,
        repeats=1,
    )

    report["methods"]["k_boundary"] = evaluate_method(
        dev,
        model,
        lambda rep, si: KBoundaryController(),
        budgets,
        repeats=1,
    )

    report["methods"]["kfirst_voi"] = evaluate_method(
        dev,
        model,
        lambda rep, si: KFirstVOIController(
            model,
            outcome,
            mc_samples=args.mc_samples,
            max_candidate_actions=args.max_candidate_actions,
            query_cost=0.0125,
            allow_stop=False,
            seed=args.seed + 7000 + si,
        ),
        budgets,
        repeats=1,
    )

    report["methods"]["oracle_one_step"] = evaluate_oracle(dev, model, budgets)

    # Separate autonomous STOP experiment, never mixed into fixed-budget curves.
    stop_rows = []
    max_budget = max(budgets)
    for si, system in enumerate(dev):
        controller = KFirstVOIController(
            model,
            outcome,
            mc_samples=max(args.mc_samples, 20),
            max_candidate_actions=args.max_candidate_actions,
            query_cost=0.0125,
            allow_stop=True,
            min_queries_before_stop=3,
            stop_k_confidence=0.70,
            stop_membership_margin=0.10,
            seed=args.seed + 9000 + si,
        )
        stop_rows.append(run_episode(system, model, controller, budget=max_budget))

    report["auto_stop"] = aggregate(stop_rows)

    report_path = out_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n===== CROSS-FITTED TRAIN/TUNING DIAGNOSTICS =====", flush=True)
    print(json.dumps(fit_diag, indent=2, sort_keys=True), flush=True)

    print("\n===== HELD-OUT DEV: PRIMARY K-FIRST RESULTS =====", flush=True)
    for method, by_budget in report["methods"].items():
        print(f"\n[{method}]", flush=True)
        for budget in budgets:
            x = by_budget[str(budget)]
            print(
                f"budget={budget:>2} "
                f"Kacc={x['k_exact']:.3f} "
                f"Kmae={x['k_abs_error']:.3f} "
                f"F1={x['set_f1']:.3f} "
                f"ExactSet={x['exact_set']:.3f} "
                f"AUC={x['role_micro_auc']:.3f} "
                f"directK={x['k_exact_direct']:.3f} "
                f"rolesK={x['k_exact_roles']:.3f}",
                flush=True,
            )

    print("\n[auto_stop]", flush=True)
    print(json.dumps(report["auto_stop"], indent=2, sort_keys=True), flush=True)
    print(f"\n[done] report={report_path}", flush=True)


if __name__ == "__main__":
    main()
