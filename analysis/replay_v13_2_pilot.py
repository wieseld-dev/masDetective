#!/usr/bin/env python3
"""Offline replay of the already-consumed V13 one-system pilot with V13.2.

No MAS/API calls are made. The script re-scores the public questions/answers already
stored in sessions.jsonl, aggregates evidence conservatively, and only then attaches
the stored eval-only truth to compute diagnostics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np

from v13_2.judge import Calibrator, DualHeadJudge, logit, sigmoid


def set_metrics(true_roles: set[str], pred_roles: set[str]) -> dict:
    tp = len(true_roles & pred_roles)
    fp = len(pred_roles - true_roles)
    fn = len(true_roles - pred_roles)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for ln, line in enumerate(Path(path).open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise RuntimeError(f"{path}:{ln}: expected object")
        rows.append(obj)
    return rows


def robust_role_probability(probs: list[float]) -> tuple[float, float]:
    """Median log-odds avoids the overconfidence from repeated log-odds summation."""
    if not probs:
        return 0.5, 0.0
    logits = [logit(p) for p in probs]
    m = float(median(logits))
    return sigmoid(m), m


def main() -> None:
    ap = argparse.ArgumentParser(description="Offline V13.2 replay on existing pilot outputs")
    ap.add_argument("--judge-dir", default="artifacts/v13_2_dual_judge")
    ap.add_argument("--sessions", default="artifacts/v13_fanout_count_pilot1/sessions.jsonl")
    ap.add_argument("--output-dir", default="artifacts/v13_2_pilot_replay")
    ap.add_argument("--eval-batch-size", type=int, default=6)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    judge_dir = Path(args.judge_dir)
    decision_path = judge_dir / "decision.json"
    calibrator_path = judge_dir / "calibrator.json"
    if not decision_path.exists() or not calibrator_path.exists():
        raise SystemExit(f"judge is incomplete: expected {decision_path} and {calibrator_path}")
    validation_decision = json.loads(decision_path.read_text(encoding="utf-8"))
    calibrator = Calibrator.from_json(calibrator_path)
    expected_features = ("evidence_delta", "answer_attribution")
    if calibrator.feature_names != expected_features:
        raise SystemExit(
            f"unexpected calibrator feature order: {calibrator.feature_names}; "
            f"expected {expected_features}"
        )

    from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES

    sessions = read_jsonl(args.sessions)
    if not sessions:
        raise SystemExit(f"no sessions in {args.sessions}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    judge = DualHeadJudge(
        judge_dir=judge_dir,
        device="cuda:0",
        use_4bit=not args.no_4bit,
        eval_batch_size=args.eval_batch_size,
    )

    reports = []
    for sess in sessions:
        history = [h for h in (sess.get("history") or []) if h.get("step") and h.get("query") and h.get("output") and h.get("role")]
        if not history:
            continue

        # Build one batched request set containing candidate and public competitors.
        requests: list[tuple[str, str, str, str]] = []
        event_index: list[dict] = []
        for h in history:
            role = str(h["role"])
            query = str(h["query"])
            output = str(h["output"])
            req_i = len(requests)
            requests.append((role, ROLE_DESCRIPTIONS[role], query, output))
            comp_indices = []
            for comp in h.get("competitors_public") or []:
                comp = str(comp)
                if comp not in ROLE_DESCRIPTIONS:
                    continue
                j = len(requests)
                requests.append((comp, ROLE_DESCRIPTIONS[comp], query, output))
                comp_indices.append((comp, j))
            event_index.append({"candidate_idx": req_i, "competitors": comp_indices})

        scored = judge.score_many(requests)
        by_role = defaultdict(lambda: {"p": [], "ev": [], "attr": [], "attr_margin": [], "events": []})
        replay_history = []

        for h, ix in zip(history, event_index):
            role = str(h["role"])
            s = scored[ix["candidate_idx"]]
            features = [s["evidence_delta"], s["attribution"]]
            p = calibrator.probability(features)

            comp_attrs = []
            for comp, j in ix["competitors"]:
                comp_attrs.append((comp, float(scored[j]["attribution"])))
            max_comp_attr = max((x[1] for x in comp_attrs), default=None)
            attr_margin = None if max_comp_attr is None else float(s["attribution"] - max_comp_attr)

            d = by_role[role]
            d["p"].append(float(p))
            d["ev"].append(float(s["evidence_delta"]))
            d["attr"].append(float(s["attribution"]))
            if attr_margin is not None:
                d["attr_margin"].append(attr_margin)
            d["events"].append(int(h["step"]))

            replay_history.append(
                {
                    "step": int(h["step"]),
                    "role": role,
                    "query": h["query"],
                    "old_probe_p": h.get("probe_p"),
                    "old_detector_score": h.get("detector_score"),
                    "old_belief_after": h.get("belief_after"),
                    "evidence_raw": s["evidence_raw"],
                    "question_prior": s["question_prior"],
                    "evidence_delta": s["evidence_delta"],
                    "answer_attribution": s["attribution"],
                    "combined_probability": p,
                    "competitor_attribution": comp_attrs,
                    "candidate_minus_max_competitor_attribution": attr_margin,
                }
            )

        role_summary = {}
        for role in TRAIN_ROLES:
            d = by_role.get(role)
            if not d or not d["p"]:
                role_summary[role] = {
                    "n": 0,
                    "probability": 0.5,
                    "median_combined_logit": 0.0,
                    "median_evidence_delta": None,
                    "median_attribution": None,
                    "median_attr_margin_to_public_competitors": None,
                }
                continue
            prob, med_logit = robust_role_probability(d["p"])
            role_summary[role] = {
                "n": len(d["p"]),
                "probability": prob,
                "median_combined_logit": med_logit,
                "mean_per_probe_probability": float(np.mean(d["p"])),
                "median_evidence_delta": float(median(d["ev"])),
                "median_attribution": float(median(d["attr"])),
                "median_attr_margin_to_public_competitors": (
                    float(median(d["attr_margin"])) if d["attr_margin"] else None
                ),
            }

        ranking = sorted(TRAIN_ROLES, key=lambda r: role_summary[r]["probability"], reverse=True)
        for rank, role in enumerate(ranking, 1):
            role_summary[role]["rank"] = rank

        pred = {r for r in TRAIN_ROLES if role_summary[r]["probability"] >= 0.5}
        old_pred = set(map(str, sess.get("predicted_roles") or []))

        # Truth is used only here, after all replay scoring/aggregation is complete.
        true = set(map(str, sess.get("true_roles_eval_only") or []))
        k_true = int(sess.get("k_true_workers_eval_only", len(true)))
        new_metrics = set_metrics(true, pred) if true else None
        old_metrics = set_metrics(true, old_pred) if true else None
        true_ranks = {r: role_summary[r]["rank"] for r in sorted(true)} if true else {}
        recall_at = {}
        for k in (3, 5, 7):
            top = set(ranking[:k])
            recall_at[str(k)] = len(top & true) / max(len(true), 1) if true else None

        report = {
            "system_id": sess.get("system_id"),
            "offline_only_no_api_calls": True,
            "n_public_answers_reused": len(history),
            "ranking": ranking,
            "role_summary": role_summary,
            "new_predicted_roles_threshold_0_5": sorted(pred),
            "new_k_hat": len(pred),
            "old_predicted_roles": sorted(old_pred),
            "old_k_hat": int(sess.get("k_hat_workers", len(old_pred))),
            "true_roles_eval_only": sorted(true),
            "k_true_eval_only": k_true,
            "new_role_metrics_eval_only": new_metrics,
            "old_role_metrics_eval_only": old_metrics,
            "new_k_abs_error_eval_only": abs(len(pred) - k_true) if true else None,
            "old_k_abs_error_eval_only": abs(int(sess.get("k_hat_workers", len(old_pred))) - k_true) if true else None,
            "true_role_ranks_eval_only": true_ranks,
            "mean_true_role_rank_eval_only": float(np.mean(list(true_ranks.values()))) if true_ranks else None,
            "recall_at_k_eval_only": recall_at,
            "replay_history": replay_history,
        }
        reports.append(report)

    if not reports:
        raise SystemExit("no replayable sessions")

    # Current development file contains one pilot, but aggregate generically.
    pilot = reports[0]
    replay_checks = {
        "f1_at_least_0_50": bool(pilot["new_role_metrics_eval_only"] and pilot["new_role_metrics_eval_only"]["f1"] >= 0.50),
        "k_abs_error_at_most_3": bool(pilot["new_k_abs_error_eval_only"] is not None and pilot["new_k_abs_error_eval_only"] <= 3),
        "all_true_roles_in_top5": bool(pilot["recall_at_k_eval_only"].get("5") == 1.0),
        "mean_true_rank_at_most_5": bool(pilot["mean_true_role_rank_eval_only"] is not None and pilot["mean_true_role_rank_eval_only"] <= 5.0),
    }
    validation_pass = bool(validation_decision.get("go_to_fresh_api_pilot_from_validation_only"))
    replay_pass = all(replay_checks.values())
    final_decision = {
        "validation_pass": validation_pass,
        "offline_replay_pass": replay_pass,
        "replay_checks": replay_checks,
        "go_to_one_fresh_api_pilot": bool(validation_pass and replay_pass),
        "reason": (
            "PASS: validation and offline replay both passed; one fresh API pilot is justified."
            if validation_pass and replay_pass
            else "STOP: do not spend fresh API calls yet. Inspect validation/replay failures."
        ),
        "api_calls_used_by_replay": 0,
    }

    output = {
        "protocol": "v13.2_offline_replay_existing_v13_pilot",
        "judge_dir": str(judge_dir),
        "sessions_path": str(args.sessions),
        "systems": reports,
        "decision": final_decision,
    }
    (outdir / "replay.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    (outdir / "decision.json").write_text(json.dumps(final_decision, indent=2, sort_keys=True), encoding="utf-8")

    print("\n=== V13.2 OFFLINE PILOT REPLAY ===", flush=True)
    print(
        json.dumps(
            {
                "system_id": pilot["system_id"],
                "true_roles": pilot["true_roles_eval_only"],
                "old_k_hat": pilot["old_k_hat"],
                "new_k_hat": pilot["new_k_hat"],
                "old_metrics": pilot["old_role_metrics_eval_only"],
                "new_metrics": pilot["new_role_metrics_eval_only"],
                "true_role_ranks": pilot["true_role_ranks_eval_only"],
                "top10": pilot["ranking"][:10],
                "decision": final_decision,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    print(f"full replay: {outdir / 'replay.json'}", flush=True)


if __name__ == "__main__":
    main()
