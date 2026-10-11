#!/usr/bin/env python3

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

# Reuse the EXACT already-tested replay implementation/classes.
from analysis.replay_v13_2_pilot import (
    DualHeadJudge,
    Calibrator,
    read_jsonl,
)

from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES


def set_metrics(true, pred):
    true, pred = set(true), set(pred)
    tp = len(true & pred)
    fp = len(pred - true)
    fn = len(true - pred)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def top_mean(xs, k):
    if not xs:
        return None
    z = sorted(xs, reverse=True)[:k]
    return float(sum(z) / len(z))


def percentile(xs, q):
    if not xs:
        return None
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def main():
    ap = argparse.ArgumentParser(
        description="V13.3c offline all-role scoring of existing pilot answers"
    )
    ap.add_argument(
        "--judge-dir",
        default="artifacts/v13_2_dual_judge_resume",
    )
    ap.add_argument(
        "--sessions",
        default="artifacts/v13_fanout_count_pilot1/sessions.jsonl",
    )
    ap.add_argument(
        "--output-dir",
        default="artifacts/v13_3c_cross_role",
    )
    ap.add_argument("--eval-batch-size", type=int, default=6)
    ap.add_argument("--request-chunk", type=int, default=160)
    ap.add_argument("--no-4bit", action="store_true")
    args = ap.parse_args()

    judge_dir = Path(args.judge_dir)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    calibrator_path = judge_dir / "calibrator.json"
    if not calibrator_path.exists():
        raise SystemExit(f"missing calibrator: {calibrator_path}")

    calibrator = Calibrator.from_json(calibrator_path)

    expected = ("evidence_delta", "answer_attribution")
    if tuple(calibrator.feature_names) != expected:
        raise SystemExit(
            f"unexpected calibrator features: {calibrator.feature_names}"
        )

    sessions = read_jsonl(args.sessions)
    if not sessions:
        raise SystemExit(f"no sessions: {args.sessions}")

    print("=" * 90)
    print("V13.3c CROSS-ROLE EVIDENCE MATRIX")
    print("API CALLS: 0")
    print("Judge:", judge_dir)
    print("Sessions:", args.sessions)
    print("Candidate roles:", len(TRAIN_ROLES))
    print("=" * 90, flush=True)

    judge = DualHeadJudge(
        judge_dir=judge_dir,
        device="cuda:0",
        use_4bit=not args.no_4bit,
        eval_batch_size=args.eval_batch_size,
    )

    reports = []

    for si, sess in enumerate(sessions):
        history = [
            h for h in (sess.get("history") or [])
            if h.get("step")
            and h.get("query")
            and h.get("output")
            and h.get("role")
        ]

        if not history:
            continue

        sid = str(sess.get("system_id", f"system_{si}"))
        truth = set(map(str, sess.get("true_roles_eval_only") or []))

        print()
        print("=" * 90)
        print("SYSTEM:", sid)
        print("events:", len(history))
        print("truth eval-only:", sorted(truth))
        print("=" * 90, flush=True)

        # Every existing answer is scored against EVERY role.
        requests = []
        request_meta = []

        for event_i, h in enumerate(history):
            q = str(h["query"])
            o = str(h["output"])

            for role in TRAIN_ROLES:
                request_meta.append((event_i, role))
                requests.append(
                    (
                        role,
                        ROLE_DESCRIPTIONS[role],
                        q,
                        o,
                    )
                )

        print(
            f"cross-role requests: {len(requests)} "
            f"({len(history)} x {len(TRAIN_ROLES)})",
            flush=True,
        )

        # Chunk calls to keep memory predictable.
        scored = []
        for start in range(0, len(requests), args.request_chunk):
            end = min(start + args.request_chunk, len(requests))
            print(
                f"scoring requests {start + 1}-{end}/{len(requests)}",
                flush=True,
            )
            scored.extend(judge.score_many(requests[start:end]))

        if len(scored) != len(requests):
            raise RuntimeError("score count mismatch")

        event_scores = [
            {"scores": {}} for _ in history
        ]

        for (event_i, role), s in zip(request_meta, scored):
            features = [
                float(s["evidence_delta"]),
                float(s["attribution"]),
            ]
            p = float(calibrator.probability(features))

            event_scores[event_i]["scores"][role] = {
                "probability": p,
                "evidence_delta": float(s["evidence_delta"]),
                "answer_attribution": float(s["attribution"]),
                "evidence_raw": float(s["evidence_raw"]),
                "question_prior": float(s["question_prior"]),
            }

        # -----------------------------------------------------------
        # Per-event diagnostics
        # -----------------------------------------------------------
        event_rows = []
        true_target_rows = []
        false_target_rows = []

        for i, (h, es) in enumerate(zip(history, event_scores)):
            scores = es["scores"]

            ranking = sorted(
                TRAIN_ROLES,
                key=lambda r: scores[r]["probability"],
                reverse=True,
            )
            ranks = {r: j + 1 for j, r in enumerate(ranking)}

            target = str(h["role"])
            target_true = target in truth
            target_p = scores[target]["probability"]

            true_ps = {
                r: scores[r]["probability"]
                for r in truth
                if r in scores
            }
            false_ps = {
                r: scores[r]["probability"]
                for r in TRAIN_ROLES
                if r not in truth
            }

            best_true_role = (
                max(true_ps, key=true_ps.get)
                if true_ps else None
            )
            best_true_p = (
                true_ps[best_true_role]
                if best_true_role is not None else None
            )

            best_false_role = max(false_ps, key=false_ps.get)
            best_false_p = false_ps[best_false_role]

            top_role = ranking[0]

            row = {
                "step": int(h["step"]),
                "target_role": target,
                "target_is_true_eval_only": target_true,
                "query": str(h["query"]),
                "target_probability": target_p,
                "target_rank": ranks[target],
                "top_role": top_role,
                "top_probability": scores[top_role]["probability"],
                "top_role_is_true_eval_only": top_role in truth,
                "best_true_role_eval_only": best_true_role,
                "best_true_probability_eval_only": best_true_p,
                "best_true_rank_eval_only": (
                    ranks[best_true_role]
                    if best_true_role is not None
                    else None
                ),
                "best_false_role_eval_only": best_false_role,
                "best_false_probability_eval_only": best_false_p,
                "true_minus_false_margin_eval_only": (
                    best_true_p - best_false_p
                    if best_true_p is not None
                    else None
                ),
                "all_role_scores": scores,
            }

            event_rows.append(row)

            if target_true:
                true_target_rows.append(row)
            else:
                false_target_rows.append(row)

        # -----------------------------------------------------------
        # Aggregate each role across ALL answers.
        # Diagnostic only — no claim this is the final aggregator.
        # -----------------------------------------------------------
        role_summary = {}

        for role in TRAIN_ROLES:
            ps = [
                row["all_role_scores"][role]["probability"]
                for row in event_rows
            ]
            attrs = [
                row["all_role_scores"][role]["answer_attribution"]
                for row in event_rows
            ]
            evs = [
                row["all_role_scores"][role]["evidence_delta"]
                for row in event_rows
            ]

            role_summary[role] = {
                "is_true_eval_only": role in truth,
                "n_answers_scored": len(ps),
                "max_probability": max(ps),
                "top2_mean_probability": top_mean(ps, 2),
                "top3_mean_probability": top_mean(ps, 3),
                "p90_probability": percentile(ps, 0.90),
                "p75_probability": percentile(ps, 0.75),
                "median_probability": float(statistics.median(ps)),
                "mean_probability": float(np.mean(ps)),
                "max_attribution": max(attrs),
                "median_attribution": float(statistics.median(attrs)),
                "median_evidence_delta": float(statistics.median(evs)),
            }

        aggregate_rankings = {}

        for metric in [
            "max_probability",
            "top2_mean_probability",
            "top3_mean_probability",
            "p90_probability",
            "p75_probability",
            "median_probability",
            "mean_probability",
        ]:
            ranking = sorted(
                TRAIN_ROLES,
                key=lambda r: role_summary[r][metric],
                reverse=True,
            )
            aggregate_rankings[metric] = ranking

            for rank, role in enumerate(ranking, 1):
                role_summary[role][f"{metric}_rank"] = rank

        # -----------------------------------------------------------
        # Key cross-role diagnostics
        # -----------------------------------------------------------
        n = len(event_rows)

        top1_true_rate = (
            sum(x["top_role_is_true_eval_only"] for x in event_rows) / n
            if n else 0.0
        )

        true_target_top1 = (
            sum(x["top_role"] == x["target_role"] for x in true_target_rows)
            / len(true_target_rows)
            if true_target_rows else 0.0
        )

        true_target_top3 = (
            sum(
                x["target_rank"] <= 3
                for x in true_target_rows
            )
            / len(true_target_rows)
            if true_target_rows else 0.0
        )

        # Particularly interesting:
        # while probing an ABSENT target, did the answer reveal a TRUE role
        # more strongly than the absent target itself?
        rescue_rows = [
            x for x in false_target_rows
            if x["best_true_probability_eval_only"] is not None
            and x["best_true_probability_eval_only"]
                > x["target_probability"]
            and x["best_true_probability_eval_only"] >= 0.5
        ]

        diagnostics = {
            "events": n,
            "true_target_events": len(true_target_rows),
            "false_target_events": len(false_target_rows),
            "event_top1_is_any_true_role_rate": top1_true_rate,
            "true_target_is_top1_rate": true_target_top1,
            "true_target_is_top3_rate": true_target_top3,
            "false_target_cross_role_rescue_count": len(rescue_rows),
            "false_target_cross_role_rescue_rate": (
                len(rescue_rows) / len(false_target_rows)
                if false_target_rows else 0.0
            ),
        }

        report = {
            "system_id": sid,
            "true_roles_eval_only": sorted(truth),
            "diagnostics_eval_only": diagnostics,
            "role_summary_eval_only": role_summary,
            "aggregate_rankings_eval_only": aggregate_rankings,
            "events": event_rows,
        }

        reports.append(report)

        print()
        print("=== CROSS-ROLE DIAGNOSTICS ===")
        print(json.dumps(diagnostics, indent=2))

        print()
        print("=== TRUE ROLE AGGREGATE RANKS ===")
        for role in sorted(truth):
            d = role_summary[role]
            print(
                f"{role:28s} "
                f"max={d['max_probability']:.3f} "
                f"top3={d['top3_mean_probability']:.3f} "
                f"p90={d['p90_probability']:.3f} "
                f"median={d['median_probability']:.3f} "
                f"| ranks "
                f"max={d['max_probability_rank']} "
                f"top3={d['top3_mean_probability_rank']} "
                f"p90={d['p90_probability_rank']} "
                f"median={d['median_probability_rank']}"
            )

        print()
        print("=== TOP 10 BY TOP-3 MEAN ===")
        rr = aggregate_rankings["top3_mean_probability"][:10]
        for rank, role in enumerate(rr, 1):
            d = role_summary[role]
            flag = "TRUE" if role in truth else "false"
            print(
                f"{rank:02d}. {role:28s} "
                f"{d['top3_mean_probability']:.3f} "
                f"[{flag}]"
            )

        print()
        print("=== FIRST 15 CROSS-ROLE RESCUES ===")
        for x in rescue_rows[:15]:
            print(
                f"step={x['step']:3d} "
                f"asked={x['target_role']:26s} "
                f"asked_p={x['target_probability']:.3f} "
                f"true={x['best_true_role_eval_only']:26s} "
                f"true_p={x['best_true_probability_eval_only']:.3f}"
            )

    output = {
        "protocol": "v13.3c_cross_role_existing_answer_matrix",
        "api_calls": 0,
        "truth_used_for_scoring": False,
        "truth_used_for_diagnostics_only": True,
        "judge_dir": str(judge_dir),
        "sessions": args.sessions,
        "candidate_roles": list(TRAIN_ROLES),
        "systems": reports,
    }

    out = outdir / "cross_role_matrix.json"
    out.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("=" * 90)
    print("DONE")
    print("saved:", out)
    print("API CALLS USED: 0")
    print("=" * 90)


if __name__ == "__main__":
    main()
