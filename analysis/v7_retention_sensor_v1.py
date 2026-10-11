from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path("analysis").resolve()))

import build_visible_realized_gain_v3 as vg

from v11.role_universe import (
    TRAIN_ROLES,
    ROLE_DESCRIPTIONS,
)


EXEC = Path(
    "artifacts/v7_retention_pilot_v1/executions.jsonl"
)

OUT = Path(
    "artifacts/v7_retention_sensor_v1"
)

ADAPTER = Path(
    "artifacts/"
    "v13_8b_balanced_repair_2048_run1/"
    "best_adapter"
)

CAL = Path(
    "artifacts/"
    "visible_realized_gain_v3/"
    "channel_calibration.json"
)

MAX_LENGTH = 1024


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def write_jsonl(p, rows):
    with Path(p).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


def mean(xs):
    xs = [
        float(x)
        for x in xs
        if x is not None
        and np.isfinite(x)
    ]

    return (
        float(np.mean(xs))
        if xs
        else None
    )


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if (
        len(a) < 2
        or float(np.std(a)) < 1e-12
        or float(np.std(b)) < 1e-12
    ):
        return None

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def row_metrics(
    scores,
    probs,
    llr,
    selected,
    roles,
):
    selected = set(selected)

    y = np.asarray(
        [
            int(r in selected)
            for r in roles
        ],
        dtype=int,
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    probs = np.asarray(
        probs,
        dtype=float,
    )

    llr = np.asarray(
        llr,
        dtype=float,
    )

    pos = y == 1
    neg = y == 0

    if (
        pos.sum() == 0
        or neg.sum() == 0
    ):
        auc = None
        raw_margin = None
        prob_margin = None
        llr_margin = None
        selected_llr = None
        absent_llr = None
        topk_recall = None
    else:
        auc = float(
            roc_auc_score(
                y,
                scores,
            )
        )

        raw_margin = float(
            scores[pos].mean()
            - scores[neg].mean()
        )

        prob_margin = float(
            probs[pos].mean()
            - probs[neg].mean()
        )

        selected_llr = float(
            llr[pos].mean()
        )

        absent_llr = float(
            llr[neg].mean()
        )

        llr_margin = float(
            selected_llr
            - absent_llr
        )

        k = int(pos.sum())

        top = np.argsort(-scores)[:k]

        topk_recall = float(
            y[top].sum() / k
        )

    return {
        "within_row_auc": auc,
        "topk_selected_recall":
            topk_recall,
        "raw_selected_absent_margin":
            raw_margin,
        "prob_selected_absent_margin":
            prob_margin,
        "selected_mean_loglr":
            selected_llr,
        "absent_mean_loglr":
            absent_llr,
        "loglr_selected_absent_margin":
            llr_margin,
        "sensor_top1_role":
            roles[int(np.argmax(scores))],
    }


def set_metrics(a, b):
    a = set(a)
    b = set(b)

    union = a | b
    inter = a & b

    return {
        "activation_exact":
            bool(a == b),

        "activation_count_equal":
            bool(len(a) == len(b)),

        "activation_bad_recall":
            (
                len(inter) / len(a)
                if a
                else None
            ),

        "activation_repair_recall":
            (
                len(inter) / len(b)
                if b
                else None
            ),

        "activation_jaccard":
            (
                len(inter) / len(union)
                if union
                else 1.0
            ),
    }


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        x for x in read_jsonl(EXEC)
        if x["execution_status"] == "ok"
    ]

    if len(rows) != 72:
        raise RuntimeError(
            f"Expected 72 successful executions, "
            f"got {len(rows)}"
        )

    roles = list(TRAIN_ROLES)

    cal = json.loads(
        CAL.read_text(
            encoding="utf-8"
        )
    )

    required_cal = {
        "intercept",
        "score_slope",
        "role_bias",
        "role_prevalence",
    }

    missing = (
        required_cal
        - set(cal)
    )

    if missing:
        raise RuntimeError(
            f"Calibration missing: {missing}"
        )

    print(
        "loading V13.8b sensor...",
        flush=True,
    )

    tok, model, dtype = (
        vg.load_sensor(ADAPTER)
    )

    scored = []

    for i, x in enumerate(rows, 1):

        scores = vg.sensor_score_row(
            model,
            tok,
            dtype,
            roles,
            ROLE_DESCRIPTIONS,
            x["query"],
            x["final_output"],
            MAX_LENGTH,
        )

        probs = vg.calibrated_probs(
            scores,
            roles,
            cal,
        )

        llr = vg.sensor_log_lr(
            scores,
            roles,
            cal,
        )

        rm = row_metrics(
            scores,
            probs,
            llr,
            x["selected_roles_eval_only"],
            roles,
        )

        scored.append({
            **x,
            "sensor_scores":
                scores.tolist(),
            "sensor_probs":
                probs.tolist(),
            "sensor_loglr":
                llr.tolist(),
            **rm,
        })

        print(
            f"[{i:02d}/72] "
            f"pair={x['pair_id']:02d} "
            f"{x['kind']:<6} "
            f"rep={x['rep']} "
            f"auc={rm['within_row_auc']:.3f} "
            f"topk={rm['topk_selected_recall']:.3f} "
            f"llr_margin="
            f"{rm['loglr_selected_absent_margin']:+.3f}",
            flush=True,
        )

    write_jsonl(
        OUT / "scored_executions.jsonl",
        scored,
    )

    by_key = {
        (
            int(x["pair_id"]),
            int(x["rep"]),
            str(x["kind"]),
        ): x
        for x in scored
    }

    matched = []

    for pair_id in range(12):
        for rep in range(3):

            b = by_key[
                (pair_id, rep, "bad")
            ]

            r = by_key[
                (pair_id, rep, "repair")
            ]

            sm = set_metrics(
                b["selected_roles_eval_only"],
                r["selected_roles_eval_only"],
            )

            score_corr = corr(
                b["sensor_scores"],
                r["sensor_scores"],
            )

            prob_corr = corr(
                b["sensor_probs"],
                r["sensor_probs"],
            )

            llr_corr = corr(
                b["sensor_loglr"],
                r["sensor_loglr"],
            )

            matched.append({
                "pair_id":
                    pair_id,

                "rep":
                    rep,

                "family":
                    b["family"],

                "session_id":
                    b["session_id"],

                "bad_query":
                    b["bad_query"],

                "repair_query":
                    r["repair_query"],

                "bad_selected":
                    b["selected_roles_eval_only"],

                "repair_selected":
                    r["selected_roles_eval_only"],

                **sm,

                "sensor_score_corr":
                    score_corr,

                "sensor_prob_corr":
                    prob_corr,

                "sensor_loglr_corr":
                    llr_corr,

                "bad_auc":
                    b["within_row_auc"],

                "repair_auc":
                    r["within_row_auc"],

                "delta_auc":
                    (
                        r["within_row_auc"]
                        - b["within_row_auc"]
                    ),

                "bad_topk":
                    b["topk_selected_recall"],

                "repair_topk":
                    r["topk_selected_recall"],

                "delta_topk":
                    (
                        r["topk_selected_recall"]
                        - b["topk_selected_recall"]
                    ),

                "bad_raw_margin":
                    b["raw_selected_absent_margin"],

                "repair_raw_margin":
                    r["raw_selected_absent_margin"],

                "delta_raw_margin":
                    (
                        r["raw_selected_absent_margin"]
                        - b["raw_selected_absent_margin"]
                    ),

                "bad_llr_margin":
                    b["loglr_selected_absent_margin"],

                "repair_llr_margin":
                    r["loglr_selected_absent_margin"],

                "delta_llr_margin":
                    (
                        r["loglr_selected_absent_margin"]
                        - b["loglr_selected_absent_margin"]
                    ),

                "bad_selected_mean_loglr":
                    b["selected_mean_loglr"],

                "repair_selected_mean_loglr":
                    r["selected_mean_loglr"],

                "delta_selected_mean_loglr":
                    (
                        r["selected_mean_loglr"]
                        - b["selected_mean_loglr"]
                    ),
            })

    write_jsonl(
        OUT / "matched_metrics.jsonl",
        matched,
    )

    # ----------------------------------------------------
    # Overall activation metrics.
    # ----------------------------------------------------
    overall = {
        "matched_executions":
            len(matched),

        "activation_exact_rate":
            mean([
                int(x["activation_exact"])
                for x in matched
            ]),

        "activation_count_equal_rate":
            mean([
                int(x["activation_count_equal"])
                for x in matched
            ]),

        "mean_activation_bad_recall":
            mean([
                x["activation_bad_recall"]
                for x in matched
            ]),

        "mean_activation_jaccard":
            mean([
                x["activation_jaccard"]
                for x in matched
            ]),

        "mean_sensor_score_corr":
            mean([
                x["sensor_score_corr"]
                for x in matched
            ]),

        "mean_sensor_loglr_corr":
            mean([
                x["sensor_loglr_corr"]
                for x in matched
            ]),

        "bad_mean_auc":
            mean([
                x["bad_auc"]
                for x in matched
            ]),

        "repair_mean_auc":
            mean([
                x["repair_auc"]
                for x in matched
            ]),

        "mean_delta_auc":
            mean([
                x["delta_auc"]
                for x in matched
            ]),

        "bad_mean_topk":
            mean([
                x["bad_topk"]
                for x in matched
            ]),

        "repair_mean_topk":
            mean([
                x["repair_topk"]
                for x in matched
            ]),

        "mean_delta_topk":
            mean([
                x["delta_topk"]
                for x in matched
            ]),

        "bad_mean_llr_margin":
            mean([
                x["bad_llr_margin"]
                for x in matched
            ]),

        "repair_mean_llr_margin":
            mean([
                x["repair_llr_margin"]
                for x in matched
            ]),

        "mean_delta_llr_margin":
            mean([
                x["delta_llr_margin"]
                for x in matched
            ]),

        "bad_mean_selected_loglr":
            mean([
                x["bad_selected_mean_loglr"]
                for x in matched
            ]),

        "repair_mean_selected_loglr":
            mean([
                x["repair_selected_mean_loglr"]
                for x in matched
            ]),

        "mean_delta_selected_loglr":
            mean([
                x["delta_selected_mean_loglr"]
                for x in matched
            ]),
    }

    by_family = {}

    for fam in sorted(set(
        x["family"]
        for x in matched
    )):
        xs = [
            x for x in matched
            if x["family"] == fam
        ]

        by_family[fam] = {
            "n": len(xs),

            "activation_exact_rate":
                mean([
                    int(x["activation_exact"])
                    for x in xs
                ]),

            "activation_bad_recall":
                mean([
                    x["activation_bad_recall"]
                    for x in xs
                ]),

            "activation_jaccard":
                mean([
                    x["activation_jaccard"]
                    for x in xs
                ]),

            "sensor_score_corr":
                mean([
                    x["sensor_score_corr"]
                    for x in xs
                ]),

            "bad_auc":
                mean([
                    x["bad_auc"]
                    for x in xs
                ]),

            "repair_auc":
                mean([
                    x["repair_auc"]
                    for x in xs
                ]),

            "bad_llr_margin":
                mean([
                    x["bad_llr_margin"]
                    for x in xs
                ]),

            "repair_llr_margin":
                mean([
                    x["repair_llr_margin"]
                    for x in xs
                ]),
        }

    summary = {
        "overall": overall,
        "by_family": by_family,
    }

    (OUT / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== SUMMARY =====")
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print()
    print("===== PER PAIR =====")

    for pair_id in range(12):
        xs = [
            x for x in matched
            if x["pair_id"] == pair_id
        ]

        print(
            f"PAIR {pair_id:02d} "
            f"{xs[0]['family']:<16} "
            f"exact={mean([int(x['activation_exact']) for x in xs]):.3f} "
            f"jac={mean([x['activation_jaccard'] for x in xs]):.3f} "
            f"score_corr={mean([x['sensor_score_corr'] for x in xs]):.3f} "
            f"AUC bad→repair="
            f"{mean([x['bad_auc'] for x in xs]):.3f}"
            f"→"
            f"{mean([x['repair_auc'] for x in xs]):.3f} "
            f"LLRmargin="
            f"{mean([x['bad_llr_margin'] for x in xs]):+.3f}"
            f"→"
            f"{mean([x['repair_llr_margin'] for x in xs]):+.3f}"
        )


if __name__ == "__main__":
    main()
