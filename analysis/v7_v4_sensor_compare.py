from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path("analysis").resolve()))

import build_visible_realized_gain_v3 as vg
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


V4_EXEC = Path(
    "artifacts/v7_substantive_repair_v4/executions.jsonl"
)

BAD_SCORED = Path(
    "artifacts/v7_retention_sensor_v1/scored_executions.jsonl"
)

OUT = Path(
    "artifacts/v7_substantive_repair_v4/sensor"
)

ADAPTER = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/best_adapter"
)

CAL = Path(
    "artifacts/visible_realized_gain_v3/channel_calibration.json"
)

MAX_LENGTH = 1024


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    with Path(p).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def mean(xs):
    xs = [
        float(x) for x in xs
        if x is not None and np.isfinite(x)
    ]
    return float(np.mean(xs)) if xs else None


def corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if len(a) < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None

    return float(np.corrcoef(a, b)[0, 1])


def row_metrics(scores, probs, llr, selected, roles):
    selected = set(selected)

    y = np.asarray(
        [int(r in selected) for r in roles],
        dtype=int,
    )

    scores = np.asarray(scores, dtype=float)
    probs = np.asarray(probs, dtype=float)
    llr = np.asarray(llr, dtype=float)

    pos = y == 1
    neg = y == 0

    if pos.sum() == 0 or neg.sum() == 0:
        return {
            "within_row_auc": None,
            "topk_selected_recall": None,
            "raw_selected_absent_margin": None,
            "prob_selected_absent_margin": None,
            "selected_mean_loglr": None,
            "absent_mean_loglr": None,
            "loglr_selected_absent_margin": None,
        }

    auc = float(roc_auc_score(y, scores))

    k = int(pos.sum())
    order = np.argsort(-scores)[:k]

    return {
        "within_row_auc":
            auc,

        "topk_selected_recall":
            float(y[order].sum() / k),

        "raw_selected_absent_margin":
            float(scores[pos].mean() - scores[neg].mean()),

        "prob_selected_absent_margin":
            float(probs[pos].mean() - probs[neg].mean()),

        "selected_mean_loglr":
            float(llr[pos].mean()),

        "absent_mean_loglr":
            float(llr[neg].mean()),

        "loglr_selected_absent_margin":
            float(llr[pos].mean() - llr[neg].mean()),
    }


def activation_metrics(bad, v4):
    a = set(bad)
    b = set(v4)

    inter = a & b
    union = a | b

    return {
        "activation_exact":
            bool(a == b),

        "activation_count_equal":
            bool(len(a) == len(b)),

        "activation_bad_recall":
            len(inter) / len(a) if a else None,

        "activation_jaccard":
            len(inter) / len(union) if union else 1.0,
    }


def summarize(rows):
    return {
        "n": len(rows),

        "activation_exact_rate":
            mean([int(x["activation_exact"]) for x in rows]),

        "activation_count_equal_rate":
            mean([int(x["activation_count_equal"]) for x in rows]),

        "activation_bad_recall":
            mean([x["activation_bad_recall"] for x in rows]),

        "activation_jaccard":
            mean([x["activation_jaccard"] for x in rows]),

        "sensor_score_corr":
            mean([x["sensor_score_corr"] for x in rows]),

        "bad_auc":
            mean([x["bad_auc"] for x in rows]),

        "v4_auc":
            mean([x["v4_auc"] for x in rows]),

        "delta_auc":
            mean([x["delta_auc"] for x in rows]),

        "bad_topk":
            mean([x["bad_topk"] for x in rows]),

        "v4_topk":
            mean([x["v4_topk"] for x in rows]),

        "delta_topk":
            mean([x["delta_topk"] for x in rows]),

        "bad_llr_margin":
            mean([x["bad_llr_margin"] for x in rows]),

        "v4_llr_margin":
            mean([x["v4_llr_margin"] for x in rows]),

        "delta_llr_margin":
            mean([x["delta_llr_margin"] for x in rows]),

        "bad_selected_loglr":
            mean([x["bad_selected_loglr"] for x in rows]),

        "v4_selected_loglr":
            mean([x["v4_selected_loglr"] for x in rows]),

        "delta_selected_loglr":
            mean([x["delta_selected_loglr"] for x in rows]),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    v4_rows = [
        x for x in read_jsonl(V4_EXEC)
        if x["execution_status"] == "ok"
    ]

    assert len(v4_rows) == 36, len(v4_rows)

    bad_rows = [
        x for x in read_jsonl(BAD_SCORED)
        if x["kind"] == "bad"
    ]

    assert len(bad_rows) == 36, len(bad_rows)

    bad_by_key = {
        (int(x["pair_id"]), int(x["rep"])): x
        for x in bad_rows
    }

    roles = list(TRAIN_ROLES)

    cal = json.loads(
        CAL.read_text(encoding="utf-8")
    )

    print("loading V13.8b...", flush=True)
    tok, model, dtype = vg.load_sensor(ADAPTER)

    scored_v4 = []

    for i, x in enumerate(v4_rows, 1):

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

        m = row_metrics(
            scores,
            probs,
            llr,
            x["selected_roles_eval_only"],
            roles,
        )

        z = {
            **x,
            "sensor_scores": scores.tolist(),
            "sensor_probs": probs.tolist(),
            "sensor_loglr": llr.tolist(),
            **m,
        }

        scored_v4.append(z)

        print(
            f"[{i:02d}/36] "
            f"pair={x['pair_id']:02d} "
            f"rep={x['rep']} "
            f"auc={m['within_row_auc']:.3f} "
            f"topk={m['topk_selected_recall']:.3f} "
            f"llr_margin={m['loglr_selected_absent_margin']:+.3f}",
            flush=True,
        )

    write_jsonl(
        OUT / "scored_v4.jsonl",
        scored_v4,
    )

    matched = []

    for v4 in scored_v4:
        key = (
            int(v4["pair_id"]),
            int(v4["rep"]),
        )

        bad = bad_by_key[key]

        am = activation_metrics(
            bad["selected_roles_eval_only"],
            v4["selected_roles_eval_only"],
        )

        matched.append({
            "pair_id":
                int(v4["pair_id"]),

            "rep":
                int(v4["rep"]),

            "family":
                v4["family"],

            "session_id":
                v4["session_id"],

            "bad_query":
                bad["query"],

            "v4_query":
                v4["query"],

            "bad_selected":
                bad["selected_roles_eval_only"],

            "v4_selected":
                v4["selected_roles_eval_only"],

            **am,

            "sensor_score_corr":
                corr(
                    bad["sensor_scores"],
                    v4["sensor_scores"],
                ),

            "sensor_loglr_corr":
                corr(
                    bad["sensor_loglr"],
                    v4["sensor_loglr"],
                ),

            "bad_auc":
                bad["within_row_auc"],

            "v4_auc":
                v4["within_row_auc"],

            "delta_auc":
                v4["within_row_auc"]
                - bad["within_row_auc"],

            "bad_topk":
                bad["topk_selected_recall"],

            "v4_topk":
                v4["topk_selected_recall"],

            "delta_topk":
                v4["topk_selected_recall"]
                - bad["topk_selected_recall"],

            "bad_llr_margin":
                bad["loglr_selected_absent_margin"],

            "v4_llr_margin":
                v4["loglr_selected_absent_margin"],

            "delta_llr_margin":
                v4["loglr_selected_absent_margin"]
                - bad["loglr_selected_absent_margin"],

            "bad_selected_loglr":
                bad["selected_mean_loglr"],

            "v4_selected_loglr":
                v4["selected_mean_loglr"],

            "delta_selected_loglr":
                v4["selected_mean_loglr"]
                - bad["selected_mean_loglr"],
        })

    write_jsonl(
        OUT / "matched_bad_v4.jsonl",
        matched,
    )

    by_family = {}

    for fam in sorted(set(
        x["family"] for x in matched
    )):
        xs = [
            x for x in matched
            if x["family"] == fam
        ]
        by_family[fam] = summarize(xs)

    exact = [
        x for x in matched
        if x["activation_exact"]
    ]

    summary = {
        "overall":
            summarize(matched),

        "exact_routing_only":
            summarize(exact),

        "by_family":
            by_family,
    }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))

    print("\n===== PER PAIR =====")

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
            f"AUC={mean([x['bad_auc'] for x in xs]):.3f}"
            f"→{mean([x['v4_auc'] for x in xs]):.3f} "
            f"LLR={mean([x['bad_llr_margin'] for x in xs]):+.3f}"
            f"→{mean([x['v4_llr_margin'] for x in xs]):+.3f}"
        )


if __name__ == "__main__":
    main()
