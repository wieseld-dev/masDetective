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


BAD_V3 = Path(
    "artifacts/v7_retention_sensor_v1/"
    "scored_executions.jsonl"
)

V4 = Path(
    "artifacts/v7_substantive_repair_v4/"
    "sensor/scored_v4.jsonl"
)

ADAPTER = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/"
    "best_adapter"
)

CAL = Path(
    "artifacts/visible_realized_gain_v3/"
    "channel_calibration.json"
)

OUT = Path(
    "artifacts/v7_query_output_ablation_v1"
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
        float(x)
        for x in xs
        if x is not None and np.isfinite(x)
    ]
    return float(np.mean(xs)) if xs else None


def score_metrics(scores, selected, roles, cal):
    scores = np.asarray(scores, dtype=float)

    y = np.asarray(
        [int(r in set(selected)) for r in roles],
        dtype=int,
    )

    llr = vg.sensor_log_lr(
        scores,
        roles,
        cal,
    )

    pos = y == 1
    neg = y == 0

    if pos.sum() == 0 or neg.sum() == 0:
        return {
            "auc": None,
            "topk": None,
            "llr_margin": None,
            "selected_loglr": None,
        }

    auc = float(
        roc_auc_score(y, scores)
    )

    k = int(pos.sum())
    order = np.argsort(-scores)[:k]

    return {
        "auc": auc,

        "topk":
            float(y[order].sum() / k),

        "llr_margin":
            float(
                llr[pos].mean()
                - llr[neg].mean()
            ),

        "selected_loglr":
            float(llr[pos].mean()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(BAD_V3)

    bad = {
        (int(x["pair_id"]), int(x["rep"])): x
        for x in rows
        if x["kind"] == "bad"
    }

    v3 = {
        (int(x["pair_id"]), int(x["rep"])): x
        for x in rows
        if x["kind"] == "repair"
    }

    v4 = {
        (int(x["pair_id"]), int(x["rep"])): x
        for x in read_jsonl(V4)
    }

    assert len(bad) == 36
    assert len(v3) == 36
    assert len(v4) == 36

    roles = list(TRAIN_ROLES)

    cal = json.loads(
        CAL.read_text(encoding="utf-8")
    )

    print("loading sensor...", flush=True)
    tok, model, dtype = vg.load_sensor(ADAPTER)

    experiments = []

    for version, repairs in [
        ("v3", v3),
        ("v4", v4),
    ]:
        for key in sorted(bad):
            b = bad[key]
            r = repairs[key]

            # Critical control:
            # only use rows where routing is identical.
            if set(b["selected_roles_eval_only"]) != set(
                r["selected_roles_eval_only"]
            ):
                continue

            pair_id, rep = key
            selected = b["selected_roles_eval_only"]

            combos = {
                # Original baseline
                "BB": (
                    b["query"],
                    b["final_output"],
                ),

                # BAD query + benign output
                "BR": (
                    b["query"],
                    r["final_output"],
                ),

                # benign query + BAD output
                "RB": (
                    r["query"],
                    b["final_output"],
                ),

                # fully benign
                "RR": (
                    r["query"],
                    r["final_output"],
                ),
            }

            for combo, (q, o) in combos.items():

                # Reuse already-computed diagonal scores where possible.
                if combo == "BB":
                    scores = np.asarray(
                        b["sensor_scores"],
                        dtype=float,
                    )

                elif combo == "RR":
                    scores = np.asarray(
                        r["sensor_scores"],
                        dtype=float,
                    )

                else:
                    scores = vg.sensor_score_row(
                        model,
                        tok,
                        dtype,
                        roles,
                        ROLE_DESCRIPTIONS,
                        q,
                        o,
                        MAX_LENGTH,
                    )

                m = score_metrics(
                    scores,
                    selected,
                    roles,
                    cal,
                )

                experiments.append({
                    "version": version,
                    "pair_id": pair_id,
                    "rep": rep,
                    "family": b["family"],
                    "combo": combo,
                    "query_source":
                        "bad" if combo[0] == "B" else version,
                    "output_source":
                        "bad" if combo[1] == "B" else version,
                    "selected_roles": selected,
                    "query": q,
                    "output": o,
                    **m,
                })

    write_jsonl(
        OUT / "rows.jsonl",
        experiments,
    )

    summary = {}

    for version in ["v3", "v4"]:
        xs = [
            x for x in experiments
            if x["version"] == version
        ]

        summary[version] = {}

        for combo in ["BB", "BR", "RB", "RR"]:
            ys = [
                x for x in xs
                if x["combo"] == combo
            ]

            summary[version][combo] = {
                "n": len(ys),
                "auc": mean([x["auc"] for x in ys]),
                "topk": mean([x["topk"] for x in ys]),
                "llr_margin":
                    mean([x["llr_margin"] for x in ys]),
                "selected_loglr":
                    mean([x["selected_loglr"] for x in ys]),
            }

        bb = summary[version]["BB"]
        br = summary[version]["BR"]
        rb = summary[version]["RB"]
        rr = summary[version]["RR"]

        summary[version]["effects"] = {
            # Hold BAD query fixed, change only output.
            "output_effect_auc":
                br["auc"] - bb["auc"],

            # Hold BAD output fixed, change only query.
            "query_effect_auc":
                rb["auc"] - bb["auc"],

            "output_effect_llr":
                br["llr_margin"]
                - bb["llr_margin"],

            "query_effect_llr":
                rb["llr_margin"]
                - bb["llr_margin"],

            "full_effect_auc":
                rr["auc"] - bb["auc"],

            "full_effect_llr":
                rr["llr_margin"]
                - bb["llr_margin"],
        }

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\n===== QUERY / OUTPUT ABLATION =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
