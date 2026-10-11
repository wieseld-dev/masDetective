from __future__ import annotations

import json
import random
import sys
from pathlib import Path

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

CAL = Path(
    "artifacts/visible_realized_gain_v3/"
    "channel_calibration.json"
)

ADAPTER = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/"
    "best_adapter"
)

OUT = Path(
    "artifacts/v7_query_dependency_ablation_v1"
)

MAX_LENGTH = 1024
SEED = 20260926

NEUTRAL_QUERY = (
    "Please help me with this request."
)


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


def metrics(scores, selected, roles, cal):
    scores = np.asarray(
        scores,
        dtype=float,
    )

    selected = set(
        map(str, selected)
    )

    y = np.asarray(
        [
            int(r in selected)
            for r in roles
        ],
        dtype=int,
    )

    pos = y == 1
    neg = y == 0

    if (
        pos.sum() == 0
        or neg.sum() == 0
    ):
        return {
            "auc": None,
            "topk": None,
            "llr_margin": None,
            "selected_loglr": None,
        }

    auc = float(
        roc_auc_score(
            y,
            scores,
        )
    )

    k = int(
        pos.sum()
    )

    top = np.argsort(
        -scores
    )[:k]

    topk = float(
        y[top].sum() / k
    )

    llr = vg.sensor_log_lr(
        scores,
        roles,
        cal,
    )

    return {
        "auc":
            auc,

        "topk":
            topk,

        "llr_margin":
            float(
                llr[pos].mean()
                - llr[neg].mean()
            ),

        "selected_loglr":
            float(
                llr[pos].mean()
            ),
    }


def build_versions():
    x = read_jsonl(
        BAD_V3
    )

    bad = [
        z for z in x
        if z["kind"] == "bad"
    ]

    v3 = [
        z for z in x
        if z["kind"] == "repair"
    ]

    v4 = read_jsonl(
        V4
    )

    assert len(bad) == 36
    assert len(v3) == 36
    assert len(v4) == 36

    return {
        "BAD": bad,
        "V3": v3,
        "V4": v4,
    }


def shuffled_queries(rows):
    rng = random.Random(SEED)

    result = {}

    indices = list(
        range(len(rows))
    )

    for i, row in enumerate(rows):
        # Force different session_id.
        candidates = [
            j
            for j in indices
            if (
                j != i
                and str(
                    rows[j]["session_id"]
                )
                != str(
                    row["session_id"]
                )
            )
        ]

        assert candidates

        j = rng.choice(
            candidates
        )

        result[i] = str(
            rows[j]["query"]
        )

    return result


def summarize(rows):
    out = {}

    for condition in [
        "original",
        "neutral",
        "shuffled",
    ]:
        xs = [
            x for x in rows
            if x["condition"] == condition
        ]

        out[condition] = {
            "n":
                len(xs),

            "auc":
                mean([
                    x["auc"]
                    for x in xs
                ]),

            "topk":
                mean([
                    x["topk"]
                    for x in xs
                ]),

            "llr_margin":
                mean([
                    x["llr_margin"]
                    for x in xs
                ]),

            "selected_loglr":
                mean([
                    x["selected_loglr"]
                    for x in xs
                ]),
        }

    orig = out["original"]
    neu = out["neutral"]
    shf = out["shuffled"]

    out["effects"] = {
        "neutral_minus_original_auc":
            neu["auc"]
            - orig["auc"],

        "shuffled_minus_original_auc":
            shf["auc"]
            - orig["auc"],

        "neutral_minus_original_llr":
            neu["llr_margin"]
            - orig["llr_margin"],

        "shuffled_minus_original_llr":
            shf["llr_margin"]
            - orig["llr_margin"],
    }

    return out


def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    versions = build_versions()

    roles = list(
        TRAIN_ROLES
    )

    cal = json.loads(
        CAL.read_text(
            encoding="utf-8"
        )
    )

    print(
        "loading V13.8b...",
        flush=True,
    )

    tok, model, dtype = (
        vg.load_sensor(
            ADAPTER
        )
    )

    results = []

    for version in [
        "BAD",
        "V3",
        "V4",
    ]:
        rows = versions[
            version
        ]

        shuffled = (
            shuffled_queries(
                rows
            )
        )

        print(
            f"\n===== {version} =====",
            flush=True,
        )

        for i, row in enumerate(
            rows
        ):
            selected = (
                row[
                    "selected_roles_eval_only"
                ]
            )

            # ORIGINAL:
            # reuse scores already computed
            # under canonical protocol.
            original_scores = (
                np.asarray(
                    row["sensor_scores"],
                    dtype=float,
                )
            )

            m = metrics(
                original_scores,
                selected,
                roles,
                cal,
            )

            results.append({
                "version":
                    version,

                "row_index":
                    i,

                "pair_id":
                    int(row["pair_id"]),

                "rep":
                    int(row["rep"]),

                "session_id":
                    str(row["session_id"]),

                "condition":
                    "original",

                "query":
                    str(row["query"]),

                "selected_roles":
                    list(selected),

                **m,
            })

            # NEUTRAL QUERY
            neutral_scores = (
                vg.sensor_score_row(
                    model,
                    tok,
                    dtype,
                    roles,
                    ROLE_DESCRIPTIONS,
                    NEUTRAL_QUERY,
                    row["final_output"],
                    MAX_LENGTH,
                )
            )

            m = metrics(
                neutral_scores,
                selected,
                roles,
                cal,
            )

            results.append({
                "version":
                    version,

                "row_index":
                    i,

                "pair_id":
                    int(row["pair_id"]),

                "rep":
                    int(row["rep"]),

                "session_id":
                    str(row["session_id"]),

                "condition":
                    "neutral",

                "query":
                    NEUTRAL_QUERY,

                "selected_roles":
                    list(selected),

                **m,
            })

            # SHUFFLED UNRELATED QUERY
            sq = shuffled[i]

            shuffled_scores = (
                vg.sensor_score_row(
                    model,
                    tok,
                    dtype,
                    roles,
                    ROLE_DESCRIPTIONS,
                    sq,
                    row["final_output"],
                    MAX_LENGTH,
                )
            )

            m = metrics(
                shuffled_scores,
                selected,
                roles,
                cal,
            )

            results.append({
                "version":
                    version,

                "row_index":
                    i,

                "pair_id":
                    int(row["pair_id"]),

                "rep":
                    int(row["rep"]),

                "session_id":
                    str(row["session_id"]),

                "condition":
                    "shuffled",

                "query":
                    sq,

                "selected_roles":
                    list(selected),

                **m,
            })

            print(
                f"[{i+1:02d}/36] "
                f"{version} "
                f"pair={row['pair_id']} "
                f"rep={row['rep']}",
                flush=True,
            )

    write_jsonl(
        OUT / "rows.jsonl",
        results,
    )

    summary = {}

    for version in [
        "BAD",
        "V3",
        "V4",
    ]:
        xs = [
            x for x in results
            if x["version"] == version
        ]

        summary[
            version
        ] = summarize(
            xs
        )

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== QUERY DEPENDENCY "
        "ABLATION ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
