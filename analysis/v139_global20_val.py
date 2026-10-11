#!/usr/bin/env python3

import gc
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import analysis.train_v13_6b_activation_judge as tr
import analysis.build_visible_realized_gain_v3 as vg
import analysis.v139_overnight as v139


BASE = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/best_adapter"
)
NEW = Path(
    "artifacts/v139_pilots/mix50_retry/best_adapter"
)
OUT = Path("artifacts/v139_global20_val.json")

role_desc = dict(tr.ROLE_DESCRIPTIONS)
roles = list(role_desc)

print("ROLE COUNT:", len(roles))
print("ROLES:", roles)

assert len(roles) == 20, (
    f"Expected 20-role universe, got {len(roles)}"
)

rows = [
    x for x in v139.load_worker_records()
    if x["split"] == "val"
]

assert len(rows) == 108, len(rows)

examples = []

for x in rows:
    examples.append({
        "query_id": x["query_id"],
        "query": x["query"],
        "answer": x["output_a"],
        "true_role": x["role_a"],
        "paired_other": x["role_b"],
        "stratum": x["semantic_stratum"],
    })

    examples.append({
        "query_id": x["query_id"],
        "query": x["query"],
        "answer": x["output_b"],
        "true_role": x["role_b"],
        "paired_other": x["role_a"],
        "stratum": x["semantic_stratum"],
    })

assert len(examples) == 216


def evaluate(adapter):
    tok, model, dtype = vg.load_sensor(adapter)

    records = []

    for i, x in enumerate(examples, 1):
        scores = vg.sensor_score_row(
            model=model,
            tok=tok,
            dtype=dtype,
            roles=roles,
            role_desc=role_desc,
            query=x["query"],
            answer=x["answer"],
            max_length=4096,
        )

        order = np.argsort(-scores)
        ranked = [roles[j] for j in order]

        true_idx = roles.index(x["true_role"])
        other_idx = roles.index(x["paired_other"])

        true_rank = ranked.index(x["true_role"]) + 1
        pred = ranked[0]

        rec = {
            "query_id": x["query_id"],
            "stratum": x["stratum"],
            "true_role": x["true_role"],
            "paired_other": x["paired_other"],
            "pred_role": pred,
            "true_rank": true_rank,
            "true_score": float(scores[true_idx]),
            "paired_other_score": float(scores[other_idx]),
            "pairwise_correct":
                bool(scores[true_idx] > scores[other_idx]),
            "top3_correct": bool(true_rank <= 3),
            "reciprocal_rank": 1.0 / true_rank,
            "top5": ranked[:5],
        }

        records.append(rec)

        if i % 20 == 0 or i == len(examples):
            print(
                f"{adapter.name}: {i}/{len(examples)}",
                flush=True,
            )

    top1 = np.mean([
        r["true_rank"] == 1 for r in records
    ])
    top3 = np.mean([
        r["true_rank"] <= 3 for r in records
    ])
    pairwise = np.mean([
        r["pairwise_correct"] for r in records
    ])
    mrr = np.mean([
        r["reciprocal_rank"] for r in records
    ])
    ranks = np.asarray([
        r["true_rank"] for r in records
    ])

    by_stratum = {}

    for s in sorted(set(r["stratum"] for r in records)):
        rr = [r for r in records if r["stratum"] == s]

        by_stratum[s] = {
            "n": len(rr),
            "top1": float(np.mean([
                z["true_rank"] == 1 for z in rr
            ])),
            "top3": float(np.mean([
                z["true_rank"] <= 3 for z in rr
            ])),
            "pairwise_accuracy": float(np.mean([
                z["pairwise_correct"] for z in rr
            ])),
            "mrr": float(np.mean([
                z["reciprocal_rank"] for z in rr
            ])),
        }

    by_true_role = {}

    for role in roles:
        rr = [
            r for r in records
            if r["true_role"] == role
        ]

        if not rr:
            continue

        by_true_role[role] = {
            "n": len(rr),
            "top1": float(np.mean([
                z["true_rank"] == 1 for z in rr
            ])),
            "mean_rank": float(np.mean([
                z["true_rank"] for z in rr
            ])),
        }

    wrong_preds = Counter(
        r["pred_role"]
        for r in records
        if r["true_rank"] != 1
    )

    confusions = Counter(
        (r["true_role"], r["pred_role"])
        for r in records
        if r["true_rank"] != 1
    )

    summary = {
        "n": len(records),
        "top1": float(top1),
        "top3": float(top3),
        "pairwise_accuracy": float(pairwise),
        "mrr": float(mrr),
        "mean_true_rank": float(np.mean(ranks)),
        "median_true_rank": float(np.median(ranks)),
        "rank_histogram": {
            str(k): int(np.sum(ranks == k))
            for k in range(1, 21)
            if np.sum(ranks == k)
        },
        "by_stratum": by_stratum,
        "by_true_role": by_true_role,
        "top_wrong_predicted_roles":
            wrong_preds.most_common(10),
        "top_confusions": [
            {
                "true": a,
                "pred": b,
                "n": n,
            }
            for (a, b), n
            in confusions.most_common(20)
        ],
    }

    del model
    del tok
    gc.collect()
    torch.cuda.empty_cache()

    return summary


print("\n===== V13.8b =====")
base = evaluate(BASE)

print("\n===== V13.9 MIX50 =====")
new = evaluate(NEW)

result = {
    "protocol": "v139_global20_val_v1",
    "split": "VAL_ONLY",
    "candidate_roles": roles,
    "n_visible_outputs": len(examples),
    "baseline": base,
    "v139_mix50": new,
    "delta": {
        "top1":
            new["top1"] - base["top1"],
        "top3":
            new["top3"] - base["top3"],
        "pairwise_accuracy":
            new["pairwise_accuracy"]
            - base["pairwise_accuracy"],
        "mrr":
            new["mrr"] - base["mrr"],
        "mean_true_rank":
            new["mean_true_rank"]
            - base["mean_true_rank"],
    },
}

OUT.write_text(
    json.dumps(
        result,
        indent=2,
        ensure_ascii=False,
    )
)

print("\n" + "=" * 100)
print(json.dumps(result, indent=2))
