#!/usr/bin/env python3
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

import analysis.build_visible_realized_gain_v3 as vg
import analysis.train_v13_6b_activation_judge as tr

TEST_ROLES = (
    "code",
    "planning",
    "research",
    "database_design",
    "writing",
    "fact_checking",
    "experiment_design",
)

TRUE_ROLES = {"code", "planning", "research"}

def main():
    src = Path("artifacts/k3_gap_sanity_v1/result.json")
    out = Path("artifacts/k3_rescore20_diag")
    out.mkdir(parents=True, exist_ok=True)

    data = json.loads(src.read_text())
    cal = json.loads(
        Path("artifacts/visible_realized_gain_v3/channel_calibration.json").read_text()
    )

    desc = dict(tr.ROLE_DESCRIPTIONS)
    full_roles = list(desc.keys())

    print("FULL SENSOR ROLE COUNT:", len(full_roles))
    print("FULL SENSOR ROLES:", full_roles)

    missing = [r for r in TEST_ROLES if r not in desc]
    if missing:
        raise RuntimeError(f"missing roles: {missing}")

    tok, sensor, dtype = vg.load_sensor(
        Path("artifacts/v13_8b_balanced_repair_2048_run1/best_adapter")
    )

    max_lr = {r: (-1e9, None) for r in TEST_ROLES}
    max_prob = {r: (-1.0, None) for r in TEST_ROLES}
    actual_values = defaultdict(list)
    confusion = Counter()
    rows = []

    for rec in data["records"]:
        q = rec["query"]
        answer = rec["answer"]
        selected = rec.get("selected_roles_eval_only", [])

        # IMPORTANT: calibration-compatible protocol:
        # all 20 roles scored together against same response.
        scores = vg.sensor_score_row(
            sensor, tok, dtype,
            full_roles, desc, q, answer, 4096
        )
        probs = vg.calibrated_probs(scores, full_roles, cal)
        lrs = vg.sensor_log_lr(scores, full_roles, cal)

        score_map = dict(zip(full_roles, map(float, scores)))
        prob_map = dict(zip(full_roles, map(float, probs)))
        lr_map = dict(zip(full_roles, map(float, lrs)))

        ranked7 = sorted(TEST_ROLES, key=lambda r: -lr_map[r])
        top1 = ranked7[0]

        for r in TEST_ROLES:
            if lr_map[r] > max_lr[r][0]:
                max_lr[r] = (lr_map[r], rec["probe"])
            if prob_map[r] > max_prob[r][0]:
                max_prob[r] = (prob_map[r], rec["probe"])

        actual_rank = None
        actual_lr = None
        if selected:
            a = selected[0]
            if a in TEST_ROLES:
                actual_rank = ranked7.index(a) + 1
                actual_lr = lr_map[a]
                actual_values[a].append(lr_map[a])
                confusion[(a, top1)] += 1

        print("\n" + "=" * 88)
        print(f"PROBE {rec['probe']}")
        print("ACTUAL [EVAL ONLY]:", selected)
        print("TOP1:", top1)
        print("ACTUAL RANK:", actual_rank)
        print("ACTUAL logLR:", actual_lr)

        for rank, r in enumerate(ranked7, 1):
            flag = " <-- ACTUAL" if r in selected else ""
            print(
                f"{rank}. {r:<22} "
                f"p={prob_map[r]:.4f} "
                f"logLR={lr_map[r]:+8.4f}{flag}"
            )

        rows.append({
            "probe": rec["probe"],
            "query": q,
            "selected_eval_only": selected,
            "top1": top1,
            "actual_rank": actual_rank,
            "actual_loglr": actual_lr,
            "scores_7": {r: score_map[r] for r in TEST_ROLES},
            "probs_7": {r: prob_map[r] for r in TEST_ROLES},
            "loglr_7": {r: lr_map[r] for r in TEST_ROLES},
        })

    ranked_max = sorted(TEST_ROLES, key=lambda r: -max_lr[r][0])
    vals = np.array([max_lr[r][0] for r in ranked_max])
    gaps = vals[:-1] - vals[1:]
    k_hat = int(np.argmax(gaps) + 1)

    print("\n" + "#" * 88)
    print("MAX-EVIDENCE RANKING — 20-ROLE SENSOR PROTOCOL")
    for i, r in enumerate(ranked_max, 1):
        mark = " TRUE" if r in TRUE_ROLES else ""
        print(
            f"{i}. {r:<22} "
            f"maxLR={max_lr[r][0]:+8.4f} "
            f"probe={max_lr[r][1]}{mark}"
        )

    print("\nGAPS:")
    for i, x in enumerate(gaps, 1):
        print(f"after {i}: {x:.6f}")

    print("\nK_hat(max-gap):", k_hat)
    print("Top-K:", ranked_max[:k_hat])

    print("\nACTUAL -> SENSOR TOP1")
    for (a, p), n in sorted(confusion.items()):
        print(f"{a:<22} -> {p:<22} n={n}")

    summary = {
        "protocol": "20_roles_same_response_replay",
        "api_calls": 0,
        "ranking_by_max_loglr": ranked_max,
        "max_loglr": {r: max_lr[r][0] for r in TEST_ROLES},
        "max_prob": {r: max_prob[r][0] for r in TEST_ROLES},
        "gaps": gaps.tolist(),
        "k_hat": k_hat,
        "top_k": ranked_max[:k_hat],
        "true_roles_eval_only": sorted(TRUE_ROLES),
        "top1_correct": sum(
            n for (a,p),n in confusion.items() if a == p
        ),
        "top1_total": sum(confusion.values()),
        "rows": rows,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

if __name__ == "__main__":
    main()
