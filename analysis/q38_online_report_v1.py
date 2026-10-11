from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(
    "artifacts/q38_online_r7match_v1"
)

BASELINE = Path(
    "artifacts/v14a_ckpt_eval_r07_v1/"
    "advance_dev_round_06_summary.json"
)

Q38 = (
    ROOT
    / "advance_dev_round_06_summary.json"
)


if not BASELINE.is_file():
    raise RuntimeError(
        f"missing baseline: {BASELINE}"
    )

if not Q38.is_file():
    raise RuntimeError(
        f"missing q38 result: {Q38}"
    )


r7 = json.loads(
    BASELINE.read_text()
)

q38 = json.loads(
    Q38.read_text()
)


keys = [
    "mean_role_auc",
    "mean_role_log_loss",
    "mean_role_brier",
    "mean_true_false_margin",
    "mean_oracle_k_role_recall",
]


comparison = {}

for k in keys:
    if k in r7 and k in q38:
        comparison[k] = {
            "R7": r7[k],
            "Q38_online": q38[k],
            "Q38_minus_R7":
                q38[k] - r7[k],
        }


out = {
    "experiment":
        "q38_online_grpo_r7match_v1",

    "training_protocol":
        {
            "rounds_1_6_group_size": 6,
            "round_7_group_size": 50,
            "rounds_1_6_lr": 3e-6,
            "rounds_1_6_epochs": 2,
            "round_7_lr": 1.5e-6,
            "round_7_epochs": 1,
            "clip": 0.20,
            "kl_beta": 0.05,
            "round_7_sign_consistent":
                True,
        },

    "evaluation":
        "same 8 held-out DEV systems, "
        "same six-query adaptive budget",

    "R7_baseline": r7,
    "Q38_online": q38,
    "comparison": comparison,
}


path = (
    ROOT
    / "q38_vs_r7_exact_dev_comparison.json"
)

path.write_text(
    json.dumps(
        out,
        indent=2,
    )
    + "\n"
)


print("=" * 100)
print("Q38 ONLINE GRPO vs ORIGINAL R7")
print("=" * 100)

for k, v in comparison.items():
    print()
    print(k)
    print("  R7        =", v["R7"])
    print("  Q38       =", v["Q38_online"])
    print("  delta     =", v["Q38_minus_R7"])

print()
print("SAVED =", path)
print("Q38_R7_EXACT_COMPARISON_OK")
