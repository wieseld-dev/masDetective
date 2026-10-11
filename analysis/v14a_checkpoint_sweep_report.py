from pathlib import Path
import json
import statistics

ROOTS = {
    "R6" : Path("artifacts/v14_adaptive_g50_v2_eval_old_r6"),
    "R7" : Path("artifacts/v14a_ckpt_eval_r07_v1"),
    "R8" : Path("artifacts/v14a_ckpt_eval_r08_v1"),
    "R9" : Path("artifacts/v14a_ckpt_eval_r09_v1"),
    "R10": Path("artifacts/v14_adaptive_g50_v2_eval_new_r10"),
}

KEYS = [
    "role_log_loss",
    "role_brier",
    "role_auc",
    "true_false_margin",
]

def load(p):
    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]

def metrics(root, rnd):
    rows = load(root / f"metrics_dev_round_{rnd:02d}.jsonl")
    return {
        k: statistics.mean(float(x[k]) for x in rows)
        for k in KEYS
    }

print("=" * 115)
print("V14 CHECKPOINT SWEEP — SAME CLEAN DEV / SAME 6-QUERY BUDGET")
print("=" * 115)

print(
    "CKPT | "
    "Q1 LogLoss  Q3 LogLoss  Q6 LogLoss | "
    "Q6 AUC   Q6 Brier  Q6 Margin"
)
print("-" * 115)

allm = {}

for name, root in ROOTS.items():

    ms = {
        r: metrics(root, r)
        for r in range(1, 7)
    }

    allm[name] = ms

    print(
        f"{name:4s} | "
        f"{ms[1]['role_log_loss']:.6f}    "
        f"{ms[3]['role_log_loss']:.6f}    "
        f"{ms[6]['role_log_loss']:.6f} | "
        f"{ms[6]['role_auc']:.4f}   "
        f"{ms[6]['role_brier']:.6f}  "
        f"{ms[6]['true_false_margin']:+.6f}"
    )

print()
print("=" * 115)
print("LOGLOSS BY QUERY BUDGET")
print("=" * 115)

header = "Q " + "".join(f"{x:>12s}" for x in ROOTS)
print(header)
print("-" * len(header))

for r in range(1, 7):
    print(
        f"{r} " +
        "".join(
            f"{allm[name][r]['role_log_loss']:12.6f}"
            for name in ROOTS
        )
    )

print()
print("=" * 115)
print("AUC BY QUERY BUDGET")
print("=" * 115)

print(header)
print("-" * len(header))

for r in range(1, 7):
    print(
        f"{r} " +
        "".join(
            f"{allm[name][r]['role_auc']:12.4f}"
            for name in ROOTS
        )
    )

print()
print("=" * 115)

best_final = min(
    ROOTS,
    key=lambda name: allm[name][6]["role_log_loss"]
)

best_auc = max(
    ROOTS,
    key=lambda name: allm[name][6]["role_auc"]
)

print("BEST FINAL LOGLOSS :", best_final,
      allm[best_final][6]["role_log_loss"])

print("BEST FINAL AUC     :", best_auc,
      allm[best_auc][6]["role_auc"])

print("=" * 115)
