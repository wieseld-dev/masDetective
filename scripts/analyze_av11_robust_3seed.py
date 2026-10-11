from pathlib import Path
import json
import itertools
import random
import statistics

ROOT = Path(
    "artifacts/adaptive_v11_role_only_v1/eval_robust_3seed"
)

METRICS = [
    ("role_log_loss", False),
    ("role_brier", False),
    ("role_auc", True),
    ("oracle_k_role_recall", True),
    ("true_false_margin", True),
]


def rows(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def load_metric_file(path):
    return {
        x["session_id"]: x
        for x in rows(path)
    }


seeds = sorted(
    x.name
    for x in ROOT.glob("seed_*")
    if x.is_dir()
)

print("=" * 76)
print("ADAPTIVE-V11 ROBUSTNESS EVALUATION")
print("=" * 76)
print("Seeds:", seeds)
print()


# ------------------------------------------------------------
# Validity
# ------------------------------------------------------------

print("VALIDITY")
print("-" * 76)

for seed in seeds:
    for policy in ("original", "grpo"):
        d = ROOT / seed / policy

        vals = []
        for r in (1, 2, 3):
            c = d / f"candidates_dev_round_{r:02d}.jsonl"
            s = d / f"scored_dev_round_{r:02d}.jsonl"

            nc = len(rows(c)) if c.exists() else 0
            ns = len(rows(s)) if s.exists() else 0
            vals.append(f"R{r}={ns}/{nc}")

        print(f"{seed:10s} {policy:8s} " + " ".join(vals))

print()


# ------------------------------------------------------------
# Curves across all seed × system trajectories
# ------------------------------------------------------------

print("MEAN TRAJECTORY")
print("-" * 76)

for r in (1, 2, 3):
    print(f"\nROUND {r}")

    for policy in ("original", "grpo"):
        all_rows = []

        for seed in seeds:
            p = (
                ROOT / seed / policy /
                f"metrics_dev_round_{r:02d}.jsonl"
            )
            if p.exists():
                all_rows.extend(rows(p))

        if not all_rows:
            continue

        print(
            f"{policy:8s} "
            f"logloss={statistics.mean(x['role_log_loss'] for x in all_rows):.6f} "
            f"brier={statistics.mean(x['role_brier'] for x in all_rows):.6f} "
            f"auc={statistics.mean(x['role_auc'] for x in all_rows):.6f} "
            f"recall={statistics.mean(x['oracle_k_role_recall'] for x in all_rows):.6f} "
            f"margin={statistics.mean(x['true_false_margin'] for x in all_rows):.6f}"
        )

print()


# ------------------------------------------------------------
# Pair Round 3 by seed + system
# ------------------------------------------------------------

pairs = []

for seed in seeds:
    op = ROOT / seed / "original" / "metrics_dev_round_03.jsonl"
    gp = ROOT / seed / "grpo" / "metrics_dev_round_03.jsonl"

    if not op.exists() or not gp.exists():
        continue

    o = load_metric_file(op)
    g = load_metric_file(gp)

    for sid in sorted(set(o) & set(g)):
        pairs.append((seed, sid, o[sid], g[sid]))

print("=" * 76)
print("ROUND-3 PAIRED RESULT")
print("=" * 76)
print("Paired trajectories:", len(pairs))
print()


for metric, higher_better in METRICS:
    ov = [p[2][metric] for p in pairs]
    gv = [p[3][metric] for p in pairs]

    delta = [g - o for o, g in zip(ov, gv)]

    print(metric)
    print(f"  Original mean : {statistics.mean(ov):.6f}")
    print(f"  GRPO mean     : {statistics.mean(gv):.6f}")
    print(f"  GRPO-Original : {statistics.mean(delta):+.6f}")

    if higher_better:
        wins = sum(d > 1e-12 for d in delta)
        losses = sum(d < -1e-12 for d in delta)
    else:
        wins = sum(d < -1e-12 for d in delta)
        losses = sum(d > 1e-12 for d in delta)

    ties = len(delta) - wins - losses

    print(
        f"  trajectory wins/ties/losses: "
        f"{wins}/{ties}/{losses}"
    )
    print()


# ------------------------------------------------------------
# PRIMARY TEST:
# role_log_loss clustered by hidden system.
#
# Each hidden system gets one delta = mean across seeds.
# This prevents pretending the repeated seeds are independent
# hidden systems.
# ------------------------------------------------------------

by_system = {}

for seed, sid, o, g in pairs:
    by_system.setdefault(sid, []).append(
        g["role_log_loss"] - o["role_log_loss"]
    )

system_delta = {
    sid: statistics.mean(v)
    for sid, v in by_system.items()
}

ds = list(system_delta.values())

print("=" * 76)
print("PRIMARY ENDPOINT: ROUND-3 ROLE LOG-LOSS")
print("GRPO - Original; NEGATIVE = GRPO BETTER")
print("=" * 76)

for sid, d in sorted(system_delta.items()):
    print(f"{sid:28s} {d:+.6f}")

obs = statistics.mean(ds)

print()
print(f"Systems                 : {len(ds)}")
print(f"Mean clustered delta    : {obs:+.6f}")
print(
    "Systems improved       : "
    f"{sum(d < 0 for d in ds)}/{len(ds)}"
)


# ------------------------------------------------------------
# Cluster bootstrap CI over systems
# ------------------------------------------------------------

rng = random.Random(20260916)
boots = []

for _ in range(20000):
    sample = [
        ds[rng.randrange(len(ds))]
        for _ in range(len(ds))
    ]
    boots.append(statistics.mean(sample))

boots.sort()

lo = boots[int(0.025 * len(boots))]
hi = boots[int(0.975 * len(boots)) - 1]

print(f"95% cluster bootstrap CI: [{lo:+.6f}, {hi:+.6f}]")


# ------------------------------------------------------------
# Exact paired sign-flip permutation test over systems
# H0: mean paired effect = 0
# ------------------------------------------------------------

abs_obs = abs(obs)
extreme = 0
total = 0

for signs in itertools.product((-1, 1), repeat=len(ds)):
    x = statistics.mean(
        d * s for d, s in zip(ds, signs)
    )
    total += 1
    if abs(x) >= abs_obs - 1e-15:
        extreme += 1

p = extreme / total

print(f"Exact sign-flip p       : {p:.6f}")

print()
if hi < 0:
    print(
        "INTERPRETATION: GRPO has a consistently lower "
        "final log-loss across these systems/seeds."
    )
elif lo > 0:
    print(
        "INTERPRETATION: Original V11 has a consistently lower "
        "final log-loss across these systems/seeds."
    )
else:
    print(
        "INTERPRETATION: effect direction is not yet stable enough "
        "to separate from zero on these 8 systems."
    )
