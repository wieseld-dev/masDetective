import json
import math
from pathlib import Path

from v11.role_universe import TRAIN_ROLES


ROOT = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1/"
    "data/run"
)

ROLES = list(map(str, TRAIN_ROLES))
ROUNDS = range(1, 26)

ALPHAS = [
    0.0,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
    0.75,
    1.00,
]


def load(path):
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sid(row):
    return str(
        row.get(
            "session_id",
            row.get("system_id", "")
        )
    )


def get_role_probs(row):
    p = row.get("role_probs")

    if isinstance(p, dict):
        return {
            str(k): float(v)
            for k, v in p.items()
        }

    for key in ["belief", "public_belief", "state"]:
        d = row.get(key)

        if isinstance(d, dict):
            p = d.get("role_probs")

            if isinstance(p, dict):
                return {
                    str(k): float(v)
                    for k, v in p.items()
                }

    raise RuntimeError(
        f"role_probs not found; keys={sorted(row.keys())}"
    )


def logit(p):
    p = min(
        max(float(p), 1e-8),
        1.0 - 1e-8
    )

    return math.log(
        p / (1.0 - p)
    )


def auc(scores, truth):
    truth = set(truth)

    pos = [
        float(scores[r])
        for r in ROLES
        if r in truth
    ]

    neg = [
        float(scores[r])
        for r in ROLES
        if r not in truth
    ]

    wins = 0.0

    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5

    return wins / (
        len(pos) * len(neg)
    )


def transform(mode, e):
    e = float(e)

    if mode == "symmetric":
        return e

    if mode == "positive_only":
        return max(0.0, e)

    if mode == "negative_only":
        return min(0.0, e)

    if mode == "sign":
        if e > 0:
            return 1.0
        if e < 0:
            return -1.0
        return 0.0

    if mode == "clip2":
        return max(
            -2.0,
            min(2.0, e)
        )

    if mode == "clip4":
        return max(
            -4.0,
            min(4.0, e)
        )

    raise ValueError(mode)


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]): set(map(str, x["roles"]))
    for x in manifest["systems"]
    if x.get("split") == "dev"
}

systems = sorted(truth)

MODES = [
    "symmetric",
    "positive_only",
    "negative_only",
    "sign",
    "clip2",
    "clip4",
]


# results[(mode, alpha)] = list of round records
results = {
    (mode, alpha): []
    for mode in MODES
    for alpha in ALPHAS
}

baseline_rounds = []


for rnd in ROUNDS:

    states = {
        sid(x): x
        for x in load(
            ROOT / f"states_dev_round_{rnd:02d}.jsonl"
        )
    }

    scored = {
        sid(x): x
        for x in load(
            ROOT / f"scored_dev_round_{rnd:02d}.jsonl"
        )
    }

    baseline_per_system = {}

    for s in systems:

        frozen = get_role_probs(
            states[s]
        )

        baseline_per_system[s] = auc(
            frozen,
            truth[s]
        )

    baseline_mean = (
        sum(baseline_per_system.values())
        / len(systems)
    )

    baseline_rounds.append({
        "round": rnd,
        "mean_auc": baseline_mean,
        "per_system": baseline_per_system,
    })


    for mode in MODES:

        for alpha in ALPHAS:

            per_system = {}

            for s in systems:

                frozen = get_role_probs(
                    states[s]
                )

                evidence = {
                    str(k): float(v)
                    for k, v in dict(
                        scored[s].get(
                            "evidence",
                            {}
                        )
                    ).items()
                }

                corrected = {}

                for role in ROLES:

                    base = logit(
                        frozen[role]
                    )

                    e = float(
                        evidence.get(
                            role,
                            0.0
                        )
                    )

                    corrected[role] = (
                        base
                        + alpha
                        * transform(
                            mode,
                            e
                        )
                    )

                per_system[s] = auc(
                    corrected,
                    truth[s]
                )

            mean_auc = (
                sum(per_system.values())
                / len(systems)
            )

            results[(mode, alpha)].append({
                "round": rnd,
                "mean_auc": mean_auc,
                "per_system": per_system,
            })


print("=" * 125)
print("FROZEN BELIEF + FRESH V13.5 CORRECTION")
print("=" * 125)

print()
print(
    "IMPORTANT: diagnostic on fixed existing trajectory; "
    "alpha is NOT selected deployment-validly here."
)

print()
print(
    f"{'MODE':16s} "
    f"{'ALPHA':>7s} "
    f"{'MEAN25':>9s} "
    f"{'LAST10':>9s} "
    f"{'FINAL':>9s} "
    f"{'PEAK':>9s} "
    f"{'PEAK_R':>7s} "
    f"{'>BASE':>7s} "
    f"{'<BASE':>7s}"
)


summary = []

baseline_mean25 = sum(
    x["mean_auc"]
    for x in baseline_rounds
) / len(baseline_rounds)

baseline_last10 = sum(
    x["mean_auc"]
    for x in baseline_rounds[-10:]
) / 10.0

baseline_final = baseline_rounds[-1]["mean_auc"]

baseline_peak = max(
    baseline_rounds,
    key=lambda x: x["mean_auc"]
)


print(
    f"{'FROZEN':16s} "
    f"{0.0:7.3f} "
    f"{baseline_mean25:9.6f} "
    f"{baseline_last10:9.6f} "
    f"{baseline_final:9.6f} "
    f"{baseline_peak['mean_auc']:9.6f} "
    f"{baseline_peak['round']:7d} "
    f"{'-':>7s} "
    f"{'-':>7s}"
)


for mode in MODES:

    for alpha in ALPHAS:

        xs = results[
            (mode, alpha)
        ]

        mean25 = sum(
            x["mean_auc"]
            for x in xs
        ) / len(xs)

        last10 = sum(
            x["mean_auc"]
            for x in xs[-10:]
        ) / 10.0

        final = xs[-1]["mean_auc"]

        peak = max(
            xs,
            key=lambda x: x["mean_auc"]
        )

        better = sum(
            x["mean_auc"] > b["mean_auc"]
            for x, b in zip(
                xs,
                baseline_rounds
            )
        )

        worse = sum(
            x["mean_auc"] < b["mean_auc"]
            for x, b in zip(
                xs,
                baseline_rounds
            )
        )

        rec = {
            "mode": mode,
            "alpha": alpha,
            "mean25": mean25,
            "last10": last10,
            "final": final,
            "peak": peak["mean_auc"],
            "peak_round": peak["round"],
            "better": better,
            "worse": worse,
        }

        summary.append(rec)

        print(
            f"{mode:16s} "
            f"{alpha:7.3f} "
            f"{mean25:9.6f} "
            f"{last10:9.6f} "
            f"{final:9.6f} "
            f"{peak['mean_auc']:9.6f} "
            f"{peak['round']:7d} "
            f"{better:7d} "
            f"{worse:7d}"
        )


print()
print("=" * 125)
print("TOP 15 DIAGNOSTIC SETTINGS BY MEAN AUC ACROSS 25 ROUNDS")
print("=" * 125)

ranked = sorted(
    summary,
    key=lambda x: (
        x["mean25"],
        x["last10"],
        x["final"],
    ),
    reverse=True
)

for i, x in enumerate(
    ranked[:15],
    1
):

    print(
        f"{i:02d}. "
        f"{x['mode']:16s} "
        f"alpha={x['alpha']:.3f} "
        f"mean25={x['mean25']:.6f} "
        f"Δmean={x['mean25']-baseline_mean25:+.6f} "
        f"last10={x['last10']:.6f} "
        f"Δlast10={x['last10']-baseline_last10:+.6f} "
        f"final={x['final']:.6f} "
        f"Δfinal={x['final']-baseline_final:+.6f} "
        f"peak={x['peak']:.6f}@R{x['peak_round']:02d}"
    )


print()
print("=" * 125)
print("BEST NONZERO SETTING — ROUND BY ROUND")
print("=" * 125)

best = next(
    x for x in ranked
    if x["alpha"] > 0
)

best_rows = results[
    (
        best["mode"],
        best["alpha"]
    )
]

print(
    "BEST DIAGNOSTIC:",
    best["mode"],
    "alpha=",
    best["alpha"]
)

print()
print(
    f"{'R':>3s} "
    f"{'FROZEN':>10s} "
    f"{'CORRECTED':>10s} "
    f"{'DELTA':>10s}"
)

for b, c in zip(
    baseline_rounds,
    best_rows
):

    print(
        f"{b['round']:3d} "
        f"{b['mean_auc']:10.6f} "
        f"{c['mean_auc']:10.6f} "
        f"{c['mean_auc']-b['mean_auc']:+10.6f}"
    )


print()
print("=" * 125)
print("BEST NONZERO SETTING — FINAL PER SYSTEM")
print("=" * 125)

for s in systems:

    b = baseline_rounds[-1][
        "per_system"
    ][s]

    c = best_rows[-1][
        "per_system"
    ][s]

    print(
        f"{s:42s} "
        f"FROZEN={b:.6f} "
        f"CORRECTED={c:.6f} "
        f"DELTA={c-b:+.6f}"
    )


print()
print("=" * 125)
print("ASYMMETRIC POSITIVE / NEGATIVE GRID")
print("=" * 125)

ASYM = [
    0.0,
    0.025,
    0.05,
    0.10,
    0.20,
    0.30,
    0.50,
]

asym_results = []

for ap in ASYM:

    for an in ASYM:

        round_aucs = []

        for rnd in ROUNDS:

            states = {
                sid(x): x
                for x in load(
                    ROOT / f"states_dev_round_{rnd:02d}.jsonl"
                )
            }

            scored = {
                sid(x): x
                for x in load(
                    ROOT / f"scored_dev_round_{rnd:02d}.jsonl"
                )
            }

            sys_aucs = []

            for s in systems:

                frozen = get_role_probs(
                    states[s]
                )

                evidence = {
                    str(k): float(v)
                    for k, v in dict(
                        scored[s].get(
                            "evidence",
                            {}
                        )
                    ).items()
                }

                corrected = {}

                for role in ROLES:

                    e = float(
                        evidence.get(
                            role,
                            0.0
                        )
                    )

                    correction = (
                        ap * max(e, 0.0)
                        +
                        an * min(e, 0.0)
                    )

                    corrected[role] = (
                        logit(
                            frozen[role]
                        )
                        + correction
                    )

                sys_aucs.append(
                    auc(
                        corrected,
                        truth[s]
                    )
                )

            round_aucs.append(
                sum(sys_aucs)
                / len(sys_aucs)
            )

        asym_results.append({
            "alpha_pos": ap,
            "alpha_neg": an,
            "mean25": sum(round_aucs) / len(round_aucs),
            "last10": sum(round_aucs[-10:]) / 10.0,
            "final": round_aucs[-1],
            "peak": max(round_aucs),
            "peak_round": 1 + max(
                range(len(round_aucs)),
                key=lambda i: round_aucs[i]
            ),
        })


for i, x in enumerate(
    sorted(
        asym_results,
        key=lambda x: (
            x["mean25"],
            x["last10"],
            x["final"],
        ),
        reverse=True
    )[:15],
    1
):

    print(
        f"{i:02d}. "
        f"alpha_pos={x['alpha_pos']:.3f} "
        f"alpha_neg={x['alpha_neg']:.3f} "
        f"mean25={x['mean25']:.6f} "
        f"Δmean={x['mean25']-baseline_mean25:+.6f} "
        f"last10={x['last10']:.6f} "
        f"final={x['final']:.6f} "
        f"peak={x['peak']:.6f}@R{x['peak_round']:02d}"
    )
