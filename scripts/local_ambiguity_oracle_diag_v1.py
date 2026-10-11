from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("artifacts/local_ambiguity_v1")
OUT = ROOT / "oracle_diag"
OUT.mkdir(parents=True, exist_ok=True)

HORDER = ["00", "10", "01", "11"]
TYPES = ["A_DOMINANT", "B_DOMINANT", "BOUNDARY", "JOINT"]


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text(encoding="utf-8")
)

truth = {
    x["session_id"]: x
    for x in manifest["systems"]
}

responses = read_jsonl(
    ROOT / "responses_dev_round_01.jsonl"
)

rows = []

for r in responses:
    if r.get("execution_status") != "ok":
        continue

    sid = str(r["session_id"])
    m = truth[sid]

    A = m["role_a"]
    B = m["role_b"]

    selected = set(
        r.get("selected_roles_training_only", [])
    )

    rows.append({
        "pair_id": m["pair_id"],
        "hypothesis": m["hypothesis"],
        "probe_type": r["probe_type"],
        "variant": int(r["variant"]),
        "A_selected": int(A in selected),
        "B_selected": int(B in selected),
        "AB_selected": int(A in selected and B in selected),
        "either_selected": int(A in selected or B in selected),
        "n_selected": len(selected),
    })


# ------------------------------------------------------------
# Aggregate routing behavior by probe type × hidden local state
# ------------------------------------------------------------

agg = {}

for typ in TYPES:
    for h in HORDER:
        rr = [
            x for x in rows
            if x["probe_type"] == typ
            and x["hypothesis"] == h
        ]

        agg[(typ, h)] = {
            "n": len(rr),
            "A_selected": float(np.mean([x["A_selected"] for x in rr])),
            "B_selected": float(np.mean([x["B_selected"] for x in rr])),
            "AB_selected": float(np.mean([x["AB_selected"] for x in rr])),
            "either_selected": float(np.mean([x["either_selected"] for x in rr])),
            "n_selected": float(np.mean([x["n_selected"] for x in rr])),
        }


# ------------------------------------------------------------
# Per-pair query utility from PRIVILEGED routing truth.
#
# Utility depends on the ambiguity we are trying to resolve.
# ------------------------------------------------------------

by_query = defaultdict(dict)

for x in rows:
    key = (
        x["pair_id"],
        x["probe_type"],
        x["variant"],
    )

    by_query[key][x["hypothesis"]] = x


def a_test(hh):
    # A absent vs present, averaging over B status
    absent = 0.5 * (
        hh["00"]["A_selected"]
        + hh["01"]["A_selected"]
    )

    present = 0.5 * (
        hh["10"]["A_selected"]
        + hh["11"]["A_selected"]
    )

    return present - absent


def b_test(hh):
    absent = 0.5 * (
        hh["00"]["B_selected"]
        + hh["10"]["B_selected"]
    )

    present = 0.5 * (
        hh["01"]["B_selected"]
        + hh["11"]["B_selected"]
    )

    return present - absent


def a_vs_b(hh):
    # Reward opposite routing on H10 vs H01.
    s10 = (
        hh["10"]["A_selected"]
        - hh["10"]["B_selected"]
    )

    s01 = (
        hh["01"]["B_selected"]
        - hh["01"]["A_selected"]
    )

    return 0.5 * (s10 + s01)


def both_test(hh):
    # Does H11 uniquely produce joint engagement?
    single = 0.5 * (
        hh["10"]["AB_selected"]
        + hh["01"]["AB_selected"]
    )

    return (
        hh["11"]["AB_selected"]
        - single
    )


FUNCS = {
    "A_TEST": a_test,
    "B_TEST": b_test,
    "A_VS_B": a_vs_b,
    "BOTH_TEST": both_test,
}


query_scores = {}

for key, hh in by_query.items():
    if set(hh) != set(HORDER):
        continue

    query_scores[key] = {
        name: float(fn(hh))
        for name, fn in FUNCS.items()
    }


# ------------------------------------------------------------
# Does best probe TYPE depend on ambiguity?
# ------------------------------------------------------------

pairs = sorted(set(x["pair_id"] for x in rows))

per_pair = []
adaptive_gains = []
switches = []

for pair in pairs:
    type_scores = {}

    for scenario in FUNCS:
        type_scores[scenario] = {}

        for typ in TYPES:
            vals = [
                query_scores[(pair, typ, v)][scenario]
                for v in range(3)
            ]

            type_scores[scenario][typ] = float(np.mean(vals))

    best_by_scenario = {
        s: max(type_scores[s], key=type_scores[s].get)
        for s in FUNCS
    }

    switches.append(
        len(set(best_by_scenario.values())) >= 2
    )

    adaptive = float(np.mean([
        max(type_scores[s].values())
        for s in FUNCS
    ]))

    fixed_scores = {
        typ: float(np.mean([
            type_scores[s][typ]
            for s in FUNCS
        ]))
        for typ in TYPES
    }

    best_fixed = max(
        fixed_scores,
        key=fixed_scores.get,
    )

    fixed = fixed_scores[best_fixed]

    gain = adaptive - fixed
    adaptive_gains.append(gain)

    per_pair.append({
        "pair_id": pair,
        "best_by_scenario": best_by_scenario,
        "best_fixed": best_fixed,
        "adaptive_value": adaptive,
        "fixed_value": fixed,
        "adaptive_gain": gain,
        "type_scores": type_scores,
    })


# bootstrap pair CI
rng = np.random.default_rng(170926)
vals = np.asarray(adaptive_gains, float)

boot = np.asarray([
    np.mean(
        rng.choice(
            vals,
            size=len(vals),
            replace=True,
        )
    )
    for _ in range(30000)
])

ci = [
    float(np.quantile(boot, .025)),
    float(np.quantile(boot, .975)),
]


# ------------------------------------------------------------
# Report
# ------------------------------------------------------------

L = []

L.append("=" * 100)
L.append("LOCAL AMBIGUITY — PRIVILEGED ROUTING ORACLE DIAGNOSTIC")
L.append("=" * 100)

L.append("")
L.append("ROUTING RESPONSE BY PROBE TYPE × TRUE LOCAL STATE")

for typ in TYPES:
    L.append("")
    L.append(typ)

    for h in HORDER:
        z = agg[(typ, h)]

        L.append(
            f"  H{h}: "
            f"A={z['A_selected']:.3f} "
            f"B={z['B_selected']:.3f} "
            f"AB={z['AB_selected']:.3f} "
            f"either={z['either_selected']:.3f} "
            f"nsel={z['n_selected']:.2f}"
        )

L.append("")
L.append("-" * 100)
L.append("ADAPTIVITY FROM PRIVILEGED ROUTING TRUTH")
L.append(
    f"pairs with best-type switch = "
    f"{sum(switches)}/{len(switches)} "
    f"({np.mean(switches):.3f})"
)

L.append(
    f"mean adaptive-type gain = "
    f"{np.mean(adaptive_gains):+.6f}"
)

L.append(
    f"95% pair bootstrap CI = "
    f"[{ci[0]:+.6f},{ci[1]:+.6f}]"
)

L.append(
    f"positive-gain pairs = "
    f"{sum(x > 1e-9 for x in adaptive_gains)}/{len(adaptive_gains)}"
)

L.append("")
L.append("PER PAIR")

for p in per_pair:
    L.append(
        f"  {p['pair_id']} | "
        + " | ".join(
            f"{s}:{p['best_by_scenario'][s]}"
            for s in FUNCS
        )
        + f" | fixed={p['best_fixed']}"
        + f" | gain={p['adaptive_gain']:+.4f}"
    )

mean_gain = float(np.mean(adaptive_gains))
switch_rate = float(np.mean(switches))
positive = sum(x > 1e-9 for x in adaptive_gains)

if (
    mean_gain > 0
    and ci[0] > 0
    and switch_rate >= .5
    and positive >= 6
):
    verdict = "PROBES_WORK_SENSOR_IS_BOTTLENECK"

elif mean_gain > 0 and switch_rate >= .5:
    verdict = "WEAK_PROBE_ADAPTIVITY"

else:
    verdict = "PROBE_DESIGN_NO_GO"

L.append("")
L.append("=" * 100)
L.append(f"VERDICT: {verdict}")

if verdict == "PROBES_WORK_SENSOR_IS_BOTTLENECK":
    L.append(
        "The optimal local experiment genuinely changes with the hidden A/B state. "
        "The previous failure is therefore primarily an observation-model problem."
    )
elif verdict == "WEAK_PROBE_ADAPTIVITY":
    L.append(
        "Some state-dependent routing response exists, but probe specialization "
        "is not yet strong enough."
    )
else:
    L.append(
        "Even privileged routing truth does not show strong state-dependent "
        "probe value. Improve probe construction before training a new sensor."
    )

L.append("=" * 100)

report = "\n".join(L)

(OUT / "summary.json").write_text(
    json.dumps({
        "aggregate": {
            f"{typ}:{h}": agg[(typ, h)]
            for typ in TYPES
            for h in HORDER
        },
        "adaptive_gain_mean": mean_gain,
        "adaptive_gain_ci95": ci,
        "switch_rate": switch_rate,
        "positive_gain_pairs": positive,
        "per_pair": per_pair,
        "verdict": verdict,
    }, indent=2),
    encoding="utf-8",
)

(OUT / "REPORT.txt").write_text(
    report + "\n",
    encoding="utf-8",
)

print(report)
