import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1/"
    "data/run"
)

OUT = ROOT.parent.parent / "reports"
OUT.mkdir(parents=True, exist_ok=True)

ROUNDS = range(1, 26)

# Fixed bins chosen before looking at labels.
EDGES = [
    float("-inf"),
    -4.0,
    -2.0,
    -1.0,
    0.0,
    1.0,
    2.0,
    4.0,
    float("inf"),
]


def load(path):
    if not path.exists():
        return []

    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sid(x):
    return str(
        x.get(
            "session_id",
            x.get("system_id", "")
        )
    )


def extract_target(row):
    prompt = str(row.get("prompt", ""))

    m = re.search(
        r"TARGET CANDIDATE CAPABILITY:\s*([^\n]+)",
        prompt
    )

    return m.group(1).strip() if m else None


def find_sensor_target(row, target):
    """
    Try the schemas we have seen in HCA scored files.
    """

    containers = [
        row.get("evidence"),
        row.get("role_evidence"),
        row.get("scores"),
        row.get("role_scores"),
    ]

    for key in ["v135", "v134", "sensor", "scoring"]:
        d = row.get(key)

        if isinstance(d, dict):
            containers.extend([
                d,
                d.get("evidence"),
                d.get("role_evidence"),
                d.get("scores"),
                d.get("role_scores"),
            ])

    for d in containers:
        if not isinstance(d, dict):
            continue

        if target in d:
            try:
                return float(d[target])
            except Exception:
                pass

    return None


def bin_index(v):
    for i in range(len(EDGES) - 1):
        lo = EDGES[i]
        hi = EDGES[i + 1]

        if i == len(EDGES) - 2:
            if lo <= v <= hi:
                return i
        elif lo <= v < hi:
            return i

    raise RuntimeError(v)


def fmt_edge(x):
    if x == float("-inf"):
        return "-inf"

    if x == float("inf"):
        return "+inf"

    return f"{x:g}"


records = []

for rnd in ROUNDS:

    candidates = {
        sid(x): x
        for x in load(
            ROOT / f"candidates_dev_round_{rnd:02d}.jsonl"
        )
    }

    responses = {
        sid(x): x
        for x in load(
            ROOT / f"responses_dev_round_{rnd:02d}.jsonl"
        )
    }

    scored = {
        sid(x): x
        for x in load(
            ROOT / f"scored_dev_round_{rnd:02d}.jsonl"
        )
    }

    for system_id, cand in candidates.items():

        target = extract_target(cand)

        if not target:
            continue

        response = responses.get(system_id, {})
        score_row = scored.get(system_id, {})

        truth = set(
            map(
                str,
                response.get(
                    "truth_roles_training_only",
                    []
                )
            )
        )

        if not truth:
            continue

        evidence = find_sensor_target(
            score_row,
            target
        )

        if evidence is None:
            continue

        records.append({
            "round": rnd,
            "system": system_id,
            "target": target,
            "present": int(target in truth),
            "evidence": float(evidence),
            "probe_style": cand.get(
                "probe_style",
                "?"
            ),
            "query": cand.get(
                "query",
                ""
            ),
        })


if not records:
    raise SystemExit(
        "ERROR: no usable records found."
    )


n_present = sum(
    x["present"]
    for x in records
)

n_absent = (
    len(records) - n_present
)

prior = (
    n_present / len(records)
)

prior_odds = (
    prior / (1.0 - prior)
)


print("=" * 125)
print("RAW TARGET SENSOR EVIDENCE -> PRESENT/ABSENT LIKELIHOOD")
print("=" * 125)

print()
print("TOTAL RECORDS :", len(records))
print("PRESENT       :", n_present)
print("ABSENT        :", n_absent)
print(f"EMPIRICAL PRIOR P(PRESENT) = {prior:.6f}")
print(f"EMPIRICAL PRIOR ODDS       = {prior_odds:.6f}")


bins = []

for i in range(len(EDGES) - 1):

    lo = EDGES[i]
    hi = EDGES[i + 1]

    xs = [
        x for x in records
        if bin_index(x["evidence"]) == i
    ]

    p = sum(
        x["present"]
        for x in xs
    )

    a = len(xs) - p

    # Jeffreys smoothing.
    alpha = 0.5

    p_bin_given_present = (
        (p + alpha) /
        (n_present + alpha * (len(EDGES) - 1))
    )

    p_bin_given_absent = (
        (a + alpha) /
        (n_absent + alpha * (len(EDGES) - 1))
    )

    lr = (
        p_bin_given_present /
        p_bin_given_absent
    )

    llr = math.log(lr)

    empirical_present = (
        p / len(xs)
        if xs
        else float("nan")
    )

    label = (
        f"[{fmt_edge(lo)}, {fmt_edge(hi)})"
        if hi != float("inf")
        else f"[{fmt_edge(lo)}, {fmt_edge(hi)}]"
    )

    bins.append({
        "bin": label,
        "lo": lo,
        "hi": hi,
        "n": len(xs),
        "present": p,
        "absent": a,
        "p_present_empirical": empirical_present,
        "lr_present_vs_absent": lr,
        "llr": llr,
    })


print()
print("=" * 125)
print("FIXED EVIDENCE BINS")
print("=" * 125)

print(
    f"{'SENSOR BIN':20s} "
    f"{'N':>5s} "
    f"{'P':>5s} "
    f"{'A':>5s} "
    f"{'P(PRESENT|BIN)':>16s} "
    f"{'LR':>10s} "
    f"{'LLR':>10s}"
)

for b in bins:

    pp = b["p_present_empirical"]

    pp_txt = (
        f"{pp:.3f}"
        if not math.isnan(pp)
        else "nan"
    )

    print(
        f"{b['bin']:20s} "
        f"{b['n']:5d} "
        f"{b['present']:5d} "
        f"{b['absent']:5d} "
        f"{pp_txt:>16s} "
        f"{b['lr_present_vs_absent']:10.3f} "
        f"{b['llr']:10.3f}"
    )


print()
print("=" * 125)
print("POSITIVE-TAIL THRESHOLDS")
print("=" * 125)

thresholds = [
    -2.0,
    -1.0,
    0.0,
    0.5,
    1.0,
    2.0,
    3.0,
    4.0,
]

print(
    f"{'EVIDENCE >= T':15s} "
    f"{'N':>5s} "
    f"{'PRESENT':>8s} "
    f"{'ABSENT':>8s} "
    f"{'PRECISION':>10s}"
)

for t in thresholds:

    xs = [
        x for x in records
        if x["evidence"] >= t
    ]

    p = sum(
        x["present"]
        for x in xs
    )

    a = len(xs) - p

    precision = (
        p / len(xs)
        if xs
        else float("nan")
    )

    print(
        f"{t:15.2f} "
        f"{len(xs):5d} "
        f"{p:8d} "
        f"{a:8d} "
        f"{precision:10.3f}"
    )


print()
print("=" * 125)
print("NEGATIVE-TAIL THRESHOLDS")
print("=" * 125)

print(
    f"{'EVIDENCE <= T':15s} "
    f"{'N':>5s} "
    f"{'PRESENT':>8s} "
    f"{'ABSENT':>8s} "
    f"{'P(ABSENT)':>10s}"
)

for t in [
    0.0,
    -0.5,
    -1.0,
    -2.0,
    -3.0,
    -4.0,
]:

    xs = [
        x for x in records
        if x["evidence"] <= t
    ]

    p = sum(
        x["present"]
        for x in xs
    )

    a = len(xs) - p

    p_absent = (
        a / len(xs)
        if xs
        else float("nan")
    )

    print(
        f"{t:15.2f} "
        f"{len(xs):5d} "
        f"{p:8d} "
        f"{a:8d} "
        f"{p_absent:10.3f}"
    )


print()
print("=" * 125)
print("TOP POSITIVE SENSOR EVIDENCE")
print("=" * 125)

for x in sorted(
    records,
    key=lambda z: z["evidence"],
    reverse=True
)[:20]:

    label = (
        "PRESENT"
        if x["present"]
        else "ABSENT "
    )

    print(
        f"R{x['round']:02d} "
        f"{label} "
        f"E={x['evidence']:+7.3f} "
        f"TARGET={x['target']:24s} "
        f"{x['system']}"
    )


print()
print("=" * 125)
print("TOP NEGATIVE SENSOR EVIDENCE")
print("=" * 125)

for x in sorted(
    records,
    key=lambda z: z["evidence"]
)[:20]:

    label = (
        "PRESENT"
        if x["present"]
        else "ABSENT "
    )

    print(
        f"R{x['round']:02d} "
        f"{label} "
        f"E={x['evidence']:+7.3f} "
        f"TARGET={x['target']:24s} "
        f"{x['system']}"
    )


csv_path = (
    OUT /
    "sensor_evidence_likelihood_records.csv"
)

with csv_path.open(
    "w",
    newline=""
) as f:

    fields = [
        "round",
        "system",
        "target",
        "present",
        "evidence",
        "probe_style",
        "query",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()
    w.writerows(records)


bin_path = (
    OUT /
    "sensor_evidence_likelihood_bins.csv"
)

with bin_path.open(
    "w",
    newline=""
) as f:

    fields = [
        "bin",
        "n",
        "present",
        "absent",
        "p_present_empirical",
        "lr_present_vs_absent",
        "llr",
    ]

    w = csv.DictWriter(
        f,
        fieldnames=fields
    )

    w.writeheader()

    for b in bins:
        w.writerow({
            k: b[k]
            for k in fields
        })


print()
print("WROTE:")
print(" ", csv_path)
print(" ", bin_path)
