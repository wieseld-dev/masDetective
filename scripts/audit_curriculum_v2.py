#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path


# ============================================================
# BASIC HELPERS
# ============================================================

def read_json(path: Path):
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x))


def mean(xs):
    xs = [float(x) for x in xs if num(x)]
    return statistics.fmean(xs) if xs else float("nan")


def median(xs):
    xs = [float(x) for x in xs if num(x)]
    return statistics.median(xs) if xs else float("nan")


def fmt(x, n=4):
    return "NA" if not num(x) else f"{float(x):.{n}f}"


def flatten_numeric(obj, prefix=""):
    out = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_numeric(v, p))

    elif isinstance(obj, list):
        # Do not flatten arbitrary vectors into thousands of columns.
        if obj and all(num(v) for v in obj) and len(obj) <= 16:
            for i, v in enumerate(obj):
                out[f"{prefix}[{i}]"] = float(v)

    elif num(obj):
        out[prefix] = float(obj)

    return out


def percentile(xs, p):
    xs = sorted(float(x) for x in xs if num(x))
    if not xs:
        return float("nan")
    if len(xs) == 1:
        return xs[0]

    z = (len(xs) - 1) * p
    lo = int(math.floor(z))
    hi = int(math.ceil(z))

    if lo == hi:
        return xs[lo]

    w = z - lo
    return xs[lo] * (1 - w) + xs[hi] * w


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if num(x) and num(y)]
    if len(pairs) < 2:
        return float("nan")

    x = [a for a, _ in pairs]
    y = [b for _, b in pairs]

    mx, my = mean(x), mean(y)

    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)

    if vx <= 0 or vy <= 0:
        return float("nan")

    cov = sum((a - mx) * (b - my) for a, b in pairs)
    return cov / math.sqrt(vx * vy)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)

    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1

        avg = (i + 1 + j) / 2.0

        for k in range(i, j):
            r[order[k]] = avg

        i = j

    return r


def spearman(xs, ys):
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    return pearson(ranks(xs), ranks(ys))


def bootstrap_mean_ci(values, iters=5000, seed=1337):
    vals = [float(v) for v in values if num(v)]

    if not vals:
        return float("nan"), float("nan"), float("nan")

    rng = random.Random(seed)
    n = len(vals)
    boots = []

    for _ in range(iters):
        boots.append(mean([vals[rng.randrange(n)] for _ in range(n)]))

    return mean(vals), percentile(boots, 0.025), percentile(boots, 0.975)


def sign_test_two_sided(values):
    vals = [float(v) for v in values if num(v) and abs(float(v)) > 1e-12]

    if not vals:
        return float("nan")

    pos = sum(v > 0 for v in vals)
    n = len(vals)
    k = min(pos, n - pos)

    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def extract_int(pattern, text, default=-1):
    m = re.search(pattern, text)
    return int(m.group(1)) if m else default


# ============================================================
# ARGUMENTS / OUTPUT
# ============================================================

ap = argparse.ArgumentParser()
ap.add_argument(
    "--root",
    default="artifacts/adaptive_v11_curriculum_v2",
)
ap.add_argument(
    "--output",
    default=None,
)
args = ap.parse_args()

ROOT = Path(args.root)

OUT = (
    Path(args.output)
    if args.output
    else ROOT / "final_analysis"
)

OUT.mkdir(parents=True, exist_ok=True)

stage_dirs = sorted(
    [p for p in ROOT.glob("stage_*") if p.is_dir()],
    key=lambda p: extract_int(r"stage_(\d+)", p.name),
)


# ============================================================
# 1. TRAINING: GENERATOR + SELECTOR
# ============================================================

group_records = []
breakdown_values = defaultdict(list)

candidate_valid = defaultdict(lambda: [0, 0])
candidate_attempts = defaultdict(list)
candidate_duplicate_rates = defaultdict(list)
preexec_forbidden = defaultdict(set)

FORBIDDEN_PREEXEC = {
    "truth_roles",
    "truth_roles_training_only",
    "selected_roles_training_only",
    "answer",
    "final_output",
    "response_id",
    "reward",
    "evidence",
}


for sdir in stage_dirs:
    stage = extract_int(r"stage_(\d+)", sdir.name)

    if stage < 0:
        continue

    # --------------------------------------------------------
    # Pre-execution audit / generator validity
    # --------------------------------------------------------

    for pool_file in sdir.glob("cycle_*/candidate_pool_train_round_*.jsonl"):
        rows = read_jsonl(pool_file)

        groups = defaultdict(list)

        for r in rows:
            candidate_valid[stage][1] += 1

            if r.get("valid", False):
                candidate_valid[stage][0] += 1

            if num(r.get("local_attempts")):
                candidate_attempts[stage].append(float(r["local_attempts"]))

            bad = FORBIDDEN_PREEXEC.intersection(r.keys())
            preexec_forbidden[stage].update(bad)

            key = (r.get("session_id"), r.get("group_id"))
            groups[key].append(r)

        for rs in groups.values():
            queries = [
                str(r.get("query", "")).strip().lower()
                for r in rs
                if str(r.get("query", "")).strip()
            ]

            if queries:
                duplicate_rate = 1.0 - len(set(queries)) / len(queries)
                candidate_duplicate_rates[stage].append(duplicate_rate)

    # --------------------------------------------------------
    # Counterfactual training pools
    # --------------------------------------------------------

    for tr_file in sdir.glob("cycle_*/train_rows_round_*.jsonl"):
        cycle = extract_int(r"cycle_(\d+)", str(tr_file))
        rnd = extract_int(r"round_(\d+)", tr_file.name)

        rows = read_jsonl(tr_file)
        groups = defaultdict(list)

        for r in rows:
            key = (
                r.get("session_id"),
                r.get("group_id"),
            )
            groups[key].append(r)

        for (session_id, group_id), rs in groups.items():

            valid_rows = [r for r in rs if num(r.get("reward"))]

            if not valid_rows:
                continue

            rewards = [float(r["reward"]) for r in valid_rows]

            best_reward = max(rewards)
            best_candidates = [
                r for r in valid_rows
                if abs(float(r["reward"]) - best_reward) < 1e-12
            ]
            best = best_candidates[0]

            chosen_candidates = [
                r for r in valid_rows
                if bool(r.get("chosen_for_continuation"))
            ]
            chosen = chosen_candidates[0] if chosen_candidates else None

            pred_rows = [
                r for r in valid_rows
                if num(r.get("selector_predicted_utility"))
            ]

            greedy = (
                max(
                    pred_rows,
                    key=lambda r: float(r["selector_predicted_utility"])
                )
                if pred_rows
                else None
            )

            reward_sorted = sorted(
                valid_rows,
                key=lambda r: float(r["reward"]),
                reverse=True,
            )

            def reward_rank(row):
                if row is None:
                    return None

                ridx = row.get("candidate_index")

                for i, x in enumerate(reward_sorted, 1):
                    if x.get("candidate_index") == ridx:
                        return i

                return None

            chosen_rank = reward_rank(chosen)
            greedy_rank = reward_rank(greedy)

            pred = [
                float(r["selector_predicted_utility"])
                for r in pred_rows
            ]
            pred_reward = [
                float(r["reward"])
                for r in pred_rows
            ]

            rec = {
                "stage": stage,
                "cycle": cycle,
                "round": rnd,
                "session_id": session_id,
                "group_id": group_id,
                "pool_size": len(valid_rows),
                "best_reward": best_reward,
                "random_expected_reward": mean(rewards),
                "pool_has_positive": best_reward > 0,
                "chosen_reward":
                    float(chosen["reward"]) if chosen else float("nan"),
                "greedy_reward":
                    float(greedy["reward"]) if greedy else float("nan"),
                "chosen_rank": chosen_rank,
                "greedy_rank": greedy_rank,
                "chosen_exact_best":
                    chosen is not None and
                    abs(float(chosen["reward"]) - best_reward) < 1e-12,
                "greedy_exact_best":
                    greedy is not None and
                    abs(float(greedy["reward"]) - best_reward) < 1e-12,
                "greedy_top2":
                    greedy_rank is not None and greedy_rank <= 2,
                "greedy_top3":
                    greedy_rank is not None and greedy_rank <= 3,
                "chosen_positive":
                    chosen is not None and float(chosen["reward"]) > 0,
                "greedy_positive":
                    greedy is not None and float(greedy["reward"]) > 0,
                "pool_spearman":
                    spearman(pred, pred_reward) if len(pred) >= 2 else float("nan"),
            }

            group_records.append(rec)

            # ------------------------------------------------
            # Reward breakdown: oracle / actual / greedy /
            # expected random.
            # ------------------------------------------------

            candidate_breakdowns = []

            for r in valid_rows:
                flat = flatten_numeric(
                    r.get("reward_breakdown_training_only", {})
                )
                candidate_breakdowns.append((r, flat))

            all_keys = set()
            for _, flat in candidate_breakdowns:
                all_keys.update(flat.keys())

            for field in all_keys:
                vals = [
                    flat[field]
                    for _, flat in candidate_breakdowns
                    if field in flat and num(flat[field])
                ]

                if vals:
                    breakdown_values[(stage, field, "random")].append(mean(vals))

                bflat = flatten_numeric(
                    best.get("reward_breakdown_training_only", {})
                )
                if field in bflat:
                    breakdown_values[(stage, field, "oracle_best")].append(
                        bflat[field]
                    )

                if chosen:
                    cflat = flatten_numeric(
                        chosen.get("reward_breakdown_training_only", {})
                    )
                    if field in cflat:
                        breakdown_values[(stage, field, "chosen")].append(
                            cflat[field]
                        )

                if greedy:
                    gflat = flatten_numeric(
                        greedy.get("reward_breakdown_training_only", {})
                    )
                    if field in gflat:
                        breakdown_values[(stage, field, "greedy")].append(
                            gflat[field]
                        )


# ============================================================
# 2. STAGE-LEVEL SELECTOR / GENERATOR SUMMARY
# ============================================================

stage_summary = []

for stage in sorted(set(r["stage"] for r in group_records)):
    rs = [r for r in group_records if r["stage"] == stage]

    pool_size = mean([r["pool_size"] for r in rs])
    random_exact_baseline = (
        mean([1.0 / r["pool_size"] for r in rs if r["pool_size"] > 0])
    )

    stage_summary.append({
        "stage": stage,
        "groups": len(rs),
        "pool_size": pool_size,

        "positive_pool_fraction":
            mean([float(r["pool_has_positive"]) for r in rs]),

        "mean_best_reward":
            mean([r["best_reward"] for r in rs]),

        "mean_random_reward":
            mean([r["random_expected_reward"] for r in rs]),

        "mean_chosen_reward":
            mean([r["chosen_reward"] for r in rs]),

        "mean_greedy_reward":
            mean([r["greedy_reward"] for r in rs]),

        "chosen_regret":
            mean([
                r["best_reward"] - r["chosen_reward"]
                for r in rs if num(r["chosen_reward"])
            ]),

        "greedy_regret":
            mean([
                r["best_reward"] - r["greedy_reward"]
                for r in rs if num(r["greedy_reward"])
            ]),

        "random_exact_best_baseline":
            random_exact_baseline,

        "chosen_exact_best":
            mean([float(r["chosen_exact_best"]) for r in rs]),

        "greedy_exact_best":
            mean([float(r["greedy_exact_best"]) for r in rs]),

        "greedy_top2":
            mean([float(r["greedy_top2"]) for r in rs]),

        "greedy_top3":
            mean([float(r["greedy_top3"]) for r in rs]),

        "chosen_positive":
            mean([float(r["chosen_positive"]) for r in rs]),

        "greedy_positive":
            mean([float(r["greedy_positive"]) for r in rs]),

        "mean_pool_spearman":
            mean([r["pool_spearman"] for r in rs]),

        "valid_candidate_fraction":
            (
                candidate_valid[stage][0] /
                candidate_valid[stage][1]
                if candidate_valid[stage][1]
                else float("nan")
            ),

        "mean_local_attempts":
            mean(candidate_attempts[stage]),

        "exact_duplicate_query_rate":
            mean(candidate_duplicate_rates[stage]),

        "preexec_forbidden_fields":
            sorted(preexec_forbidden[stage]),
    })


# ============================================================
# 3. DEV TRAJECTORIES + PAIRED EFFECT SIZES
# ============================================================

dev_round_rows = []
paired_effects = []

METRICS = {
    "role_log_loss": -1,          # lower is better
    "role_brier": -1,
    "role_auc": +1,
    "true_false_margin": +1,
    "oracle_k_role_recall": +1,
}

for sdir in stage_dirs:
    stage = extract_int(r"stage_(\d+)", sdir.name)

    report_path = sdir / "dev_report.json"

    if report_path.exists():
        report = read_json(report_path)

        for r in report.get("rounds", []):
            row = {"stage": stage}
            row.update(r)
            dev_round_rows.append(row)

    metric_files = sorted(
        (sdir / "dev_eval").glob("metrics_dev_round_*.jsonl")
    )

    if not metric_files:
        continue

    rounds = {}

    for p in metric_files:
        rnd = extract_int(r"round_(\d+)", p.name)

        if rnd >= 0:
            rounds[rnd] = {
                r["session_id"]: r
                for r in read_jsonl(p)
                if "session_id" in r
            }

    if len(rounds) < 2:
        continue

    first_round = min(rounds)
    last_round = max(rounds)

    first = rounds[first_round]
    last = rounds[last_round]

    common = sorted(set(first).intersection(last))

    for metric, direction in METRICS.items():
        improvements = []
        raw_deltas = []

        for sid in common:
            a = first[sid].get(metric)
            b = last[sid].get(metric)

            if not num(a) or not num(b):
                continue

            delta = float(b) - float(a)
            raw_deltas.append(delta)
            improvements.append(direction * delta)

        m, lo, hi = bootstrap_mean_ci(improvements)

        paired_effects.append({
            "stage": stage,
            "metric": metric,
            "first_round": first_round,
            "last_round": last_round,
            "systems": len(improvements),
            "raw_delta_mean": mean(raw_deltas),
            "oriented_improvement_mean": m,
            "improvement_median": median(improvements),
            "ci95_low": lo,
            "ci95_high": hi,
            "fraction_improved":
                mean([float(x > 0) for x in improvements]),
            "sign_test_p":
                sign_test_two_sided(improvements),
        })


# ============================================================
# 4. STAGE 10 PEAK / STOPPING ANALYSIS
# ============================================================

stage10_rounds = [
    r for r in dev_round_rows
    if r.get("stage") == 10
]

stopping = {}

if stage10_rounds:

    def best_round(metric, higher=True):
        valid = [
            r for r in stage10_rounds
            if num(r.get(metric))
        ]

        if not valid:
            return None

        return (
            max(valid, key=lambda r: r[metric])
            if higher
            else min(valid, key=lambda r: r[metric])
        )

    for metric, direction in METRICS.items():
        b = best_round(metric, higher=(direction > 0))

        if b:
            stopping[metric] = {
                "best_round": b["round"],
                "best_value": b[metric],
                "final_round": stage10_rounds[-1]["round"],
                "final_value": stage10_rounds[-1][metric],
            }


# ============================================================
# 5. REWARD BREAKDOWN SUMMARY
# ============================================================

breakdown_rows = []

for (stage, field, kind), vals in sorted(breakdown_values.items()):
    breakdown_rows.append({
        "stage": stage,
        "field": field,
        "kind": kind,
        "n": len(vals),
        "mean": mean(vals),
        "median": median(vals),
        "p25": percentile(vals, 0.25),
        "p75": percentile(vals, 0.75),
    })


# ============================================================
# 6. WRITE CSVs
# ============================================================

def write_csv(path, rows):
    if not rows:
        return

    keys = []
    seen = set()

    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)

    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()

        for r in rows:
            x = dict(r)

            for k, v in x.items():
                if isinstance(v, (list, dict)):
                    x[k] = json.dumps(v, sort_keys=True)

            w.writerow(x)


write_csv(OUT / "selector_groups.csv", group_records)
write_csv(OUT / "stage_summary.csv", stage_summary)
write_csv(OUT / "dev_trajectory.csv", dev_round_rows)
write_csv(OUT / "paired_effects.csv", paired_effects)
write_csv(OUT / "reward_breakdown.csv", breakdown_rows)


# ============================================================
# 7. JSON SUMMARY
# ============================================================

summary_json = {
    "root": str(ROOT),
    "stage_summary": stage_summary,
    "paired_effects": paired_effects,
    "stage10_stopping": stopping,
}

(OUT / "summary.json").write_text(
    json.dumps(summary_json, indent=2, sort_keys=True)
)


# ============================================================
# 8. HUMAN-READABLE REPORT
# ============================================================

lines = []

lines.append("# Curriculum V2 Final Audit")
lines.append("")
lines.append("This report uses existing artifacts only; no MAS/LLM execution.")
lines.append("")

lines.append("## Generator / Selector")
lines.append("")
lines.append(
    "| Stage | Groups | G | Positive pool | Oracle reward | Random reward | "
    "Chosen reward | Greedy reward | Chosen regret | Greedy regret | "
    "Greedy exact-best | Random exact-best | Greedy top-2 | Spearman |"
)
lines.append(
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
)

for r in stage_summary:
    lines.append(
        f"| {r['stage']} "
        f"| {r['groups']} "
        f"| {fmt(r['pool_size'],2)} "
        f"| {fmt(r['positive_pool_fraction'],3)} "
        f"| {fmt(r['mean_best_reward'],3)} "
        f"| {fmt(r['mean_random_reward'],3)} "
        f"| {fmt(r['mean_chosen_reward'],3)} "
        f"| {fmt(r['mean_greedy_reward'],3)} "
        f"| {fmt(r['chosen_regret'],3)} "
        f"| {fmt(r['greedy_regret'],3)} "
        f"| {fmt(r['greedy_exact_best'],3)} "
        f"| {fmt(r['random_exact_best_baseline'],3)} "
        f"| {fmt(r['greedy_top2'],3)} "
        f"| {fmt(r['mean_pool_spearman'],3)} |"
    )

lines.append("")
lines.append("## Generator validity / leakage sanity")
lines.append("")

for r in stage_summary:
    lines.append(
        f"- Stage {r['stage']}: "
        f"candidate-valid={fmt(r['valid_candidate_fraction'],3)}, "
        f"duplicate-query-rate={fmt(r['exact_duplicate_query_rate'],3)}, "
        f"mean-local-attempts={fmt(r['mean_local_attempts'],2)}, "
        f"forbidden-preexecution-fields={r['preexec_forbidden_fields']}"
    )

lines.append("")
lines.append("## Paired DEV effects")
lines.append("")
lines.append(
    "| Stage | Metric | Rfirst→Rlast | N | Raw delta | Improvement | "
    "95% CI | Fraction improved | sign-test p |"
)
lines.append(
    "|---:|---|---|---:|---:|---:|---|---:|---:|"
)

for r in paired_effects:
    lines.append(
        f"| {r['stage']} "
        f"| {r['metric']} "
        f"| {r['first_round']}→{r['last_round']} "
        f"| {r['systems']} "
        f"| {fmt(r['raw_delta_mean'],5)} "
        f"| {fmt(r['oriented_improvement_mean'],5)} "
        f"| [{fmt(r['ci95_low'],5)}, {fmt(r['ci95_high'],5)}] "
        f"| {fmt(r['fraction_improved'],3)} "
        f"| {fmt(r['sign_test_p'],4)} |"
    )

lines.append("")
lines.append("## Stage 10 stopping / saturation")
lines.append("")

for metric, r in stopping.items():
    lines.append(
        f"- {metric}: best round={r['best_round']} "
        f"value={fmt(r['best_value'],5)}; "
        f"final R{r['final_round']}={fmt(r['final_value'],5)}"
    )

# Stage 10 direct interpretation
s10 = next((r for r in stage_summary if r["stage"] == 10), None)

if s10:
    lines.append("")
    lines.append("## Stage 10 automatic checks")
    lines.append("")

    selector_gain = (
        s10["mean_greedy_reward"] - s10["mean_random_reward"]
        if num(s10["mean_greedy_reward"]) and num(s10["mean_random_reward"])
        else float("nan")
    )

    oracle_gap = s10["greedy_regret"]

    lines.append(
        f"- Generator positive-pool fraction: "
        f"{fmt(s10['positive_pool_fraction'],3)}"
    )
    lines.append(
        f"- Oracle best reward: {fmt(s10['mean_best_reward'],3)}"
    )
    lines.append(
        f"- Random expected reward: {fmt(s10['mean_random_reward'],3)}"
    )
    lines.append(
        f"- Greedy selector reward: {fmt(s10['mean_greedy_reward'],3)}"
    )
    lines.append(
        f"- Greedy selector uplift over random: {fmt(selector_gain,3)}"
    )
    lines.append(
        f"- Remaining oracle gap after greedy selector: {fmt(oracle_gap,3)}"
    )
    lines.append(
        f"- Greedy exact-best rate: {fmt(s10['greedy_exact_best'],3)} "
        f"vs random baseline {fmt(s10['random_exact_best_baseline'],3)}"
    )
    lines.append(
        f"- Greedy top-2 rate: {fmt(s10['greedy_top2'],3)}"
    )
    lines.append(
        f"- Mean within-pool predicted-utility/reward Spearman: "
        f"{fmt(s10['mean_pool_spearman'],3)}"
    )

lines.append("")
lines.append("## Reward breakdown fields")
lines.append("")
lines.append(
    "See `reward_breakdown.csv`. It separates oracle-best, actual chosen, "
    "greedy-selector and expected-random values for every numeric field "
    "available in `reward_breakdown_training_only`."
)

lines.append("")
lines.append("## Files")
lines.append("")
lines.append("- `stage_summary.csv`")
lines.append("- `selector_groups.csv`")
lines.append("- `dev_trajectory.csv`")
lines.append("- `paired_effects.csv`")
lines.append("- `reward_breakdown.csv`")
lines.append("- `summary.json`")

report = "\n".join(lines) + "\n"

(OUT / "REPORT.md").write_text(report)

print(report)
print(f"\nWROTE: {OUT}")
