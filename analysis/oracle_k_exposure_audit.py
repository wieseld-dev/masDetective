from pathlib import Path
from collections import defaultdict, Counter
import json
import statistics

ROOT = Path("artifacts/adaptive_v11_curriculum_v2")

files = sorted(ROOT.rglob("responses_train_round_*.jsonl"))

systems = defaultdict(lambda: defaultdict(list))
truths = {}


def is_chosen(v):
    if v is True or v == 1:
        return True
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes"}
    return False


for p in files:
    with p.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)

            sid = r.get("system_id") or r.get("session_id")
            rnd = r.get("round")
            truth = tuple(sorted(
                r.get("truth_roles_training_only") or []
            ))

            if not sid or rnd is None or len(truth) not in {3,4,5,6}:
                continue

            selected = tuple(sorted(
                r.get("selected_roles_training_only") or []
            ))

            output = (
                r.get("output")
                or r.get("answer")
                or r.get("final_output")
            )

            if not output:
                continue

            if sid in truths and truths[sid] != truth:
                raise RuntimeError(
                    f"truth mismatch {sid}: "
                    f"{truths[sid]} vs {truth}"
                )

            truths[sid] = truth

            systems[sid][int(rnd)].append({
                "selected": selected,
                "candidate_index": r.get("candidate_index"),
                "chosen": is_chosen(
                    r.get("chosen_for_continuation")
                ),
                "valid": r.get("valid"),
                "query": r.get("query"),
            })


def union_selected(rows):
    out = set()
    for r in rows:
        out.update(r["selected"])
    return out


def evaluate(name, get_rows):
    records = []

    for sid, round_map in systems.items():
        truth = set(truths[sid])
        k_true = len(truth)

        rows = get_rows(round_map)

        if rows is None:
            continue

        selected = union_selected(rows)

        k_hat = len(selected)

        records.append({
            "sid": sid,
            "k_true": k_true,
            "k_hat": k_hat,
            "coverage": len(selected & truth) / k_true,
            "exact": selected == truth,
            "n_rows": len(rows),
        })

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)

    print(f"systems={len(records)}")

    if not records:
        return

    acc = sum(x["k_hat"] == x["k_true"] for x in records) / len(records)
    exact = sum(x["exact"] for x in records) / len(records)
    mae = sum(abs(x["k_hat"] - x["k_true"]) for x in records) / len(records)
    cov = sum(x["coverage"] for x in records) / len(records)

    print(f"K accuracy             = {acc:.4f}")
    print(f"K MAE                  = {mae:.4f}")
    print(f"exact installed set    = {exact:.4f}")
    print(f"mean role coverage     = {cov:.4f}")

    print()
    print("BY K")

    for k in (3,4,5,6):
        rr = [x for x in records if x["k_true"] == k]

        if not rr:
            continue

        acc_k = sum(
            x["k_hat"] == k for x in rr
        ) / len(rr)

        mae_k = sum(
            abs(x["k_hat"] - k) for x in rr
        ) / len(rr)

        cov_k = sum(
            x["coverage"] for x in rr
        ) / len(rr)

        hats = Counter(x["k_hat"] for x in rr)

        print(
            f"K={k}: n={len(rr):3d} "
            f"acc={acc_k:.3f} "
            f"mae={mae_k:.3f} "
            f"coverage={cov_k:.3f} "
            f"pred={dict(sorted(hats.items()))}"
        )

    print()
    print("N ROWS HISTOGRAM")
    print(dict(sorted(
        Counter(x["n_rows"] for x in records).items()
    )))


def first_two_all(round_map):
    rounds = sorted(round_map)

    if len(rounds) < 2:
        return None

    rows = []
    for rnd in rounds[:2]:
        rows.extend(round_map[rnd])

    return rows


def first_two_chosen(round_map):
    rounds = sorted(round_map)

    if len(rounds) < 2:
        return None

    rows = []

    for rnd in rounds[:2]:
        chosen = [
            r for r in round_map[rnd]
            if r["chosen"]
        ]

        # Strict: exactly one actual continuation candidate.
        if len(chosen) != 1:
            return None

        rows.extend(chosen)

    return rows


def first_one_all(round_map):
    rounds = sorted(round_map)

    if not rounds:
        return None

    return list(round_map[rounds[0]])


def all_available(round_map):
    rows = []

    for rnd in sorted(round_map):
        rows.extend(round_map[rnd])

    return rows


print("=" * 100)
print("ORACLE K EXPOSURE AUDIT")
print("=" * 100)

print(f"systems={len(systems)}")
print(f"files={len(files)}")

round_candidate_hist = Counter()
chosen_hist = Counter()

for sid, rm in systems.items():
    for rnd, rows in rm.items():
        round_candidate_hist[len(rows)] += 1
        chosen_hist[sum(r["chosen"] for r in rows)] += 1

print()
print("CANDIDATES PER SYSTEM/ROUND")
print(dict(sorted(round_candidate_hist.items())))

print()
print("CHOSEN ROWS PER SYSTEM/ROUND")
print(dict(sorted(chosen_hist.items())))


evaluate(
    "ONE ROUND / ALL CANDIDATES",
    first_one_all,
)

evaluate(
    "TWO ROUNDS / ALL CANDIDATES",
    first_two_all,
)

evaluate(
    "TWO ROUNDS / CHOSEN-ONLY",
    first_two_chosen,
)

evaluate(
    "ALL AVAILABLE ROUNDS / ALL CANDIDATES",
    all_available,
)
