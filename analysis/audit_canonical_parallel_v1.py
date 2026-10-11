from pathlib import Path
from collections import defaultdict, Counter
import json
import numpy as np

BASE = Path("artifacts/canonical_parallel_v1")

def readj(p):
    return [json.loads(x) for x in p.open() if x.strip()]

def chao(counts):
    counts = np.asarray(counts)
    s = int(np.sum(counts > 0))
    f1 = int(np.sum(counts == 1))
    f2 = int(np.sum(counts == 2))
    return s + f1 * (f1 - 1) / (2 * (f2 + 1))

def greedy(rows, budget):
    remain = list(rows)
    seen = set()

    for _ in range(min(budget, len(remain))):
        best_i = None
        best_gain = -1

        for i, x in enumerate(remain):
            sel = set(x.get("selected_roles_training_only") or [])
            gain = len(sel - seen)

            if gain > best_gain:
                best_gain = gain
                best_i = i

        x = remain.pop(best_i)
        seen.update(x.get("selected_roles_training_only") or [])

    return seen

for cohort in ["k3", "k4", "high"]:
    root = BASE / cohort

    meta_path = root / "meta.json"
    resp_path = root / "responses_train_round_01.jsonl"

    if not meta_path.exists():
        raise RuntimeError(f"missing {meta_path}")
    if not resp_path.exists():
        raise RuntimeError(f"missing {resp_path}")

    meta = json.loads(meta_path.read_text())

    probe_role = {
        int(x["candidate_index"]): x["role"]
        for x in meta["probes"]
    }

    rows = readj(resp_path)

    by = defaultdict(list)

    for x in rows:
        sid = x.get("system_id") or x.get("session_id")
        if sid:
            by[sid].append(x)

    exact = []
    coverage = []
    count_exact = []
    chao_exact = []

    target_hit_num = Counter()
    target_hit_den = Counter()

    mode_stats = defaultdict(list)

    print("\n" + "=" * 110)
    print("COHORT", cohort.upper())
    print("=" * 110)
    print("systems =", len(by))
    print("rows    =", len(rows))

    for sid, rr in sorted(by.items()):
        rr = sorted(rr, key=lambda x: int(x["candidate_index"]))

        gt = set(rr[0].get("truth_roles_training_only") or [])
        mode = rr[0].get("mode")
        K = len(gt)

        seen = set()
        counts = Counter()

        for x in rr:
            sel = set(x.get("selected_roles_training_only") or [])
            seen |= sel

            for r in sel:
                counts[r] += 1

            idx = int(x["candidate_index"])
            target = probe_role[idx]

            if target in gt:
                target_hit_den[target] += 1
                target_hit_num[target] += int(target in sel)

        cov = len(seen & gt) / K
        ex = (seen == gt)
        kex = (len(seen) == K)

        c = chao(list(counts.values()))
        khat = round(c)
        cex = (khat == K)

        exact.append(ex)
        count_exact.append(kex)
        coverage.append(cov)
        chao_exact.append(cex)

        mode_stats[mode].append((cov, ex))

        print(
            f"{sid} mode={str(mode):8s} "
            f"K={K} seen={len(seen)} "
            f"coverage={cov:.3f} exact={ex} "
            f"chao={c:.2f} khat={khat}"
        )

    print()
    print("SUMMARY")
    print(f"exact set      = {np.mean(exact):.4f}")
    print(f"exact union-K  = {np.mean(count_exact):.4f}")
    print(f"mean coverage  = {np.mean(coverage):.4f}")
    print(f"Chao rounded K = {np.mean(chao_exact):.4f}")

    print()
    print("BY MODE")

    for m, v in sorted(mode_stats.items()):
        print(
            m,
            "n=", len(v),
            "coverage=", round(np.mean([x[0] for x in v]), 3),
            "exact=", round(np.mean([x[1] for x in v]), 3),
        )

    print()
    print("TARGETED-PROBE HIT RATE WHEN ROLE IS INSTALLED")

    for r in sorted(target_hit_den):
        print(
            f"{r:25s} "
            f"{target_hit_num[r]}/{target_hit_den[r]} "
            f"= {target_hit_num[r] / target_hit_den[r]:.3f}"
        )

    print()
    print("ORACLE GREEDY FROM SAME 20 CANONICAL PROBES")

    for b in [1,2,3,4,5,6,8,10,12,16,20]:
        ex = []
        cov = []

        for sid, rr in by.items():
            gt = set(rr[0].get("truth_roles_training_only") or [])
            seen = greedy(rr, b)

            ex.append(seen == gt)
            cov.append(len(seen & gt) / len(gt))

        print(
            f"budget={b:2d} "
            f"exact={np.mean(ex):.3f} "
            f"coverage={np.mean(cov):.3f}"
        )
