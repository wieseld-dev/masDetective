from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


ART = Path("artifacts")
KMIN, KMAX = 3, 6


def read_jsonl(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def clip_k(k):
    return max(KMIN, min(KMAX, int(k)))


def map_k(k_probs):
    return int(max(k_probs, key=lambda k: float(k_probs[k])))


def gap_k(role_probs):
    """
    Sort p1 >= ... >= p20.
    Choose the biggest gap after positions 3..6.
    """
    vals = sorted(map(float, role_probs.values()), reverse=True)

    candidates = []
    for k in range(KMIN, KMAX + 1):
        gap = vals[k - 1] - vals[k]
        candidates.append((gap, k))

    gap, k = max(candidates)
    return int(k), float(gap)


def threshold_k(role_probs, tau):
    raw = sum(float(p) >= tau for p in role_probs.values())
    return clip_k(raw)


def sum_k(role_probs):
    return clip_k(round(sum(map(float, role_probs.values()))))


def two_cluster_k(role_probs):
    """
    Actual 1D 2-cluster separation.
    Higher-probability cluster size = estimated K.
    """
    x = np.asarray(list(map(float, role_probs.values())), dtype=float)

    lo = float(np.quantile(x, 0.25))
    hi = float(np.quantile(x, 0.75))

    for _ in range(100):
        dlo = np.abs(x - lo)
        dhi = np.abs(x - hi)
        high = dhi < dlo

        if high.all() or (~high).all():
            break

        nlo = float(x[~high].mean())
        nhi = float(x[high].mean())

        if abs(nlo - lo) + abs(nhi - hi) < 1e-10:
            lo, hi = nlo, nhi
            break

        lo, hi = nlo, nhi

    if hi < lo:
        high = ~high

    return clip_k(int(high.sum()))


def truth_map(manifest):
    obj = json.loads(manifest.read_text(encoding="utf-8"))
    out = {}

    for s in obj.get("systems", []):
        roles = s.get("roles", s.get("worker_roles", []))
        if not roles:
            continue

        k = len(roles)

        for key in ("session_id", "system_id"):
            if s.get(key) is not None:
                out[str(s[key])] = k

    return out


def evaluate_root(root):
    manifest = root / "manifest_privileged.json"
    if not manifest.exists():
        return None

    truth = truth_map(manifest)
    if not truth:
        return None

    state_files = sorted(root.glob("states_dev_round_*.jsonl"))
    if not state_files:
        return None

    results = []

    for sf in state_files:
        rows = read_jsonl(sf)
        round_no = int(sf.stem.split("_")[-1])

        good = [
            r for r in rows
            if str(r.get("session_id")) in truth
            and r.get("role_probs")
            and r.get("k_probs")
        ]

        if not good:
            continue

        methods = defaultdict(list)
        gaps = []

        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50,
                      0.55, 0.60, 0.65, 0.70]

        for r in good:
            sid = str(r["session_id"])
            y = truth[sid]
            rp = r["role_probs"]

            kgap, gap = gap_k(rp)
            gaps.append(gap)

            preds = {
                "k_map": map_k(r["k_probs"]),
                "largest_gap": kgap,
                "two_cluster": two_cluster_k(rp),
                "round_sum": sum_k(rp),
                "threshold_0.50": threshold_k(rp, 0.50),
            }

            for tau in thresholds:
                preds[f"threshold_{tau:.2f}"] = threshold_k(rp, tau)

            for name, pred in preds.items():
                methods[name].append(int(pred == y))

        summary = {
            "round": round_no,
            "n": len(good),
            "mean_gap": float(np.mean(gaps)),
            "median_gap": float(np.median(gaps)),
            "accuracy": {
                k: float(np.mean(v))
                for k, v in methods.items()
            },
        }

        results.append(summary)

    return results


def main():
    datasets = []

    for manifest in ART.rglob("manifest_privileged.json"):
        root = manifest.parent
        result = evaluate_root(root)

        if result:
            datasets.append((root, result))

    if not datasets:
        raise SystemExit(
            "No DEV state roots with matching manifest_privileged.json found."
        )

    output = {}

    for root, rounds in datasets:
        print("\n" + "=" * 110)
        print("DATASET:", root)
        print("=" * 110)

        output[str(root)] = rounds

        for s in rounds:
            a = s["accuracy"]

            print(
                f"\nROUND {s['round']:02d}  n={s['n']}  "
                f"median_gap={s['median_gap']:.4f}"
            )

            print(
                f"  K-MAP         {a['k_map']:.3f}\n"
                f"  LARGEST-GAP   {a['largest_gap']:.3f}\n"
                f"  TWO-CLUSTER   {a['two_cluster']:.3f}\n"
                f"  ROUND-SUM     {a['round_sum']:.3f}\n"
                f"  THRESHOLD .50 {a['threshold_0.50']:.3f}"
            )

            th = sorted(
                (
                    (float(k.split("_")[1]), v)
                    for k, v in a.items()
                    if k.startswith("threshold_")
                ),
                key=lambda z: z[0],
            )

            print("  threshold sweep:")
            print(
                "    " +
                "  ".join(f"{t:.2f}:{acc:.3f}" for t, acc in th)
            )

    out = Path("artifacts/k_decision_rule_benchmark.json")
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nSAVED:", out)


if __name__ == "__main__":
    main()
