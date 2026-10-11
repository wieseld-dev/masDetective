from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from adaptive_v11_active.io import read_jsonl, write_json

METRICS = [
    "role_log_loss",
    "role_brier",
    "role_auc",
    "true_false_margin",
    "oracle_k_role_recall",
]


def paired_rows(root: Path, split: str, rnd: int) -> dict[str, dict]:
    p = root / f"kboot_metrics_{split}_round_{rnd:02d}.jsonl"
    if not p.exists():
        raise SystemExit(f"missing metrics: {p}")
    return {str(x["session_id"]): x for x in read_jsonl(p)}


def mean(xs):
    arr = np.asarray(xs, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def bootstrap_delta(a: np.ndarray, b: np.ndarray, *, seed: int, nboot: int = 20000):
    # delta = adaptive - static
    d = np.asarray(a - b, dtype=float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")]}
    rng = np.random.default_rng(seed)
    vals = np.empty(nboot, dtype=float)
    n = len(d)
    for i in range(nboot):
        vals[i] = d[rng.integers(0, n, size=n)].mean()
    return {
        "mean": float(d.mean()),
        "ci95": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
    }


def targets(root: Path, split: str, rnd: int) -> dict[str, str]:
    p = root / f"candidates_{split}_round_{rnd:02d}.jsonl"
    if not p.exists():
        return {}
    return {str(x["session_id"]): str(x.get("target_role", "")) for x in read_jsonl(p)}


def main():
    ap = argparse.ArgumentParser(description="Paired static-vs-adaptive controller report")
    ap.add_argument("--static-root", required=True)
    ap.add_argument("--adaptive-root", required=True)
    ap.add_argument("--split", default="dev", choices=["train", "dev"])
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--seed", type=int, default=917511)
    args = ap.parse_args()

    sr = Path(args.static_root)
    ar = Path(args.adaptive_root)
    outroot = Path(args.output_root)
    outroot.mkdir(parents=True, exist_ok=True)

    rounds = []
    lines = []
    lines.append("SIMPLE ADAPTIVE V11 CONTROLLER — STATIC vs ADAPTIVE")
    lines.append("delta = adaptive - static; lower is better for log-loss/Brier, higher for AUC/margin/recall")
    lines.append("")

    for rnd in range(1, args.rounds + 1):
        sm = paired_rows(sr, args.split, rnd)
        am = paired_rows(ar, args.split, rnd)
        ids = sorted(set(sm) & set(am))
        if not ids:
            raise SystemExit(f"no paired systems at round {rnd}")

        rec = {"round": rnd, "paired_systems": len(ids), "metrics": {}}
        lines.append(f"ROUND {rnd} paired={len(ids)}")
        for j, key in enumerate(METRICS):
            s = np.asarray([float(sm[x][key]) for x in ids], dtype=float)
            a = np.asarray([float(am[x][key]) for x in ids], dtype=float)
            bd = bootstrap_delta(a, s, seed=args.seed + 1000 * rnd + j)
            rec["metrics"][key] = {
                "static_mean": mean(s),
                "adaptive_mean": mean(a),
                "adaptive_minus_static": bd["mean"],
                "paired_bootstrap_ci95": bd["ci95"],
            }
            lines.append(
                f"  {key:22s} static={mean(s):.6f} adaptive={mean(a):.6f} "
                f"delta={bd['mean']:+.6f} CI95=[{bd['ci95'][0]:+.6f},{bd['ci95'][1]:+.6f}]"
            )

        st = targets(sr, args.split, rnd)
        at = targets(ar, args.split, rnd)
        tids = sorted(set(st) & set(at))
        same = sum(st[x] == at[x] for x in tids)
        rec["target_selection"] = {
            "paired": len(tids),
            "same_target": same,
            "different_target": len(tids) - same,
            "static_counts": dict(Counter(st[x] for x in tids)),
            "adaptive_counts": dict(Counter(at[x] for x in tids)),
        }
        lines.append(f"  target-role same={same}/{len(tids)} different={len(tids)-same}/{len(tids)}")
        lines.append("")
        rounds.append(rec)

    final = rounds[-1]
    ids = sorted(set(paired_rows(sr, args.split, args.rounds)) & set(paired_rows(ar, args.split, args.rounds)))
    sm = paired_rows(sr, args.split, args.rounds)
    am = paired_rows(ar, args.split, args.rounds)
    d = np.asarray([float(am[x]["role_log_loss"]) - float(sm[x]["role_log_loss"]) for x in ids])
    wins = int(np.sum(d < -1e-12))
    ties = int(np.sum(np.abs(d) <= 1e-12))
    losses = int(np.sum(d > 1e-12))

    obj = {
        "version": "simple-adaptive-v11-controller-v1",
        "split": args.split,
        "rounds": rounds,
        "primary_endpoint": {
            "metric": "role_log_loss",
            "round": args.rounds,
            **final["metrics"]["role_log_loss"],
            "adaptive_better_tie_worse_systems": [wins, ties, losses],
        },
        "interpretation_guardrail": (
            "This isolates controller adaptivity: both arms use the same frozen original V11 generator, "
            "same judges, same belief updater, same systems, same budget, and a shared exact round-1 trajectory."
        ),
    }
    write_json(outroot / "controller_report.json", obj)
    (outroot / "controller_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("PRIMARY:", json.dumps(obj["primary_endpoint"], indent=2), flush=True)
    print(f"[done] {outroot / 'controller_report.json'}", flush=True)


if __name__ == "__main__":
    main()
