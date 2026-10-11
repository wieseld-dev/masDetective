from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--log", default="artifacts/v10_probe_discovery.jsonl")
ap.add_argument("--summary", default="artifacts/v10_summary.json")
args = ap.parse_args()

rows = [json.loads(x) for x in Path(args.log).open(encoding="utf-8") if x.strip()]
train = [r for r in rows if r.get("phase") == "train"]
dev = [r for r in rows if r.get("phase") == "holdout"]
evaluated = [r for r in train if r.get("reward_breakdown_training_only")]
strict = [r for r in evaluated if r["reward_breakdown_training_only"].get("strict_specific_pass")]
elite = [r for r in train if r.get("elite")]
confirmed = [r for r in train if r.get("confirmation_pass_training_only")]

print("=== V10.1 DIAGNOSTIC ===")
print("train candidates:", len(train))
print("fully evaluated:", len(evaluated), f"({len(evaluated)/max(1,len(train)):.1%})")
print("strict primary probes:", len(strict), f"({len(strict)/max(1,len(evaluated)):.1%})")
print("confirmation passes:", len(confirmed))
print("confirmed elite probes:", len(elite))

if evaluated:
    print("mean robust activation:", f"{np.mean([r['reward_breakdown_training_only']['activation_rate'] for r in evaluated]):.3f}")
    print("mean plus activation:", f"{np.mean([r['reward_breakdown_training_only'].get('plus_activation_rate', r['reward_breakdown_training_only']['activation_rate']) for r in evaluated]):.3f}")
    print("mean sham activation:", f"{np.mean([r['reward_breakdown_training_only'].get('sham_activation_rate', r['reward_breakdown_training_only']['activation_rate']) for r in evaluated]):.3f}")
    print("mean robust SPEC:", f"{np.mean([r['reward_breakdown_training_only']['robust_specific'] for r in evaluated]):+.4f}")

by = defaultdict(list)
for r in elite:
    by[r["target_role"]].append(r)
for role in sorted(by):
    print(f"\n{role.upper()} CONFIRMED ELITES")
    for r in sorted(by[role], key=lambda x: x["reward"], reverse=True)[:6]:
        b = r["reward_breakdown_training_only"]
        c = r.get("confirmation_reward_breakdown_training_only") or {}
        print(
            f" R={r['reward']:+.3f} ACT={b['activation_rate']:.2f} ROB={b['robust_specific']:+.3f} "
            f"MED={b['specific_median']:+.3f} Q25={b['specific_q25']:+.3f} "
            f"PAIR+={b['positive_specific_pair_rate']:.2f} SIM={b['semantic_max_similarity']:.3f} "
            f"| confirm ACT={c.get('activation_rate', float('nan')):.2f} ROB={c.get('robust_specific', float('nan')):+.3f}"
        )
        print(" ", r["query"])

if dev:
    print("\n=== EXHAUSTIVE DEVELOPMENT VALIDATION (INSPECTED; NOT FINAL TEST) ===")
    byh = defaultdict(list)
    for r in dev:
        byh[r["target_role"]].append(r)
    role_strict = role_act = role_delta = 0
    for role in sorted(byh):
        rr = byh[role]
        good = [r for r in rr if r.get("reward_breakdown_training_only")]
        bs = [r["reward_breakdown_training_only"] for r in good]
        any_strict = any(b.get("strict_specific_pass") for b in bs)
        any_delta = any(b.get("strict_delta_pass") for b in bs)
        any_act = any(float(b.get("activation_rate", 0)) >= 0.75 for b in bs)
        role_strict += int(any_strict)
        role_delta += int(any_delta)
        role_act += int(any_act)
        best = max((float(b["robust_specific"]) for b in bs), default=float("nan"))
        print(
            f" {role:<10} valid={len(good)}/{len(rr)} anyACT={any_act} anySTRICT={any_strict} "
            f"anyDELTA={any_delta} bestROB={best:+.3f}"
        )
        for r in rr:
            b = r.get("reward_breakdown_training_only")
            if b:
                print(
                    f"   ACT={b['activation_rate']:.2f} (P={b.get('plus_activation_rate', b['activation_rate']):.2f},S={b.get('sham_activation_rate', b['activation_rate']):.2f}) "
                    f"ROB={b['robust_specific']:+.3f} MED={b['specific_median']:+.3f} Q25={b['specific_q25']:+.3f} "
                    f"PAIR+={b['positive_specific_pair_rate']:.2f} strict={b['strict_specific_pass']} | {r['query']}"
                )
            else:
                print(f"   {r['status']} | {r['query']}")
    n = max(1, len(byh))
    print(f"role any activation>=.75: {role_act}/{n} = {role_act/n:.1%}")
    print(f"role any strict SPEC:     {role_strict}/{n} = {role_strict/n:.1%}")
    print(f"role any strict DELTA:    {role_delta}/{n} = {role_delta/n:.1%}")

if Path(args.summary).exists():
    s = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    print("\nBASELINE COMPARISON")
    b = s.get("initial_v9_exhaustive_baseline", {})
    print("V9 exhaustive strict SPEC roles:", b.get("role_any_strict_specific"))
    print("V10.1 dev strict SPEC roles:", s.get("holdout_role_any_strict_specific"))
    print("V9 exhaustive strict DELTA roles:", b.get("role_any_strict_delta"))
    print("V10.1 dev strict DELTA roles:", s.get("holdout_role_any_strict_delta"))
    print("final untouched test:", s.get("final_untouched_test"))
