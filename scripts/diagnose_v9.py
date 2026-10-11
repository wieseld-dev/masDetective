from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument("--log",default="artifacts/v9_probe_discovery.jsonl")
args=ap.parse_args()
rows=[json.loads(x) for x in Path(args.log).open(encoding="utf-8") if x.strip()]
train=[r for r in rows if r.get("phase")=="train"]
hold=[r for r in rows if r.get("phase")=="holdout"]
ok=[r for r in train if r.get("reward_breakdown_training_only")]
pos=[r for r in ok if r["reward_breakdown_training_only"]["specific_separation"]>0]
elite=[r for r in train if r.get("elite")]
print("=== V9 DIAGNOSTIC ===")
print("train candidates:",len(train))
print("fully evaluated:",len(ok))
print("positive target-specific separation:",len(pos),f"/{len(ok)} = {len(pos)/max(1,len(ok)):.1%}")
if ok:
    print("mean specific separation:",f"{np.mean([r['reward_breakdown_training_only']['specific_separation'] for r in ok]):+.4f}")
    print("mean activation rate:",f"{np.mean([r['reward_breakdown_training_only']['activation_rate'] for r in ok]):.3f}")
print("elite probes:",len(elite))
by=defaultdict(list)
for r in elite: by[r["target_role"]].append(r)
for role in sorted(by):
    print("\n",role.upper())
    for r in sorted(by[role],key=lambda x:x["reward"],reverse=True)[:5]:
        b=r["reward_breakdown_training_only"]
        print(f" R={r['reward']:+.3f} SPEC={b['specific_separation']:+.3f} ACT={b['activation_rate']:.2f} REL={b['relative_to_anchor']:+.3f}")
        print(" ",r["query"])
if hold:
    hgood=[r for r in hold if r.get("reward_breakdown_training_only")]
    hpos=[r for r in hgood if r["reward_breakdown_training_only"]["specific_separation"]>0]
    print("\nHOLDOUT")
    print("valid/evaluated:",len(hgood),"/",len(hold))
    print("positive specific:",len(hpos),f"/{len(hgood)} = {len(hpos)/max(1,len(hgood)):.1%}")
    for r in hold:
        b=r.get("reward_breakdown_training_only")
        if b:
            print(f" {r['target_role']:<10} R={r['reward']:+.3f} SPEC={b['specific_separation']:+.3f} ACT={b['activation_rate']:.2f} | {r['query']}")
        else:
            print(f" {r['target_role']:<10} {r['status']} | {r['query']}")
