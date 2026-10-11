from pathlib import Path
from collections import defaultdict
import json
import numpy as np

ROOT=Path("artifacts/adaptive_v11_curriculum_v2")

by=defaultdict(list)

for p in ROOT.glob("stage_*/cycle_*/responses_train_round_*.jsonl"):
    for line in p.open():
        if not line.strip():
            continue
        r=json.loads(line)
        sid=r.get("system_id") or r.get("session_id")
        if sid:
            by[sid].append(r)

print("="*95)
print("EXISTING TRAINED-POLICY EXPOSURE — SMALL MAS")
print("="*95)

for K in [3,4,5,6,7,8]:
    vals=[]

    for sid,rr in by.items():
        truths={
            tuple(sorted(x.get("truth_roles_training_only") or []))
            for x in rr
            if x.get("truth_roles_training_only")
        }
        if len(truths)!=1:
            continue

        gt=set(next(iter(truths)))
        if len(gt)!=K:
            continue

        # Chosen continuation only.
        chosen=[
            x for x in rr
            if x.get("chosen_for_continuation")
            and x.get("execution_status")=="ok"
        ]

        cseen=set()
        for x in chosen:
            cseen.update(x.get("selected_roles_training_only") or [])

        # All generated candidates.
        aseen=set()
        for x in rr:
            if x.get("execution_status")=="ok":
                aseen.update(x.get("selected_roles_training_only") or [])

        vals.append((
            len(cseen & gt)/K,
            cseen==gt,
            len(aseen & gt)/K,
            aseen==gt,
            len(chosen),
            len(rr),
        ))

    if not vals:
        continue

    A=np.asarray(vals,float)

    print(
        f"K={K} n={len(vals):3d} | "
        f"chosen coverage={A[:,0].mean():.3f} "
        f"chosen exact={A[:,1].mean():.3f} | "
        f"all-candidate coverage={A[:,2].mean():.3f} "
        f"all-candidate exact={A[:,3].mean():.3f} | "
        f"mean chosen probes={A[:,4].mean():.2f}"
    )
