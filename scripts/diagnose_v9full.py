from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--log',default='artifacts/v9_full_probe_discovery.jsonl')
args=ap.parse_args()
rows=[json.loads(x) for x in Path(args.log).open(encoding='utf-8') if x.strip()]
train=[r for r in rows if r.get('phase')=='train']
hold=[r for r in rows if r.get('phase')=='holdout']
ok=[r for r in train if r.get('reward_breakdown_training_only')]
pos=[r for r in ok if r['reward_breakdown_training_only']['specific_separation']>0]
elite=[r for r in train if r.get('elite')]
print('=== V9-FULL DIAGNOSTIC ===')
print('train candidates:',len(train))
print('fully evaluated:',len(ok))
print('valid/evaluated rate:',f'{len(ok)/max(1,len(train)):.1%}')
print('positive target-specific separation:',len(pos),f'/{len(ok)} = {len(pos)/max(1,len(ok)):.1%}')
if ok:
    print('mean specific separation:',f"{np.mean([r['reward_breakdown_training_only']['specific_separation'] for r in ok]):+.4f}")
    print('mean activation rate:',f"{np.mean([r['reward_breakdown_training_only']['activation_rate'] for r in ok]):.3f}")
print('elite probes:',len(elite))
by=defaultdict(list)
for r in elite: by[r['target_role']].append(r)
for role in sorted(by):
    print('\n',role.upper())
    for r in sorted(by[role],key=lambda x:x['reward'],reverse=True)[:8]:
        b=r['reward_breakdown_training_only']
        print(f" R={r['reward']:+.3f} SPEC={b['specific_separation']:+.3f} ACT={b['activation_rate']:.2f} REL={b['relative_to_anchor']:+.3f}")
        print(' ',r['query'])
if hold:
    hgood=[r for r in hold if r.get('reward_breakdown_training_only')]
    hpos=[r for r in hgood if r['reward_breakdown_training_only']['specific_separation']>0]
    print('\nHOLDOUT ALL PROBES')
    print('valid/evaluated:',len(hgood),'/',len(hold))
    print('positive specific:',len(hpos),f'/{len(hgood)} = {len(hpos)/max(1,len(hgood)):.1%}')
    byh=defaultdict(list)
    for r in hold: byh[r['target_role']].append(r)
    role_any=role_mean=0
    for role in sorted(byh):
        rr=byh[role]
        good=[r for r in rr if r.get('reward_breakdown_training_only')]
        specs=[r['reward_breakdown_training_only']['specific_separation'] for r in good]
        anypos=any(x>0 for x in specs)
        mean=float(np.mean(specs)) if specs else float('nan')
        role_any += int(anypos); role_mean += int(bool(specs) and mean>0)
        print(f" {role:<10} valid={len(good)}/{len(rr)} meanSPEC={mean:+.3f} bestSPEC={(max(specs) if specs else float('nan')):+.3f} any+={anypos}")
        for r in rr:
            b=r.get('reward_breakdown_training_only')
            if b: print(f"   R={r['reward']:+.3f} SPEC={b['specific_separation']:+.3f} ACT={b['activation_rate']:.2f} | {r['query']}")
            else: print(f"   {r['status']} | {r['query']}")
    print('role any-positive:',role_any,'/',len(byh),f'= {role_any/max(1,len(byh)):.1%}')
    print('role mean-positive:',role_mean,'/',len(byh),f'= {role_mean/max(1,len(byh)):.1%}')
