from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from adaptive_v11_active.io import read_jsonl, write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',required=True); ap.add_argument('--horizon',type=int,required=True); ap.add_argument('--output',required=True); a=ap.parse_args()
    rounds=[]
    for r in range(1,a.horizon+1):
        p=Path(a.root)/f"metrics_dev_round_{r:02d}.jsonl"; xs=read_jsonl(p)
        row={'round':r,'systems':len(xs)}
        for k in ['role_log_loss','role_brier','role_auc','true_false_margin','oracle_k_role_recall']:
            vals=[float(x[k]) for x in xs if x.get(k) is not None and np.isfinite(float(x[k]))]
            row[k]=float(np.mean(vals)) if vals else float('nan')
        row['valid_updates']=sum(x.get('status')=='ok' for x in xs); rounds.append(row)
    out={'horizon':a.horizon,'rounds':rounds,'final':rounds[-1]}
    write_json(a.output,out); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
