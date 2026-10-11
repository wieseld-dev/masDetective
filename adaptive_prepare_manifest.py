from __future__ import annotations
import argparse,json
from pathlib import Path
from adaptive_v1.specs import make_locked_adaptive_specs,write_manifest


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=97531)
    ap.add_argument('--n-train',type=int,default=24)
    ap.add_argument('--n-dev',type=int,default=8)
    ap.add_argument('--strong-manifest',default='artifacts/v11_2_strong_eval/benchmark_manifest.json')
    ap.add_argument('--output',default='artifacts/adaptive_v1/manifest.json')
    args=ap.parse_args()
    specs=make_locked_adaptive_specs(seed=args.seed,n_train=args.n_train,n_dev=args.n_dev,strong_manifest=args.strong_manifest)
    obj=write_manifest(args.output,specs,args.seed)
    print(f"[manifest] systems={len(specs)} train={args.n_train} dev={args.n_dev} sha256={obj['sha256']}")
    from collections import Counter
    rc=Counter(r for s in specs for r in s.worker_roles); mc=Counter(s.execution_mode for s in specs); kc=Counter(len(s.worker_roles) for s in specs)
    print('[role-coverage]',json.dumps(dict(sorted(rc.items())),sort_keys=True))
    print('[mode-coverage]',dict(mc))
    print('[k-coverage]',dict(kc))
    print('NOTE: this is controller-development data, NOT final paper test data.')
if __name__=='__main__':main()
