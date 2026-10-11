from __future__ import annotations
import argparse,json,os
from pathlib import Path
from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_fanout_mas import SingleRouterSystemSpec
from v8.execution import execute_cached


def load_jsonl(p):
    out=[]
    with Path(p).open(encoding='utf-8') as f:
        for line in f:
            if line.strip():out.append(json.loads(line))
    return out


def main():
    ap=argparse.ArgumentParser(description='CPU/API collection for adaptive controller replay bank')
    ap.add_argument('--manifest',default='artifacts/adaptive_v1/manifest.json')
    ap.add_argument('--actions',default='artifacts/adaptive_v1/action_library.jsonl')
    ap.add_argument('--output',default='artifacts/adaptive_v1/public_observations.jsonl')
    ap.add_argument('--public-cache',default='artifacts/adaptive_v1/response_public.jsonl')
    ap.add_argument('--truth-cache',default='artifacts/adaptive_v1/response_truth.jsonl')
    args=ap.parse_args()
    if not os.environ.get('OPENAI_API_KEY'):raise SystemExit('OPENAI_API_KEY is not set')
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'));actions=[x for x in load_jsonl(args.actions) if x.get('valid') and x.get('query')]
    specs=[]
    for x in manifest['systems']:
        specs.append((str(x['split']),SingleRouterSystemSpec(system_id=x['system_id'],split=f"adaptive_v1_{x['split']}",execution_mode=x['mode'],style=x['style'],worker_roles=tuple(x['roles']))))
    cache=ResponseCacheV7(args.public_cache,args.truth_cache)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);tmp=out.with_suffix(out.suffix+'.tmp');done=0;total=len(specs)*len(actions)
    with tmp.open('w',encoding='utf-8') as f:
        for split,spec in specs:
            for a in actions:
                r=execute_cached(spec,str(a['query']),cache);done+=1
                # Public response is stored here. Privileged labels are written only for offline training/eval metadata.
                rec={'system_id':spec.system_id,'split':split,'mode':spec.execution_mode,'truth_roles_training_only':list(spec.worker_roles),'selected_roles_training_only':list(r.selected_roles_training_only),'action':{'target':a['target'],'competitor':a.get('competitor'),'variant':int(a.get('variant',0))},'query':a['query'],'output':r.output,'cache_hit':bool(r.cache_hit),'manifest_sha256':manifest.get('sha256')}
                f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                print(f"[collect] {done}/{total} split={split} system={spec.system_id} target={a['target']:<24} selected={list(r.selected_roles_training_only)} cache={r.cache_hit}",flush=True)
    tmp.replace(out);print(f'[done] public_observations={out} rows={done}',flush=True)
if __name__=='__main__':main()
