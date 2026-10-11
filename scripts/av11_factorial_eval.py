from __future__ import annotations

import argparse, gc, hashlib, json, math, random, statistics
from collections import Counter, defaultdict
from pathlib import Path

from adaptive_v11_active.io import read_jsonl, stable_int, write_json, write_jsonl, sha256_json
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
from v11.role_universe import TRAIN_ROLES

ARMS = ["A_original_static", "B_original_adaptive", "C_grpo_static", "D_grpo_adaptive"]


def key(roles): return tuple(sorted(map(str, roles)))
def mean(xs): return statistics.mean(xs) if xs else float("nan")


def manifest_role_sets(path: Path):
    out=set()
    try: obj=json.loads(path.read_text())
    except Exception: return out
    if not isinstance(obj,dict): return out
    for fld in ("systems","known_specs","dev_specs","train_specs","test_specs"):
        for r in obj.get(fld,[]) if isinstance(obj.get(fld,[]),list) else []:
            if isinstance(r,dict):
                rs=r.get("roles",r.get("worker_roles",()))
                if rs: out.add(key(rs))
    return out


def prepare(args):
    from targets.single_router_zoo_v11 import MODES, KS, _make_train_specs, get_v11_split, validate_v11_design
    from adaptive_v11_active.belief import FrozenBeliefBundle
    from adaptive_v11_active.role_belief import role_only_predict

    validate_v11_design()
    root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
    forbidden=set()
    for split in ("train","dev"):
        try:
            forbidden |= {key(s.worker_roles) for s in get_v11_split(split)}
        except Exception: pass
    # Explicit strong-eval guard + every existing artifact manifest.
    try:
        sc=_make_train_specs(per_mode=8,seed=24681357)
        vt={key(s.worker_roles) for s in get_v11_split("train")}
        sf=[s for s in sc if key(s.worker_roles) not in vt]
        for mode in MODES:
            forbidden |= {key(s.worker_roles) for s in [x for x in sf if x.execution_mode==mode][:6]}
    except Exception: pass
    sources=[]
    for p in sorted(Path("artifacts").rglob("*.json")):
        if "manifest" in p.name.lower():
            before=len(forbidden); forbidden |= manifest_role_sets(p)
            if len(forbidden)>before: sources.append(str(p))

    rng=random.Random(args.design_seed)
    candidates=_make_train_specs(per_mode=max(256,args.per_cell*96),seed=args.design_seed)
    candidates=[s for s in candidates if key(s.worker_roles) not in forbidden]
    counts=Counter({r:0 for r in TRAIN_ROLES}); used=set(); chosen=[]
    for mode in MODES:
        for kval in KS:
            pool=[s for s in candidates if s.execution_mode==mode and len(s.worker_roles)==kval and key(s.worker_roles) not in used]
            for _ in range(args.per_cell):
                if not pool: raise RuntimeError(f"no fresh pool mode={mode} K={kval}")
                def cost(s):
                    return sum((counts[r]+1)**2-counts[r]**2 for r in s.worker_roles)+rng.random()*1e-6
                b=min(pool,key=cost); pool.remove(b); used.add(key(b.worker_roles)); counts.update(b.worker_roles); chosen.append(b)

    rows=[]
    for s in chosen:
        opaque=hashlib.sha256(f"heldout|{args.design_seed}|{s.execution_mode}|{s.style}|{','.join(sorted(s.worker_roles))}".encode()).hexdigest()[:16]
        rows.append({"session_id":f"av11h_{opaque}","split":"dev","mode":s.execution_mode,"style":s.style,"roles":list(s.worker_roles),"k":len(s.worker_roles)})
    assert len(rows)==len(MODES)*len(KS)*args.per_cell
    assert len({key(r['roles']) for r in rows})==len(rows)
    assert not ({key(r['roles']) for r in rows}&forbidden)
    assert set(Counter((r['mode'],r['k']) for r in rows).values())=={args.per_cell}

    manifest={"version":"adaptive_v11_factorial_heldout_v1","seed":args.design_seed,"systems":rows,"sha256":sha256_json(rows),"notes":{"privileged":True,"policy_must_never_read_this_file":True,"forbidden_role_sets":len(forbidden),"manifest_sources_excluded":sources}}
    write_json(root/"manifest_privileged.json",manifest)

    belief=FrozenBeliefBundle.load(args.belief); kp,rp=role_only_predict(belief,[])
    seeds=[int(x) for x in args.run_seeds.split(',') if x.strip()]
    for seed in seeds:
        for arm in ARMS:
            rr=root/"runs"/f"seed_{seed:03d}"/arm; rr.mkdir(parents=True,exist_ok=True)
            sts=[]
            for s in rows:
                st=PublicBeliefState(session_id=s['session_id'],split='dev',round_index=0,k_probs=dict(kp),role_probs=dict(rp),history=[])
                st.validate(TRAIN_ROLES,belief.ks); sts.append(st.to_public_dict())
            write_jsonl(rr/"states_dev_round_00.jsonl",sts)
    print(f"[prepare] heldout={len(rows)} forbidden={len(forbidden)}")
    print("[mode]",dict(Counter(r['mode'] for r in rows))); print("[K]",dict(Counter(r['k'] for r in rows))); print("[role]",dict(sorted(counts.items())))


def generate(args):
    import torch
    from adaptive_v11_active.reward import surface_valid
    from adaptive_v11_active.role_policy import RoleAdaptiveProbePolicyV11
    if not torch.cuda.is_available(): raise SystemExit("CUDA required")
    root=Path(args.root); cur=read_jsonl(root/f"states_dev_round_{args.round-1:02d}.jsonl")
    initial={x['session_id']:PublicBeliefState.from_public_dict(x) for x in read_jsonl(root/"states_dev_round_00.jsonl")}
    policy=RoleAdaptiveProbePolicyV11("Qwen/Qwen2.5-1.5B-Instruct",adapter_path=args.adapter,trainable=False,use_4bit=False,max_seq_length=2048)
    out=[]
    for i,raw in enumerate(cur):
        assert_policy_safe(raw); state=PublicBeliefState.from_public_dict(raw); seen=[h.query for h in state.history]
        if args.static:
            b=initial[state.session_id]
            ps=PublicBeliefState(session_id=state.session_id,split=state.split,round_index=state.round_index,k_probs=dict(b.k_probs),role_probs=dict(b.role_probs),history=list(state.history))
        else: ps=state
        seed=stable_int(args.seed,"dev",args.round,state.session_id,0); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        sample=policy.sample_probe({"belief_state":ps.to_public_dict(),"candidate_index":0,"max_history_turns":2,"max_answer_chars":0},temperature=.90,top_p=.95,max_new_tokens=64)
        q=str(sample.query or '').strip(); ok,why=surface_valid(q); valid=bool(sample.valid) and ok and q not in seen
        out.append({"session_id":state.session_id,"split":"dev","round":args.round,"candidate_index":0,"group_id":f"{state.session_id}::r{args.round:02d}","query":q,"valid":valid,"invalid_reason":"" if valid else (str(getattr(sample,'invalid_reason','')) or why or 'duplicate'),"prompt":str(sample.prompt),"completion":str(sample.completion),"old_logp":float(getattr(sample,'old_mean_logprob',0.0)),"ref_logp":0.0,"raw_completion":str(sample.raw_completion),"local_attempts":int(getattr(sample,'local_attempts',1)),"seed":int(seed),"adapter":args.adapter,"phase":"factorial_eval","probe_condition":"static" if args.static else "adaptive"})
        print(f"[gen] {i+1}/{len(cur)} r={args.round} static={args.static} valid={valid} q={q[:110]!r}",flush=True)
    write_jsonl(root/f"candidates_dev_round_{args.round:02d}.jsonl",out); print(f"[done] valid={sum(int(x['valid']) for x in out)}/{len(out)}")
    del policy; gc.collect(); torch.cuda.empty_cache()


def boot(ds,n=30000,seed=7):
    q=random.Random(seed); m=len(ds); z=[]
    for _ in range(n): z.append(mean([ds[q.randrange(m)] for __ in range(m)]))
    z.sort(); return z[int(.025*n)],z[min(n-1,int(.975*n))]

def perm(ds,n=100000,seed=9):
    q=random.Random(seed); obs=mean(ds); a=abs(obs); ext=1; low=1
    for _ in range(n):
        x=mean([d if q.random()<.5 else -d for d in ds])
        if abs(x)>=a-1e-15: ext+=1
        if x<=obs+1e-15: low+=1
    return ext/(n+1),low/(n+1)

def show(name,A,B,V,lower=True):
    ids=sorted(set(V[A])&set(V[B])); rows=[]; ds=[]
    for sid in ids:
        a=mean(V[A][sid]); b=mean(V[B][sid]); d=b-a; rows.append((sid,a,b,d)); ds.append(d)
    lo,hi=boot(ds); p2,p1=perm(ds); wins=sum((d<0 if lower else d>0) for d in ds); ties=sum(abs(d)<=1e-12 for d in ds)
    print(f"\n{name}: {B}-{A}\n  n={len(ds)} delta={mean(ds):+.6f} CI95=[{lo:+.6f},{hi:+.6f}] W/T/L={wins}/{ties}/{len(ds)-wins-ties} p2={p2:.6f}"+(f" p1lower={p1:.6f}" if lower else ''))
    return rows,ds


def analyze(args):
    root=Path(args.root); M=json.loads((root/"manifest_privileged.json").read_text()); truth={x['session_id']:x for x in M['systems']}; seeds=sorted((root/"runs").glob("seed_*"))
    print("="*88); print("AV11 HELD-OUT 2x2 FACTORIAL"); print("systems",len(truth),"seeds",[s.name for s in seeds]); print("A=Original+Static B=Original+Adaptive C=GRPO+Static D=GRPO+Adaptive")
    print("\nVALIDITY")
    for arm in ARMS:
        for r in (1,2,3):
            g=s=0
            for sd in seeds:
                g+=len(read_jsonl(sd/arm/f"candidates_dev_round_{r:02d}.jsonl")); s+=len(read_jsonl(sd/arm/f"scored_dev_round_{r:02d}.jsonl"))
            print(f"{arm:22s} R{r} {s}/{g} {100*s/max(1,g):.1f}%")
    print("\nTRAJECTORY")
    for r in (1,2,3):
        print('ROUND',r)
        for arm in ARMS:
            z=[]
            for sd in seeds: z+=read_jsonl(sd/arm/f"metrics_dev_round_{r:02d}.jsonl")
            print(f" {arm:22s} LL={mean([x['role_log_loss'] for x in z]):.6f} Brier={mean([x['role_brier'] for x in z]):.6f} AUC={mean([x['role_auc'] for x in z]):.6f} Recall={mean([x['oracle_k_role_recall'] for x in z]):.6f}")
    metrics=['role_log_loss','role_brier','role_auc','oracle_k_role_recall','true_false_margin']; V={m:{a:defaultdict(list) for a in ARMS} for m in metrics}
    for sd in seeds:
        for arm in ARMS:
            for x in read_jsonl(sd/arm/'metrics_dev_round_03.jsonl'):
                for m in metrics: V[m][arm][x['session_id']].append(float(x[m]))
    print("\n"+'='*88); print("PRIMARY ROUND-3 LOG-LOSS (negative=second arm better)"); print('='*88)
    rows,_=show('PRIMARY GRPO under adaptivity',ARMS[1],ARMS[3],V['role_log_loss'],True)
    show('Adaptivity original',ARMS[0],ARMS[1],V['role_log_loss'],True); show('GRPO static',ARMS[0],ARMS[2],V['role_log_loss'],True); show('Adaptivity GRPO',ARMS[2],ARMS[3],V['role_log_loss'],True); show('Full vs original-static',ARMS[0],ARMS[3],V['role_log_loss'],True)
    inter=[]
    for sid in truth:
        A=mean(V['role_log_loss'][ARMS[0]][sid]); B=mean(V['role_log_loss'][ARMS[1]][sid]); C=mean(V['role_log_loss'][ARMS[2]][sid]); D=mean(V['role_log_loss'][ARMS[3]][sid]); inter.append((D-C)-(B-A))
    lo,hi=boot(inter,seed=11); p2,_=perm(inter,seed=13); print(f"\nINTERACTION (D-C)-(B-A): {mean(inter):+.6f} CI95=[{lo:+.6f},{hi:+.6f}] p2={p2:.6f}")
    print("\nSECONDARY D-B")
    for m in metrics[1:]: show(m,ARMS[1],ARMS[3],V[m],m=='role_brier')
    db={sid:d for sid,_,_,d in rows}
    print("\nD-B BY K")
    for k in sorted({x['k'] for x in truth.values()}):
        z=[db[sid] for sid,x in truth.items() if x['k']==k]; print(f" K={k} n={len(z)} delta={mean(z):+.6f} improved={sum(d<0 for d in z)}/{len(z)}")
    print("\nD-B BY MODE")
    for mode in sorted({x['mode'] for x in truth.values()}):
        z=[db[sid] for sid,x in truth.items() if x['mode']==mode]; print(f" {mode} n={len(z)} delta={mean(z):+.6f} improved={sum(d<0 for d in z)}/{len(z)}")


def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    p=sub.add_parser('prepare'); p.add_argument('--root',required=True); p.add_argument('--belief',required=True); p.add_argument('--design-seed',type=int,default=2026091601); p.add_argument('--per-cell',type=int,default=2); p.add_argument('--run-seeds',default='101,202,303')
    g=sub.add_parser('generate'); g.add_argument('--root',required=True); g.add_argument('--adapter',required=True); g.add_argument('--seed',type=int,required=True); g.add_argument('--round',type=int,required=True); g.add_argument('--static',action='store_true')
    a=sub.add_parser('analyze'); a.add_argument('--root',required=True)
    args=ap.parse_args(); {'prepare':prepare,'generate':generate,'analyze':analyze}[args.cmd](args)
if __name__=='__main__': main()
