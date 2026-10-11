from __future__ import annotations

import argparse, hashlib, json, random
from collections import Counter
from itertools import combinations
from pathlib import Path

from adaptive_v11_active.io import write_json
from adaptive_v11_curriculum.config import STAGES
from v11.role_universe import TRAIN_ROLES

FAMILIES = {
    "engineering": {"code","database_design","qa_testing","operations"},
    "language": {"writing","translation","documentation","accessibility_review"},
    "science": {"research","fact_checking","experiment_design","data"},
    "planning": {"planning","requirements_analysis","risk_assessment","design"},
    "business": {"financial_analysis","customer_support_triage","incident_management"},
    "math": {"math"},
}
ROLE_FAMILY = {r:f for f,rs in FAMILIES.items() for r in rs}
for r in TRAIN_ROLES:
    ROLE_FAMILY.setdefault(r, r)

try:
    from v11_2.counterfactual import PREFERRED_CONFUSERS
    CONF = {r:set(map(str,vs)) for r,vs in PREFERRED_CONFUSERS.items()}
except Exception:
    CONF={r:set() for r in TRAIN_ROLES}


def comp_key(roles): return tuple(sorted(map(str,roles)))

def conf_edges(roles):
    rr=set(roles); n=0
    for a,b in combinations(sorted(rr),2):
        if b in CONF.get(a,set()) or a in CONF.get(b,set()) or ROLE_FAMILY[a]==ROLE_FAMILY[b]:
            n+=1
    return n


def semantic_ok(roles, rule):
    fams=[ROLE_FAMILY[r] for r in roles]
    if rule=="distinct":
        return len(set(fams))==len(fams) and conf_edges(roles)==0
    if rule=="one_confuser":
        return conf_edges(roles)>=1 and len(set(fams))>=2
    if rule=="multi_confuser":
        return conf_edges(roles)>=2 and len(set(fams))>=2
    return True


def existing_compositions(search_root: Path) -> set[tuple[str,...]]:
    out=set()
    for p in search_root.rglob('*.json'):
        if 'manifest' not in p.name.lower():
            continue
        try: obj=json.loads(p.read_text(encoding='utf-8'))
        except Exception: continue
        rows=obj.get('systems',[]) if isinstance(obj,dict) else []
        for x in rows:
            if isinstance(x,dict) and isinstance(x.get('roles'),list):
                out.add(comp_key(x['roles']))
    return out


def make_rows(*,rng, n, k_choices, modes, rule, split, stage, cycle, forbidden, prefix):
    rows=[]; role_count=Counter(); cell_count=Counter(); attempts=0
    while len(rows)<n and attempts<200000:
        attempts+=1
        k=rng.choice(tuple(k_choices)); mode=rng.choice(tuple(modes))
        # Balanced-ish role coverage: sample a small pool biased toward under-used roles.
        ranked=sorted(TRAIN_ROLES,key=lambda r:(role_count[r],rng.random()))
        pool=ranked[:max(k+5,12)]
        roles=tuple(sorted(rng.sample(pool,k)))
        key=comp_key(roles)
        if key in forbidden or any(comp_key(x['roles'])==key for x in rows): continue
        if not semantic_ok(roles,rule): continue
        opaque=hashlib.sha256(f"{prefix}|s{stage}|c{cycle}|{split}|{mode}|{','.join(roles)}".encode()).hexdigest()[:16]
        rec={
            'session_id':f"av11cur_{opaque}", 'split':split, 'mode':mode, 'style':'s1',
            'roles':list(roles), 'k':len(roles), 'curriculum_stage':stage, 'curriculum_cycle':cycle,
        }
        rows.append(rec); forbidden.add(key); role_count.update(roles); cell_count[(mode,k)]+=1
    if len(rows)!=n:
        raise RuntimeError(f"could not build stage={stage} cycle={cycle} split={split}: {len(rows)}/{n} after {attempts}")
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='artifacts/adaptive_v11_curriculum_v1')
    ap.add_argument('--search-root',default='artifacts')
    ap.add_argument('--seed',type=int,default=20260918)
    args=ap.parse_args()
    root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
    forbidden=existing_compositions(Path(args.search_root))
    # Do not let a previous partial prepare of THIS curriculum block deterministic reruns.
    for p in root.rglob('manifest*.json'):
        try:
            obj=json.loads(p.read_text())
            for x in obj.get('systems',[]):
                forbidden.discard(comp_key(x.get('roles',[])))
        except Exception: pass
    rng=random.Random(args.seed)
    report={'seed':args.seed,'existing_forbidden_before_new':len(forbidden),'stages':[]}
    all_new=[]
    for s in STAGES:
        sdir=root/f"stage_{s.stage:02d}"; sdir.mkdir(parents=True,exist_ok=True)
        stage_info={'stage':s.stage,'name':s.name,'horizon':s.horizon,'group_size':s.group_size,'cycles':[]}
        for c in range(1,s.cycles+1):
            rows=make_rows(rng=rng,n=s.train_systems_per_cycle,k_choices=s.k_choices,modes=s.modes,rule=s.semantic_rule,
                           split='train',stage=s.stage,cycle=c,forbidden=forbidden,prefix=f"{args.seed}|train")
            obj={'version':'adaptive_v11_curriculum_manifest_v1','seed':args.seed,'stage':s.stage,'cycle':c,
                 'semantic_rule':s.semantic_rule,'systems':rows,
                 'sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest(),
                 'notes':{'privileged':True,'policy_must_never_read_this_file':True}}
            out=sdir/f"manifest_train_cycle_{c:02d}.json"; write_json(out,obj); all_new.extend(rows)
            stage_info['cycles'].append({'cycle':c,'systems':len(rows),'file':str(out),'K':dict(Counter(map(str,[x['k'] for x in rows]))),'mode':dict(Counter(x['mode'] for x in rows))})
        dev=make_rows(rng=rng,n=s.dev_systems,k_choices=s.k_choices,modes=s.modes,rule=s.semantic_rule,
                      split='dev',stage=s.stage,cycle=0,forbidden=forbidden,prefix=f"{args.seed}|dev")
        dobj={'version':'adaptive_v11_curriculum_manifest_v1','seed':args.seed,'stage':s.stage,'cycle':0,
              'semantic_rule':s.semantic_rule,'systems':dev,
              'sha256':hashlib.sha256(json.dumps(dev,sort_keys=True).encode()).hexdigest(),
              'notes':{'privileged':True,'policy_must_never_read_this_file':True,'stage_dev':True}}
        dout=sdir/'manifest_dev.json'; write_json(dout,dobj); all_new.extend(dev)
        stage_info['dev']={'systems':len(dev),'file':str(dout),'K':dict(Counter(map(str,[x['k'] for x in dev]))),'mode':dict(Counter(x['mode'] for x in dev))}
        report['stages'].append(stage_info)
    # Untouched final test. This is created but submit script never executes it.
    final_rows=make_rows(rng=rng,n=32,k_choices=(3,4,5,6,7,8),modes=('single','fanout2','fanout3','mixed'),rule='any',
                         split='dev',stage=99,cycle=0,forbidden=forbidden,prefix=f"{args.seed}|final")
    final={'version':'adaptive_v11_curriculum_final_test_v1','seed':args.seed,'systems':final_rows,
           'sha256':hashlib.sha256(json.dumps(final_rows,sort_keys=True).encode()).hexdigest(),
           'notes':{'privileged':True,'UNTOUCHED_BY_TRAINING_SUBMITTER':True}}
    write_json(root/'final_test_manifest_UNTOUCHED.json',final)
    report['final_test_systems']=len(final_rows)
    report['new_compositions']=len(all_new)+len(final_rows)
    write_json(root/'manifest_report.json',report)
    print(json.dumps(report,indent=2))
    print(f"[done] root={root} untouched_final_test=32")

if __name__=='__main__': main()
