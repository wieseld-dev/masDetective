from __future__ import annotations

import hashlib,json,random
from collections import Counter
from pathlib import Path
from typing import Sequence

from targets.single_router_fanout_mas import SingleRouterSystemSpec
from targets.single_router_zoo_v11 import _make_train_specs,get_v11_split,MODES,KS
from v11.role_universe import TRAIN_ROLES


def _key(roles):return tuple(sorted(map(str,roles)))


def make_locked_adaptive_specs(*,seed:int=97531,n_train:int=24,n_dev:int=8,strong_manifest:str="artifacts/v11_2_strong_eval/benchmark_manifest.json"):
    """Fresh balanced systems for controller development.

    The default 24/8 design is exactly balanced across the four execution modes
    and K=3..6 in aggregate (8 systems per mode, 8 per K).  Development receives
    two systems per mode and two per K.  Compositions are disjoint from V11 train,
    from the strong generator benchmark when its manifest exists, and from each
    other.  Other train/dev sizes are accepted but lose the exact balancing
    guarantee and use a generic greedy fallback.
    """
    forbidden={_key(s.worker_roles) for s in get_v11_split("train")}
    # Reconstruct the already-defined strong-eval locked compositions even if its
    # manifest has not been materialized yet. This prevents ordering-dependent
    # overlap when the adaptive manifest is locked while other jobs are pending.
    strong_candidates=_make_train_specs(per_mode=8,seed=24681357)
    v11_train_keys={_key(s.worker_roles) for s in get_v11_split("train")}
    strong_fresh=[s for s in strong_candidates if _key(s.worker_roles) not in v11_train_keys]
    for mode in MODES:
        for s in [x for x in strong_fresh if x.execution_mode==mode][:6]: forbidden.add(_key(s.worker_roles))
    mp=Path(strong_manifest)
    if mp.exists():
        obj=json.loads(mp.read_text(encoding="utf-8"))
        for x in obj.get("known_specs",[]): forbidden.add(_key(x.get("roles",())))
    candidates=_make_train_specs(per_mode=48,seed=seed)
    candidates=[s for s in candidates if _key(s.worker_roles) not in forbidden]
    rng=random.Random(seed)

    if n_train==24 and n_dev==8:
        # Select two systems from every (mode,K) cell while balancing role counts.
        counts=Counter({r:0 for r in TRAIN_ROLES}); chosen=[]; used=set()
        for mode in MODES:
            for kval in KS:
                pool=[s for s in candidates if s.execution_mode==mode and len(s.worker_roles)==kval and _key(s.worker_roles) not in used]
                for _ in range(2):
                    if not pool: raise RuntimeError(f"not enough fresh specs for mode={mode} K={kval}")
                    def cost(s):
                        return sum((counts[r]+1)**2-counts[r]**2 for r in s.worker_roles)+rng.random()*1e-4
                    best=min(pool,key=cost); pool.remove(best); used.add(_key(best.worker_roles)); chosen.append(best); counts.update(best.worker_roles)
        # Two dev systems per mode and exactly two dev systems per K.
        dev_k_by_mode={
            MODES[0]:(3,4),
            MODES[1]:(5,6),
            MODES[2]:(3,5),
            MODES[3]:(4,6),
        }
        dev=[]; train=[]; dev_role_counts=Counter({r:0 for r in TRAIN_ROLES})
        for mode in MODES:
            designated=set(dev_k_by_mode[mode])
            for kval in KS:
                cell=[s for s in chosen if s.execution_mode==mode and len(s.worker_roles)==kval]
                if kval in designated:
                    # Pick the dev member that keeps dev role coverage flatter.
                    d=min(cell,key=lambda s:sum((dev_role_counts[r]+1)**2-dev_role_counts[r]**2 for r in s.worker_roles)); dev.append(d);dev_role_counts.update(d.worker_roles)
                    train.extend([s for s in cell if s is not d])
                else:
                    train.extend(cell)
        assert len(train)==24 and len(dev)==8
        selected=[("train",s) for s in train]+[("dev",s) for s in dev]
    else:
        total=n_train+n_dev; counts=Counter({r:0 for r in TRAIN_ROLES});modec=Counter();kc=Counter();picked=[];used=set()
        while len(picked)<total:
            rem=[s for s in candidates if _key(s.worker_roles) not in used]
            if not rem: raise RuntimeError("not enough fresh adaptive specs")
            def cost(s):
                roles=list(s.worker_roles)
                return sum((counts[r]+1)**2-counts[r]**2 for r in roles)+2.0*modec[s.execution_mode]+2.0*kc[len(roles)]+rng.random()*1e-3
            s=min(rem,key=cost);used.add(_key(s.worker_roles));picked.append(s);counts.update(s.worker_roles);modec[s.execution_mode]+=1;kc[len(s.worker_roles)]+=1
        selected=[("train" if i<n_train else "dev",s) for i,s in enumerate(picked)]

    out=[]
    for i,(split,s) in enumerate(selected):
        out.append(SingleRouterSystemSpec(system_id=f"adaptv1_{split}_{i:03d}_{s.execution_mode}_k{len(s.worker_roles)}",split=f"adaptive_v1_{split}",execution_mode=s.execution_mode,style=s.style,worker_roles=tuple(s.worker_roles)))
    return out


def write_manifest(path,specs,seed):
    rows=[{"system_id":s.system_id,"split":"train" if "_train_" in s.system_id else "dev","mode":s.execution_mode,"style":s.style,"k":len(s.worker_roles),"roles":list(s.worker_roles)} for s in specs]
    obj={"version":"adaptive_controller_bank_v1","seed":seed,"systems":rows,"notes":{"truth_roles":"privileged labels only; never controller input","final_test":"NOT final paper test; reserve a new locked seed/composition set"}}
    canon=json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode();obj["sha256"]=hashlib.sha256(canon).hexdigest();p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8");return obj
