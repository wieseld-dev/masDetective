from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np

from v11.role_universe import TRAIN_ROLES, DEV_UNSEEN_ROLES, ROLE_DESCRIPTIONS, anchor_probe
from targets.single_router_zoo_v11 import get_v11_split
from v8.counterfactual import get_k, get_mode, get_roles, split_base_specs
from v11_2.counterfactual import build_hard_counterfactual_pairs


def _norm_text(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def _word_jaccard(a: str, b: str) -> float:
    wa=set(_norm_text(a).split()); wb=set(_norm_text(b).split())
    if not wa or not wb: return 0.0
    return len(wa & wb) / max(1, len(wa | wb))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    aa=np.asarray(a,dtype=np.float32).reshape(-1); bb=np.asarray(b,dtype=np.float32).reshape(-1)
    aa=aa/max(float(np.linalg.norm(aa)),1e-8); bb=bb/max(float(np.linalg.norm(bb)),1e-8)
    return float(np.clip(float(np.dot(aa,bb)),-1.0,1.0))


def _stable_jitter(seed: int, *parts: object) -> float:
    txt="|".join(map(str,(seed,)+parts)); h=hashlib.sha256(txt.encode("utf-8")).hexdigest()[:12]
    return int(h,16)/float(16**12)


def _pair_signature(p: object):
    return (get_mode(p.plus_spec), get_k(p.plus_spec), str(getattr(p,"negative_replacement","")), tuple(sorted(get_roles(p.plus_spec))))


def _build_stratified_schedule(by_role, roles: Sequence[str], *, rounds: int, bundle_size: int, confirmation_size: int, seed: int):
    primary_usage={}; overall_usage={}; primary={}; confirm={}
    def choose(pairs,n,round_i,role,blocked,usage_map):
        chosen=[]; used_modes=set(); used_k=set(); used_neg=set(); used_comp=set()
        available_n=len([p for p in pairs if str(getattr(p,"pair_id")) not in blocked])
        while len(chosen)<min(max(0,n),available_n):
            remaining=[p for p in pairs if p not in chosen and str(getattr(p,"pair_id")) not in blocked]
            if not remaining: break
            def score(p):
                pid=str(getattr(p,"pair_id")); mode,kval,neg,comp=_pair_signature(p)
                return (1 if usage_map.get(pid,0)==0 else 0,1 if mode not in used_modes else 0,1 if kval not in used_k else 0,1 if neg not in used_neg else 0,1 if comp not in used_comp else 0,-usage_map.get(pid,0),-overall_usage.get(pid,0),_stable_jitter(seed,round_i,role,pid))
            best=max(remaining,key=score); chosen.append(best)
            mode,kval,neg,comp=_pair_signature(best); used_modes.add(mode); used_k.add(kval); used_neg.add(neg); used_comp.add(comp)
        return chosen
    for round_i in range(1,rounds+1):
        for role in roles:
            pairs=list(by_role.get(role,[]))
            if not pairs:
                primary[(round_i,role)]=[]; confirm[(round_i,role)]=[]; continue
            psel=choose(pairs,int(bundle_size),round_i,role,set(),primary_usage)
            blocked={str(getattr(p,"pair_id")) for p in psel}
            csel=choose(pairs,int(confirmation_size),round_i+10000,role,blocked,overall_usage)
            for p in psel:
                pid=str(getattr(p,"pair_id")); primary_usage[pid]=primary_usage.get(pid,0)+1; overall_usage[pid]=overall_usage.get(pid,0)+1
            for p in csel:
                pid=str(getattr(p,"pair_id")); overall_usage[pid]=overall_usage.get(pid,0)+1
            primary[(round_i,role)]=psel; confirm[(round_i,role)]=csel
    return primary,confirm


def _load_seed_elites(path: str, roles: Sequence[str], max_per_role: int=3):
    out={r:[] for r in roles}; p=Path(path)
    if not p.exists(): return out
    try:
        obj=json.loads(p.read_text(encoding="utf-8")); raw=obj.get("elite_probes",{})
        for role in roles:
            rows=list(raw.get(role,[])); rows.sort(key=lambda x:float(x.get("reward",-1e9)),reverse=True); seen=set()
            for row in rows:
                q=str(row.get("query","")).strip(); k=_norm_text(q)
                if not q or k in seen: continue
                seen.add(k); out[role].append(q)
                if len(out[role])>=max_per_role: break
    except Exception as exc:
        print("WARNING: could not load initial elite summary:",type(exc).__name__,flush=True)
    return out


def _merge_unique(a: Sequence[str], b: Sequence[str], limit: int=8):
    out=[]; seen=set()
    for x in list(a)+list(b):
        q=str(x).strip(); k=_norm_text(q)
        if q and k not in seen: seen.add(k); out.append(q)
    return out[-limit:]


def _copy_cache_seed(src: str, dst: str):
    s,d=Path(src),Path(dst)
    if s.exists() and not d.exists():
        d.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(s,d); print(f"seeded cache {d} from {s}",flush=True)


def build_context(*, profile_path: str, rounds: int, bundle_size: int, confirmation_size: int, seed: int):
    profile=json.loads(Path(profile_path).read_text(encoding="utf-8"))
    train_roles=tuple(profile.get("train_candidate_roles",TRAIN_ROLES)); dev_roles=tuple(profile.get("dev_unseen_candidate_roles",DEV_UNSEEN_ROLES))
    if tuple(train_roles)!=tuple(TRAIN_ROLES): raise SystemExit("V11 profile training roles differ from reviewed TRAIN_ROLES")
    if tuple(dev_roles)!=tuple(DEV_UNSEEN_ROLES): raise SystemExit("V11 profile dev roles differ from reviewed DEV_UNSEEN_ROLES")
    if set(train_roles)&set(dev_roles): raise SystemExit("V11 role leakage: train/dev overlap")
    all_specs=list(get_v11_split("train")); rl_specs,detector_reserved_specs=split_base_specs(all_specs,detector_fraction=1/8,seed=1109)
    train_pairs=build_hard_counterfactual_pairs(rl_specs,train_roles,include_sham=True)
    by_role={r:[] for r in train_roles}
    for p in train_pairs: by_role.setdefault(p.target_role,[]).append(p)
    usable_roles=[r for r in train_roles if by_role.get(r)]
    if set(dev_roles)&{r for spec in rl_specs for r in get_roles(spec)}: raise SystemExit("V11 leakage: unseen dev role appears in RL training specs")
    if set(dev_roles)&{str(getattr(p,"negative_replacement","")) for p in train_pairs}: raise SystemExit("V11 leakage: unseen dev role used as training replacement")
    schedule,confirmation_schedule=_build_stratified_schedule(by_role,usable_roles,rounds=rounds,bundle_size=bundle_size,confirmation_size=confirmation_size,seed=seed)
    return {"profile":profile,"train_roles":train_roles,"dev_roles":dev_roles,"usable_roles":usable_roles,"rl_specs":rl_specs,"detector_reserved_specs":detector_reserved_specs,"train_pairs":train_pairs,"by_role":by_role,"schedule":schedule,"confirmation_schedule":confirmation_schedule}


def role_order(usable_roles: Sequence[str], round_i: int, seed: int):
    xs=list(usable_roles); random.Random(seed+7919*round_i).shuffle(xs); return xs


def load_previous_elites(initial_summary: str, previous_eval_jsonl: str|None, roles: Sequence[str]):
    seed_elites=_load_seed_elites(initial_summary,roles,max_per_role=3); prior={r:[] for r in roles}
    if previous_eval_jsonl and Path(previous_eval_jsonl).exists():
        with Path(previous_eval_jsonl).open(encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                row=json.loads(line); role=str(row.get("target_role",""))
                if role in prior and row.get("elite"):
                    q=str(row.get("query","")).strip()
                    if q and _norm_text(q) not in {_norm_text(x) for x in prior[role]}: prior[role].append(q)
    return {r:_merge_unique(seed_elites.get(r,[]),prior.get(r,[]),limit=8) for r in roles}


def pair_meta(p):
    return {"pair_id":str(p.pair_id),"mode":get_mode(p.plus_spec),"k":get_k(p.plus_spec),"negative_replacement":str(getattr(p,"negative_replacement",""))}
