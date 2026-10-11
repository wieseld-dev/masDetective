#!/usr/bin/env python3
"""V5.11.3: one intact V137 DEV evaluation; validation is held out from TRAIN.

The old DEV-as-validation-and-test design cannot split a giant exact-observation
connected component. Instead keep all DEV compositions together as EVALUATION,
hold whole separate training-system compositions out for VALIDATION, and
quarantine exact visible observations from the remaining training rows.
No target labels/traces/system IDs enter role-verifier feature inputs.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, random, runpy
from pathlib import Path
from analysis.role_direct_within_v4 import v5111_rolewise_verifier as baseline
from analysis.role_direct_within_v4 import v510_generalizing_sensor as v510

ROOT=baseline.ROOT
BASE=ROOT/'artifacts/role_direct_within_v4/sensor_v5113'
SEED=5113001


def sha(s):
    return hashlib.sha256(str(s).encode()).hexdigest()


def unique_keys(systems):
    return {baseline.record_id(r) for observations,_ in systems.values() for r in observations.values()}


def train_counts(rows,roles):
    return baseline.group_counts(rows,roles)


def data_for_validation(pools,roles):
    # Source systems solely from V137 TRAIN. V56 must remain in train.
    pairs=pools.get('hist_within_system',[])
    seen={}
    for pos,neg in pairs:
        for row in (pos,neg):
            sid=row['system_id']
            seen[(sid,row['role'],baseline.record_id(row))]=row
    systems,invalid=v510.complete_systems(list(seen.values()),roles)
    return systems,invalid


def choose_train_validation(roles,rows,train_systems,test,all_systems=None):
    """Select heldout complete systems by role composition, with zero train overlap.

    Selection uses metadata and labels only, not model predictions/test scores.
    Keep heldout systems intact for Within-MAS AUC and exact top3.
    """
    all_systems=all_systems or train_systems
    if len(train_systems)<8:
        raise RuntimeError(f'Only {len(train_systems)} complete V137 TRAIN systems remain; need >=8 for internal validation')
    test_compositions={tuple(sorted(x[1])) for x in test.values()}
    groups=collections.defaultdict(list)
    for sid,(obs,positives) in train_systems.items():
        comp=tuple(sorted(positives))
        if comp in test_compositions:
            continue
        if sid in test:continue
        groups[comp].append(sid)
    if len(groups)<3:
        raise RuntimeError(f'Only {len(groups)} candidate V137 TRAIN compositions for validation; need >=3')
    comps=sorted(groups)
    # Fixed single split selected exclusively by non-model data coverage.
    rng=random.Random(SEED)
    candidates=[]
    for i in range(450):
        order=comps[:];rng.shuffle(order)
        take=set();target=max(5,min(10,round(len(train_systems)*.18)))
        for comp in order:
            take.update(groups[comp])
            if len(take)>=target:break
        val={sid:train_systems[sid] for sid in sorted(take)}
        val_keys=unique_keys(val)
        # Nothing from validation systems, compositions, or identical visible inputs
        # is permitted in model training, including V56 historical overlaps.
        val_comps={tuple(sorted(x[1])) for x in val.values()}
        remaining=[r for r in rows if r['system_id'] not in take
                   and baseline.record_id(r) not in val_keys
                   and not _same_known_composition(r,val_comps,all_systems)]
        ct=train_counts(remaining,roles)
        thin={r:c for r,c in ct.items() if min(c['positive'],c['negative'])<3}
        if thin:continue
        # More roles represented with both positive and negative systems in
        # validation is desirable for diagnostic role-wise binary AUC.
        cov=0
        for role in roles:
            ps=sum(role in p for _,p in val.values())
            cov+=int(0<ps<len(val))
        score=(cov,-abs(len(val)-target),len(remaining))
        candidates.append((score,remaining,val))
    if not candidates:
        original_ct=train_counts(rows,roles)
        raise RuntimeError(f'No leak-free train/validation composition split leaves >=3 examples of each label for all roles. '
                           f'Candidate systems={len(train_systems)}, compositions={len(comps)}, '
                           f'raw training by role={original_ct}. Cannot train safely.')
    _,tr,val=max(candidates,key=lambda x:x[0])
    return tr,val,{'candidate_train_systems':len(train_systems),
                   'candidate_train_compositions':len(comps),
                   'validation_systems':len(val),
                   'train_rows_before_quarantine':len(rows),
                   'train_rows_after_quarantine':len(tr),
                   'quarantined_train_rows':len(rows)-len(tr),
                   'validation_role_binary_coverage':sum(0<sum(r in pos for _,pos in val.values())<len(val) for r in roles)}


def _same_known_composition(r,val_comps,train_systems):
    s=train_systems.get(r['system_id'])
    return bool(s and tuple(sorted(s[1])) in val_comps)


def identical_observation_ambiguity(systems):
    bykey=collections.defaultdict(lambda:{'labels':set(),'n':0,'role':''})
    for observations,_ in systems.values():
        for r in observations.values():
            entry=bykey[baseline.record_id(r)]
            entry['labels'].add(int(r['label']))
            entry['n']+=1
            entry['role']=r['role']
    conflicted={k:x for k,x in bykey.items() if len(x['labels'])>1}
    return {'systems':len(systems),'unique_visible_observations':len(bykey),
            'unique_ambiguous_observations':len(conflicted),
            'occurrences_of_ambiguous_observations':sum(x['n'] for x in conflicted.values()),
            'by_role':dict(sorted(collections.Counter(x['role'] for x in conflicted.values()).items())),
            'example_fingerprints_sha256':sorted(conflicted)[:6],
            'meaning':'A single role/query/answer with opposing labels has no deterministic single-observation presence indicator.'}


def prep(override=None):
    if override is None:
        roles,descs,oldval,oldtest,pools,upstream=v510.obtain_sources()
    else:
        roles,descs,oldval,oldtest,pools,upstream=override
    dev={**oldval,**oldtest}
    if len(dev)!=len(oldval)+len(oldtest):raise RuntimeError('Duplicate DEV system IDs')
    train,conflicts=baseline.source_rows(pools,return_audit=True)
    original_count=len(train)
    dev_keys=unique_keys(dev)
    # Quarantine overlap at the highest priority evaluation boundary.
    train=[r for r in train if r['system_id'] not in dev and baseline.record_id(r) not in dev_keys]
    systems,invalid=data_for_validation(pools,roles)
    # Read known training compositions from *all* sources, not just V137.
    # This prevents V56 systems with matching compositions remaining in TRAIN
    # while the same composition is used to select on internal validation.
    all_raw={}
    for pairs in pools.values():
        for pos,neg in pairs:
            for row in (pos,neg):
                all_raw[(row['system_id'],row['role'],baseline.record_id(row))]=row
    all_systems,_=v510.complete_systems(list(all_raw.values()),roles)
    # Candidate train systems may contain observations with mixed labels across
    # configurations. Such instances are not eligible for validation.
    # No DEV query/response pair can participate in validation either.
    valid={}
    for sid,(obs,positives) in systems.items():
        if sid in dev or unique_keys({sid:(obs,positives)}) & dev_keys:
            continue
        if len(set(baseline.record_id(r) for r in obs.values()))!=len(roles):
            continue
        valid[sid]=(obs,positives)
    tr,val,split=choose_train_validation(roles,train,valid,dev,all_systems)
    trk={baseline.record_id(r) for r in tr}
    vk=unique_keys(val)
    dk=dev_keys
    if trk & vk or trk & dk or vk & dk:
        raise RuntimeError('Exact visible observation crosses TRAIN/VAL/DEV boundary')
    trids={r['system_id'] for r in tr}
    if trids & (set(val)|set(dev)) or (set(val)&set(dev)):
        raise RuntimeError('Same system ID crosses split boundary')
    def comps(sys):return {tuple(sorted(p)) for _,p in sys.values()}
    if comps(val)&comps(dev):raise RuntimeError('VALIDATION/DEV composition overlap')
    # Some train systems may have been omitted from 'valid' because incomplete;
    # their role compositions are unknown, so avoid claiming all train comps disjoint.
    priority={baseline.record_id(r) for pos,neg in pools.get('matched_same_query',[]) for r in (pos,neg)}
    report={'status':'SPLIT_VERIFIED','split_protocol':'V137_TRAIN_group_holdout_validation__WHOLE_V137_DEV_evaluation',
            'roles':len(roles),'train_rows':len(tr),'validation_systems':len(val),'dev_evaluation_systems':len(dev),
            'unambiguous_train_rows_before_dev_quarantine':original_count,
            'train_rows_after_dev_quarantine_before_val':len(train),
            'dev_systems_kept_intact':True,
            'dev_not_used_for_selection':True,
            'previously_inspected_dev_not_pristine':True,
            'train_val_dev_exact_overlap':{'train_val':len(trk&vk),'train_dev':len(trk&dk),'val_dev':len(vk&dk)},
            'validation_dev_composition_overlap':len(comps(val)&comps(dev)),
            'train_conflicts':conflicts,
            'visible_label_conflicts':conflicts,
            'evaluation_ambiguity':identical_observation_ambiguity(dev),
            'validation_ambiguity':identical_observation_ambiguity(val),
            'train_candidates_missing_complete_roles':len(invalid),
            'val_split':split,
            'train_by_role':train_counts(tr,roles),
            'test_macro_per_role_possible':sum(0<sum(role in pos for _,pos in dev.values())<len(dev) for role in roles),
            'upstream':upstream,
            'caveat':'V137 DEV has been examined in previous experiments; this is exploratory retrospective evaluation. Exact overlap is removed; query template overlap may remain and is assessed by query-only control. No claims of unseen query distribution.'}
    audit_file=os.environ.get('V5113_AUDIT_FILE')
    if audit_file:baseline.dump(audit_file,report)
    return roles,tr,val,dev,report,priority


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['selftest','preflight','run'])
    ap.add_argument('--out',type=Path)
    x=ap.parse_args()
    if x.stage=='selftest':
        runpy.run_path(str(Path(__file__).resolve().parents[2] / "tests/test_v5113_cpu.py"), run_name="__main__");return
    if x.stage=='preflight':
        roles,train,val,test,report,priority=prep()
        print('V5113_PREFLIGHT',json.dumps({'status':'READY','train':len(train),'val_systems':len(val),
            'whole_v137_dev_systems':len(test),'cross_split_exact_overlap':report['train_val_dev_exact_overlap'],
            'validation_role_coverage':report['val_split']['validation_role_binary_coverage'],
            'no_api_calls':True}),flush=True)
        return
    if not x.out:raise RuntimeError('--out required')
    # Reuse known V5111 role classifiers and controls, replace only source split.
    baseline.BASE=BASE
    baseline.prepare_sources=prep
    baseline.run(x.out)

if __name__=='__main__':main()
