#!/usr/bin/env python3
"""V5.11.2: quarantine cross-split exact duplicates and make leakage-blocked DEV split.

Reuses V5111 classical role-wise verifiers. NEVER changes labels and NEVER permits
identical role/query/answer in multiple Train/Validation/Test partitions.
When DEV cannot be split into independent sets without discarding observations,
fails CLOSED, prints an actionable audit, and submits no SLURM training job.
"""
from __future__ import annotations
import argparse, collections, json, os, random
from pathlib import Path
from analysis.role_direct_within_v4 import v5111_rolewise_verifier as prev

ROOT=prev.ROOT
BASE=ROOT/'artifacts/role_direct_within_v4/sensor_v5112'
SEED=5112026
MIN_SPLIT_SYSTEMS=3

def row_keys(systems):
    return {prev.record_id(r) for selected,_ in systems.values() for r in selected.values()}

def overlap_stats(train,val,test):
    tk={prev.record_id(x) for x in train};vk=row_keys(val);ek=row_keys(test)
    return {'train_validation_unique':len(tk&vk),
            'train_test_unique':len(tk&ek),
            'validation_test_unique':len(vk&ek),
            'train_unique':len(tk),'validation_unique':len(vk),'test_unique':len(ek)}

def composition_key(s):
    return tuple(sorted(s[1]))

def cross_eval_overlap_detail(val,test):
    a={prev.record_id(r):r for selected,_ in val.values() for r in selected.values()}
    b={prev.record_id(r):r for selected,_ in test.values() for r in selected.values()}
    shared=set(a)&set(b)
    return {'unique_shared':len(shared),
            'shared_by_target_role':dict(sorted(collections.Counter(a[k]['role'] for k in shared).items())),
            'same_labels':sum(int(a[k]['label'])==int(b[k]['label']) for k in shared),
            'contradictory_labels':sum(int(a[k]['label'])!=int(b[k]['label']) for k in shared),
            'sample_key_sha256':sorted(shared)[:8]}

def component_split(systems, before_val, before_test, roles):
    """Group *whole MAS systems* by role composition and repeated exact observations.

    A connected component cannot straddle validation and test, even if the
    repeated role/query/answer has different labels. Never delete evaluation
    rows or silently alter the denominator of Within-MAS AUC.
    """
    allids=sorted(systems)
    parent={sid:sid for sid in allids}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a=find(a);b=find(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    first_comp={};first_obs={}
    duplicate_cross_edges=set();comp_edges=set()
    for sid in allids:
        cfg=composition_key(systems[sid])
        if cfg in first_comp:
            comp_edges.add(tuple(sorted([sid,first_comp[cfg]])))
            union(sid,first_comp[cfg])
        else:first_comp[cfg]=sid
        for row in systems[sid][0].values():
            key=prev.record_id(row)
            if key in first_obs:
                duplicate_cross_edges.add(tuple(sorted([sid,first_obs[key]])))
                union(sid,first_obs[key])
            else:first_obs[key]=sid
    connected=collections.defaultdict(list)
    for sid in allids:connected[find(sid)].append(sid)
    components=[sorted(xs) for xs in connected.values()]
    components.sort(key=lambda x:(-len(x),x))
    sizes=[len(x) for x in components]
    audit={'systems_before':len(allids),'composition_groups':len(first_comp),
           'connected_components':len(components),'component_sizes':sizes,
           'cross_system_exact_observation_edges':len(duplicate_cross_edges),
           'same_composition_edges':len(comp_edges),
           'initial_val_systems':len(before_val),'initial_test_systems':len(before_test),
           'component_system_ids_sha256':__import__('hashlib').sha256(json.dumps(components,sort_keys=True).encode()).hexdigest()}
    if len(components)<2:
        raise ValueError('All V137 DEV systems are connected by shared observations/compositions; independent validation/test impossible with this corpus')
    # Purely metadata-driven split search: no response model scores or TEST AUC.
    rng=random.Random(SEED)
    candidates=[]
    for trial in range(2500):
        order=list(range(len(components)));rng.shuffle(order)
        v_ids=set();t_ids=set()
        for idx in order:
            group=set(components[idx])
            # Slightly favor balanced populations of MAS systems.
            if len(v_ids)<=len(t_ids):v_ids.update(group)
            else:t_ids.update(group)
        if min(len(v_ids),len(t_ids))<MIN_SPLIT_SYSTEMS:continue
        val={sid:systems[sid] for sid in sorted(v_ids)}
        test={sid:systems[sid] for sid in sorted(t_ids)}
        train_truth=lambda p: set().union(*(systems[x][1] for x in p)) if p else set()
        # Prefer labels of all 20 roles represented in both halves, when possible.
        def coverage(ids):
            c=collections.Counter(r for sid in ids for r in systems[sid][1]);
            return sum(1 for role in roles if 0<c[role]<len(ids))
        cval=coverage(v_ids);ctest=coverage(t_ids)
        # Among otherwise equivalent splits prefer minimal disturbance of original DEV split.
        moved=len(v_ids.symmetric_difference(before_val))
        score=(min(cval,ctest),cval+ctest,-abs(len(v_ids)-len(t_ids)),-moved)
        candidates.append((score,val,test))
    if not candidates:
        raise ValueError('Cannot form two DEV subsets with >=3 MAS systems each while preventing cross-split exact observation duplicates')
    candidates.sort(key=lambda x:x[0],reverse=True)
    _,val,test=candidates[0]
    if row_keys(val)&row_keys(test):raise AssertionError('DEV component split still overlaps')
    if {composition_key(s) for s in val.values()} & {composition_key(s) for s in test.values()}:
        raise AssertionError('DEV component split leaks same composition')
    audit.update({'val_systems_after':len(val),'test_systems_after':len(test),
                  'systems_removed_from_evaluation':0,
                  'validation_systems_reassigned':len(set(val).symmetric_difference(before_val)),
                  'role_coverage_val':sum(0<sum(r in pos for _,pos in val.values())<len(val) for r in roles),
                  'role_coverage_test':sum(0<sum(r in pos for _,pos in test.values())<len(test) for r in roles),
                  'warning':'DEV grouping chosen using exact-input overlap and role composition metadata only, not model performance or test AUC; old V137 DEV was inspected previously.'})
    return val,test,audit

def prepare_sources():
    roles,desc,oldval,oldtest,pools,stats=prev.v510.obtain_sources()
    train,conf=prev.source_rows(pools,return_audit=True)
    orig=overlap_stats(train,oldval,oldtest)
    dev={**oldval,**oldtest}
    if len(dev)!=len(oldval)+len(oldtest):raise RuntimeError('Duplicate DEV system_id')
    # Model never gets data-derived labels or system IDs as feature inputs.
    report={'status':'AUDITING','initial_overlap':orig,'training_conflicts':conf,
            'systems':{'train_pre':len({r['system_id'] for r in train}),
                       'val_pre':len(oldval),'test_pre':len(oldtest)},
            'val_test_overlap_detail_before':cross_eval_overlap_detail(oldval,oldtest)}
    try:
        if orig['validation_test_unique']:
            val,test,dev_audit=component_split(dev,set(oldval),set(oldtest),roles)
            report['split_method']='exact_observation_and_composition_connected_components'
            report['dev_split_audit']=dev_audit
        else:
            val,test=oldval,oldtest
            report['split_method']='original_V510_clean_dev_split'
            report['dev_split_audit']={'systems_removed_from_evaluation':0}
        # Remove TRAIN samples duplicated by either heldout split; NEVER remove
        # test or validation data to make training fit. Log every removal.
        heldout=row_keys(val)|row_keys(test)
        overlap_train=[r for r in train if prev.record_id(r) in heldout]
        train=[r for r in train if prev.record_id(r) not in heldout]
        report['exact_train_rows_quarantined']=len(overlap_train)
        report['training_quarantine_by_role']=dict(collections.Counter(r['role'] for r in overlap_train))
        trk={prev.record_id(r) for r in train};vk=row_keys(val);tk=row_keys(test)
        if (trk&vk) or (trk&tk) or (vk&tk):raise RuntimeError('Unexpected remaining cross-split observation overlap')
        trids={r['system_id'] for r in train}
        if trids&(set(val)|set(test)):raise RuntimeError('Same system_id in TRAIN and DEV after quarantine')
        comps_val={composition_key(x) for x in val.values()}
        comps_test={composition_key(x) for x in test.values()}
        if comps_val & comps_test:raise RuntimeError('Same role composition in DEV validation and test')
        if min(len(val),len(test))<MIN_SPLIT_SYSTEMS:raise RuntimeError('Insufficient DEV MAS systems after leakage-safe split')
        role_count=prev.group_counts(train,roles)
        thin={role:c for role,c in role_count.items() if min(c['positive'],c['negative'])<3}
        if thin:raise RuntimeError(f'Train missing at least 3 positive/negative examples per role: {thin}')
        # Old priority keys remain safe; stripped training rows simply do not
        # get used. Training labels only enter the optimizer.
        priority={prev.record_id(r) for pos,neg in pools.get('matched_same_query',[]) for r in (pos,neg)}
        report['final_overlap']=overlap_stats(train,val,test)
        report['train_rows_after']=len(train)
        report['val_systems_after']=len(val)
        report['test_systems_after']=len(test)
        report['status']='SPLIT_VERIFIED'
        final_stat={'train_rows':len(train),'val_systems':len(val),'test_systems':len(test),
                    'source_rows':dict(collections.Counter(r.get('source','unknown') for r in train)),
                    'matched_priority_unique_rows':len(priority),
                    'training_by_role':role_count,
                    'val_by_role':prev.group_counts(prev.config_rows(val,roles),roles),
                    'test_by_role':prev.group_counts(prev.config_rows(test,roles),roles),
                    'note':'Heldout V137 DEV aggregate was inspected before; exploratory generalization, not pristine prospective test',
                    'v510_split_stats':stats,'visible_label_conflicts':conf,'leakage_audit':report}
        save_audit(report)
        return roles,train,val,test,final_stat,priority
    except Exception as exc:
        report['status']='DATA_SPLIT_NOT_IDENTIFIABLE'
        report['reason']=str(exc)
        save_audit(report)
        raise RuntimeError(f'V5112 stopped BEFORE TRAINING: {exc}; see leakage audit JSON') from exc

def save_audit(d):
    dest=os.environ.get('V5112_AUDIT_FILE','')
    if dest:prev.dump(dest,d)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['selftest','preflight','run']);ap.add_argument('--out',type=Path)
    args=ap.parse_args()
    if args.stage=='selftest':
        selftest();return
    # Monkeypatch only this process's V5111 run entrypoints; no files overwritten.
    prev.BASE=BASE
    prev.prepare_sources=prepare_sources
    if args.stage=='preflight':
        roles,train,val,test,stat,priority=prepare_sources()
        print('V5112_PREFLIGHT',json.dumps({'roles':len(roles),'train':len(train),'val_systems':len(val),
             'test_systems':len(test),'split_method':stat['leakage_audit']['split_method'],
             'duplicates_quarantined':stat['leakage_audit']['exact_train_rows_quarantined'],
             'no_MAS_API_calls':True}),flush=True)
        return
    if args.out is None:raise RuntimeError('--out is required')
    prev.run(args.out)

def selftest():
    roles=[f'r{i}' for i in range(20)]
    def fixture(i):
        return ({r:{'system_id':f's{i:02d}','role':r,'query':f'q{i} {r}',
                'answer':f'a{i} {r}','label':int(r in {roles[(3*i+j)%20] for j in range(3)})} for r in roles},
                {roles[(3*i+j)%20] for j in range(3)})
    allsys={f's{i:02d}':fixture(i) for i in range(18)}
    # Shared exact observation forces different compositions into same component.
    a,b='s00','s09';role=roles[0];allsys[b][0][role]['query']=allsys[a][0][role]['query'];allsys[b][0][role]['answer']=allsys[a][0][role]['answer']
    val,test,audit=component_split(allsys,{'s00','s01'},set(allsys)-{'s00','s01'},roles)
    assert not row_keys(val)&row_keys(test)
    assert not {composition_key(x) for x in val.values()} & {composition_key(x) for x in test.values()}
    assert len(val)>=3 and len(test)>=3
    train=[{'system_id':'train','role':'r1','query':'shared q','answer':'shared a','label':1}]
    overlaps=overlap_stats(train,{'test':({'r1':train[0]},{'r1'})},{})
    assert overlaps['train_validation_unique']==1
    tied={f's{i:02d}':fixture(i) for i in range(6)}
    for i in range(1,6):
        tied[f's{i:02d}'][0][role]['query']=tied['s00'][0][role]['query'];tied[f's{i:02d}'][0][role]['answer']=tied['s00'][0][role]['answer']
    try:component_split(tied,{}, {},roles)
    except ValueError:pass
    else:raise AssertionError('Should fail closed when all DEV systems share exact observations')
    print('V5112_SELFTEST_OK: groupwise split, no overlap, cannot split test')

if __name__=='__main__':main()
