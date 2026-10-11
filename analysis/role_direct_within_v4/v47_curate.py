#!/usr/bin/env python3
"""MAS5 V4.7: audit -> manually curate -> regenerate -> review -> finalize.

DOES NOT execute MAS, does not read hidden target routing, does not train models.
Never silently overwrite a probe bank. Conservatively triages weak proposals.
"""
from __future__ import annotations
import argparse, collections, csv, hashlib, json, os, re, shutil, sys
from pathlib import Path
from analysis.role_direct_within_v4 import v47_quality as qc
from analysis.role_direct_within_v4 import v46_generator as oldgen
from analysis.role_direct_within_v4 import pipeline as old

SOURCE_FILE='v46_hmhy_candidate_probes_round_01.jsonl'
REPAIR_DIR='v47_quality'


def rows(path):
    p=Path(path)
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def write_jsonl(path,items):
    path=Path(path)
    if path.exists():raise RuntimeError(f'Refusing to overwrite {path}')
    with path.open('x') as f:
        for r in items:f.write(json.dumps(r,ensure_ascii=False)+'\n')


def write_json(path,obj):
    p=Path(path)
    if p.exists():raise RuntimeError('Refusing to overwrite '+str(p))
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def root_paths(root, allow_prepared=False):
    root=Path(root).expanduser().resolve(strict=True)
    project=Path('/home/wieseld/mas_rl_detective')
    if not root.is_relative_to(project/'artifacts/role_direct_within_v4/overnight_v46'):
        raise RuntimeError('This script only accepts a MAS5 V46 existing run root, not arbitrary paths')
    source=root/'train/probes'/SOURCE_FILE
    if not source.is_file():raise RuntimeError('Missing probe bank '+str(source))
    if not allow_prepared and ((root/'train/candidates_train_round_01.jsonl').exists() or (root/'train/responses_train_round_01.jsonl').exists()):
        raise RuntimeError('Training data already prepared/executed; not safe to mutate query bank')
    checkpoint=root/'train/generator_bootstrap_adapter/adapter_config.json'
    if not checkpoint.is_file():raise RuntimeError('Existing bootstrap Generator adapter not present')
    return root,source,root/'train'/REPAIR_DIR


def validate_source(raw):
    catalog=old.catalog();roles=set(catalog['ROLES'])
    seen=set()
    if len(raw)>160:raise RuntimeError('Overfull source bank')
    for r in raw:
        role=r.get('target_role'); ix=int(r.get('strategy_index',-1));key=(role,ix)
        if role not in roles or ix not in range(8):raise RuntimeError('Unknown role/slot: '+str(key))
        if key in seen:raise RuntimeError('Duplicate role/slot: '+str(key))
        seen.add(key)
        if r.get('probe_family')!=('help_me_help_you' if ix<6 else 'direct_task'):
            raise RuntimeError('Incorrect probe family: '+str(key))
    return catalog


def quality_for_row(r):
    role=r['target_role'];idx=int(r['strategy_index']);q=r['query']
    state,reasons=qc.assess(q,role,idx)
    valid,why=old.quality(q)
    if not valid:return 'REJECT',['old_quality_'+str(why)]+reasons
    if idx<6:
        valid,why=oldgen.hmhy_quality(q)
        if not valid:return 'REJECT',['old_hmhy_'+str(why)]+reasons
    return state,reasons


def write_tsv(p,rows_):
    fields=['role','slot','family','decision','reasons','query']
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter='\t',lineterminator='\n')
        w.writeheader();w.writerows(rows_)


def audit(root):
    root,source,folder=root_paths(root, allow_prepared=True)
    if folder.exists():raise RuntimeError('Audit folder already exists; preserve manual decisions. '+str(folder))
    raw=rows(source);catalog=validate_source(raw)
    folder.mkdir(parents=True)
    records=[]
    for r in raw:
        status,reasons=quality_for_row(r)
        records.append({'role':r['target_role'],'slot':int(r['strategy_index']),
                        'family':r['probe_family'],'decision':status,
                        'reasons':','.join(reasons),'query':' '.join(r['query'].split())})
    # File only; user can edit the decision column for borderline rows.
    write_tsv(folder/'review.tsv',records)
    write_json(folder/'source_manifest.json',{'source':str(source),'sha256':sha(source),
        'source_count':len(raw),'roles':len(catalog['ROLES']),
        'decision_counts':dict(collections.Counter(r['decision'] for r in records)),
        'missing_slots':sorted([list((role,i)) for role in catalog['ROLES'] for i in range(8)
                                if (role,i) not in {(r['target_role'],int(r['strategy_index'])) for r in raw}]),
        'disclaimer':'This is textual triage. Only matched target execution can establish fingerprint discrimination.'})
    print('MAS5_V47_AUDIT',json.dumps(collections.Counter(r['decision'] for r in records)))
    print('REVIEW_FILE',folder/'review.tsv')
    print('Original bank is unchanged:',source)
    print('Edit REVIEW decisions to KEEP or REJECT. REJECT decisions can be revised to KEEP only after manual inspection.')


def curate(root,reject_review=False):
    root,source,folder=root_paths(root, allow_prepared=True)
    manifest=json.loads((folder/'source_manifest.json').read_text())
    if sha(source)!=manifest['sha256']:
        raise RuntimeError('Source bank changed since audit. DO NOT reuse stale review.tsv')
    orig=rows(source);catalog=validate_source(orig)
    by_key={(r['target_role'],int(r['strategy_index'])):r for r in orig}
    with (folder/'review.tsv').open(newline='') as f:
        decisions=list(csv.DictReader(f,delimiter='\t'))
    if len(decisions)!=len(orig):raise RuntimeError('Review TSV row count mismatch')
    kept=[];dropped=[];seen=set();unresolved=[]
    for dec in decisions:
        key=(dec['role'],int(dec['slot']))
        if key not in by_key or key in seen:raise RuntimeError('Unknown or repeated decision row: '+str(key))
        seen.add(key);row=by_key[key]
        if ' '.join(row['query'].split())!=dec['query']:
            raise RuntimeError('Question text changed in TSV: '+str(key))
        decision=dec['decision'].strip().upper()
        if decision=='REVIEW':
            if not reject_review:unresolved.append(key);continue
            decision='REJECT'
        if decision not in ('KEEP','REJECT'):
            raise RuntimeError('Decision must be KEEP, REJECT or REVIEW: '+str(key))
        if decision=='KEEP':
            curr,why=quality_for_row(row)
            if curr=='REJECT':
                raise RuntimeError('Cannot retain known-invalid probe '+str(key)+': '+','.join(why))
            if any(a['target_role']==row['target_role'] and oldgen.jaccard(a['query'],row['query'])>.62 for a in kept):
                raise RuntimeError('Same-role near duplicate retained: '+str(key))
            kept.append(row)
        else:dropped.append({**row,'v47_reasons':dec['reasons']})
    if unresolved:
        print('UNRESOLVED REVIEW COUNT',len(unresolved),'first',unresolved[:10],file=sys.stderr)
        raise RuntimeError('Inspect review.tsv and change REVIEW to KEEP or REJECT; or use --reject-review')
    kept.sort(key=lambda r:(catalog['ROLES'].index(r['target_role']),int(r['strategy_index'])))
    write_jsonl(folder/'kept.jsonl',kept)
    write_jsonl(folder/'rejected.jsonl',dropped)
    m={'status':'CURATED','source_count':len(orig),'kept':len(kept),'rejected':len(dropped),
       'new_needed':160-len(kept),'source_sha256':sha(source),'review_sha256':sha(folder/'review.tsv'),
       'rejected_preview':[{'role':r['target_role'],'slot':r['strategy_index'],'reason':r['v47_reasons']} for r in dropped],
       'no_mas_api_calls':True}
    write_json(folder/'curation.json',m)
    print('MAS5_V47_CURATED',json.dumps({'kept':len(kept),'rejected':len(dropped),'missing_to_generate':160-len(kept)}))
    print('Kept exact original rows:',folder/'kept.jsonl')


def inspect(root):
    root,source,folder=root_paths(root, allow_prepared=True)
    p=folder/'gpu_candidate'/SOURCE_FILE
    if not p.exists():raise RuntimeError('Regeneration has not completed')
    rr=rows(p);catalog=validate_source(rr)
    original=rows(folder/'kept.jsonl');orig={(r['target_role'],int(r['strategy_index'])):r for r in original}
    new=[];flagged=[];bad=[]
    for r in rr:
        key=(r['target_role'],int(r['strategy_index']))
        if key in orig:
            if orig[key]!=r:raise RuntimeError('A kept row was altered '+str(key))
        else:
            new.append(r)
        st,why=quality_for_row(r)
        if st=='REJECT':bad.append((key,why))
        if st=='REVIEW':flagged.append((key,why))
    if bad:raise RuntimeError('Generated bank contains known invalid candidates: '+str(bad[:10]))
    if len(rr)!=160 or len(original)+len(new)!=160:raise RuntimeError('Incomplete candidate bank')
    roles=catalog['ROLES']
    for role in roles:
        relevant=[r for r in rr if r['target_role']==role]
        if len(relevant)!=8 or sum(r['probe_family']=='help_me_help_you' for r in relevant)!=6:
            raise RuntimeError('Wrong family or slot count '+role)
    print('MAS5_V47_CANDIDATE_OK',{'original_kept':len(original),'new':len(new),
            'format_review_rows':len(flagged),'total':len(rr)})
    for r in new:
        print('\nNEW',r['target_role'],r['strategy_index'],r['strategy'])
        print(r['query'])
    print('\nHuman review still required before any paid MAS execution.')


def finalize(root,confirm):
    if confirm!='I_REVIEWED_PROBES':raise RuntimeError('Explicit manual review acknowledgement required')
    root,source,folder=root_paths(root)
    candidate=folder/'gpu_candidate'/SOURCE_FILE
    oldrows=rows(folder/'kept.jsonl'); newrows=rows(candidate)
    # Validations and byte-for-byte preservation must pass.
    inspect(root)
    if (root/'train/candidates_train_round_01.jsonl').exists():
        raise RuntimeError('Training preparation already started; not safe to replace bank')
    if (folder/'MAS5_V47_APPROVED.json').exists():raise RuntimeError('Already approved')
    backup=folder/'original_bank_backup.jsonl'
    shutil.copy2(source,backup)
    temp=source.with_suffix(source.suffix+'.v47.tmp')
    shutil.copy2(candidate,temp)
    os.replace(temp,source)
    # Keep generation summary up to date if present, not required by downstream prep.
    write_json(folder/'MAS5_V47_APPROVED.json',{'status':'APPROVED_BY_USER',
        'kept':len(oldrows),'regenerated':160-len(oldrows),
        'original_backup':str(backup),'bank_sha256':sha(source),
        'source_adapter':str(root/'train/generator_bootstrap_adapter'),
        'limits':'Textual review only: actual discrimination not yet measured'})
    print('MAS5_V47_APPROVED_BUT_NO_API_RUNS',source)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['audit','curate','inspect','finalize'])
    ap.add_argument('--root',required=True)
    ap.add_argument('--reject-review',action='store_true',help='Conservatively reject all unresolved REVIEW rows')
    ap.add_argument('--confirm',default='')
    a=ap.parse_args()
    if a.command=='audit':audit(a.root)
    elif a.command=='curate':curate(a.root,a.reject_review)
    elif a.command=='inspect':inspect(a.root)
    elif a.command=='finalize':finalize(a.root,a.confirm)

if __name__=='__main__':main()
