#!/usr/bin/env python3
"""V4.7.3 isolated MAS5 question regeneration from 102 human-reviewed probes.
Preserves original bank/experiment artifacts; no target MAS/API execution.
"""
from __future__ import annotations
import argparse, json, os, hashlib, tempfile
from pathlib import Path
from collections import Counter
from analysis.role_direct_within_v4 import v473_generator

REL='train/v47_quality/v472_reviewed_second_pass'
BANK='v46_hmhy_candidate_probes_round_01.jsonl'
PROGRESS='v473_progress.partial.jsonl'

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(p):return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
def key(r):return (str(r['target_role']),int(r['strategy_index']))

def check(rows,approved,limit=160,require_all_approved=True):
    if len(rows)>limit:raise RuntimeError('Too many candidates')
    seen={}
    for r in rows:
        k=key(r)
        if k in seen:raise RuntimeError('Duplicate role-slot '+str(k))
        if not 0<=k[1]<8:raise RuntimeError('Invalid slot '+str(k))
        expected='help_me_help_you' if k[1]<6 else 'direct_task'
        if r['probe_family']!=expected:raise RuntimeError('Wrong probe family '+str(k))
        seen[k]=r
    for k,r in approved.items():
        if k in seen:
            if seen[k]!=r:
                raise RuntimeError('Approved question altered '+str(k))
        elif require_all_approved:
            raise RuntimeError('Approved question missing '+str(k))
    return seen

def write_jsonl_atomic(path, rows):
    # No changes to approved_102.jsonl, original 160-bank, or previous checkpoints.
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row,ensure_ascii=False)+'\n')
        out.flush()
        os.fsync(out.fileno())
    os.replace(tmp,path)

def union_resume(original, sources):
    result=dict(original)
    for label, rows in sources:
        check(rows,original,require_all_approved=False)
        for r in rows:
            k=key(r)
            if k in result and result[k]!=r:
                raise RuntimeError('Conflicting versions for role-slot '+str(k)+' in '+label)
            result[k]=r
    if len(result)>160:
        raise RuntimeError('Union exceeds 160 role-slots')
    return result

def main(root,attempts):
    root=root.expanduser().resolve(strict=True)
    project=Path.cwd().resolve()
    if not root.is_relative_to(project/'artifacts/role_direct_within_v4/overnight_v46'):
        raise RuntimeError('Unexpected MAS5 RUN path')
    folder=root/REL
    source=folder/'approved_102.jsonl'
    manifest_file=folder/'manifest.json'
    if not (source.is_file() and manifest_file.is_file()):
        raise RuntimeError('V4.7.2 approved bank / manifest missing')
    manifest=json.loads(manifest_file.read_text())
    if manifest.get('approved_so_far')!=102 or manifest.get('needs_regeneration')!=58:
        raise RuntimeError('Unexpected approved bank manifest')
    if sha(source)!=manifest['approved_sha256']:
        raise RuntimeError('Approved bank changed since review')
    preserved=load(source)
    original={key(r):r for r in preserved}
    if len(preserved)!=102 or len(original)!=102:
        raise RuntimeError('Approved bank duplicate or wrong count')
    generator_adapter=root/'train/generator_bootstrap_adapter'
    if not (generator_adapter/'adapter_config.json').is_file():
        raise RuntimeError('HMHY-SFT Generator checkpoint missing')
    work=folder/'v473_regeneration'
    work.mkdir(exist_ok=True)
    complete=work/BANK
    progress=work/PROGRESS
    if complete.exists():
        complete_rows=load(complete)
        check(complete_rows,original)
        if len(complete_rows)==160:
            print('MAS5_V473_ALREADY_COMPLETE 160/160',flush=True)
            return
    # A checkpoint from interrupted generation is a ROLE-ORDERED PREFIX:
    # it does not yet contain approved entries from later roles. Merge it with
    # the complete approved set instead of incorrectly requiring all 102 within it.
    sources=[]
    resume=work/'v474_merged_resume.jsonl'
    # CRITICAL: A retry starts writing a new prefix checkpoint from role 1.
    # If it crashes early, that checkpoint may be SHORTER than the previous one.
    # Preserve the last full merged set across repeated retries.
    if resume.is_file():
        sources.append(('previous_merged_resume',load(resume)))
    if complete.is_file():
        sources.append(('previous_complete',load(complete)))
    if progress.is_file():
        sources.append(('checkpoint',load(progress)))
    merged=union_resume(original,sources)
    write_jsonl_atomic(resume,[merged[k] for k in sorted(merged)])
    print('MAS5_V474_SAFE_MERGE',json.dumps({
        'approved_unchanged':len(original),'checkpoint_rows':
        len(load(progress)) if progress.is_file() else 0,
        'recovered_unique':len(merged),'remaining':160-len(merged),
        'resume_path':str(resume)}),flush=True)
    print('MAS5_V473_START',json.dumps({'approved':len(original),'resume':len(merged),
           'missing':160-len(merged),'attempts_per_slot':attempts,'resume_path':str(resume)}),flush=True)
    os.environ['V46_GEN_ADAPTER']=str(generator_adapter)
    os.environ['V46_SEED']='20261019'
    v473_generator.SEED=20261019  # Module imported before environment assignment.
    # May raise if coverage <160, but its final candidate file and per-slot progress survive.
    try:
        v473_generator.generate(work,1,None,.92,resume_bank=resume,max_attempts=attempts)
    finally:
        if progress.is_file():
            salvaged=load(progress)
            # Incomplete checkpoints legitimately omit approved later-role slots.
            # Verify any approved row they DO contain is byte-for-byte identical.
            check(salvaged,original,require_all_approved=False)
            recovered=union_resume(original,[('checkpoint',salvaged)])
            print('MAS5_V474_SAVED_PROGRESS',len(recovered),'/160',
                  'checkpoint_prefix',len(salvaged),flush=True)
    final=load(complete)
    final_map=check(final,original)
    if len(final)!=160:
        raise RuntimeError(f'Expected 160 total candidates, got {len(final)}')
    # Exactly 20 roles x 8 slots. All original 102 are byte-for-byte unchanged at row level.
    roles=Counter(r['target_role'] for r in final)
    if len(roles)!=20 or any(n!=8 for n in roles.values()):
        raise RuntimeError('Per-role candidate coverage incomplete')
    family=Counter(r['probe_family'] for r in final)
    if family!=Counter({'help_me_help_you':120,'direct_task':40}):
        raise RuntimeError('Incorrect 6:2 ratio')
    accepted_new=[r for r in final if key(r) not in original]
    result={'status':'READY_FOR_HUMAN_REVIEW_NOT_MAS',
            'original_approved':len(original),'generated_replacements':len(accepted_new),
            'total':len(final),'families':dict(family),'candidate_sha256':sha(complete),
            'source_approved_sha256':sha(source),'no_mas_api_calls':True,
            'discrimination_verified':False,'output_path':str(complete)}
    target=work/'v473_manifest.json'
    temp=target.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    os.replace(temp,target)
    print('MAS5_V473_DONE_AWAITING_HUMAN_REVIEW',json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--attempts',type=int,default=36)
    args=ap.parse_args()
    if not 18<=args.attempts<=120:raise ValueError('attempts must be 18..120')
    main(args.root,args.attempts)
