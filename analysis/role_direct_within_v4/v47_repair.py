#!/usr/bin/env python3
"""GPU-only MAS5 V4.7 regeneration of REJECTED/missing slots from existing QLoRA.
Does not overwrite original bank and makes NO external MAS/API calls.
"""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from analysis.role_direct_within_v4 import v47_generator as generator
from analysis.role_direct_within_v4 import v47_curate as curate


def main(root,retry=False):
    root,source,folder=curate.root_paths(root, allow_prepared=True)
    conf=folder/'curation.json'
    if not conf.exists():raise RuntimeError('First run audit and curate (review decisions)')
    meta=json.loads(conf.read_text())
    if curate.sha(source)!=meta['source_sha256']:
        raise RuntimeError('Original source bank changed; refusing stale curation')
    if curate.sha(folder/'review.tsv')!=meta['review_sha256']:
        raise RuntimeError('Review.tsv modified AFTER curation. Re-audit to prevent stale decisions')
    adapter=root/'train/generator_bootstrap_adapter'
    if not (adapter/'adapter_config.json').is_file():raise RuntimeError('No successful generator SFT checkpoint')
    candidate=folder/'gpu_candidate'
    if candidate.exists():
        if not retry:raise RuntimeError('Repair already attempted. Inspect output or submit a retry.')
        partial=candidate/'v46_hmhy_candidate_probes_round_01.jsonl'
        if not partial.exists() or len(curate.rows(partial))>=160:
            raise RuntimeError('Candidate already complete or missing; do not blindly retry.')
        n=1
        while (folder/f'gpu_retry_{n:02d}').exists() or (folder/f'gpu_candidate_before_retry_{n:02d}').exists():n+=1
        work=folder/f'gpu_retry_{n:02d}'
        resume=partial
    else:
        work=candidate
        resume=folder/'kept.jsonl'
        n=0
    work.mkdir()
    os.environ['V46_GEN_ADAPTER']=str(adapter)
    os.environ['V46_SEED']='20261017'
    print('V47_GPU_REPAIR_BEGIN',{'original_kept':meta['kept'],'current_resume':len(curate.rows(resume)),
          'to_regenerate':160-len(curate.rows(resume)),'retry':retry},flush=True)
    generator.generate(work,1,None,.92,resume_bank=resume,max_attempts=120)
    if retry:
        backup=folder/f'gpu_candidate_before_retry_{n:02d}'
        os.rename(candidate,backup)
        os.rename(work,candidate)
    curate.inspect(root)
    print('V47_GPU_REPAIR_DONE_AWAITING_HUMAN_REVIEW',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True)
    p.add_argument('--retry',action='store_true')
    a=p.parse_args();main(a.root,a.retry)
