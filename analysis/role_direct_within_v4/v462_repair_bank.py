#!/usr/bin/env python3
"""Recover an incomplete V4.6 MAS5 probe bank without discarding accepted probes.

NO MAS/API calls. Only samples the MISSING slots from the existing trained adapter.
The 158 accepted rows are compared byte-for-byte as JSON objects before replacement.
"""
from __future__ import annotations
import argparse, json, os, shutil
from collections import Counter
from pathlib import Path
from analysis.role_direct_within_v4 import v46_generator as g, pipeline as old


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(root:Path):
    root=root.resolve()
    project=Path('/home/wieseld/mas_rl_detective').resolve()
    if not root.is_relative_to(project/'artifacts/role_direct_within_v4/overnight_v46'):
        raise RuntimeError('Unexpected MAS5 training root')
    report=root/'train/generator_bootstrap_report.json'
    if not report.is_file() or json.loads(report.read_text()).get('status')!='PASS':
        raise RuntimeError('No successful existing Generator SFT bootstrap')
    adapter=root/'train/generator_bootstrap_adapter/adapter_config.json'
    if not adapter.is_file():raise RuntimeError('Missing actual Generator bootstrap adapter')
    probe_dir=root/'train/probes'
    source=probe_dir/'v46_hmhy_candidate_probes_round_01.jsonl'
    summary=probe_dir/'v46_hmhy_generation_round_01.json'
    if not source.is_file():raise RuntimeError('No existing incomplete bank to resume')
    original=rows(source)
    keys={(x['target_role'],int(x['strategy_index'])) for x in original}
    if len(keys)!=len(original) or len(original)>=160:
        raise RuntimeError('Expected unique incomplete original bank, refusing to modify')
    if (root/'train/candidates_train_round_01.jsonl').exists() or (root/'train/responses_train_round_01.jsonl').exists():
        raise RuntimeError('Training already prepared/executed. Refusing bank mutation')
    tag=str(os.environ.get('SLURM_JOB_ID',os.getpid()))
    work=root/'train'/('v462_repair_work_'+tag)
    work.mkdir(parents=True,exist_ok=False)
    print('MAS5_V462_BEGIN_REPAIR',{'kept':len(original),'missing':160-len(original),'work':str(work)},flush=True)
    g.generate(work,1,None,.9,resume_bank=source,max_attempts=96)
    candidate=work/source.name
    updated=rows(candidate)
    newkeys={(x['target_role'],int(x['strategy_index'])) for x in updated}
    cat=old.catalog()
    if len(updated)!=160 or len(newkeys)!=160 or any(sum(x['target_role']==role for x in updated)!=8 for role in cat['ROLES']):
        raise RuntimeError('Repaired output failed full cardinality check')
    current={(x['target_role'],int(x['strategy_index'])):x for x in updated}
    for x in original:
        key=(x['target_role'],int(x['strategy_index']))
        if current[key]!=x:raise RuntimeError('Existing probe unexpectedly mutated: '+str(key))
    backup=probe_dir/('backup_before_v462_'+tag)
    backup.mkdir(exist_ok=False)
    shutil.copy2(source,backup/source.name)
    if summary.is_file():shutil.copy2(summary,backup/summary.name)
    # Replace the original bank only after the full validated replacement exists.
    os.replace(candidate,source)
    os.replace(work/summary.name,summary)
    marker=probe_dir/'MAS5_V462_REPAIRED.json'
    if marker.exists():raise RuntimeError('Repair marker already exists')
    marker.write_text(json.dumps({'status':'PASS','kept':len(original),
        'new':len(updated)-len(original),
        'missing_slots_filled':sorted([list(z) for z in (newkeys-keys)]),
        'backup':str(backup)},indent=2)+'\n')
    print('MAS5_V462_REPAIR_COMPLETE',marker,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True)
    args=ap.parse_args();main(Path(args.root))
