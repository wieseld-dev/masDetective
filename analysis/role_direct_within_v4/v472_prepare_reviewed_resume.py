#!/usr/bin/env python3
"""MAS5 V4.7.2 offline selection of original 61 plus reviewed new candidates.

No API, model inference, source edits, checkpoint writes or Slurm submission.
Refuses to silently overwrite an existing reviewed bank.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re
from pathlib import Path
from collections import Counter

def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]

def key(r: dict) -> tuple[str,int]:
    return (str(r['target_role']),int(r['strategy_index']))

def expected_family(ix:int) -> str:
    return 'help_me_help_you' if ix<6 else 'direct_task'

def jaccard(a: str, b: str) -> float:
    def toks(s):return set(re.sub(r'\W+', ' ', s.lower()).split())
    aa,bb=toks(a),toks(b)
    return len(aa&bb)/max(1,len(aa|bb))

def write_jsonl_exclusive(path: Path, items: list[dict]):
    with path.open('x', encoding='utf-8') as f:
        for row in items:
            f.write(json.dumps(row,ensure_ascii=False)+'\n')

def main(root: Path,review: Path):
    root=root.expanduser().resolve(strict=True)
    review=review.expanduser().resolve(strict=True)
    project=Path.cwd().resolve()
    expected=project/'artifacts/role_direct_within_v4/overnight_v46'
    if not root.is_relative_to(expected):
        raise RuntimeError('Unexpected RUN directory; execute from MAS project root')
    quality=root/'train/v47_quality'
    candidate=quality/'gpu_candidate/v46_hmhy_candidate_probes_round_01.jsonl'
    originals=quality/'kept.jsonl'
    assert candidate.is_file() and originals.is_file(), 'Missing original curation or candidate bank'
    rr=load_jsonl(candidate)
    preserved=load_jsonl(originals)
    if len(rr)!=160 or len(preserved)!=61:
        raise RuntimeError(f'Unexpected candidate/original counts {len(rr)}/{len(preserved)}; no mutation')
    cand={key(r):r for r in rr}
    orig={key(r):r for r in preserved}
    if len(cand)!=160 or len(orig)!=61:
        raise RuntimeError('Duplicate role/slot in candidate or preserved data')
    for k,r in orig.items():
        if cand.get(k)!=r:
            raise RuntimeError(f'Previously approved probe altered {k}')
    for r in rr:
        if r['probe_family']!=expected_family(key(r)[1]):
            raise RuntimeError(f'Wrong family in source candidate {key(r)}')
    with review.open(encoding='utf-8-sig',newline='') as f:
        decisions=list(csv.DictReader(f,delimiter='\t'))
    if len(decisions)!=99:
        raise RuntimeError('Expected exactly 99 second-round reviews')
    review_keys=set()
    retained=list(preserved)
    rejected=[]
    for d in decisions:
        k=(d['role'],int(d['slot']))
        if k in review_keys or k not in cand or k in orig:
            raise RuntimeError(f'Unexpected or duplicate review slot {k}')
        review_keys.add(k)
        if d['query']!=cand[k]['query'] or d['strategy']!=cand[k]['strategy']:
            raise RuntimeError(f'Query or strategy differs from reviewed source {k}')
        if d['family']!=cand[k]['probe_family']:
            raise RuntimeError(f'Probe family mismatch {k}')
        if d['decision']=='KEEP':
            retained.append(cand[k])
        elif d['decision']=='REJECT':
            rejected.append({**cand[k], 'manual_review_reason':d['reason']})
        else:
            raise RuntimeError(f'Unknown review decision {k}')
    if review_keys != set(cand)-set(orig):
        raise RuntimeError('Reviews did not cover exactly the 99 regenerated rows')
    if len(retained)!=102 or len(rejected)!=58:
        raise RuntimeError(f'Unexpected post-review split {len(retained)}/{len(rejected)}')
    per_role={}
    for r in retained:
        per_role.setdefault(r['target_role'],[]).append(r)
    for role, rs in per_role.items():
        for i,a in enumerate(rs):
            for b in rs[i+1:]:
                if jaccard(a['query'],b['query'])>.62:
                    raise RuntimeError(f'Within-role near-duplicate retained for {role}: {key(a)} and {key(b)}')
    # Ensure the 99-question review file wasn't silently changed from the known upload.
    # The file's own SHA is independent of the original upload because of added decisions.
    folder=quality/'v472_reviewed_second_pass'
    folder.mkdir(parents=False,exist_ok=False)
    order={role:i for i,role in enumerate(dict.fromkeys(r['target_role'] for r in rr))}
    retained.sort(key=lambda r:(order[r['target_role']],int(r['strategy_index'])))
    rejected.sort(key=lambda r:(order[r['target_role']],int(r['strategy_index'])))
    write_jsonl_exclusive(folder/'approved_102.jsonl',retained)
    write_jsonl_exclusive(folder/'rejected_58.jsonl',rejected)
    summary={'status':'TEXT_REVIEW_READY_FOR_REGENERATION','kept_previous':61,'kept_new':41,
             'approved_so_far':102,'needs_regeneration':58,
             'original_candidate_sha256':digest(candidate), 'review_decisions_sha256':digest(review),
             'approved_sha256':digest(folder/'approved_102.jsonl'),
             'no_mas_or_model_execution':True,
             'discrimination_verified':False,
             'warning':'Use a separate output folder when generating replacements. Do not overwrite the original 160 candidate bank.'}
    (folder/'manifest.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('MAS5_V472_REVIEWED',json.dumps(summary,ensure_ascii=False))
    print('RESUME_BANK',folder/'approved_102.jsonl')
    print('REJECTED_BANK',folder/'rejected_58.jsonl')

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--review',type=Path,default=Path('mas5_v472_new_questions_review.tsv'))
    args=ap.parse_args()
    main(args.root,args.review)
