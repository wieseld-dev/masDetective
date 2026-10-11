#!/usr/bin/env python3
"""Read-only V4.7.5 resume health check: detects conflicts, counts unique role/slots."""
import argparse, json, hashlib
from pathlib import Path
from analysis.role_direct_within_v4.v473_regenerate import REL,BANK,PROGRESS,load,key,union_resume

def main(root):
    folder=root/REL
    approved_file=folder/'approved_102.jsonl'
    manifest=json.loads((folder/'manifest.json').read_text())
    assert hashlib.sha256(approved_file.read_bytes()).hexdigest()==manifest['approved_sha256'], 'Approved bank SHA mismatch'
    a=load(approved_file); approved={key(x):x for x in a}
    assert len(a)==len(approved)==102
    work=folder/'v473_regeneration'
    sources=[]
    for name,p in [('merged',work/'v474_merged_resume.jsonl'),('completed',work/BANK),('progress',work/PROGRESS)]:
        if p.is_file():
            records=load(p)
            sources.append((name,records))
            print('SOURCE',name,len(records),flush=True)
    merged=union_resume(approved,sources)
    print('MAS5_V475_PREFLIGHT',json.dumps({'approved':len(approved),'unique_available':len(merged),
           'remaining':160-len(merged),'protected_adapter':str(root/'train/generator_bootstrap_adapter'),
           'writes_performed':False},ensure_ascii=False),flush=True)
    assert all(merged[k]==v for k,v in approved.items())
    assert len(merged)<=160

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();main(a.root.resolve(strict=True))
