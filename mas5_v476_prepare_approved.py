#!/usr/bin/env python3
"""Merge V4.7.5 question decisions without mutating original banks.

Usage:
  python mas5_v476_prepare_approved.py --root "$V46_ROOT" --review mas5_v476_new_58_decisions.tsv
"""
import argparse, csv, hashlib, json
from collections import Counter
from pathlib import Path

def load_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]

def key(row):
    return (str(row["target_role"]), int(row["strategy_index"]))

def dump_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--review",type=Path,required=True)
    a=ap.parse_args()
    base=a.root/"train/v47_quality/v472_reviewed_second_pass"
    original=base/"approved_102.jsonl"
    candidate=base/"v473_regeneration/v46_hmhy_candidate_probes_round_01.jsonl"
    approved=load_jsonl(original)
    generated=load_jsonl(candidate)
    originals={key(r):r for r in approved}
    finals={key(r):r for r in generated}
    if len(originals)!=102 or len(finals)!=160 or len(generated)!=160:
        raise RuntimeError("Expected 102 unique original approvals and 160 unique candidate rows")
    if any(finals.get(k)!=v for k,v in originals.items()):
        raise RuntimeError("Previously approved questions were changed: refusing to proceed")
    with a.review.open(encoding="utf-8-sig",newline="") as f:
        decisions=list(csv.DictReader(f,delimiter="\t"))
    if len(decisions)!=58:
        raise RuntimeError("Expected exactly 58 reviewed replacement questions")
    seen=set(); keep=[]; reject=[]
    for d in decisions:
        k=(d["role"],int(d["slot"]))
        if k in seen or k in originals or k not in finals:
            raise RuntimeError("Repeated/unexpected reviewed key: "+str(k))
        seen.add(k)
        current=finals[k]
        if d["query"]!=current["query"] or d["strategy"]!=current["strategy"]:
            raise RuntimeError("Question/strategy mismatch for "+str(k))
        if d["decision"]=="KEEP":keep.append(current)
        elif d["decision"]=="REJECT":reject.append(current)
        else:raise RuntimeError("Invalid decision for "+str(k))
    if seen != (set(finals)-set(originals)):
        raise RuntimeError("Review does not cover precisely the 58 new questions")
    merged=sorted(approved+keep,key=lambda r:(r["target_role"],int(r["strategy_index"])))
    if len(merged)!=len({key(r) for r in merged}) or len(merged)+len(reject)!=160:
        raise RuntimeError("Invalid merged coverage")
    out=base/"v476_manual_review"
    out.mkdir(parents=True,exist_ok=True)
    dest=out/("approved_%d.jsonl"%len(merged))
    if dest.exists():
        if load_jsonl(dest)!=merged:raise RuntimeError("Existing curated output conflicts; refusing overwrite")
    else:dump_jsonl(dest,merged)
    rejected=out/("rejected_%d.jsonl"%len(reject))
    if rejected.exists():
        if load_jsonl(rejected)!=reject:raise RuntimeError("Existing rejected output conflicts; refusing overwrite")
    else:dump_jsonl(rejected,reject)
    manifest={"status":"REVIEWED_NOT_DISCRIMINATION_VERIFIED","approved_previous":len(approved),"approved_new":len(keep),"approved_total":len(merged),"regenerate":len(reject),"no_mas_calls":True,"approved_source_sha256":hashlib.sha256(original.read_bytes()).hexdigest(),"candidate_source_sha256":hashlib.sha256(candidate.read_bytes()).hexdigest(),"review_sha256":hashlib.sha256(a.review.read_bytes()).hexdigest(),"approved_path":str(dest),"rejected_path":str(rejected)}
    m=out/"manifest.json"
    if m.exists() and json.loads(m.read_text())!=manifest:raise RuntimeError("Existing manifest conflicts; refusing overwrite")
    m.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
