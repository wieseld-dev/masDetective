#!/usr/bin/env python3
"""Apply reviewed final 42 decisions to existing MAS5 artifacts without overwriting prior data.
Usage: python mas5_v480_apply_review.py --root "$V46_ROOT" --review mas5_v480_final42_decisions.tsv
"""
import argparse, csv, hashlib, json
from pathlib import Path
from collections import Counter

def load(p):
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def key(r):
    return r["target_role"], int(r["strategy_index"])

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--review",type=Path,required=True)
    args=ap.parse_args()
    base=args.root / "train/v47_quality/v472_reviewed_second_pass/v476_manual_review"
    approved_path=base / "approved_118.jsonl"
    final_path=base / "v479_manual_completion/MAS5_candidate_bank_160.jsonl"
    approved_rows=load(approved_path)
    final_rows=load(final_path)
    assert len(approved_rows)==118 and len(final_rows)==160, "Unexpected bank sizes"
    approved={key(r):r for r in approved_rows}
    final={key(r):r for r in final_rows}
    assert len(approved)==118 and len(final)==160, "Duplicate slots in source"
    assert all(final.get(k)==v for k,v in approved.items()), "Approved bank changed!"
    with args.review.open(newline="",encoding="utf-8") as f:
        review=list(csv.DictReader(f,delimiter="\t"))
    assert len(review)==42, "Review must have 42 rows"
    decisions={}
    for r in review:
        k=(r["role"],int(r["slot"]))
        assert k not in decisions and k in final and k not in approved, f"Invalid review slot: {k}"
        assert r["query"]==final[k]["query"], f"Question text mismatched: {k}"
        assert r["strategy"]==final[k].get("strategy",""), f"Strategy mismatched: {k}"
        assert r["decision"] in ("KEEP","REJECT"), f"Invalid decision: {k}"
        decisions[k]=r["decision"]
    assert set(decisions)==set(final)-set(approved), "Incomplete review coverage"
    selected={**approved}
    rejected=[]
    for k in decisions:
        if decisions[k]=="KEEP": selected[k]=final[k]
        else: rejected.append(final[k])
    assert len(selected)==134 and len(rejected)==26, "Unexpected final counts"
    assert all(selected[k]==v for k,v in approved.items())
    output=base / "v480_final42_review"
    output.mkdir(parents=True,exist_ok=True)
    kept_file=output / "approved_134.jsonl"
    rej_file=output / "rejected_26.jsonl"
    manifest_file=output / "manifest.json"
    # Do not overwrite existing outputs; audit them separately on a rerun.
    if any(p.exists() for p in (kept_file,rej_file,manifest_file)):
        raise RuntimeError("Output already exists; refusing overwrite: " + str(output))
    with kept_file.open("x",encoding="utf-8") as f:
        for k in sorted(selected): f.write(json.dumps(selected[k],ensure_ascii=False)+"\n")
    with rej_file.open("x",encoding="utf-8") as f:
        for r in rejected: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    manifest={
        "status":"TEXT_REVIEWED_PENDING_MATCHED_EVALUATION",
        "approved_previous":118,"approved_from_final42":16,"approved_total":134,
        "rejected_from_final42":26,"open_slots":26,
        "source_full_bank_sha256":sha(final_path),"source_approved_sha256":sha(approved_path),
        "review_sha256":sha(args.review),"approved_134_sha256":sha(kept_file),
        "no_mas_calls":True,"discrimination_verified":False,
        "approved_path":str(kept_file.resolve()),"rejected_path":str(rej_file.resolve())
    }
    manifest_file.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=="__main__":main()
