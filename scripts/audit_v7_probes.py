from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from detective.benign_gate_v7 import check_benign_hard_v7, normalize_query_v7


def main() -> None:
    ap=argparse.ArgumentParser(description="Audit every free-text V7 probe generated during training/evaluation")
    ap.add_argument("paths", nargs="+", help="V7 training JSONL and/or evaluation histories JSONL")
    ap.add_argument("--csv", default="artifacts/v7_probe_audit.csv")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if any accepted probe fails the hard gate")
    args=ap.parse_args()
    rows=[]
    for path in args.paths:
        p=Path(path)
        with p.open(encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line: continue
                obj=json.loads(line)
                if "actions" in obj:
                    for a in obj.get("actions",[]):
                        if a.get("action") == "ask": rows.append({"source":str(p),**a})
                elif obj.get("parsed_action") == "ask":
                    rows.append({"source":str(p),"query":obj.get("query",""),"allowed":obj.get("allowed",False),"gate_reason":obj.get("gate_reason","")})
    failures=[]; reasons=Counter(); unique=set(); accepted=blocked=0
    out=[]
    for r in rows:
        q=normalize_query_v7(r.get("query","")); hard=check_benign_hard_v7(q); allowed=bool(r.get("allowed",False))
        accepted += int(allowed); blocked += int(not allowed); unique.add(q); reasons[str(r.get("gate_reason",""))]+=1
        if allowed and not hard.allowed: failures.append(q)
        out.append({"source":r.get("source",""),"allowed":int(allowed),"hard_recheck":int(hard.allowed),"hard_reason":hard.reason,"query":q})
    Path(args.csv).parent.mkdir(parents=True,exist_ok=True)
    with Path(args.csv).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["source","allowed","hard_recheck","hard_reason","query"]);w.writeheader();w.writerows(out)
    print("generated ASK probes:",len(rows))
    print("accepted:",accepted,"blocked:",blocked,"unique:",len(unique))
    print("accepted-but-hard-gate-failures:",len(failures))
    print("top gate reasons:",reasons.most_common(10))
    print("csv:",args.csv)
    if args.strict and failures:
        raise SystemExit("STRICT AUDIT FAILED: accepted probes failed hard-gate recheck")


if __name__ == "__main__": main()
