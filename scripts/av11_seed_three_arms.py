from __future__ import annotations
import argparse, shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--common-root",required=True); ap.add_argument("--arms",nargs="+",required=True); ap.add_argument("--split",default="dev"); args=ap.parse_args()
    c=Path(args.common_root); names=[f"candidates_{args.split}_round_01.jsonl",f"candidate_pool_{args.split}_round_01.jsonl",f"responses_{args.split}_round_01.jsonl",f"score_input_{args.split}_round_01.jsonl",f"scored_{args.split}_round_01.jsonl",f"states_{args.split}_round_01.jsonl",f"kboot_metrics_{args.split}_round_01.jsonl",f"kboot_advance_{args.split}_round_01_summary.json","target_public_cache.jsonl","target_truth_cache_training_only.jsonl"]
    for a in args.arms:
        d=Path(a); d.mkdir(parents=True,exist_ok=True)
        for n in names:
            s=c/n
            if s.exists(): shutil.copy2(s,d/n)
    print("PASS: three arms seeded from exact shared round 1")
if __name__=="__main__": main()
