from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
from detective.benign_gate_v7 import check_benign_hard_v7

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--log',default='artifacts/v8_grpo_train.jsonl');ap.add_argument('--strict',action='store_true');args=ap.parse_args();rows=[json.loads(x) for x in Path(args.log).read_text(encoding='utf-8').splitlines() if x.strip()];status=Counter(r.get('status','') for r in rows);qs=[r.get('query','') for r in rows if r.get('status')=='ok'];fails=[q for q in qs if not check_benign_hard_v7(q).allowed]
 print('rows:',len(rows));print('status:',dict(status));print('unique accepted queries:',len(set(qs)));print('accepted-but-hard-gate-failures:',len(fails));print('mean reward:',sum(float(r.get('reward',0)) for r in rows)/max(len(rows),1))
 if args.strict and fails: raise SystemExit('STRICT AUDIT FAILED')
if __name__=='__main__':main()
