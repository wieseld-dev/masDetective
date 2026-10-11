from __future__ import annotations
import argparse, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--update-summary',required=True); ap.add_argument('--eval-report',required=True); ap.add_argument('--strict',type=int,default=1); a=ap.parse_args()
    u=json.loads(Path(a.update_summary).read_text()); e=json.loads(Path(a.eval_report).read_text()); f=e['final']
    reasons=[]
    if float(u['mean_kl_surrogate_to_v11'])>0.50: reasons.append(f"KL surrogate too high {u['mean_kl_surrogate_to_v11']:.4f}")
    if int(f['valid_updates']) < max(1,int(.90*int(f['systems']))): reasons.append(f"dev valid updates too low {f['valid_updates']}/{f['systems']}")
    if float(f['role_log_loss'])>0.62: reasons.append(f"dev role_log_loss catastrophic {f['role_log_loss']:.4f}")
    status='PASS' if not reasons else 'FAIL'
    print(json.dumps({'gate':status,'reasons':reasons,'final_dev':f,'update':u},indent=2))
    if reasons and a.strict: raise SystemExit(3)
if __name__=='__main__': main()
