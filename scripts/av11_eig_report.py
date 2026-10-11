from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from adaptive_v11_active.io import read_jsonl
METRICS=["role_log_loss","role_brier","role_auc","true_false_margin","oracle_k_role_recall"]
def m(root,r): return {str(x["session_id"]):x for x in read_jsonl(Path(root)/f"kboot_metrics_dev_round_{r:02d}.jsonl")}
def boot(d,seed,n=20000):
    d=np.asarray(d,float); rng=np.random.default_rng(seed); z=np.asarray([d[rng.integers(0,len(d),len(d))].mean() for _ in range(n)]); return float(d.mean()),[float(np.quantile(z,.025)),float(np.quantile(z,.975))]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); ap.add_argument("--rounds",type=int,default=6); args=ap.parse_args(); root=Path(args.root)
    arms=["static","maxent","utility"]; lines=["QUERY-UTILITY ADAPTIVE CONTROLLER V2","delta = first arm minus second arm; lower is better for log-loss/Brier","" ]; obj={"version":"av11-query-utility-v2","rounds":[]}
    for r in range(1,args.rounds+1):
        mm={a:m(root/a,r) for a in arms}; ids=sorted(set.intersection(*(set(mm[a]) for a in arms))); rec={"round":r,"paired":len(ids),"comparisons":{}}
        lines.append(f"ROUND {r} paired={len(ids)}")
        for A,B in [("utility","maxent"),("utility","static")]:
            lines.append(f"  {A} vs {B}")
            rec["comparisons"][f"{A}_minus_{B}"]={}
            for j,k in enumerate(METRICS):
                da=np.asarray([float(mm[A][sid][k])-float(mm[B][sid][k]) for sid in ids]); mean,ci=boot(da,917700+r*100+j)
                rec["comparisons"][f"{A}_minus_{B}"][k]={"delta":mean,"ci95":ci,"a_mean":float(np.mean([mm[A][sid][k] for sid in ids])),"b_mean":float(np.mean([mm[B][sid][k] for sid in ids]))}
                lines.append(f"    {k:22s} delta={mean:+.6f} CI95=[{ci[0]:+.6f},{ci[1]:+.6f}]")
        lines.append(""); obj["rounds"].append(rec)
    pe=obj["rounds"][-1]["comparisons"]["utility_minus_maxent"]["role_log_loss"]; obj["primary_endpoint"]={"metric":"role_log_loss","round":args.rounds,"comparison":"utility_minus_maxent",**pe}
    out=root/"report"; out.mkdir(parents=True,exist_ok=True); (out/"eig_report.txt").write_text("\n".join(lines)+"\n"); (out/"eig_report.json").write_text(json.dumps(obj,indent=2,sort_keys=True)); print("\n".join(lines)); print("PRIMARY",json.dumps(obj["primary_endpoint"],indent=2))
if __name__=="__main__": main()
