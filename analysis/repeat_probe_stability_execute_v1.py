#!/usr/bin/env python3
"""
repeat_probe_stability_execute_v1.py

Fresh repeated execution pilot for the SAME (hidden MAS, candidate probe).

Default pilot:
  - round 1 only (cleanest: no prior adaptive-history confound)
  - 8 held-out test systems, stratified ~2 per K in {3,4,5,6}
  - 4 candidate probes/system spanning current selector scores
  - 3 FRESH repetitions/probe
  = at most 96 fresh MAS executions.

IMPORTANT:
- This intentionally BYPASSES the old query cache.
- Training-only trace/true-role fields are saved only for diagnostics.
- Runtime output itself is still the only black-box observation used by the
  downstream fingerprint detector.
"""

from __future__ import annotations
import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Register V11 design exactly as the original executor does.
from targets.single_router_zoo_v11 import validate_v11_design
validate_v11_design()
from targets.single_router_fanout_mas import SingleRouterFanoutMAS, SingleRouterSystemSpec

import sys
sys.path.insert(0, str(Path("analysis").resolve()))
import observation_model_v1 as om


def read_jsonl(p: Path):
    out=[]
    with p.open(encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: out.append(json.loads(line))
    return out


def append_jsonl(p: Path, row: dict):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False)+"\n")
        f.flush()


def load_test_systems(obs_dir: Path):
    d=json.loads((obs_dir/"split_systems.json").read_text())
    return set(map(str,d["test"]))


def load_rank_scores(path: Path):
    out={}
    if not path.exists():
        return out
    for x in read_jsonl(path):
        key=(str(x["system_id"]),int(x["round"]),int(x["candidate_index"]))
        out[key]=x
    return out


def discover_system_specs(artifact_root: Path):
    """
    Build session_id -> privileged manifest record from existing manifests.
    Prefer exact manifest_privileged.json, then other manifest*.json.
    """
    paths=list(artifact_root.rglob("manifest_privileged.json"))
    if not paths:
        paths=list(artifact_root.rglob("manifest*.json"))
    by_sid={}
    src={}
    for p in paths:
        try:
            obj=json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        systems=obj.get("systems",[]) if isinstance(obj,dict) else []
        for m in systems:
            sid=str(m.get("session_id",""))
            if not sid or not all(k in m for k in ["mode","style","roles"]):
                continue
            sig=(str(m["mode"]),str(m["style"]),tuple(map(str,m["roles"])))
            if sid in by_sid:
                old=by_sid[sid]
                oldsig=(str(old["mode"]),str(old["style"]),tuple(map(str,old["roles"])))
                if sig != oldsig:
                    raise RuntimeError(f"Conflicting privileged specs for session {sid}: {src[sid]} vs {p}")
            else:
                by_sid[sid]=m
                src[sid]=str(p)
    return by_sid,src,paths


def execute_fresh(m: dict, sid: str, q: str, model: str|None, retries: int):
    spec=SingleRouterSystemSpec(
        system_id=sid,
        split=f"adaptive_v11_{m.get('split','train')}",
        execution_mode=str(m["mode"]),
        style=str(m["style"]),
        worker_roles=tuple(map(str,m["roles"])),
    )
    last=None
    for attempt in range(retries+1):
        try:
            # FRESH object every repetition; NO cache.
            target=SingleRouterFanoutMAS(spec, model=model)
            output=str(target.query(q))
            tr=getattr(target,"_last_trace",None)
            if tr is None:
                raise RuntimeError("SingleRouterFanoutMAS did not expose _last_trace")
            selected=list(map(str,getattr(tr,"selected_roles",())))
            return output,selected,attempt
        except Exception as e:
            last=e
            if attempt>=retries:
                raise
            time.sleep(min(15.0,2.0**attempt))
    raise last


def select_groups(examples,test_systems,rank_scores,round_no,n_groups,seed):
    # Group current examples by held-out system+round.
    groups=defaultdict(list)
    for i,x in enumerate(examples):
        sid=str(x["system_id"])
        if sid in test_systems and int(x["round"])==round_no:
            groups[sid].append((i,x))

    # Stratify systems by true K.
    by_k=defaultdict(list)
    for sid,rows in groups.items():
        rows=sorted(rows,key=lambda ix:int(ix[1]["candidate_index"]))
        if len(rows)<4: continue
        k=len(rows[0][1]["truth_roles"])
        by_k[k].append((sid,rows))

    rng=random.Random(seed)
    for k in by_k: rng.shuffle(by_k[k])

    chosen=[]
    ks=sorted(by_k)
    # Round-robin over K so pilot is balanced.
    ptr={k:0 for k in ks}
    while len(chosen)<n_groups:
        progress=False
        for k in ks:
            if ptr[k] < len(by_k[k]) and len(chosen)<n_groups:
                chosen.append(by_k[k][ptr[k]])
                ptr[k]+=1; progress=True
        if not progress: break

    panel=[]
    for sid,rows in chosen:
        scored=[]
        for i,x in rows:
            ci=int(x["candidate_index"])
            rs=rank_scores.get((sid,round_no,ci),{})
            scored.append({
                "example_index":i,
                "x":x,
                "visible_mi":float(rs.get("visible_mi",0.0)),
                "learned_score":float(rs.get("learned_score",0.0)),
                "old_score":float(rs.get("old_score",x.get("selector_predicted_utility") or 0.0)),
            })

        selected=[]
        def add(z):
            if z["example_index"] not in {a["example_index"] for a in selected}:
                selected.append(z)

        # Span different acquisition preferences, not hindsight utility.
        add(max(scored,key=lambda z:z["visible_mi"]))
        add(max(scored,key=lambda z:z["learned_score"]))
        add(min(scored,key=lambda z:z["visible_mi"]))
        # Fill with median visible-MI candidate(s).
        for z in sorted(scored,key=lambda z:z["visible_mi"])[len(scored)//2:]:
            add(z)
            if len(selected)>=4: break
        for z in scored:
            add(z)
            if len(selected)>=4: break

        panel.append((sid,selected[:4]))
    return panel


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--obs-dir",default="artifacts/observation_model_v1")
    ap.add_argument("--rank-preds",default="artifacts/learned_visible_probe_ranker_v1/test_predictions.jsonl")
    ap.add_argument("--out",default="artifacts/repeat_probe_stability_v1")
    ap.add_argument("--round",type=int,default=1)
    ap.add_argument("--groups",type=int,default=8)
    ap.add_argument("--repeats",type=int,default=3)
    ap.add_argument("--workers",type=int,default=4)
    ap.add_argument("--retries",type=int,default=2)
    ap.add_argument("--model",default=None)
    ap.add_argument("--seed",type=int,default=73)
    args=ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in this batch job.")

    root=Path(args.root); obsdir=Path(args.obs_dir); out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    results_path=out/"fresh_repeats.jsonl"
    manifest_path=out/"panel.json"

    examples,roles,_,_=om.collect_examples(root)
    test_systems=load_test_systems(obsdir)
    rank_scores=load_rank_scores(Path(args.rank_preds))
    panel=select_groups(examples,test_systems,rank_scores,args.round,args.groups,args.seed)

    specs,spec_src,manifest_files=discover_system_specs(Path("artifacts"))
    needed={sid for sid,_ in panel}
    missing=sorted(needed-set(specs))
    if missing:
        raise RuntimeError(
            f"Could not find privileged system specs for {len(missing)} selected sessions: {missing[:8]}. "
            f"Found {len(manifest_files)} manifest files."
        )

    manifest={"setting":vars(args),"systems":[],"calls_planned":0}
    tasks=[]
    for sid,cands in panel:
        m=dict(specs[sid])
        m["split"]="train"
        sysrow={
            "system_id":sid,
            "true_k":len(cands[0]["x"]["truth_roles"]),
            "truth_roles_training_only":cands[0]["x"]["truth_roles"],
            "privileged_manifest_source":spec_src[sid],
            "candidates":[]
        }
        for z in cands:
            x=z["x"]; ci=int(x["candidate_index"])
            crow={
                "example_index":int(z["example_index"]),
                "round":int(x["round"]),
                "candidate_index":ci,
                "query":str(x["query"]),
                "visible_mi":z["visible_mi"],
                "learned_score":z["learned_score"],
                "old_score":z["old_score"],
            }
            sysrow["candidates"].append(crow)
            for rep in range(args.repeats):
                tasks.append((sid,m,crow,rep))
        manifest["systems"].append(sysrow)
    manifest["calls_planned"]=len(tasks)
    manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")

    # Fresh run: overwrite previous partial pilot intentionally only at start.
    if results_path.exists():
        backup=out/f"fresh_repeats.previous.{int(time.time())}.jsonl"
        results_path.rename(backup)
        print("[backup]",backup,flush=True)

    print(f"[pilot] systems={len(panel)} candidates/system=4 repeats={args.repeats} fresh_calls={len(tasks)}",flush=True)
    print("[pilot] round=1 only; old target cache is BYPASSED",flush=True)

    failures=[]
    completed=0
    with ThreadPoolExecutor(max_workers=max(1,args.workers)) as ex:
        futs={}
        for sid,m,c,rep in tasks:
            fut=ex.submit(execute_fresh,m,sid,c["query"],args.model,args.retries)
            futs[fut]=(sid,m,c,rep)
        for fut in as_completed(futs):
            sid,m,c,rep=futs[fut]
            try:
                output,selected,retries_used=fut.result()
                row={
                    **c,
                    "system_id":sid,
                    "repeat_index":rep,
                    "final_output":output,
                    "output":output,
                    "selected_roles_training_only":selected,
                    "truth_roles_training_only":list(map(str,m["roles"])),
                    "execution_status":"ok",
                    "retries_used":retries_used,
                    "fresh_execution":True,
                }
                append_jsonl(results_path,row)
            except Exception as e:
                failures.append({"system_id":sid,"candidate_index":c["candidate_index"],"repeat_index":rep,"error":repr(e)})
                append_jsonl(results_path,{
                    **c,"system_id":sid,"repeat_index":rep,
                    "execution_status":"error","error":repr(e),"fresh_execution":True,
                })
            completed+=1
            if completed%8==0 or completed==len(tasks):
                print(f"[execute] {completed}/{len(tasks)} failures={len(failures)}",flush=True)

    (out/"execute_summary.json").write_text(json.dumps({
        "planned":len(tasks),"completed":completed,"failures":len(failures),
        "failure_examples":failures[:10]
    },indent=2),encoding="utf-8")
    print("[done]",results_path,flush=True)


if __name__=="__main__":
    main()
