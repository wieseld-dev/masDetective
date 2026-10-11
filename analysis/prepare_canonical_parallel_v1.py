from pathlib import Path
from collections import defaultdict, Counter
from copy import deepcopy
import json

BASE = Path("artifacts/adaptive_v11_curriculum_v2")
OUT  = Path("artifacts/canonical_parallel_v1")
OUT.mkdir(parents=True, exist_ok=True)

PROBES = [
("accessibility_review",
 "Review this login form with Email, Password, Remember me, and Sign in fields. What concrete changes would improve keyboard and screen-reader accessibility?"),

("code",
 "Write a Python function deduplicate_records(records) that keeps the first dictionary for each unique 'id', and include two small tests?"),

("customer_support_triage",
 "A customer says they were charged twice, cannot access their account, and need the service today. How should this ticket be triaged and what should the first response say?"),

("data",
 "Given daily signups [120, 125, 119, 123, 310, 128, 126], summarize the pattern and identify any data point that deserves investigation?"),

("database_design",
 "Design a normalized relational schema for customers, orders, order items, and products, including primary and foreign keys?"),

("design",
 "Redesign a mobile checkout flow that currently has five screens so users can complete a purchase with less friction while preserving necessary information?"),

("documentation",
 "Write concise API documentation for a function create_user(email, display_name, send_welcome=True), including parameters, return value, and likely errors?"),

("experiment_design",
 "Design an A/B experiment to test whether a shorter signup form improves completed registrations without reducing user quality?"),

("fact_checking",
 "Evaluate these two claims for internal consistency: 'The service launched in 2022' and 'The company reported three full years of service history in January 2024'. What should be verified?"),

("financial_analysis",
 "Revenue increased from $2.4M to $2.9M while cost of goods increased from $1.4M to $1.9M. Compare gross margin between the two periods and interpret the change?"),

("incident_management",
 "A production API suddenly has 40% errors after a deployment and checkout is affected. Give the immediate incident-response steps, priorities, and ownership?"),

("math",
 "Solve 3x^2 - 11x - 4 = 0 and show enough calculation to verify the roots?"),

("operations",
 "A support team has 14 employees covering 24/7 service with demand peaking from 16:00 to 23:00. Propose a practical staffing approach and operational checks?"),

("planning",
 "Create a four-week implementation plan for launching a small internal analytics dashboard, including milestones, dependencies, and deliverables?"),

("qa_testing",
 "Create a compact but thorough test plan for a password-reset feature, covering happy path, edge cases, failures, and security-relevant behavior?"),

("requirements_analysis",
 "Turn this stakeholder request into clear functional and non-functional requirements: 'We need customers to upload invoices, search them later, and the system should feel fast and secure'?"),

("research",
 "Given three short observations—A users prefer fewer steps, B completion falls on long forms, C optional fields are rarely used—synthesize the evidence and state what further evidence would strengthen the conclusion?"),

("risk_assessment",
 "Create a risk assessment for migrating a customer-facing database during a weekend maintenance window, including likelihood, impact, mitigation, and contingency?"),

("translation",
 "Translate this sentence naturally into Spanish while preserving tone: 'Please send the revised contract by Friday so we can review it before the meeting.'?"),

("writing",
 "Rewrite this message to be concise and professional: 'Hey, just wanted to check because I still don't have the report and we really need it soon, so can you send it ASAP please.'?"),
]

def readj(p):
    out=[]
    with p.open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def sid(r):
    return r.get("system_id") or r.get("session_id")

def truth(r):
    return tuple(sorted(r.get("truth_roles_training_only") or []))

def source_cycles():
    ans=[]
    for rp in sorted(BASE.glob("stage_*/cycle_*/responses_train_round_01.jsonl")):
        cp=rp.parent/"candidates_train_round_01.jsonl"
        if not cp.exists():
            continue

        systems={}
        for r in readj(rp):
            s=sid(r)
            if not s or s in systems:
                continue
            systems[s]={
                "K":len(truth(r)),
                "mode":r.get("mode"),
                "truth":truth(r),
            }

        cycle=rp.parent.name.split("_")[-1]
        manifest=rp.parent.parent/f"manifest_train_cycle_{cycle}.json"
        if manifest.exists():
            ans.append((rp.parent,manifest,systems))
    return ans

sources=source_cycles()

def best_source_for_k(k):
    candidates=[]
    for root,manifest,systems in sources:
        n=sum(v["K"]==k for v in systems.values())
        if n:
            candidates.append((n,root,manifest,systems))
    if not candidates:
        raise RuntimeError(f"No source for K={k}")
    candidates.sort(key=lambda x:x[0],reverse=True)
    return candidates[0]

def balanced_pick(systems, allowed_k, n=12):
    buckets=defaultdict(list)
    for s,v in systems.items():
        if v["K"] in allowed_k:
            buckets[str(v["mode"])].append(s)

    for m in buckets:
        buckets[m].sort()

    modes=sorted(buckets)
    out=[]
    while len(out)<n:
        progressed=False
        for m in modes:
            if buckets[m] and len(out)<n:
                out.append(buckets[m].pop(0))
                progressed=True
        if not progressed:
            break
    return out

def build(name, source_root, manifest, systems, chosen):
    root=OUT/name
    root.mkdir(parents=True,exist_ok=True)

    cand=readj(source_root/"candidates_train_round_01.jsonl")
    tmpl={}
    for r in cand:
        s=r.get("session_id")
        if s and s not in tmpl:
            tmpl[s]=r

    rows=[]
    for s in chosen:
        if s not in tmpl:
            raise RuntimeError(f"missing candidate template {s}")

        for j,(role,q) in enumerate(PROBES):
            x=deepcopy(tmpl[s])

            x["session_id"]=s
            x["split"]="train"
            x["round"]=1
            x["candidate_index"]=j
            x["group_id"]=f"{s}::canonical20"
            x["query"]=q
            x["valid"]=True
            x["invalid_reason"]=""

            # These are metadata only for execution; no policy generation is used.
            x["prompt"]=f"[CANONICAL_PROBE role={role}]"
            x["completion"]=q
            x["raw_completion"]=q
            x["local_attempts"]=1
            x["chosen_for_continuation"]=False

            if "selector_predicted_utility" in x:
                x["selector_predicted_utility"]=0.0
            if "selector_model_samples" in x:
                x["selector_model_samples"]=0
            if "selector_mode" in x:
                x["selector_mode"]="canonical_diagnostic"
            if "selector_epsilon" in x:
                x["selector_epsilon"]=0.0

            rows.append(x)

    with (root/"candidates_train_round_01.jsonl").open("w") as f:
        for x in rows:
            f.write(json.dumps(x)+"\n")

    meta={
        "name":name,
        "source_root":str(source_root),
        "manifest":str(manifest),
        "systems":chosen,
        "system_info":{s:systems[s] for s in chosen},
        "probes":[{"candidate_index":i,"role":r,"query":q}
                  for i,(r,q) in enumerate(PROBES)]
    }

    (root/"meta.json").write_text(json.dumps(meta,indent=2))

    print(
        name,
        "source=",source_root,
        "manifest=",manifest,
        "systems=",len(chosen),
        "K=",Counter(systems[s]["K"] for s in chosen),
        "modes=",Counter(systems[s]["mode"] for s in chosen),
        "rows=",len(rows),
    )


# K=3
_,r3,m3,s3 = best_source_for_k(3)
pick3=balanced_pick(s3,{3},12)
build("k3",r3,m3,s3,pick3)

# K=4
_,r4,m4,s4 = best_source_for_k(4)
pick4=balanced_pick(s4,{4},12)
build("k4",r4,m4,s4,pick4)

# High-K: use Stage 10 cycle 02 explicitly.
rh=BASE/"stage_10/cycle_02"
mh=BASE/"stage_10/manifest_train_cycle_02.json"

sh={}
for r in readj(rh/"responses_train_round_01.jsonl"):
    s=sid(r)
    if s and s not in sh:
        sh[s]={
            "K":len(truth(r)),
            "mode":r.get("mode"),
            "truth":truth(r),
        }

pickh=balanced_pick(sh,{6,7,8},12)
build("high",rh,mh,sh,pickh)

print("\nREADY:",OUT)
