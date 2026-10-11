#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, itertools, json, math, random, re, shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sys

# Standalone V13.4 role universe: no dependency on v11 being a Python package.
ROLE_UNIVERSE = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
    "qa_testing",
    "translation",
    "financial_analysis",
    "database_design",
    "accessibility_review",
    "fact_checking",
    "requirements_analysis",
    "incident_management",
    "experiment_design",
    "risk_assessment",
    "documentation",
    "customer_support_triage",
]

# Fallback descriptions are intentionally simple. During data loading we replace
# them with candidate_description_public / target_description_public whenever
# those descriptions are present in the training artifacts.
ROLE_DESCRIPTIONS = {
    r: r.replace("_", " ")
    for r in ROLE_UNIVERSE
}

HARD_DIAGNOSTIC_ROLES = {"code","planning","database_design","fact_checking","writing"}

PLUS_ROLE_KEYS = ("plus_roles_training_only","plus_roles")
MINUS_ROLE_KEYS = ("minus_roles_training_only","minus_roles")
SHAM_ROLE_KEYS = ("sham_roles_training_only","sham_roles")
PLUS_SEL_KEYS = ("plus_selected_training_only","plus_selected")
MINUS_SEL_KEYS = ("minus_selected_training_only","minus_selected")
SHAM_SEL_KEYS = ("sham_selected_training_only","sham_selected")
HARDNEG_KEYS = (
    "hard_negative_role_training_only","hardneg_training_only",
    "hard_neighbor_role_training_only","hard_negative_role","hardneg","hard_neighbor",
)
GROUP_KEYS = ("source_probe_group_id","probe_group_id","group_id","probe_family_id")
DESC_KEYS = ("candidate_description_public","target_description_public","candidate_description","target_desc")

def dump(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")

def read_jsonl(path):
    rows=[]
    with Path(path).open("r", encoding="utf-8") as f:
        for i,line in enumerate(f,1):
            line=line.strip()
            if not line: continue
            x=json.loads(line)
            if isinstance(x,dict): rows.append(x)
    return rows

def learn_role_descriptions(rows13, rows11):
    desc = dict(ROLE_DESCRIPTIONS)
    for row in list(rows13) + list(rows11):
        # Search recursively because artifacts may wrap branch records.
        stack = [row]
        while stack:
            x = stack.pop()
            if isinstance(x, Mapping):
                target = norm(first(x, ("target_role", "candidate_role")))
                d = first(x, DESC_KEYS)
                if target in desc and d:
                    desc[target] = str(d).strip()
                stack.extend(x.values())
            elif isinstance(x, list):
                stack.extend(x)
    return desc


def first(d, keys, default=None):
    for k in keys:
        if k in d and d[k] not in (None,"",[],{}):
            return d[k]
    return default

def tup(x):
    if x is None: return ()
    if isinstance(x,str): return (x,)
    if isinstance(x,Mapping): return tuple(str(k) for k,v in x.items() if v)
    try: return tuple(str(v) for v in x)
    except TypeError: return (str(x),)

def norm(x): return re.sub(r"\s+"," ",str(x or "")).strip()
def h(x): return hashlib.sha256(norm(x).encode()).hexdigest()[:20]

@dataclass
class BranchCase:
    source:str; row_index:int; group_id:str; target:str; target_desc:str; query:str; pair_id:str
    plus_output:str; minus_output:str; sham_output:str
    plus_roles:tuple[str,...]; minus_roles:tuple[str,...]; sham_roles:tuple[str,...]
    plus_selected:tuple[str,...]; minus_selected:tuple[str,...]; sham_selected:tuple[str,...]
    hardneg:str|None; quality_weight:float; component:str=""

def looks_branch(d):
    ks=set(d)
    return bool({"plus_output","minus_output","sham_output"} & ks) and (
        "pair_id" in ks or any(k in ks for k in PLUS_ROLE_KEYS+MINUS_ROLE_KEYS+SHAM_ROLE_KEYS)
    )

def find_branches(obj):
    if isinstance(obj,Mapping):
        if looks_branch(obj):
            yield dict(obj); return
        for v in obj.values(): yield from find_branches(v)
    elif isinstance(obj,list):
        for v in obj: yield from find_branches(v)

def normalize_rows(rows, source, role_desc, source_weight):
    out=[]; c=Counter(); known=set(role_desc)
    for ri,row in enumerate(rows):
        bs=list(find_branches(row))
        if not bs:
            c["rows_without_branch_dict"]+=1; continue
        for bi,b in enumerate(bs):
            target=norm(first(b,("target_role","candidate_role")) or first(row,("target_role","candidate_role")))
            query=str(first(b,("query","probe","prompt")) or first(row,("query","probe","prompt")) or "").strip()
            if not target or not query:
                c["missing_target_or_query"]+=1; continue
            if target not in known:
                c["unknown_target"]+=1; continue
            desc=str(first(b,DESC_KEYS) or first(row,DESC_KEYS) or role_desc.get(target,target)).strip()
            pair_id=norm(first(b,("pair_id","counterfactual_pair_id")) or f"{source}:{ri}:{bi}")
            group_id=str(first(b,GROUP_KEYS) or first(row,GROUP_KEYS) or f"{source}:row:{ri}:target:{target}")

            po=str(b.get("plus_output") or "").strip()
            mo=str(b.get("minus_output") or "").strip()
            so=str(b.get("sham_output") or "").strip()
            pr=tup(first(b,PLUS_ROLE_KEYS)); mr=tup(first(b,MINUS_ROLE_KEYS)); sr=tup(first(b,SHAM_ROLE_KEYS))
            ps=tup(first(b,PLUS_SEL_KEYS)); ms=tup(first(b,MINUS_SEL_KEYS)); ss=tup(first(b,SHAM_SEL_KEYS))

            # Composition truth is mandatory. Never silently replace it with selected roles.
            if po and not pr: c["plus_output_missing_composition"]+=1; po=""
            if mo and not mr: c["minus_output_missing_composition"]+=1; mo=""
            if so and not sr: c["sham_output_missing_composition"]+=1; so=""
            if not any((po,mo,so)):
                c["no_usable_composition_labeled_outputs"]+=1; continue

            unknown=(set(pr)|set(mr)|set(sr))-known
            if unknown:
                raise RuntimeError(f"unknown roles {sorted(unknown)} in {source} row={ri} pair={pair_id}")

            hard=norm(first(b,HARDNEG_KEYS) or first(row,HARDNEG_KEYS)) or None
            qw=source_weight
            for d in (b,row):
                q=first(d,("quality_weight","training_weight"))
                if q is not None:
                    try: qw=max(.05,float(q))*source_weight
                    except: pass
                    break

            out.append(BranchCase(
                source,ri,group_id,target,desc,query,pair_id,
                po,mo,so,pr,mr,sr,ps,ms,ss,hard,float(qw)
            ))
            c["cases"]+=1
    return out,c

class DSU:
    def __init__(self,n): self.p=list(range(n)); self.r=[0]*n
    def find(self,x):
        while self.p[x]!=x:
            self.p[x]=self.p[self.p[x]]; x=self.p[x]
        return x
    def union(self,a,b):
        a,b=self.find(a),self.find(b)
        if a==b:return
        if self.r[a]<self.r[b]: a,b=b,a
        self.p[b]=a
        if self.r[a]==self.r[b]: self.r[a]+=1

def attach_components(cases):
    d=DSU(len(cases)); owner={}
    for i,c in enumerate(cases):
        tags=[
            f"g:{c.source}:{c.group_id}",
            f"p:{c.source}:{c.pair_id}",
            f"q:{h(c.query)}",
        ]
        for o in (c.plus_output,c.minus_output,c.sham_output):
            if o: tags.append(f"o:{h(o)}")
        for tag in tags:
            if tag in owner:d.union(i,owner[tag])
            else: owner[tag]=i
    names={}; out=[]
    for i,c in enumerate(cases):
        r=d.find(i)
        names.setdefault(r,f"component_{len(names):04d}")
        out.append(replace(c,component=names[r]))
    return out

def split_cases(cases,args):
    from sklearn.model_selection import StratifiedGroupKFold
    idx=np.arange(len(cases)); y=np.array([c.target for c in cases]); g=np.array([c.component for c in cases])
    splits=list(StratifiedGroupKFold(n_splits=args.n_splits,shuffle=True,random_state=args.seed).split(idx,y,g))
    ti,vi=splits[args.fold]
    tc={cases[i].component for i in ti}; vc={cases[i].component for i in vi}
    if tc&vc: raise RuntimeError("component leakage")
    tr=[c for c in cases if c.component in tc]; va=[c for c in cases if c.component in vc]
    tp={(c.source,c.pair_id) for c in tr}; vp={(c.source,c.pair_id) for c in va}
    tq={norm(c.query) for c in tr}; vq={norm(c.query) for c in va}
    if tp&vp: raise RuntimeError("pair leakage")
    if tq&vq: raise RuntimeError("exact-query leakage")
    return tr,va,{
        "train_cases":len(tr),"val_cases":len(va),
        "train_components":sorted(tc),"val_components":sorted(vc),
        "component_overlap":[],"pair_overlap":[],"exact_query_overlap_count":0,
    }

@dataclass(frozen=True)
class PresenceExample:
    text:str; label:int; role:str; probe_target:str; branch_type:str
    source:str; component:str; pair_id:str; base_weight:float
    selected_flag:int; self_target_absent:int; hard_neighbor_absent:int

@dataclass(frozen=True)
class CounterfactualPair:
    pos_text:str; neg_text:str; role:str; pair_type:str
    source:str; component:str; pair_id:str; weight:float

def presence_text(role,desc,target,query,answer):
    return (
        "[TASK]\n"
        "Determine whether the candidate specialist actually PARTICIPATED in producing "
        "the visible response to this specific benign probe. "
        "Do NOT infer participation merely because the query or answer discusses the "
        "candidate topic. A different general-purpose specialist may answer a question "
        "about that topic. Semantic topic match alone is NOT evidence of activation.\n\n"
        f"[CANDIDATE ROLE]\n{role}\n\n"
        f"[PUBLIC ROLE DESCRIPTION]\n{desc}\n\n"
        f"[PROBE TARGET]\n{target}\n\n"
        f"[BENIGN PROBE]\n{query}\n\n"
        f"[VISIBLE MAS RESPONSE]\n{answer}\n\n"
        "[DECISION]\n"
        "Did the candidate role actually participate in producing this response?"
    )

def iter_branches(c):
    yield "plus",c.plus_output,c.plus_roles,c.plus_selected
    yield "minus",c.minus_output,c.minus_roles,c.minus_selected
    yield "sham",c.sham_output,c.sham_roles,c.sham_selected

def build_presence(cases,role_desc):
    out=[]; cnt=Counter(); roles=list(role_desc)
    for c in cases:
        for bt,o,rt,st in iter_branches(c):
            if not o:continue
            comp=set(rt); sel=set(st)
            for r in roles:
                y=int(r in sel)
                sa=int(r==c.target and y==0)
                ha=int(bool(c.hardneg) and r==c.hardneg and y==0)
                out.append(PresenceExample(
                    presence_text(r,role_desc[r],c.target,c.query,o),y,r,c.target,bt,
                    c.source,c.component,c.pair_id,c.quality_weight,int(r in sel),sa,ha
                ))
                cnt[f"label:{y}"]+=1; cnt[f"role:{r}:label:{y}"]+=1
                cnt[f"branch:{bt}:label:{y}"]+=1; cnt[f"source:{c.source}:label:{y}"]+=1
                cnt["self_target_absent"]+=sa; cnt["hard_neighbor_absent"]+=ha
                cnt["self_target_present"]+=int(r==c.target and y==1)
                cnt["positive_selected"]+=int(y==1 and r in sel)
                cnt["positive_present_not_selected"]+=int(y==1 and r not in sel)
    return out,cnt

def sample_train(examples,ratio,seed):
    pos=[e for e in examples if e.label==1]
    mandatory=[e for e in examples if e.label==0 and (e.self_target_absent or e.hard_neighbor_absent)]
    optional=[e for e in examples if e.label==0 and not (e.self_target_absent or e.hard_neighbor_absent)]
    target=max(len(mandatory),round(ratio*len(pos))); need=max(0,target-len(mandatory))
    rng=random.Random(seed); by=defaultdict(list)
    for e in optional:by[e.role].append(e)
    for r in by:rng.shuffle(by[r])
    kept_role=Counter(e.role for e in mandatory); active={r for r,x in by.items() if x}; samp=[]
    while active and len(samp)<need:
        r=min(active,key=lambda z:(kept_role[z],z))
        samp.append(by[r].pop()); kept_role[r]+=1
        if not by[r]:active.remove(r)
    kept=pos+mandatory+samp; rng.shuffle(kept)
    return kept,{
        "raw_total":len(examples),"raw_positive":len(pos),"raw_negative":len(mandatory)+len(optional),
        "mandatory_negative":len(mandatory),"self_target_absent":sum(e.self_target_absent for e in mandatory),
        "hard_neighbor_absent":sum(e.hard_neighbor_absent for e in mandatory),
        "optional_negative_available":len(optional),"optional_negative_retained":len(samp),
        "retained_total":len(kept),"retained_positive":len(pos),
        "retained_negative":len(mandatory)+len(samp),
        "retained_neg_pos_ratio":(len(mandatory)+len(samp))/max(1,len(pos)),
    }

def build_pairs(cases,role_desc):
    roles=list(role_desc); out=[]; cnt=Counter()
    for c in cases:
        comps=[
            ("plus_minus",c.plus_output,set(c.plus_selected),c.minus_output,set(c.minus_selected)),
            ("sham_minus",c.sham_output,set(c.sham_selected),c.minus_output,set(c.minus_selected)),
        ]
        for pt,ao,ar,bo,br in comps:
            if not ao or not bo:continue
            for r in roles:
                a=r in ar; b=r in br
                if a==b:continue
                po,no=(ao,bo) if a else (bo,ao)
                out.append(CounterfactualPair(
                    presence_text(r,role_desc[r],c.target,c.query,po),
                    presence_text(r,role_desc[r],c.target,c.query,no),
                    r,pt,c.source,c.component,c.pair_id,c.quality_weight
                ))
                cnt[pt]+=1; cnt[f"role:{r}"]+=1
    return out,cnt

def summary(examples):
    c=Counter()
    for e in examples:
        c["total"]+=1;c[f"label:{e.label}"]+=1;c[f"role:{e.role}:label:{e.label}"]+=1
        c["self_target_absent"]+=e.self_target_absent;c["hard_neighbor_absent"]+=e.hard_neighbor_absent
        c["self_target_present"]+=int(e.role==e.probe_target and e.label==1)
    return c

def train_weights(examples,self_w,hard_w):
    n=len(examples); lc=Counter(e.label for e in examples); rc=Counter(e.role for e in examples)
    cw={y:n/(2*max(1,lc[y])) for y in (0,1)}; meanr=float(np.mean(list(rc.values())))
    ws=[]
    for e in examples:
        w=e.base_weight*cw[e.label]*(meanr/max(1,rc[e.role]))
        if e.self_target_absent:w*=self_w
        if e.hard_neighbor_absent:w*=hard_w
        ws.append(w)
    m=max(float(np.mean(ws)),1e-8)
    return [w/m for w in ws]

class PresenceDataset:
    def __init__(self,e,w):self.e=list(e);self.w=list(w)
    def __len__(self):return len(self.e)
    def __getitem__(self,i):return self.e[i],self.w[i]

class PairDataset:
    def __init__(self,p):self.p=list(p)
    def __len__(self):return len(self.p)
    def __getitem__(self,i):return self.p[i]

def collate_presence(batch):
    import torch
    e,w=zip(*batch)
    return {
        "text":[x.text for x in e],
        "label":torch.tensor([x.label for x in e],dtype=torch.float32),
        "weight":torch.tensor(w,dtype=torch.float32),
        "role":[x.role for x in e],"probe_target":[x.probe_target for x in e],
    }

def collate_pairs(batch):
    import torch
    return {
        "pos_text":[x.pos_text for x in batch],"neg_text":[x.neg_text for x in batch],
        "weight":torch.tensor([x.weight for x in batch],dtype=torch.float32),
        "pair_type":[x.pair_type for x in batch],"role":[x.role for x in batch],
    }

def make_model(args):
    import torch
    from transformers import AutoModelForSequenceClassification,AutoTokenizer,BitsAndBytesConfig
    from peft import LoraConfig,TaskType,get_peft_model,prepare_model_for_kbit_training
    tok=AutoTokenizer.from_pretrained(args.base_model,use_fast=True)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    tok.padding_side="right"
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    base=AutoModelForSequenceClassification.from_pretrained(
        args.base_model,num_labels=1,quantization_config=bnb,device_map={"":0},torch_dtype=dtype
    )
    base.config.pad_token_id=tok.pad_token_id;base.config.use_cache=False
    base=prepare_model_for_kbit_training(base,use_gradient_checkpointing=True)
    cfg=LoraConfig(r=args.lora_r,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,bias="none",
        task_type=TaskType.SEQ_CLS,target_modules=["q_proj","k_proj","v_proj","o_proj"],modules_to_save=["score"])
    model=get_peft_model(base,cfg)
    print("[init] clean base + fresh 1-logit presence head + fresh LoRA; V13.2 adapter/calibrator NOT loaded",flush=True)
    model.print_trainable_parameters()
    return tok,model,dtype

def logits(model,tok,texts,maxlen,device):
    x=tok(texts,padding=True,truncation=True,max_length=maxlen,return_tensors="pt").to(device)
    return model(**x).logits[:,0].float()

def safe_auc(y,s):
    from sklearn.metrics import roc_auc_score
    y=np.asarray(y,dtype=int);s=np.asarray(s,dtype=float)
    return None if len(np.unique(y))<2 else float(roc_auc_score(y,s))

def eval_presence(model,tok,examples,args,device,dtype):
    import torch
    from torch.utils.data import DataLoader
    dl=DataLoader(PresenceDataset(examples,[1.]*len(examples)),batch_size=args.eval_batch_size,shuffle=False,collate_fn=collate_presence)
    ys=[];zs=[];rs=[];ts=[];model.eval()
    with torch.no_grad():
        for b in dl:
            with torch.autocast(device_type="cuda",dtype=dtype,enabled=True):
                z=logits(model,tok,b["text"],args.max_length,device)
            ys+=b["label"].cpu().numpy().astype(int).tolist();zs+=z.cpu().numpy().tolist()
            rs+=b["role"];ts+=b["probe_target"]
    y=np.array(ys);z=np.array(zs);p=1/(1+np.exp(-np.clip(z,-50,50)));pred=(p>=.5).astype(int)
    pos=y==1;neg=y==0
    rec=float(np.mean(pred[pos]==1));spec=float(np.mean(pred[neg]==0))
    per={};aucs=[]
    for r in sorted(ROLE_DESCRIPTIONS):
        idx=np.array([x==r for x in rs]); ry=y[idx];rz=z[idx];a=safe_auc(ry,rz)
        if a is not None:aucs.append(a)
        per[r]={"auc":a,"n":int(idx.sum()),"positive_n":int((ry==1).sum()),"negative_n":int((ry==0).sum())}
    si=np.array([r==t for r,t in zip(rs,ts)]); sf=si&(y==0)
    return {
        "presence_micro_auc":safe_auc(y,z),"presence_macro_auc":float(np.mean(aucs)) if aucs else None,
        "balanced_accuracy_at_0_5":.5*(rec+spec),"positive_recall_at_0_5":rec,
        "negative_specificity_at_0_5":spec,"self_target_present_vs_absent_auc":safe_auc(y[si],z[si]),
        "self_target_absent_fpr_at_0_5":float(np.mean(pred[sf]==1)) if sf.any() else None,
        "n_examples":len(y),"positive_n":int(pos.sum()),"negative_n":int(neg.sum()),
    },per

def eval_pairs(model,tok,pairs,args,device,dtype):
    import torch
    from torch.utils.data import DataLoader
    if not pairs:return {"counterfactual_accuracy":None,"plus_minus_accuracy":None,"sham_minus_accuracy":None,"n_pairs":0}
    dl=DataLoader(PairDataset(pairs),batch_size=args.eval_pair_batch_size,shuffle=False,collate_fn=collate_pairs)
    ms=[];types=[];model.eval()
    with torch.no_grad():
        for b in dl:
            with torch.autocast(device_type="cuda",dtype=dtype,enabled=True):
                z=logits(model,tok,b["pos_text"]+b["neg_text"],args.max_length,device)
            n=len(b["pos_text"]);ms+=(z[:n]-z[n:]).cpu().numpy().tolist();types+=b["pair_type"]
    m=np.array(ms)
    def acc(t):
        idx=np.array([x==t for x in types]);return float(np.mean(m[idx]>0)) if idx.any() else None
    return {"counterfactual_accuracy":float(np.mean(m>0)),"plus_minus_accuracy":acc("plus_minus"),
        "sham_minus_accuracy":acc("sham_minus"),"mean_counterfactual_margin":float(np.mean(m)),"n_pairs":len(pairs)}

def select_score(m):
    ma=.5 if m["presence_macro_auc"] is None else m["presence_macro_auc"]
    sa=.5 if m["self_target_present_vs_absent_auc"] is None else m["self_target_present_vs_absent_auc"]
    rej=0 if m["self_target_absent_fpr_at_0_5"] is None else 1-m["self_target_absent_fpr_at_0_5"]
    cf=.5 if m["counterfactual_accuracy"] is None else m["counterfactual_accuracy"]
    return .45*ma+.25*sa+.20*rej+.10*cf

def train(tr,va,tp,vp,args,out):
    import torch,torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup
    if not torch.cuda.is_available():raise RuntimeError("CUDA required; use --audit-only on login node")
    dev=torch.device("cuda:0");tok,model,dtype=make_model(args)
    ws=train_weights(tr,args.self_target_neg_weight,args.hardneg_weight)
    dl=DataLoader(PresenceDataset(tr,ws),batch_size=args.batch_size,shuffle=True,collate_fn=collate_presence)
    pdl=DataLoader(PairDataset(tp),batch_size=args.pair_batch_size,shuffle=True,collate_fn=collate_pairs) if tp else None
    params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=args.lr,weight_decay=.01)
    steps=math.ceil(len(dl)/args.grad_accum)*args.epochs;warm=max(10,int(.05*steps))
    sch=get_cosine_schedule_with_warmup(opt,warm,max(1,steps))
    scaler=torch.amp.GradScaler("cuda",enabled=dtype==torch.float16)
    best=-1;bad=0;hist=[]
    for ep in range(1,args.epochs+1):
        model.train();opt.zero_grad(set_to_none=True);pit=itertools.cycle(pdl) if pdl else None
        run=rb=rp=0.;seen=0
        for step,b in enumerate(dl,1):
            y=b["label"].to(dev);w=b["weight"].to(dev);pb=next(pit) if pit else None
            with torch.autocast(device_type="cuda",dtype=dtype,enabled=True):
                z=logits(model,tok,b["text"],args.max_length,dev)
                be=F.binary_cross_entropy_with_logits(z,y,reduction="none");bce=(be*w).sum()/w.sum().clamp_min(1e-8)
                pl=torch.zeros((),device=dev,dtype=bce.dtype)
                if pb:
                    pw=pb["weight"].to(dev);q=logits(model,tok,pb["pos_text"]+pb["neg_text"],args.max_length,dev);n=len(pb["pos_text"])
                    pe=F.softplus(args.pair_margin-(q[:n]-q[n:]));pl=(pe*pw).sum()/pw.sum().clamp_min(1e-8)
                loss=bce+args.pair_loss_weight*pl
            scaler.scale(loss/args.grad_accum).backward();bs=len(b["text"])
            run+=float(loss.detach().cpu())*bs;rb+=float(bce.detach().cpu())*bs;rp+=float(pl.detach().cpu())*bs;seen+=bs
            if step%args.grad_accum==0 or step==len(dl):
                scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(params,1.0);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);sch.step()
        pm,per=eval_presence(model,tok,va,args,dev,dtype);qm=eval_pairs(model,tok,vp,args,dev,dtype);m={**pm,**qm,"epoch":ep}
        m["train_loss"]=run/seen;m["train_bce_loss"]=rb/seen;m["train_pair_loss"]=rp/seen;m["selection_score"]=select_score(m);hist.append(m)
        f=lambda x:"NA" if x is None else f"{x:.4f}"
        print(f"epoch={ep:02d} train={m['train_loss']:.4f} bce={m['train_bce_loss']:.4f} pair={m['train_pair_loss']:.4f} "
              f"micro_auc={f(pm['presence_micro_auc'])} macro_auc={f(pm['presence_macro_auc'])} "
              f"bal_acc={f(pm['balanced_accuracy_at_0_5'])} self_auc={f(pm['self_target_present_vs_absent_auc'])} "
              f"self_abs_fpr={f(pm['self_target_absent_fpr_at_0_5'])} cf_acc={f(qm['counterfactual_accuracy'])} select={m['selection_score']:.4f}",flush=True)
        print("[hard-roles] "+" ".join(f"{r}={f(per.get(r,{}).get('auc'))}" for r in sorted(HARD_DIAGNOSTIC_ROLES)),flush=True)
        last=out/"last_adapter";shutil.rmtree(last,ignore_errors=True);model.save_pretrained(last);tok.save_pretrained(last);dump(out/"training_history.json",hist)
        if m["selection_score"]>best+args.min_delta:
            best=m["selection_score"];bad=0;bd=out/"best_adapter";shutil.rmtree(bd,ignore_errors=True);model.save_pretrained(bd);tok.save_pretrained(bd)
            dump(out/"validation_metrics.json",m);dump(out/"per_role_metrics.json",per);print(f"[best] epoch={ep} score={best:.4f}",flush=True)
        else:
            bad+=1
            if bad>=args.early_stop_patience:print(f"[early-stop] epoch={ep} best={best:.4f}",flush=True);break

def parse_args():
    a=argparse.ArgumentParser()
    a.add_argument("--v13-discovery",default="artifacts/v13_probe_disambiguation.jsonl")
    a.add_argument("--v11-discovery",default="artifacts/v11_probe_discovery.jsonl")
    a.add_argument("--output-dir",default="artifacts/v13_6_activation_judge")
    a.add_argument("--audit-only",action="store_true")
    a.add_argument("--seed",type=int,default=1331);a.add_argument("--n-splits",type=int,default=5);a.add_argument("--fold",type=int,default=0)
    a.add_argument("--v11-source-weight",type=float,default=.70);a.add_argument("--neg-ratio",type=float,default=2.0)
    a.add_argument("--self-target-neg-weight",type=float,default=2.5);a.add_argument("--hardneg-weight",type=float,default=1.5)
    a.add_argument("--base-model",default="Qwen/Qwen2.5-1.5B-Instruct");a.add_argument("--lora-r",type=int,default=16)
    a.add_argument("--lora-alpha",type=int,default=32);a.add_argument("--lora-dropout",type=float,default=.05)
    a.add_argument("--max-length",type=int,default=1024);a.add_argument("--batch-size",type=int,default=4);a.add_argument("--pair-batch-size",type=int,default=2)
    a.add_argument("--eval-batch-size",type=int,default=8);a.add_argument("--eval-pair-batch-size",type=int,default=4);a.add_argument("--grad-accum",type=int,default=4)
    a.add_argument("--epochs",type=int,default=4);a.add_argument("--lr",type=float,default=1e-4);a.add_argument("--pair-loss-weight",type=float,default=.25)
    a.add_argument("--pair-margin",type=float,default=.75);a.add_argument("--early-stop-patience",type=int,default=2);a.add_argument("--min-delta",type=float,default=1e-4)
    return a.parse_args()


# ============================================================================
# V13.4 V13.2-LOADER OVERRIDE
# ============================================================================
# Reuse the already validated V13.2 artifact parser and leakage-safe DSU.
#
# This fixes two problems seen in the first V13.4 audit:
#   - V11 was parsed as 0 cases.
#   - hard_negative_replacement_training_only was not recognized.
#
# The V13.4 objective/dataset/training code remains unchanged.
# ============================================================================

# Ensure repository root is on sys.path when executing analysis/*.py directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import analysis.train_v13_2_dual_judge as v132


# Replace the local fallback role descriptions with the canonical universe used
# by V13.2 / V11.
_ALL_ROLE_DESCRIPTIONS = v132.load_role_descriptions()

# V13.4 deployment/evaluation universe is the fixed 20-role universe used by
# V13.3/V12 cross-role inference, not the larger 26-role V11 catalog.
ROLE_DESCRIPTIONS = {
    role: _ALL_ROLE_DESCRIPTIONS[role]
    for role in ROLE_UNIVERSE
}

if len(ROLE_DESCRIPTIONS) != 20:
    raise RuntimeError(
        f"Expected exactly 20 V13.4 candidate roles, got {len(ROLE_DESCRIPTIONS)}"
    )


def read_jsonl(path):
    """Use V13.2's validated JSONL reader."""
    return v132.read_jsonl(path)


def normalize_rows(rows, source, role_desc, source_weight):
    """
    Use the EXACT V13.2 discovery normalization.

    Important fields recovered here include:
      branches_training_only
      plus/minus/sham_roles_training_only
      plus/minus/sham_selected
      hard_negative_replacement_training_only
      quality weighting / specificity filtering
    """

    # Preserve the strong V13 public/truth integrity check when its artifacts
    # are available. This does not use V12 and performs no API calls.
    if source == "v13":
        public_path = Path("artifacts/v13_train_response_public.jsonl")
        truth_path = Path("artifacts/v13_train_response_truth.jsonl")

        if public_path.exists() and truth_path.exists():
            report = v132.audit_v13(
                rows,
                public_path,
                truth_path,
            )
            print(
                "[v13-integrity] "
                + json.dumps(report, sort_keys=True),
                flush=True,
            )
        else:
            print(
                "[v13-integrity] skipped: public/truth audit files not found",
                flush=True,
            )

    return v132.normalize_discovery_rows(
        rows=rows,
        source=source,
        role_desc=role_desc,
        source_weight=source_weight,
    )


def attach_components(cases):
    """
    Use V13.2's leakage-safe connected components:
      source/group + pair + exact query + exact duplicate outputs.
    """
    return v132.attach_components(cases)

# ============================================================================
# END V13.2-LOADER OVERRIDE
# ============================================================================

def main():
    args=parse_args();random.seed(args.seed);np.random.seed(args.seed)
    for p in (args.v13_discovery,args.v11_discovery):
        if "v12" in p.lower():raise RuntimeError(f"PROTOCOL VIOLATION: V12 training path: {p}")
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);rd=dict(ROLE_DESCRIPTIONS)
    print("="*100);print("V13.6 QUERY-CONDITIONED ALL-ROLE ACTIVATION JUDGE");print("="*100)
    print("[protocol] V12 training use: FORBIDDEN");print("[protocol] label = role in selected_roles for this exact response")
    r13=read_jsonl(args.v13_discovery);r11=read_jsonl(args.v11_discovery)
    c13,p13=normalize_rows(r13,"v13",rd,1.0);c11,p11=normalize_rows(r11,"v11",rd,args.v11_source_weight)
    cases=attach_components(c13+c11)
    if not cases:raise RuntimeError("No usable BranchCases")
    trc,vac,srep=split_cases(cases,args);tra,traw=build_presence(trc,rd);vaa,vraw=build_presence(vac,rd)
    tr,sampling=sample_train(tra,args.neg_ratio,args.seed);tp,tpc=build_pairs(trc,rd);vp,vpc=build_pairs(vac,rd)
    ts=summary(tr)
    if ts["self_target_absent"]<=0:raise RuntimeError("AUDIT FAIL: zero self-target absent negatives")
    miss=[r for r in rd if ts[f"role:{r}:label:0"]==0]
    if miss:raise RuntimeError("AUDIT FAIL no negatives for: "+",".join(miss))
    audit={"protocol":"v13.6_query_conditioned_all_role_activation","v12_used_for_training":False,"role_universe_size":len(rd),
           "total_cases":len(cases),"v13_cases":len(c13),"v11_cases":len(c11),"parse_v13":dict(p13),"parse_v11":dict(p11),
           "split_report":srep,"train_raw_counts":dict(traw),"val_raw_counts":dict(vraw),"train_sampling":sampling,
           "train_counterfactual_pairs":len(tp),"val_counterfactual_pairs":len(vp),"train_pair_counts":dict(tpc),"val_pair_counts":dict(vpc)}
    dump(out/"split_report.json",srep);dump(out/"data_audit.json",audit)
    print(f"[data] cases={len(cases)} v13={len(c13)} v11={len(c11)}")
    print(f"[split] train={len(trc)} val={len(vac)} train_components={len(srep['train_components'])} val_components={len(srep['val_components'])} overlap=0")
    print(f"[raw-train] total={len(tra)} pos={traw['label:1']} neg={traw['label:0']} self_target_absent={traw['self_target_absent']} hard_neighbor_absent={traw['hard_neighbor_absent']}")
    print("[retained-train] "+json.dumps(sampling,sort_keys=True))
    print(f"[val-full] total={len(vaa)} pos={vraw['label:1']} neg={vraw['label:0']} self_target_absent={vraw['self_target_absent']}")
    print(f"[counterfactual] train={len(tp)} val={len(vp)}")
    print("[per-role-supervision]")
    for r in sorted(rd):print(f"  {r:25s} pos={ts[f'role:{r}:label:1']:5d} neg={ts[f'role:{r}:label:0']:5d}")
    if args.audit_only:
        print("AUDIT-ONLY PASS: no model loaded, no CUDA, no API calls.");return
    dump(out/"judge_config.json",{**vars(args),"protocol":"v13.6_query_conditioned_all_role_activation","v12_used_for_training":False,"init":"clean base + fresh LoRA; activation supervision"})
    train(tr,vaa,tp,vp,args,out)

if __name__=="__main__":main()
