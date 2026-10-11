#!/usr/bin/env python3
"""
observation_model_v1.py

Stage-1 diagnostic for the black-box MAS active-probing project.

Learn:
    P(Y | Z, q, b_t)

Z   = hidden 20-role MAS map (training/evaluation only)
q   = candidate benign probe
b_t = current belief state (role_probs + k_probs)
Y   = roles actually selected/exposed by the MAS

This is intentionally an ORACLE-Z upper-bound diagnostic.  At deployment Z is
unknown; if this works, the next step is to integrate the observation model
over posterior hypotheses Z~P_t(Z) and score probes by expected information
gain.

Data:
  artifacts/adaptive_v11_curriculum_v2/**/
      states_train_round_*.jsonl
      responses_train_round_*.jsonl

The split is grouped by system_id to prevent row-level leakage.
"""

from __future__ import annotations
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from sklearn.metrics import f1_score, roc_auc_score, hamming_loss, average_precision_score
except Exception:
    f1_score = roc_auc_score = hamming_loss = average_precision_score = None

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_chosen(v):
    if v is True or v == 1:
        return True
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return False


def read_jsonl(path):
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def parse_round(path):
    m = re.search(r"round_(\d+)", Path(path).name)
    if not m:
        raise ValueError(f"cannot parse round from {path}")
    return int(m.group(1))


def load_parent(parent):
    parent = Path(parent)
    states = {}
    for p in sorted(parent.glob("states_train_round_*.jsonl")):
        fr = parse_round(p)
        for x in read_jsonl(p):
            sid = x.get("session_id") or x.get("system_id")
            if sid:
                states[(sid, int(x.get("round_index", fr)))] = x

    responses = []
    for p in sorted(parent.glob("responses_train_round_*.jsonl")):
        fr = parse_round(p)
        for x in read_jsonl(p):
            sid = x.get("system_id") or x.get("session_id")
            if not sid:
                continue
            xx = dict(x)
            xx["_sid"] = sid
            xx["_round"] = int(x.get("round", fr))
            xx["_parent"] = str(parent)
            responses.append(xx)
    return states, responses


def collect_examples(root):
    root = Path(root)
    parents = sorted({p.parent for p in root.rglob("responses_train_round_*.jsonl")})
    examples = []
    role_order = None
    skipped = defaultdict(int)

    for parent in parents:
        states, responses = load_parent(parent)
        for x in responses:
            sid, rnd = x["_sid"], x["_round"]
            state = states.get((sid, rnd - 1)) or states.get((sid, rnd))
            if state is None:
                skipped["missing_state"] += 1
                continue

            role_probs = state.get("role_probs") or {}
            k_probs = state.get("k_probs") or {}
            query = x.get("query") or x.get("completion")
            truth = x.get("truth_roles_training_only") or []
            selected = x.get("selected_roles_training_only") or []

            if not role_probs or not k_probs:
                skipped["missing_belief"] += 1
                continue
            if not query:
                skipped["missing_query"] += 1
                continue
            if not truth:
                skipped["missing_truth"] += 1
                continue

            roles = list(role_probs.keys())
            if role_order is None:
                role_order = roles
            elif set(roles) != set(role_order):
                skipped["role_set_mismatch"] += 1
                continue

            role_set = set(role_order)
            if not set(truth).issubset(role_set) or not set(selected).issubset(role_set):
                skipped["unknown_role"] += 1
                continue

            examples.append({
                "parent": str(parent),
                "system_id": sid,
                "split_key": sid,
                "round": rnd,
                "group_id": x.get("group_id") or f"{parent}::{sid}::r{rnd:02d}",
                "candidate_index": int(x.get("candidate_index") or 0),
                "query": str(query),
                "truth_roles": tuple(sorted(truth)),
                "selected_roles": tuple(sorted(selected)),
                "chosen": is_chosen(x.get("chosen_for_continuation")),
                "selector_predicted_utility": x.get("selector_predicted_utility"),
                "role_probs": {r: float(role_probs[r]) for r in role_order},
                "k_probs": {str(k): float(v) for k, v in k_probs.items()},
            })

    if role_order is None:
        raise RuntimeError("No usable examples.")
    return examples, role_order, parents, dict(skipped)


def split_systems(examples, val_frac, test_frac, seed):
    systems = sorted({x["split_key"] for x in examples})
    rng = random.Random(seed)
    rng.shuffle(systems)
    n = len(systems)
    nt = max(1, int(round(n * test_frac)))
    nv = max(1, int(round(n * val_frac)))
    test = set(systems[:nt])
    val = set(systems[nt:nt+nv])
    train = set(systems[nt+nv:])
    return train, val, test


def make_arrays(examples, roles):
    r2i = {r:i for i,r in enumerate(roles)}
    max_round = max(x["round"] for x in examples)
    Z, B, K, R, Y = [], [], [], [], []

    for x in examples:
        z = np.zeros(len(roles), np.float32)
        y = np.zeros(len(roles), np.float32)
        for r in x["truth_roles"]:
            z[r2i[r]] = 1
        for r in x["selected_roles"]:
            y[r2i[r]] = 1
        b = np.array([x["role_probs"][r] for r in roles], np.float32)
        kp = np.array([x["k_probs"].get(str(k), 0.0) for k in [3,4,5,6]], np.float32)
        rr = np.array([x["round"]/max(max_round,1)], np.float32)
        Z.append(z); B.append(b); K.append(kp); R.append(rr); Y.append(y)

    return tuple(np.stack(a) for a in [Z,B,K,R,Y])


class ObsDS(Dataset):
    def __init__(self, idx, Q, Z, B, K, R, Y):
        self.idx = np.asarray(idx, np.int64)
        self.Q, self.Z, self.B, self.K, self.R, self.Y = Q,Z,B,K,R,Y
    def __len__(self): return len(self.idx)
    def __getitem__(self, j):
        i = int(self.idx[j])
        return (
            torch.from_numpy(self.Q[i]),
            torch.from_numpy(self.Z[i]),
            torch.from_numpy(self.B[i]),
            torch.from_numpy(self.K[i]),
            torch.from_numpy(self.R[i]),
            torch.from_numpy(self.Y[i]),
            i,
        )


class ObservationNet(nn.Module):
    def __init__(self, qdim, n_roles, hidden=512, dropout=.2):
        super().__init__()
        in_dim = qdim + 4*n_roles + 4 + 1
        self.body = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, n_roles),
        )
    def forward(self, q, z, b, k, r):
        x = torch.cat([q, z, b, z*b, b*(1-b), k, r], dim=-1)
        logits = self.body(x)
        # Y must be a subset of Z.
        return logits.masked_fill(z < .5, -12.0)


def encode_queries(texts, model_name, device, batch):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    enc = SentenceTransformer(model_name, device=device)
    return enc.encode(
        texts, batch_size=batch, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    ).astype(np.float32)


@torch.no_grad()
def predict_all(model, loader, device, n, nr):
    P = np.zeros((n,nr), np.float32)
    Y = np.zeros((n,nr), np.float32)
    model.eval()
    for q,z,b,k,r,y,idx in loader:
        q,z,b,k,r = [t.to(device) for t in [q,z,b,k,r]]
        p = torch.sigmoid(model(q,z,b,k,r)).cpu().numpy()
        ii = idx.numpy()
        P[ii] = p
        Y[ii] = y.numpy()
    return P,Y


def optimize_thresholds(y,p):
    th = np.full(y.shape[1], .5, np.float32)
    if f1_score is None:
        return th
    grid = np.arange(.05,.951,.05)
    for j in range(y.shape[1]):
        if y[:,j].sum() == 0:
            continue
        best = (-1,.5)
        for t in grid:
            f = f1_score(y[:,j], (p[:,j]>=t).astype(int), zero_division=0)
            if f > best[0]:
                best = (f,t)
        th[j] = best[1]
    return th


def multilabel_metrics(y,p,th):
    pred = (p >= th[None,:]).astype(int)
    out = {
        "true_cardinality_mean": float(y.sum(1).mean()),
        "pred_cardinality_mean": float(pred.sum(1).mean()),
        "prob_cardinality_mean": float(p.sum(1).mean()),
        "exact_multilabel_set_accuracy": float(np.mean(np.all(pred==y,axis=1))),
    }
    if f1_score is not None:
        out.update({
            "micro_f1": float(f1_score(y,pred,average="micro",zero_division=0)),
            "macro_f1": float(f1_score(y,pred,average="macro",zero_division=0)),
            "samples_f1": float(f1_score(y,pred,average="samples",zero_division=0)),
            "hamming_loss": float(hamming_loss(y,pred)),
        })
    if roc_auc_score is not None:
        try: out["micro_auroc"] = float(roc_auc_score(y,p,average="micro"))
        except Exception: out["micro_auroc"] = None
        aucs, aps = [], []
        for j in range(y.shape[1]):
            if len(np.unique(y[:,j])) < 2:
                continue
            try: aucs.append(roc_auc_score(y[:,j],p[:,j]))
            except Exception: pass
            if average_precision_score is not None:
                try: aps.append(average_precision_score(y[:,j],p[:,j]))
                except Exception: pass
        out["macro_auroc_valid_roles"] = float(np.mean(aucs)) if aucs else None
        out["macro_ap_valid_roles"] = float(np.mean(aps)) if aps else None
    return out


def safe_spearman(a,b):
    a,b = np.asarray(a,float), np.asarray(b,float)
    if len(a)<2 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return None
    if spearmanr is not None:
        v = spearmanr(a,b).correlation
        return None if np.isnan(v) else float(v)
    ra=np.empty(len(a)); rb=np.empty(len(b))
    ra[np.argsort(a)] = np.arange(len(a))
    rb[np.argsort(b)] = np.arange(len(b))
    v=np.corrcoef(ra,rb)[0,1]
    return None if np.isnan(v) else float(v)


def ranking_eval(examples, probs, test_idx, roles):
    r2i = {r:i for i,r in enumerate(roles)}
    testset = set(map(int,test_idx))
    traj = defaultdict(lambda: defaultdict(list))
    for i,x in enumerate(examples):
        if i in testset:
            traj[(x["parent"],x["system_id"])][x["round"]].append((i,x))

    recs=[]
    blocks={"r1":defaultdict(list),"all":defaultdict(list)}

    def add(block,k,v):
        if v is not None: block[k].append(v)

    for (parent,sid), rm in traj.items():
        seen=set()
        for rnd in sorted(rm):
            rows=rm[rnd]
            if len(rows)<2:
                ch=[x for _,x in rows if x["chosen"]]
                if len(ch)==1: seen.update(ch[0]["selected_roles"])
                continue

            cand=[]
            for i,x in rows:
                unseen=np.ones(len(roles),np.float32)
                for rr in seen:
                    if rr in r2i: unseen[r2i[rr]]=0
                reward=len(set(x["selected_roles"])-seen)
                cand.append({
                    "candidate_index":x["candidate_index"],
                    "query":x["query"],
                    "selected_roles":list(x["selected_roles"]),
                    "oracle_new":reward,
                    "predicted_new":float((probs[i]*unseen).sum()),
                    "selector_score":(
                        float(x["selector_predicted_utility"])
                        if x["selector_predicted_utility"] is not None else None
                    ),
                    "chosen_actual":x["chosen"],
                })

            oracle=max(cand,key=lambda c:(c["oracle_new"],len(c["selected_roles"]),-c["candidate_index"]))
            chosen=max(cand,key=lambda c:(c["predicted_new"],-c["candidate_index"]))
            orw, prw = oracle["oracle_new"], chosen["oracle_new"]
            rho=safe_spearman([c["predicted_new"] for c in cand],[c["oracle_new"] for c in cand])
            random_reward=float(np.mean([c["oracle_new"] for c in cand]))

            sel=[c for c in cand if c["selector_score"] is not None]
            sel_reward=None
            if len(sel)==len(cand):
                ss=max(sel,key=lambda c:(c["selector_score"],-c["candidate_index"]))
                sel_reward=ss["oracle_new"]

            act=[c for c in cand if c["chosen_actual"]]
            act_reward=act[0]["oracle_new"] if len(act)==1 else None

            rec={
                "parent":parent,"system_id":sid,"round":rnd,
                "seen_actual_before":sorted(seen),"pool_size":len(cand),
                "oracle_candidate":oracle["candidate_index"],"oracle_reward":orw,
                "predicted_candidate":chosen["candidate_index"],"predicted_reward":prw,
                "predicted_reward_optimal":int(prw==orw),
                "predicted_exact_oracle_candidate":int(chosen["candidate_index"]==oracle["candidate_index"]),
                "predicted_regret":orw-prw,
                "within_group_spearman":rho,
                "random_expected_reward":random_reward,
                "existing_selector_reward":sel_reward,
                "actual_chosen_reward":act_reward,
                "candidates":cand,
            }
            recs.append(rec)

            for name in ["all"]+(["r1"] if rnd==1 else []):
                b=blocks[name]
                add(b,"oracle",orw); add(b,"pred",prw)
                add(b,"reward_opt",int(prw==orw))
                add(b,"exact_match",int(chosen["candidate_index"]==oracle["candidate_index"]))
                add(b,"regret",orw-prw); add(b,"rho",rho)
                add(b,"random",random_reward); add(b,"selector",sel_reward); add(b,"actual",act_reward)
                if sel_reward is not None: add(b,"selector_opt",int(sel_reward==orw))

            if len(act)==1:
                seen.update(act[0]["selected_roles"])

    def summarize(b):
        def m(k):
            return float(np.mean(b[k])) if b.get(k) else None
        return {
            "groups":len(b.get("oracle",[])),
            "mean_oracle_new":m("oracle"),
            "mean_predicted_new_realized":m("pred"),
            "reward_optimal_hit_rate":m("reward_opt"),
            "exact_oracle_candidate_match_rate":m("exact_match"),
            "mean_regret":m("regret"),
            "mean_within_group_spearman":m("rho"),
            "spearman_groups":len(b.get("rho",[])),
            "random_mean_reward":m("random"),
            "existing_selector_mean_reward":m("selector"),
            "existing_selector_reward_optimal_rate":m("selector_opt"),
            "actual_chosen_mean_reward":m("actual"),
        }
    return recs, {
        "round_1_clean":summarize(blocks["r1"]),
        "all_rounds_teacher_forced_actual_history":summarize(blocks["all"]),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="artifacts/adaptive_v11_curriculum_v2")
    ap.add_argument("--out",default="artifacts/observation_model_v1")
    ap.add_argument("--text-model",default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device",default="auto")
    ap.add_argument("--epochs",type=int,default=35)
    ap.add_argument("--patience",type=int,default=6)
    ap.add_argument("--batch-size",type=int,default=256)
    ap.add_argument("--embed-batch-size",type=int,default=128)
    ap.add_argument("--hidden",type=int,default=512)
    ap.add_argument("--dropout",type=float,default=.2)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--val-frac",type=float,default=.10)
    ap.add_argument("--test-frac",type=float,default=.20)
    ap.add_argument("--seed",type=int,default=7)
    args=ap.parse_args()

    seed_all(args.seed)
    device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)

    print("device:",device)
    examples,roles,parents,skipped=collect_examples(Path(args.root))
    train_sys,val_sys,test_sys=split_systems(examples,args.val_frac,args.test_frac,args.seed)

    tr=[i for i,x in enumerate(examples) if x["split_key"] in train_sys]
    va=[i for i,x in enumerate(examples) if x["split_key"] in val_sys]
    te=[i for i,x in enumerate(examples) if x["split_key"] in test_sys]

    print("parents:",len(parents),"examples:",len(examples),"roles:",len(roles))
    print("systems train/val/test:",len(train_sys),len(val_sys),len(test_sys))
    print("rows train/val/test:",len(tr),len(va),len(te))
    print("skipped:",skipped)

    Z,B,K,R,Y=make_arrays(examples,roles)
    emb_path=out/"query_embeddings.npy"
    meta_path=out/"query_embeddings_meta.json"
    Q=None
    if emb_path.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if meta.get("n")==len(examples) and meta.get("text_model")==args.text_model:
            print("loading cached embeddings:",emb_path)
            Q=np.load(emb_path)
    if Q is None:
        print("encoding queries...")
        Q=encode_queries([x["query"] for x in examples],args.text_model,device,args.embed_batch_size)
        np.save(emb_path,Q)
        meta_path.write_text(json.dumps({"n":len(examples),"qdim":int(Q.shape[1]),"text_model":args.text_model},indent=2))

    mkloader=lambda idx,shuffle: DataLoader(ObsDS(idx,Q,Z,B,K,R,Y),batch_size=args.batch_size,shuffle=shuffle,num_workers=0)
    trl,val,test,alll=mkloader(tr,True),mkloader(va,False),mkloader(te,False),mkloader(range(len(examples)),False)

    model=ObservationNet(Q.shape[1],len(roles),args.hidden,args.dropout).to(device)
    ytr=Y[np.asarray(tr)]
    pos=ytr.sum(0); neg=len(ytr)-pos
    pw=np.clip(neg/np.maximum(pos,1),1,20).astype(np.float32)
    lossfn=nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(pw).to(device))
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)

    best=None; bestval=float("inf"); bad=0
    for ep in range(1,args.epochs+1):
        model.train(); total=n=0
        for q,z,b,k,r,y,idx in trl:
            q,z,b,k,r,y=[t.to(device) for t in [q,z,b,k,r,y]]
            opt.zero_grad(set_to_none=True)
            loss=lossfn(model(q,z,b,k,r),y)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            total+=loss.detach().item()*len(y); n+=len(y)
        model.eval(); vt=vn=0
        with torch.no_grad():
            for q,z,b,k,r,y,idx in val:
                q,z,b,k,r,y=[t.to(device) for t in [q,z,b,k,r,y]]
                loss=lossfn(model(q,z,b,k,r),y)
                vt+=loss.item()*len(y); vn+=len(y)
        tl=total/max(n,1); vl=vt/max(vn,1)
        print(f"[epoch {ep:03d}] train={tl:.6f} val={vl:.6f}")
        if vl < bestval-1e-5:
            bestval=vl; bad=0
            best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad+=1
            if bad>=args.patience:
                print("early stopping"); break

    if best is None: raise RuntimeError("no checkpoint")
    model.load_state_dict(best)

    P,L=predict_all(model,alll,device,len(examples),len(roles))
    th=optimize_thresholds(L[np.asarray(va)],P[np.asarray(va)])
    val_metrics=multilabel_metrics(L[np.asarray(va)],P[np.asarray(va)],th)
    test_metrics=multilabel_metrics(L[np.asarray(te)],P[np.asarray(te)],th)
    rank_records,rank_summary=ranking_eval(examples,P,te,roles)

    torch.save({
        "state_dict":best,"role_order":roles,"query_embedding_dim":int(Q.shape[1]),
        "text_model":args.text_model,"hidden":args.hidden,"dropout":args.dropout,
        "thresholds":th.tolist(),"best_val_loss":bestval,"args":vars(args)
    },out/"model.pt")
    (out/"role_order.json").write_text(json.dumps(roles,indent=2))
    (out/"split_systems.json").write_text(json.dumps({
        "train":sorted(train_sys),"val":sorted(val_sys),"test":sorted(test_sys)
    },indent=2))
    with (out/"per_group_test.jsonl").open("w",encoding="utf-8") as f:
        for r in rank_records: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    summary={
        "dataset":{
            "root":args.root,"parents":len(parents),"examples":len(examples),"roles":len(roles),
            "systems_train":len(train_sys),"systems_val":len(val_sys),"systems_test":len(test_sys),
            "rows_train":len(tr),"rows_val":len(va),"rows_test":len(te),"skipped":skipped
        },
        "model":{
            "type":"oracle-Z multilabel observation model",
            "text_model":args.text_model,
            "best_val_loss":bestval,
            "structural_constraint":"Y subset of Z enforced in logits"
        },
        "validation_multilabel":val_metrics,
        "test_multilabel":test_metrics,
        "test_selector_ranking":rank_summary,
        "interpretation":{
            "important":"ORACLE-Z diagnostic only; not yet a deployable selector.",
            "next_if_good":"Integrate P(Y|Z,q,b) over posterior Z and compute expected information gain."
        }
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))

    print("\n===== OBSERVATION MODEL V1 SUMMARY =====")
    print("TEST MULTILABEL")
    for k,v in test_metrics.items(): print(f"{k}: {v}")
    print("\nTEST SELECTOR RANKING")
    for name,block in rank_summary.items():
        print("\n"+name)
        for k,v in block.items(): print(f"{k}: {v}")
    print("\n[saved]",out/"summary.json")
    print("[saved]",out/"per_group_test.jsonl")
    print("[saved]",out/"model.pt")


if __name__=="__main__":
    main()
