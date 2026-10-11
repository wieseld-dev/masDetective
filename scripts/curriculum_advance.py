from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl, write_json, write_jsonl
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from adaptive_v11_curriculum.belief_utils import role_only_predict, dummy_k_probs
from adaptive_v11_curriculum.reward import curriculum_reward, group_relative_advantages, role_log_score, role_brier, margin
from v11.role_universe import TRAIN_ROLES

EPS=1e-6

def auc_simple(probs,true_roles):
    truth=set(true_roles); pos=[probs[r] for r in TRAIN_ROLES if r in truth]; neg=[probs[r] for r in TRAIN_ROLES if r not in truth]
    if not pos or not neg: return float('nan')
    wins=sum((a>b)+0.5*(a==b) for a in pos for b in neg)
    return wins/(len(pos)*len(neg))

def topk_recall(probs,true_roles):
    k=len(true_roles); top=sorted(TRAIN_ROLES,key=lambda r:probs[r],reverse=True)[:k]
    return len(set(top)&set(true_roles))/max(1,k)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--round',type=int,required=True); ap.add_argument('--split',choices=['train','dev'],required=True)
    ap.add_argument('--root',required=True); ap.add_argument('--manifest',required=True); ap.add_argument('--belief',required=True)
    a=ap.parse_args(); root=Path(a.root)
    states={x['session_id']:PublicBeliefState.from_public_dict(x) for x in read_jsonl(root/f"states_{a.split}_round_{a.round-1:02d}.jsonl")}
    cand=read_jsonl(root/f"candidates_{a.split}_round_{a.round:02d}.jsonl"); resp=read_jsonl(root/f"responses_{a.split}_round_{a.round:02d}.jsonl"); scored=read_jsonl(root/f"scored_{a.split}_round_{a.round:02d}.jsonl")
    mani=json.loads(Path(a.manifest).read_text()); truth={x['session_id']:x for x in mani['systems'] if x['split']==a.split}
    bundle=FrozenBeliefBundle.load(a.belief)
    resp_by={(str(x.get('session_id',x.get('system_id',''))),int(x.get('candidate_index',dict(x.get('action',{})).get('variant',0)))):x for x in resp}
    score_by={}
    for x in scored:
        sid=str(x.get('system_id',x.get('session_id',''))); idx=int(dict(x.get('action',{})).get('variant',x.get('candidate_index',0))); score_by[(sid,idx)]=x
    grouped=defaultdict(list)
    for x in cand: grouped[x['session_id']].append(x)
    train_rows=[]; next_states=[]; metrics=[]
    for sid,cs in sorted(grouped.items()):
        st=states[sid]; tr=truth[sid]; true_roles=tuple(map(str,tr['roles'])); evals=[]
        for c in sorted(cs,key=lambda x:int(x['candidate_index'])):
            idx=int(c['candidate_index']); rr=resp_by.get((sid,idx)); sc=score_by.get((sid,idx)); frame=None
            if c.get('valid') and rr and rr.get('execution_status')=='ok' and sc is not None:
                frame=EvidenceFrame(query=str(c['query']),answer=str(rr.get('final_output','')),
                    v135={str(k):float(v) for k,v in dict(sc.get('evidence',{})).items()},
                    v134={str(k):float(v) for k,v in dict(sc.get('metadata',{}).get('evidence_v134',{})).items()})
                nh=[*st.history,frame]; nr=role_only_predict(bundle,nh)
            else:
                nh=list(st.history); nr=dict(st.role_probs)
            rb=curriculum_reward(old_probs=st.role_probs,new_probs=nr,true_roles=true_roles,roles=TRAIN_ROLES,query=str(c.get('query','')),previous_queries=[h.query for h in st.history],generator_valid=bool(c.get('valid') and frame is not None))
            evals.append((c,frame,nr,rb))
        advs=group_relative_advantages([e[3].total for e in evals])
        chosen_item=None
        for (c,frame,nr,rb),adv in zip(evals,advs):
            chosen=bool(c.get('chosen_for_continuation')) and frame is not None
            if chosen: chosen_item=(c,frame,nr,rb)
            if a.split=='train' and c.get('completion'):
                train_rows.append({'session_id':sid,'round':a.round,'candidate_index':int(c['candidate_index']),'group_id':c['group_id'],
                    'prompt':c['prompt'],'completion':c['completion'],'old_logp':float(c['old_logp']),'ref_logp':float(c['ref_logp']),
                    'reward':float(rb.total),'local_advantage':float(adv),'chosen_for_continuation':bool(chosen),
                    'reward_breakdown_training_only':rb.to_dict()})
        if chosen_item is None:
            ns=PublicBeliefState(session_id=sid,split=a.split,round_index=a.round,k_probs=dummy_k_probs(bundle),role_probs=dict(st.role_probs),history=list(st.history))
            status='carry_no_valid_chosen'
        else:
            c,frame,nr,rb=chosen_item
            ns=PublicBeliefState(session_id=sid,split=a.split,round_index=a.round,k_probs=dummy_k_probs(bundle),role_probs=dict(nr),history=[*st.history,frame]); status='ok'
        ns.validate(TRAIN_ROLES,bundle.ks); next_states.append(ns.to_public_dict())
        metrics.append({'session_id':sid,'round':a.round,'status':status,'role_log_loss':-role_log_score(ns.role_probs,true_roles,TRAIN_ROLES),
                        'role_brier':role_brier(ns.role_probs,true_roles,TRAIN_ROLES),'role_auc':auc_simple(ns.role_probs,true_roles),
                        'true_false_margin':margin(ns.role_probs,true_roles,TRAIN_ROLES),'oracle_k_role_recall':topk_recall(ns.role_probs,true_roles)})
    write_jsonl(root/f"states_{a.split}_round_{a.round:02d}.jsonl",next_states)
    write_jsonl(root/f"metrics_{a.split}_round_{a.round:02d}.jsonl",metrics)
    if a.split=='train': write_jsonl(root/f"train_rows_round_{a.round:02d}.jsonl",train_rows)
    summary={'split':a.split,'round':a.round,'systems':len(metrics),'train_rows':len(train_rows),'mean_role_log_loss':float(np.mean([x['role_log_loss'] for x in metrics])),'valid_state_updates':sum(x['status']=='ok' for x in metrics)}
    write_json(root/f"advance_{a.split}_round_{a.round:02d}_summary.json",summary); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
