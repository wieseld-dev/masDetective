#!/usr/bin/env python3
"""Role-directed, 8-strategy probe bank from the existing V4 trained LoRA.

No MAS executions, no API calls, no privileged runtime observation.
R2 requires explicitly supplied PUBLIC state. Nothing is silently assumed.
"""
from __future__ import annotations
import argparse, collections, json, os, random, re
from pathlib import Path

from analysis.role_direct_within_v4 import pipeline as old

SEED=20261009
DENY_PRIVATE={'truth','installed_roles','selected_roles_training_only','routing_trace','router_trace',
    'system_prompt','hidden_prompt','privileged','internal_agent_outputs','labels','label','pair_id'}


def public_state(path):
    if path is None:return None
    data=json.loads(Path(path).read_text())
    def check(x):
        if isinstance(x,dict):
            for k,v in x.items():
                if str(k).lower() in DENY_PRIVATE:raise ValueError(f'Nonpublic state field: {k}')
                check(v)
        if isinstance(x,list):
            for v in x:check(v)
    check(data)
    # Strict whitelist at the very top: no identity, no hidden traces or ground truth.
    allowed={'role_belief','history','round'}
    if set(data)-allowed:raise ValueError(f'Unexpected public state keys: {set(data)-allowed}')
    if 'role_belief' in data:
        if not isinstance(data['role_belief'],dict) or any(not isinstance(v,(float,int)) or not 0<=v<=1 for v in data['role_belief'].values()):
            raise ValueError('Invalid public belief probabilities')
    if 'history' in data:
        assert isinstance(data['history'],list) and len(data['history'])<=160
        for item in data['history']:
            if set(item)-{'target_role','query','final_response','round'}:
                raise ValueError('History contains nonpublic or unsupported fields')
    return data


def state_context(state,role):
    if not state:return 'ROUND ONE. No previous observations are available.'
    previous=[h for h in state.get('history',[]) if h.get('target_role')==role]
    previous=previous[-3:]
    if not previous:return 'ROUND TWO. No prior evidence for this role; pursue a distinct diagnostic task.'
    text=[]
    for h in previous:
        text.append('PREVIOUS QUERY (public): '+str(h['query'])[:1800])
        text.append('VISIBLE RESPONSE (public): '+str(h['final_response'])[:2600])
    prior=float(state.get('role_belief',{}).get(role,.5))
    return ('ROUND TWO. Current public-only belief for target role: '+f'{prior:.3f}.\n'+
            '\n'.join(text)+'\nWrite a NEW, non-redundant benign task targeting exactly the same capability. '
            'Explore a different behavioral signature to disambiguate the presence of the target.')


def norm(q):return re.sub(r'\W+',' ',q.lower()).strip()

def jaccard(a,b):
    x=set(norm(a).split());y=set(norm(b).split())
    return len(x&y)/max(1,len(x|y))


def clean_generated(q):
    q=re.sub(r'^\s*(?:request|probe|question|user query)\s*:\s*','',str(q),flags=re.I).strip()
    q=q.strip('"\'` \n')
    return old.clean(q)


def generate(root,round_num,prior_path,temperature):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
    from peft import PeftModel
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    name=torch.cuda.get_device_name(0)
    if '4090' not in name:raise RuntimeError('This environment supports RTX4090: '+name)
    state=public_state(prior_path)
    if round_num==2 and state is None:raise RuntimeError('R2 requires --public-state from prior visible observations')
    if round_num==1 and state is not None:raise RuntimeError('R1 must not read a prior public state')
    cat=old.catalog()
    adapter=root.parent.parent/'checkpoints/generator_activation_sft'
    if not old.original_adapter(adapter):
        raise RuntimeError('V4 trained Generator adapter missing: '+str(adapter))
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True)
    cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
    base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',quantization_config=cfg,
                                                device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False).eval()
    strategies=cat['STRATEGIES']
    assert len(strategies)==8
    rows=[];rejects=collections.Counter()
    for rid,role in enumerate(cat['ROLES']):
        current=[]
        prev=[str(z['query']) for z in (state.get('history',[]) if state else []) if z.get('target_role')==role]
        for strategy_ix,(strategy,explanation) in enumerate(strategies):
            accepted=False
            for attempt in range(5):
                msgs=old.gen_messages(role,strategy,cat['DESCRIPTIONS'],cat['CONFUSABLES'])
                msgs[0]['content']+=(' The task must be realistically solvable and demand a CHECKABLE specialist deliverable. '
                    'When that specialist is present it should naturally be routed to their capability. '
                    'Choose input details that leave room for observable output differences versus a system lacking that specialist. '
                    'Neither mention the target role nor ask the target to identify its architecture.')
                msgs[1]['content']+=('\nSTRATEGY EXPLANATION: '+explanation+'\n'+state_context(state,role)+
                  '\nDo not repeat previous queries. Provide the user task only.')
                seed=SEED+round_num*100000+rid*1000+strategy_ix*50+attempt
                random.seed(seed);torch.manual_seed(seed)
                prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                enc=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
                with torch.inference_mode():
                    sample=model.generate(**enc,max_new_tokens=230,do_sample=True,temperature=temperature,
                          top_p=.92,pad_token_id=tok.eos_token_id)
                q=clean_generated(tok.decode(sample[0,enc['input_ids'].shape[1]:],skip_special_tokens=True))
                valid,why=old.quality(q)
                if not valid:rejects[why]+=1;continue
                if any(jaccard(q,oldq)>.62 for oldq in (current+prev)):
                    rejects['too_similar_to_same_role']+=1;continue
                if len(set(re.findall(r'\b[a-zA-Z]+\b',q)))<10:
                    rejects['low_content_diversity']+=1;continue
                rows.append({'target_role':role,'strategy':strategy,'query':q,
                        'round':round_num,'strategy_index':strategy_ix,'seed':seed,
                        'source_adapter':str(adapter),'activation_verified':False,
                        'present_absent_discrimination_verified':False})
                current.append(q);accepted=True;break
            if not accepted:rejects['missing_strategy_'+strategy]+=1
        print('PROBES ROLE',rid+1,'/',len(cat['ROLES']),role,'accepted',len(current),'of 8',flush=True)
    out=root/f'candidate_probes_round_{round_num:02d}.jsonl';old.writel(out,rows)
    summary={'round':round_num,'candidate_count':len(rows),'expected_max':160,'roles':len(cat['ROLES']),
           'per_role':dict(collections.Counter(r['target_role'] for r in rows)),
           'rejections':dict(rejects),'source_adapter':str(adapter),
           'purpose':'fresh pool for matched PRESENT/ABSENT evaluation; no new MAS calls',
           'no_claim_of_activation_or_discrimination':True,
           'status':'READY_FOR_CONTROLLED_MATCHED_EVALUATION' if all(sum(x['target_role']==role for x in rows)>=6 for role in cat['ROLES']) else 'INSUFFICIENT_DIVERSITY_REVIEW_REQUIRED'}
    old.writej(root/f'probe_generation_round_{round_num:02d}.json',summary)
    print('V42_PROBE_BANK',out,summary['status'],'n',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--round',type=int,choices=[1,2],default=1)
    p.add_argument('--public-state');p.add_argument('--temperature',type=float,default=.9)
    a=p.parse_args();generate(Path(a.root).resolve(),a.round,a.public_state,a.temperature)
