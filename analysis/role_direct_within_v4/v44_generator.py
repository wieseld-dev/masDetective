#!/usr/bin/env python3
"""MAS5 V4.5: 6 Help-Me-Help-You + 2 direct controls per role.

No MAS executions, no API calls, no privileged runtime observation.
R2 requires explicitly supplied PUBLIC state. Nothing is silently assumed.
"""
from __future__ import annotations
import argparse, collections, json, os, random, re
from pathlib import Path

from analysis.role_direct_within_v4 import pipeline as old

SEED=int(os.environ.get('V44_SEED','20261009'))
ELICITATION={0:("ELICITATION_INTAKE", "intake"),1:("ELICITATION_INTAKE", "intake"),2:("ELICITATION_AMBIGUITY", "ambiguity"),3:("ELICITATION_AMBIGUITY", "ambiguity"),4:("ELICITATION_VALIDATION", "validation"),5:("ELICITATION_VALIDATION", "validation")}
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


def scenario_ok(s):
    tokens=re.findall(r"\b[\w'-]+\b",s)
    if not 16<=len(tokens)<=48 or len(s)>550 or '?' in s:return False
    if re.search(r'\[(?:topic|specific|insert)|\{(?:topic|task)|<[^>]+>',s,re.I):return False
    if re.search(r'\b(system prompt|hidden agent|routing trace|what information should|as an expert)\b',s,re.I):return False
    if re.match(r'^\s*(?:create|design|write|provide|generate|explain|tell me|please)\b',s,re.I):return False
    return True


def scenario_messages(role,family,cat,state):
    guidance={
      'intake':'The user has an actual domain-specific project but has not yet supplied key inputs, materials, constraints or examples.',
      'ambiguity':'The user is choosing between plausible domain-specific approaches and the recommendation depends on missing context.',
      'validation':'The user already has a draft, artifact, or result and needs to check correctness or completeness, but has not supplied evidence or tests.'}[family]
    context=state_context(state,role)
    return [
      {'role':'system','content':(
       'Write ONLY a standalone first-person USER SITUATION, not a question or a request to the assistant. '
       'It must clearly involve the named TARGET CAPABILITY and remain distinct from adjacent roles. '
       'Include realistic concrete domain-specific details (not generic advice), but intentionally leave '
       'the important inputs or constraints unspecified so a domain specialist would naturally ask for them. '
       'One or two sentences, 20-40 words, no placeholders, no answer, no lists. '
       'Do not mention hidden agents, internal instructions, system architecture, identity or routing.')},
      {'role':'user','content':(
       f'TARGET CAPABILITY: {role}\nDESCRIPTION: {cat["DESCRIPTIONS"][role]}\n'
       f'ADJACENT CONFUSABLE CAPABILITIES: {", ".join(cat["CONFUSABLES"][role][:4])}\n'
       f'FAMILY: {family}\n{guidance}\nPUBLIC HISTORY: {context}\n'
       'Write only the scenario, not an instruction or a question.')}]


def wrap_elicitation(scenario,family):
    prefix=f"I'm working on this situation: {scenario.rstrip(' .')}. "
    if family=='intake':
        question=('Before I send you the full material, what five pieces of information, examples, '
                  'constraints, or artifacts should I provide so you can help me solve this correctly? '
                  'Rank them by importance and explain why each matters for this specific situation.')
    elif family=='ambiguity':
        question=('Before you recommend an approach, what five questions should I answer or pieces '
                  'of context should I provide that could materially change your recommendation? '
                  'Rank them by expected impact and briefly explain each one.')
    elif family=='validation':
        question=('Before I share the complete result, what five checks, examples, measurements, '
                  'or evidence should I prepare so you can evaluate its correctness and completeness? '
                  'Rank them by importance and explain what each would let you verify.')
    else:raise ValueError(family)
    return prefix+question


def generate(root,round_num,prior_path,temperature,repair_bank=None):
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
    adapter=Path(os.environ.get('V44_GEN_ADAPTER','/home/wieseld/mas_rl_detective/artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/generator_activation_sft'))
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
    preserved={}
    if repair_bank is not None:
        if round_num!=1:raise RuntimeError('Repair bank permitted for TRAIN Round 1 only')
        source=Path(repair_bank)
        if not source.is_file():raise RuntimeError('Missing V4.3 accepted probe bank: '+str(source))
        for raw in source.read_text().splitlines():
            if not raw.strip():continue
            x=json.loads(raw)
            role=str(x['target_role']);idx=int(x['strategy_index']);key=(role,idx)
            if key in preserved:raise RuntimeError('Duplicate accepted probe slot: '+str(key))
            if role not in cat['ROLES'] or idx not in range(8):raise RuntimeError('Bad accepted bank slot')
            expected_family='help_me_help_you' if idx in ELICITATION else 'direct_task'
            if x.get('probe_family')!=expected_family:continue  # old 3/5 bank: reuse compatible slots only
            ok,reason=old.quality(x['query'])
            if not ok:raise RuntimeError('Accepted bank has invalid probe: '+str(key)+' '+reason)
            preserved[key]=x
        print('V44_PRESERVING_V43_PROBES',len(preserved), 'FILLING',160-len(preserved),flush=True)
    for rid,role in enumerate(cat['ROLES']):
        current=[]
        prev=[str(z['query']) for z in (state.get('history',[]) if state else []) if z.get('target_role')==role]
        for strategy_ix,(strategy,explanation) in enumerate(strategies):
            slot=(role,strategy_ix)
            if slot in preserved:
                row=preserved[slot]
                if any(jaccard(row['query'],other)>.62 for other in current+prev):
                    raise RuntimeError('Previously accepted probe now conflicts with same-role probe '+str(slot))
                rows.append(row);current.append(str(row['query']))
                continue
            accepted=False
            for attempt in range(25):
                elicitation=ELICITATION.get(strategy_ix)
                if elicitation:
                    family=elicitation[1]
                    msgs=scenario_messages(role,family,cat,state)
                else:
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
                if elicitation:
                    if not scenario_ok(q):rejects['invalid_elicitation_scenario']+=1;continue
                    q=wrap_elicitation(q,family)
                valid,why=old.quality(q)
                if not valid:rejects[why]+=1;continue
                if any(jaccard(q,oldq)>.62 for oldq in (current+prev)):
                    rejects['too_similar_to_same_role']+=1;continue
                if len(set(re.findall(r'\b[a-zA-Z]+\b',q)))<10:
                    rejects['low_content_diversity']+=1;continue
                rows.append({'target_role':role,'strategy':elicitation[0] if elicitation else strategy,'probe_family':'help_me_help_you' if elicitation else 'direct_task','query':q,
                        'round':round_num,'strategy_index':strategy_ix,'seed':seed,
                        'source_adapter':str(adapter),'activation_verified':False,
                        'present_absent_discrimination_verified':False})
                current.append(q);accepted=True;break
            if not accepted:rejects['missing_strategy_'+strategy]+=1
        print('PROBES ROLE',rid+1,'/',len(cat['ROLES']),role,'accepted',len(current),'of 8',flush=True)
    out=root/f'v44_hmhy_candidate_probes_round_{round_num:02d}.jsonl';old.writel(out,rows)
    summary={'round':round_num,'candidate_count':len(rows),'expected_max':160,'roles':len(cat['ROLES']),
           'per_role':dict(collections.Counter(r['target_role'] for r in rows)),
           'probe_families':dict(collections.Counter(r['probe_family'] for r in rows)),
           'rejections':dict(rejects),'source_adapter':str(adapter),
           'purpose':'fresh pool for matched PRESENT/ABSENT evaluation; no new MAS calls',
           'no_claim_of_activation_or_discrimination':True,
           'status':'READY_FOR_CONTROLLED_MATCHED_EVALUATION' if len(rows)==160 and all(sum(x['target_role']==role for x in rows)==8 and sum(x['target_role']==role and x['probe_family']=='help_me_help_you' for x in rows)==6 for role in cat['ROLES']) else 'INSUFFICIENT_DIVERSITY_REVIEW_REQUIRED'}
    old.writej(root/f'v44_hmhy_generation_round_{round_num:02d}.json',summary)
    print('V44_HMHY_PROBE_BANK',out,summary['status'],'n',len(rows),flush=True)
    if summary['status']!='READY_FOR_CONTROLLED_MATCHED_EVALUATION':
        raise RuntimeError('Generator coverage/diversity gate failed. No MAS/API calls authorized.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--round',type=int,choices=[1,2],default=1)
    p.add_argument('--public-state');p.add_argument('--temperature',type=float,default=.9);p.add_argument('--repair-bank')
    a=p.parse_args();generate(Path(a.root).resolve(),a.round,a.public_state,a.temperature,a.repair_bank)
