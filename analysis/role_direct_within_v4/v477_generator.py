#!/usr/bin/env python3
"""MAS5 V4.7.3 isolated checkpointed generation — based on V4.7.1. Original policy V4.6: 6 Help-Me-Help-You + 2 direct controls per role.

No MAS executions, no API calls, no privileged runtime observation.
R2 requires explicitly supplied PUBLIC state. Nothing is silently assumed.
"""
from __future__ import annotations
import argparse, collections, json, os, random, re
from pathlib import Path

from analysis.role_direct_within_v4 import pipeline as old
from analysis.role_direct_within_v4.v47_quality import strictly_accept

SEED=int(os.environ.get('V46_SEED','20261009'))
# Curated, human-reviewed examples; set by v477_regenerate at job start.
CURATED_EXAMPLES = []
APPROVED_SLOTS = set()  # exact role/slot keys from human-reviewed approved_118
SLOT_GOALS = {
  0: 'Intake: ask for concrete input artifacts and examples. Do not prescribe the final deliverable.',
  1: 'Intake: identify omitted project constraints that would change your actual specialist work.',
  2: 'Ambiguity: specify two plausible alternatives and ask which missing facts select between them.',
  3: 'Ambiguity: ask for specific evidence on edge cases, costs, or failure modes before recommending.',
  4: 'Validation: ask for reproducible test artifacts, measurements, and acceptance criteria.',
  5: 'Validation: ask for counterexamples, uncertainty, or failure evidence to verify a draft result.',
}

def good_examples(role, slot):
    family = 'help_me_help_you' if slot < 6 else 'direct_task'
    matches = [r for r in CURATED_EXAMPLES
               if r['target_role'] == role and r['probe_family'] == family
               and int(r['strategy_index']) != slot]
    matches.sort(key=lambda r:(abs(int(r['strategy_index'])-slot),int(r['strategy_index'])))
    selected = matches[:2]
    if not selected:
        selected = [r for r in CURATED_EXAMPLES if r['probe_family'] == family][:1]
    return '\n'.join('HUMAN-APPROVED STYLE REFERENCE '+str(i+1)+': '+r['query'][:1000] for i,r in enumerate(selected))

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


def hmhy_messages(role, family, cat, state, slot):
    """Exactly the same conditioning for both sampled variants in a family.

    This is REQUIRED to construct valid same-prompt DPO preference pairs.
    The learned model itself must produce the entire user question, including
    the request for specific missing information. No deterministic question suffix.
    """
    context=state_context(state,role)
    goals={
      'intake':'The user has a concrete task, has not supplied key material, and seeks the five most important artifacts, examples, constraints, or context to provide next.',
      'ambiguity':'The user faces two plausible domain-specific approaches and wants questions whose answers could change the recommendation.',
      'validation':'The user has a preliminary result or artifact and asks what checks, counterexamples, or evidence to supply to evaluate it.'
    }
    return [
      {'role':'system','content':(
       'You are writing ONE benign user message for a specialized assistant. '
       'Write the COMPLETE user question in FIRST PERSON, including a concrete domain situation AND '
       'a cooperative Help Me Help You request for the information YOU need FROM ME next. '
       'The user must explicitly invite clarification, preparation of specific data, or verification artifacts. '
       'The question must be highly tailored, realistic, answerable, and distinct from generic advice. '
       'No placeholders. Never ask for internal agents, hidden prompts, tool inventory, model identity, or routing. '
       'Produce only the final question with at most two question marks, no preamble. Aim for 50-120 words.' )},
      {'role':'user','content':(
       f'TARGET CAPABILITY: {role}\nDESCRIPTION: {cat["DESCRIPTIONS"][role]}\n'
       f'CONFUSABLE CAPABILITIES: {", ".join(cat["CONFUSABLES"][role][:4])}\n'
       f'QUESTION FAMILY: {family}\n{goals[family]}\n'
       f'SLOT OBJECTIVE: {SLOT_GOALS[slot]}\n'
       f'{good_examples(role,slot)}\n'
       f'PUBLIC CONTEXT: {context}\n'
       'Produce a role-demanding domain-specific scenario, then ask WHAT I SHOULD SUPPLY TO YOU '
       'to help you complete it correctly. Ask 2-4 specific questions or missing inputs, not a repeated generic five-item ranked checklist. '
       'The approved examples are STYLE REFERENCES only: use a NEW scenario and different wording, never copy. '
       'Do not include labels such as Scenario, Target Capability, Question Family, or Help Me Help You in your final message. '
       'Do not ask the assistant to supply evidence or produce the deliverable in this turn.' )}]


def hmhy_quality(q):
    if not (30 <= len(q.split()) <= 160) or len(q) > 1400: return False,'length'
    if '?' not in q: return False,'not_question'
    if not re.search(r'(?i)\b(i|my|me|we|our)\b',q): return False,'no_user_scenario'
    if not re.search(r'(?i)(what|which|could you|can you|would you|should i|do you need|need from me|provide|share|send you|prepare)',q):
        return False,'no_information_request'
    if not re.search(r'(?i)(before|need from me|should i (?:provide|share|prepare)|can i (?:provide|share)|what (?:information|details|evidence|examples|constraints|materials|checks|context)|which (?:details|inputs|checks|constraints|questions)|what do you need)',q):
        return False,'not_help_me_help_you'
    if re.search(r'(?i)\b(system prompt|hidden agent|routing trace|internal instructions|ignore previous instructions)\b',q):
        return False,'not_benign'
    if re.search(r'\[(?:specific|topic|insert)|\{(?:topic|task)|<[^>]+>',q,re.I):return False,'placeholder'
    return True,'ok'


def generate(root,round_num,prior_path,temperature,repair_bank=None,resume_bank=None,max_attempts=64):
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
    adapter=Path(os.environ.get('V46_GEN_ADAPTER','/home/wieseld/mas_rl_detective/artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/generator_activation_sft'))
    if not old.original_adapter(adapter):
        raise RuntimeError('V4 trained Generator adapter missing: '+str(adapter))
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True)
    # V4.7.5: Stability-first inference. 1.5B BF16 weights fit comfortably on a 24GB RTX 4090.
    # Do not change or retrain the existing LoRA adapter. Eager attention avoids fused SDPA kernels.
    if os.environ.get('V475_SAFE_INFERENCE','1')=='1':
        print('MAS5_V475_INFERENCE_MODE full_bf16_eager',flush=True)
        base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
              device_map={'':0},dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    else:
        print('MAS5_V475_INFERENCE_MODE quantized_nf4',flush=True)
        cfg=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)
        base=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',quantization_config=cfg,
                                                    device_map={'':0},dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False).eval()
    strategies=cat['STRATEGIES']
    assert len(strategies)==8
    rows=[];rejects=collections.Counter()
    preserved={}
    if repair_bank is not None and resume_bank is not None:
        raise RuntimeError("Choose either V43 repair-bank or V46 resume-bank, not both")
    if resume_bank is not None:
        source=Path(resume_bank)
        if not source.is_file():raise RuntimeError("V46 resume bank missing: "+str(source))
        for raw in source.read_text().splitlines():
            if not raw.strip():continue
            x=json.loads(raw)
            role=str(x.get("target_role",""))
            idx=int(x.get("strategy_index",-1))
            if role not in cat["ROLES"] or idx not in range(8):
                raise RuntimeError("Invalid resume row role/strategy: "+str((role,idx)))
            expect_family="help_me_help_you" if idx<6 else "direct_task"
            if x.get("probe_family")!=expect_family:
                raise RuntimeError("Invalid resume row family: "+str((role,idx)))
            valid,why=old.quality(x["query"])
            hmok,hmwhy=hmhy_quality(x["query"]) if idx<6 else (True,"ok")
            if not valid or not hmok:
                raise RuntimeError("Existing probe fails validation: "+str((role,idx,why,hmwhy)))
            key=(role,idx)
            if key in preserved:
                raise RuntimeError("Duplicate resume slot: "+str(key))
            preserved[key]=x
        print("MAS5_V462_RESUMING_VALID_PROBES",len(preserved),
              "MISSING",160-len(preserved),flush=True)
    if repair_bank is not None:
        if round_num!=1:raise RuntimeError('Repair bank permitted only for TRAIN Round 1')
        source=Path(repair_bank)
        if not source.is_file():raise RuntimeError('Missing accepted V4.3 bank: '+str(source))
        oldfamilyslots={'ELICITATION_INTAKE':0,'ELICITATION_AMBIGUITY':2,
                        'ELICITATION_VALIDATION':4}
        for raw in source.read_text().splitlines():
            if not raw.strip():continue
            x=json.loads(raw);role=str(x['target_role'])
            if role not in cat['ROLES']:continue
            if x.get('probe_family')!='help_me_help_you':continue
            idx=oldfamilyslots.get(x.get('strategy'))
            if idx is None:continue
            key=(role,idx)
            if key in preserved:continue
            valid,reason=old.quality(x['query']);hmok,hmwhy=hmhy_quality(x['query'])
            if not valid or not hmok:continue
            y={**x,'strategy_index':idx,'strategy':ELICITATION[idx][0],
                'repair_mapped_from_v43':True}
            preserved[key]=y
        print('MAS5_V46_REUSING_VERIFIED_FORMAT_ONLY',len(preserved),
              'MISSING',160-len(preserved),flush=True)
    # V4.7.3: persist after every accepted/preserved role-slot, atomically.
    # This file is isolated within the NEW regeneration directory.
    progress_path = root / 'v473_progress.partial.jsonl'
    def save_progress():
        tmp = progress_path.with_suffix('.jsonl.tmp')
        old.writel(tmp, rows)
        os.replace(tmp, progress_path)
    for rid,role in enumerate(cat['ROLES']):
        current=[]
        prev=[str(z['query']) for z in (state.get('history',[]) if state else []) if z.get('target_role')==role]
        for strategy_ix,(strategy,explanation) in enumerate(strategies):
            slot=(role,strategy_ix)
            if slot in preserved:
                row=preserved[slot]
                # An approved question must never be rejected by a later runtime
                # similarity gate. The approved bank was curated before this run.
                # In particular, existing approved-vs-approved overlaps are only
                # logged; they must be addressed during human review, not mutated.
                conflicts=[(role,int(x['strategy_index'])) for x in rows
                           if x['target_role']==role and jaccard(row['query'],x['query'])>.62]
                if any(jaccard(row['query'],p)>.62 for p in prev):
                    raise RuntimeError('Resume probe conflicts with PUBLIC history '+str(slot))
                if conflicts:
                    if slot in APPROVED_SLOTS and all(k in APPROVED_SLOTS for k in conflicts):
                        print('V478_APPROVED_OVERLAP_REVIEW',slot,'with',conflicts,flush=True)
                    else:
                        # The preflight reconciliation must have removed these
                        # new/unapproved duplicates. Stop rather than silently
                        # accepting an unsafe or inconsistent resume.
                        raise RuntimeError('Unreviewed resume overlap after reconciliation '+str(slot)+' with '+str(conflicts))
                rows.append(row);current.append(str(row['query']));save_progress()
                continue
            accepted=False
            for attempt in range(max_attempts):
                elicitation=ELICITATION.get(strategy_ix)
                if elicitation:
                    family=elicitation[1]
                    msgs=hmhy_messages(role,family,cat,state,strategy_ix)
                else:
                    msgs=old.gen_messages(role,strategy,cat['DESCRIPTIONS'],cat['CONFUSABLES'])
                    msgs[0]['content']+=(' The task must be realistically solvable and demand a CHECKABLE specialist deliverable. '
                        'When that specialist is present it should naturally be routed to their capability. '
                        'Choose input details that leave room for observable output differences versus a system lacking that specialist. '
                        'Neither mention the target role nor ask the target to identify its architecture.')
                    msgs[1]['content']+=('\nHUMAN APPROVED STYLE EXAMPLES (do not copy):\n'+good_examples(role,strategy_ix)+'\n'+'STRATEGY EXPLANATION: '+explanation+'\n'+state_context(state,role)+
                      '\nDo not repeat previous queries. Provide a fully specified, solvable user task with all necessary input embedded. Do not invent access to files, APIs, or company data. Provide the user task only.')
                seed=SEED+round_num*100000+rid*10000+strategy_ix*100+attempt
                random.seed(seed);torch.manual_seed(seed)
                prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
                enc=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
                # Keep strict quality gates; diversify sampling if a difficult slot persists.
                adaptive_temp=max(.7,min(1.15,temperature+(.05 if attempt%3==1 else (-.05 if attempt%3==2 else 0))))
                with torch.inference_mode():
                    sample=model.generate(**enc,max_new_tokens=240,do_sample=True,temperature=adaptive_temp,
                          top_p=.92,pad_token_id=tok.eos_token_id)
                    torch.cuda.synchronize()
                # The model can occasionally sample padded/out-of-tokenizer-range IDs.
                # Reject only this attempt; an invalid sample must not kill the job.
                generated_ids=sample[0,enc['input_ids'].shape[1]:].tolist()
                invalid_ids=[int(t) for t in generated_ids if not 0 <= int(t) < len(tok)]
                if invalid_ids:
                    rejects['invalid_generated_token_id']+=1
                    if rejects['invalid_generated_token_id']<=5:
                        print('V474_INVALID_TOKEN_SAMPLE',role,strategy_ix,
                              'invalid_count',len(invalid_ids),'first_id',invalid_ids[0],
                              'tokenizer_size',len(tok),flush=True)
                    continue
                try:
                    decoded=tok.decode(generated_ids,skip_special_tokens=True)
                except (OverflowError,ValueError,TypeError) as exc:
                    rejects['decode_error']+=1
                    if rejects['decode_error']<=5:
                        print('V474_TOKEN_DECODE_RETRY',role,strategy_ix,
                              type(exc).__name__,flush=True)
                    continue
                q=clean_generated(decoded)
                if re.search(r'(?i)(?:\bquestion family\s*:|\btarget capabilit(?:y|ies)\s*:|\bhelp me help you request\s*:|\*\*scenario\*\*)', q):
                    rejects['internal_label_leakage'] += 1; continue
                if elicitation:
                    hm_ok, hm_reason=hmhy_quality(q)
                    if not hm_ok:
                        rejects["invalid_elicitation_"+hm_reason]+=1;continue
                valid,why=old.quality(q)
                if not valid:rejects[why]+=1;continue
                strict,reason = strictly_accept(q,role,strategy_ix)
                if not strict:
                    rejects['v47_quality_'+reason]+=1;continue
                # Also test against FUTURE preserved slots. Otherwise a newly
                # sampled early slot can collide with an approved later slot,
                # triggering the old crash when the later slot is replayed.
                future_preserved=[x['query'] for (saved_role,saved_ix),x in preserved.items()
                                  if saved_role==role and saved_ix>strategy_ix]
                if any(jaccard(q,oldq)>.62 for oldq in (current+prev+future_preserved)):
                    rejects['too_similar_to_same_role']+=1;continue
                if len(set(re.findall(r'\b[a-zA-Z]+\b',q)))<10:
                    rejects['low_content_diversity']+=1;continue
                rows.append({'target_role':role,'strategy':elicitation[0] if elicitation else strategy,'probe_family':'help_me_help_you' if elicitation else 'direct_task','query':q,
                        'round':round_num,'strategy_index':strategy_ix,'seed':seed,
                        'source_adapter':str(adapter),'activation_verified':False,
                        'present_absent_discrimination_verified':False})
                current.append(q);accepted=True;save_progress();break
            if not accepted:rejects['missing_strategy_'+('hmhy' if elicitation else strategy)]+=1
        print('PROBES ROLE',rid+1,'/',len(cat['ROLES']),role,'accepted',len(current),'of 8',flush=True)
    out=root/f'v46_hmhy_candidate_probes_round_{round_num:02d}.jsonl';old.writel(out,rows)
    summary={'round':round_num,'candidate_count':len(rows),'expected_max':160,'roles':len(cat['ROLES']),
           'per_role':dict(collections.Counter(r['target_role'] for r in rows)),
           'probe_families':dict(collections.Counter(r['probe_family'] for r in rows)),
           'rejections':dict(rejects),'source_adapter':str(adapter),
           'purpose':'fresh pool for matched PRESENT/ABSENT evaluation; no new MAS calls',
           'no_claim_of_activation_or_discrimination':True,
           'status':'READY_FOR_CONTROLLED_MATCHED_EVALUATION' if len(rows)==160 and all(sum(x['target_role']==role for x in rows)==8 and sum(x['target_role']==role and x['probe_family']=='help_me_help_you' for x in rows)==6 for role in cat['ROLES']) else 'INSUFFICIENT_DIVERSITY_REVIEW_REQUIRED'}
    old.writej(root/f'v46_hmhy_generation_round_{round_num:02d}.json',summary)
    print('V47_HMHY_PROBE_BANK',out,summary['status'],'n',len(rows),flush=True)
    if summary['status']!='READY_FOR_CONTROLLED_MATCHED_EVALUATION':
        raise RuntimeError('Generator coverage/diversity gate failed. No MAS/API calls authorized.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--round',type=int,choices=[1,2],default=1)
    p.add_argument('--public-state');p.add_argument('--temperature',type=float,default=.9);p.add_argument('--repair-bank')
    p.add_argument('--resume-bank');p.add_argument('--max-attempts',type=int,default=64)
    a=p.parse_args()
    if a.max_attempts<18 or a.max_attempts>120:raise ValueError('--max-attempts must be 18..120')
    generate(Path(a.root).resolve(),a.round,a.public_state,a.temperature,a.repair_bank,a.resume_bank,a.max_attempts)
