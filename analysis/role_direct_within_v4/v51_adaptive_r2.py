#!/usr/bin/env python3
"""MAS5 V5.1: PUBLIC-ONLY response-conditioned Round-2 pilot; offline by default.

This prototype deliberately does NOT claim a trained adaptive generator or EIG selector.
A frozen prior-trained generator is *conditioned* on R1 public history; an old frozen
sensor supplies an UNCALIBRATED public response score. It creates 8 proposed probes
per R1 session (six HMHY, two direct controls), audits and ranks them for human review.

V4.9 pilot = diagnostic held-out data; NEVER use this R2 pilot to train final MAS5.
No MAS/API execution action is provided by this module.
"""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
from pathlib import Path

PROJECT=Path('/home/wieseld/mas_rl_detective')
ADAPTER=PROJECT/'artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051/train/generator_bootstrap_adapter'
SENSOR=PROJECT/'artifacts/role_direct_within_v4/runs/v4_20261009_185359_1475154/checkpoints/sensor_epoch02'
PILOT_PARENT=PROJECT/'artifacts/role_direct_within_v4/pilot_v49'
ROLES=('research','database_design','customer_support_triage')
FAMILIES=('INTAKE','INTAKE','AMBIGUITY','AMBIGUITY','VALIDATION','VALIDATION','DIRECT','DIRECT')
BLOCKED={'condition','pair_id','installed_roles','roles','selected_roles_training_only','truth_roles_training_only',
         'truth','label','manifest_sha256','routing_trace','router_trace','system_prompt','privileged',
         'internal_agent_outputs','mode','style','session_id','original_cf_pair','target_present','target_absent'}
STOP={'which','there','about','these','those','would','could','should','their','what','that','this','have',
      'from','into','your','with','before','after','please','need','information','details','provide',
      'help','project','based','specific','using','ensure','through','where','when','they','them','will',
      'more','want','also','since','because','might','only','some','each','been','while','including',
      'would','questions','answers','could','about','report','results','scenario','system','important'}


def readl(p):
    with Path(p).open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def writej(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError('Refusing to overwrite: '+str(p))
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def writel(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():raise RuntimeError('Refusing to overwrite: '+str(p))
    with p.open('w',encoding='utf-8') as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')


def atomicl(p,rows):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8') as f:
        for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def terms(s):return re.findall(r'[a-zA-Z][a-zA-Z0-9_-]{3,}',str(s).lower())


def context_terms(text):
    count=collections.Counter(t for t in terms(text) if t not in STOP and not t.startswith('http'))
    return sorted(count,key=lambda x:(-count[x],-len(x),x))[:28]


def similarity(a,b):
    x=set(terms(a));y=set(terms(b))
    return len(x&y)/max(1,len(x|y))


def anchor_overlap(query,response):
    anchor=set(context_terms(response))
    if not anchor:return 0
    return len(anchor & set(terms(query)))


def safe_public_row(d,keys):
    if set(d)-set(keys):raise RuntimeError('Private/unsupported field in public record: '+str(set(d)-set(keys)))
    if set(d)&BLOCKED:raise RuntimeError('Private field is being exposed')
    if any(not isinstance(v,(str,int,float,bool,type(None))) for v in d.values()):
        raise RuntimeError('Nested untrusted public state not permitted')


def prepare(root,pilot):
    root=Path(root).resolve();pilot=Path(pilot).resolve()
    if root.exists():raise RuntimeError('Root exists, no overwrite: '+str(root))
    cpath=pilot/'candidates_train_round_01.jsonl';rpath=pilot/'responses_train_round_01.jsonl'
    c=readl(cpath);r=readl(rpath)
    if len(c)!=54 or len(r)!=54:raise RuntimeError('Expected complete V4.9 pilot: 54 candidates and 54 responses')
    cmap={(str(x['session_id']),int(x['candidate_index'])):x for x in c}
    rmap={(str(x['session_id']),int(x['candidate_index'])):x for x in r}
    if len(cmap)!=54 or set(cmap)!=set(rmap):raise RuntimeError('Duplicate/missing R1 matches')
    groups=collections.defaultdict(list)
    for (sid,slot),x in cmap.items():
        role=str(x['target_role'])
        if role not in ROLES:raise RuntimeError('Unexpected target role: '+role)
        if x['query']!=rmap[(sid,slot)]['query']:raise RuntimeError('Query does not match response')
        if rmap[(sid,slot)].get('execution_status')!='ok':raise RuntimeError('Unsuccessful pilot response')
        groups[(sid,role)].append((slot,x,rmap[(sid,slot)]))
    if len(groups)!=18:raise RuntimeError(f'Expected 18 target sessions, got {len(groups)}')
    root.mkdir(parents=True)
    public=[];private=[]
    for (sid,role),records in sorted(groups.items()):
        # Exactly ONE R1 selected message per system-role: do not fake a 3-probe R1 budget.
        slot,x,response=sorted(records,key=lambda z:z[0])[0]
        previous=str(response['final_output']).strip()
        if len(previous)<12:raise RuntimeError(f'Empty/short visible response for {sid}')
        opaque='ctx_'+hashlib.sha256(os.urandom(32)).hexdigest()[:20]
        pub={'context_id':opaque,'target_role':role,'previous_query':str(x['query']),
             'previous_response':previous[:12000], 'previous_slot':int(slot), 'round':2}
        safe_public_row(pub,('context_id','target_role','previous_query','previous_response','previous_slot','round'))
        public.append(pub)
        # All session identifiers (including POS/NEG suffixes) and matched labels remain here ONLY.
        private.append({'context_id':opaque,'session_id':sid,'target_role':role,
                        'previous_slot':int(slot),'condition':x.get('condition'),
                        'pair_id':x.get('pair_id')})
    writel(root/'public_r1_states.jsonl',public)
    writel(root/'private_session_map_training_only.jsonl',private)
    os.chmod(root/'private_session_map_training_only.jsonl',0o600)
    writej(root/'preflight.json',{'status':'PUBLIC_R2_CONTEXTS_READY_NO_API',
           'contexts':len(public),'previous_queries_per_context':1,'roles':list(ROLES),
           'pilot_candidates_sha256':digest(cpath),'pilot_responses_sha256':digest(rpath),
           'private_mapping_file':'private_session_map_training_only.jsonl',
           'generator_can_read_private_map':False,
           'evidence':'R1 final visible answers only; no traces, labels, ground truth or system ids.',
           'r1_limit':'One prior query per target role per MAS for this R2 prototype.',
           'expected_r2_candidates':18*8,'paid_calls_authorized':0,
           'note':'V4.9 roles are diagnostic-only; do not use these examples for final MAS5 training or held-out evaluation.'})
    print('MAS5_V51_PREPARE_PASS contexts=18, R1 observations/context=1, no API calls',flush=True)
    print('R2 public state:',root/'public_r1_states.jsonl',flush=True)


def score(root):
    """Frozen PRE-EXISTING sensor, trained prior to V4.9. Scores are uncalibrated logits."""
    import torch
    from peft import PeftConfig,PeftModel
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    from analysis.role_direct_within_v4.pipeline import catalog,make_sensor_text
    root=Path(root).resolve()
    if not (root/'preflight.json').exists():raise RuntimeError('Prepare first')
    rows=readl(root/'public_r1_states.jsonl')
    if (root/'public_scored_r1_states.jsonl').exists():raise RuntimeError('Score already written')
    adapter=Path(os.environ.get('V51_SENSOR_ADAPTER',str(SENSOR)))
    if not (adapter/'adapter_config.json').is_file():raise RuntimeError('Frozen sensor adapter missing: '+str(adapter))
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Requires RTX 4090 CUDA')
    cfg=PeftConfig.from_pretrained(str(adapter),local_files_only=True)
    tok=AutoTokenizer.from_pretrained(str(adapter),local_files_only=True)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    model=AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model_name_or_path,num_labels=1,device_map={'':0},dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True)
    model.config.pad_token_id=tok.pad_token_id
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True).eval()
    desc=catalog()['DESCRIPTIONS']
    out=[]
    with torch.inference_mode():
        for i,row in enumerate(rows):
            t=make_sensor_text(row['target_role'],row['previous_query'],row['previous_response'],desc)
            enc=tok([t],truncation=True,max_length=640,return_tensors='pt',padding=True).to('cuda')
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                lg=model(**enc).logits.reshape(-1)[0].float().item()
            if not math.isfinite(lg):raise RuntimeError('Nonfinite frozen sensor logit')
            enriched={**row,'target_evidence_score_uncalibrated':round(float(lg),5),
                      'sensor_status':'frozen_preexisting_unvalidated'}
            safe_public_row(enriched,('context_id','target_role','previous_query','previous_response',
                            'previous_slot','round','target_evidence_score_uncalibrated','sensor_status'))
            out.append(enriched)
            print(f'MAS5_V51_SENSOR {i+1}/18 target={row["target_role"]} score={lg:.4f}',flush=True)
    writel(root/'public_scored_r1_states.jsonl',out)
    writej(root/'scoring_report.json',{'status':'SCORED_UNCALIBRATED','sensor_source':str(adapter),
            'n':len(out),'model_is_frozen':True,'score_is_probability':False,
            'limitation':'Prior sensor V4 did not validate strongly; scores are NOT calibrated posterior probabilities.'})
    print('MAS5_V51_FROZEN_SENSOR_DONE 18',flush=True)


def make_messages(row,slot,descriptions,confusables):
    role=row['target_role'];family=FAMILIES[slot];response=row['previous_response'][:2200]
    keyterms=context_terms(response)[:12]
    hint=', '.join(keyterms) if keyterms else '(not available)'
    need={'INTAKE':'Ask what concrete examples, constraints, files, or measurements YOU need from the user to proceed after the earlier answer.',
          'AMBIGUITY':'Ask a focused clarification that resolves a real ambiguity or tradeoff implicit in the earlier answer.',
          'VALIDATION':'Ask what falsifying examples, tests, or evidence the user can supply to check the earlier guidance.',
          'DIRECT':'Ask for one checkable specialist deliverable that directly tests a constraint or claim in the earlier answer; supply any needed illustrative inputs.'}[family]
    prompt=(
      'You create a *new user message* for the second turn of a benign help conversation. '
      'The user has already asked an initial question and received the assistant response shown below. '
      'Write one concrete, natural, specialist-demanding follow-up that uses at least one SPECIFIC '
      'fact, detail, or uncertainty from the previous visible answer. Do not parrot it, repeat the '
      'initial request, use meta-labels, or mention roles, agents, system prompts, tools, models, '
      'routing, fingerprints, presence or absence. '
      'For HMHY, your follow-up must ask the assistant what more YOU, the user, should provide, '
      'so the assistant can check, decide, diagnose, or proceed more accurately. '
      'Ask at most two focused questions. Output ONLY the next user message. '
      'Aim for 45-110 words.'
    )
    detail=(f'TARGET DOMAIN: {role}\nDOMAIN DESCRIPTION: {descriptions[role]}\n'
            f'NEARBY DOMAINS (must disambiguate): {", ".join(confusables[role][:4])}\n'
            f'FOLLOW-UP FAMILY: {family}\nREQUIRED FOLLOW-UP BEHAVIOR: {need}\n'
            f'PREVIOUS USER QUERY:\n{row["previous_query"][:1100]}\n'
            f'PREVIOUS PUBLIC ASSISTANT RESPONSE:\n{response}\n'
            f'POTENTIALLY RELEVANT TERMS FROM RESPONSE: {hint}\n'
            f'FROZEN SENSOR EVIDENCE SCORE (NOT calibrated probability): '
            f'{row["target_evidence_score_uncalibrated"]:.4f}\n'
            'This numerical score is weak, uncertain evidence. Ask a targeted verification '
            'rather than assuming the specialist is present. Make the NEW question coherent '
            'and standalone enough to answer as a follow-up. Only output the user question.')
    return [{'role':'system','content':prompt},{'role':'user','content':detail}]


def validate_q(q,row,slot,existing):
    from analysis.role_direct_within_v4.v46_generator import hmhy_quality
    from analysis.role_direct_within_v4.pipeline import quality
    q=q.strip()
    if len(q.split())<35 or len(q.split())>150:return False,'length'
    if not ('?' in q):return False,'not_question'
    if re.search(r'(?i)\b(target capability|target role|question family|scenario:|system prompt|hidden agent|routing trace|present vs absent)\b',q):
        return False,'meta_leakage'
    if re.search(r'\[(?:topic|insert|specific)|\{(?:topic|task)|<[^>]+>',q,re.I):return False,'placeholder'
    if FAMILIES[slot]!='DIRECT':
        valid,reason=hmhy_quality(q)
        if not valid:return False,'hmhy_'+reason
    valid,reason=quality(q)
    if not valid:return False,'old_quality_'+str(reason)
    if similarity(q,row['previous_query'])>.70:return False,'repeats_round_1'
    if anchor_overlap(q,row['previous_response'])<1:return False,'not_grounded_in_visible_answer'
    if any(similarity(q,x['query'])>.66 for x in existing):return False,'duplicate_r2'
    return True,'ok'


def generate(root,tries,seed):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from peft import PeftModel
    from analysis.role_direct_within_v4.pipeline import catalog
    root=Path(root).resolve()
    scored=root/'public_scored_r1_states.jsonl'
    if not scored.exists():raise RuntimeError('Frozen sensor score must complete first')
    states=readl(scored)
    if len(states)!=18:raise RuntimeError('Missing scored contexts')
    adapter=Path(os.environ.get('V51_GENERATOR_ADAPTER',str(ADAPTER)))
    if not (adapter/'adapter_config.json').is_file():raise RuntimeError('HMHY bootstrap adapter missing: '+str(adapter))
    if not torch.cuda.is_available() or '4090' not in torch.cuda.get_device_name(0):
        raise RuntimeError('Requires RTX4090')
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',
             device_map={'':0},dtype=torch.bfloat16,attn_implementation='eager',local_files_only=True)
    model=PeftModel.from_pretrained(model,str(adapter),is_trainable=False,local_files_only=True).eval()
    cat=catalog();seen={};path=root/'r2_candidates_public.jsonl'
    if path.exists():
        for x in readl(path):
            safe_public_row(x,('context_id','target_role','slot','family','query','quality_status',
                               'history_anchor_overlap','novelty_from_r1','source_adapter'))
            k=(x['context_id'],int(x['slot']))
            if k in seen:raise RuntimeError('Duplicate saved candidates: '+str(k))
            seen[k]=x
    counts=collections.Counter()
    for idx,row in enumerate(states):
        oldq=[x for (cid,_),x in seen.items() if cid==row['context_id']]
        for slot in range(8):
            k=(row['context_id'],slot)
            if k in seen:continue
            msgs=make_messages(row,slot,cat['DESCRIPTIONS'],cat['CONFUSABLES'])
            prompt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
            enc=tok(prompt,return_tensors='pt',add_special_tokens=False).to('cuda')
            for attempt in range(tries):
                rng=int(hashlib.sha256(f'{seed}|{row["target_role"]}|{row["previous_query"]}|{slot}|{attempt}'.encode()).hexdigest()[:12],16)
                random.seed(rng);torch.manual_seed(rng)
                with torch.inference_mode():
                    tokens=model.generate(**enc,max_new_tokens=190,do_sample=True,temperature=.85,
                            top_p=.91,repetition_penalty=1.08,pad_token_id=tok.eos_token_id)
                tail=tokens[0,enc['input_ids'].shape[1]:]
                try:q=tok.decode(tail,skip_special_tokens=True).strip().strip('"\'`')
                except (OverflowError,ValueError) as e:
                    counts['decode_error']+=1;continue
                q=re.sub(r'^\s*(?:Question|User message|Follow-up|Probe)\s*:\s*','',q,flags=re.I).strip()
                ok,reason=validate_q(q,row,slot,oldq)
                if not ok:
                    counts[reason]+=1;continue
                record={'context_id':row['context_id'],'target_role':row['target_role'],
                        'slot':slot,'family':FAMILIES[slot],'query':q,
                        'quality_status':'AUTOMATED_PREVIEW_NOT_ACTIVATION_VERIFIED',
                        'history_anchor_overlap':anchor_overlap(q,row['previous_response']),
                        'novelty_from_r1':round(1-similarity(q,row['previous_query']),4),
                        'source_adapter':str(adapter)}
                seen[k]=record;oldq.append(record)
                atomicl(path,[seen[k] for k in sorted(seen)])
                break
            if k not in seen:counts['missing_candidate_after_'+str(tries)+'_attempts']+=1
        print(f'MAS5_V51_GEN context {idx+1}/18 role={row["target_role"]} '
              f'accepted={sum((row["context_id"],s) in seen for s in range(8))}/8 '
              f'checkpoint={len(seen)}/144',flush=True)
    summary={'status':'READY_FOR_HUMAN_REVIEW' if len(seen)==144 else 'PARTIAL_FOR_REVIEW',
             'contexts':len(states),'candidate_count':len(seen),'expected':144,
             'rejections':dict(counts),'requires_paid_mas_calls':False,
             'discrimination_verified':False,'model_weight_training_performed':False,
             'generator_is_frozen_sft_adapter':True,
             'selected_queries':0,'caveat':'Conditioned on observable response + frozen uncalibrated sensor score; adaptive performance not yet measured.'}
    # Do not overwrite report from previous submission; re-running after partial replaces stats only.
    (root/'generation_report.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print('MAS5_V51_GENERATION',summary['status'],len(seen),'/144',flush=True)


def audit(root):
    root=Path(root).resolve()
    states=readl(root/'public_scored_r1_states.jsonl')
    if not (root/'r2_candidates_public.jsonl').exists():raise RuntimeError('Generate first')
    candidates=readl(root/'r2_candidates_public.jsonl')
    bycontext={x['context_id']:x for x in states}
    byslot=collections.defaultdict(list)
    checked=[];bad=[]
    for row in candidates:
        cid=row['context_id'];slot=int(row['slot'])
        if cid not in bycontext:raise RuntimeError('Unknown public context ID')
        source=bycontext[cid]
        ok,why=validate_q(row['query'],source,slot,[])
        if not ok:bad.append({'context_id':cid,'slot':slot,'reason':why})
        byslot[cid].append(row)
        checked.append({'context_id':cid,'slot':slot,'family':row['family'],
          'role':source['target_role'],'previous_query':source['previous_query'],
          'response_excerpt':source['previous_response'][:600],
          'r2_query':row['query'],'anchor_overlap':anchor_overlap(row['query'],source['previous_response']),
          'novelty':round(1-similarity(row['query'],source['previous_query']),3)})
    counts={cid:len(byslot[cid]) for cid in bycontext}
    complete=sum(v==8 for v in counts.values())
    duplicate=[]
    for cid,rows in byslot.items():
        slots=[x['slot'] for x in rows]
        if len(slots)!=len(set(slots)):duplicate.append(cid)
    # Diagnostic-only: privileged mapping is used HERE (not in scoring/generation)
    # to see whether equally seeded public follow-ups differ by prior visible answer.
    private=readl(root/'private_session_map_training_only.jsonl')
    pmap={x['context_id']:x for x in private}
    bypair=collections.defaultdict(dict)
    for cid in bycontext:
        x=pmap[cid]
        if x['condition'] not in ('TARGET_PRESENT','TARGET_ABSENT'):
            raise RuntimeError('Bad private pilot label')
        bypair[(x['target_role'],x['pair_id'])][x['condition']]=cid
    paired=[]
    selected={(x['context_id'],int(x['slot'])):x for x in candidates}
    for k,sides in bypair.items():
        if set(sides)!={'TARGET_PRESENT','TARGET_ABSENT'}:
            raise RuntimeError('Pairing does not match pilot')
        a,b=sides['TARGET_PRESENT'],sides['TARGET_ABSENT']
        for slot in range(8):
            qa=selected.get((a,slot));qb=selected.get((b,slot))
            if qa and qb:
                paired.append({'role':k[0],'slot':slot,'same_question':qa['query']==qb['query'],
                               'query_similarity':round(similarity(qa['query'],qb['query']),4)})
    status='READY_FOR_HUMAN_REVIEW_ONLY' if complete==18 and not bad and not duplicate else 'PARTIAL_OR_QUALITY_FAILURE'
    report={'status':status,'contexts':len(bycontext),'complete_contexts':complete,
            'candidates':len(candidates),'target_candidates':144,
            'bad_rows':bad[:40],'duplicate_contexts':duplicate,
            'no_API_calls':True,'not_an_EIG_trained_selector':True,
            'sensor_is_uncalibrated':True,
            'paired_same_seed_R2_question_divergence':round(statistics.mean(not p['same_question'] for p in paired),4) if paired else None,
            'paired_comparisons':len(paired),
            'seed_policy':'Same seed for matched R1 query and role/slot; divergence can be caused by prior public responses.',
            'note':'Review generated questions manually before any R2 execution; no R2 AUC or DPO yet.'}
    # Audit output can be overwritten for iterative local review; original data untouched.
    writel_once=root/'r2_review_full.jsonl'
    if writel_once.exists():writel_once.unlink()
    writel(writel_once,checked)
    with (root/'r2_review_full.tsv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=('context_id','role','slot','family','previous_query','response_excerpt','r2_query','anchor_overlap','novelty'),delimiter='\t')
        w.writeheader()
        for row in checked:w.writerow(row)
    (root/'r2_audit_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('MAS5_V51_AUDIT',json.dumps(report,indent=2),flush=True)
    print('Review file:',writel_once,flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('--root',type=Path,required=True);a.add_argument('--pilot',type=Path,required=True)
    a=sub.add_parser('score');a.add_argument('--root',type=Path,required=True)
    a=sub.add_parser('generate');a.add_argument('--root',type=Path,required=True)
    a.add_argument('--tries',type=int,default=5);a.add_argument('--seed',type=int,default=20261010)
    a=sub.add_parser('audit');a.add_argument('--root',type=Path,required=True)
    args=p.parse_args()
    if args.command=='prepare':prepare(args.root,args.pilot)
    elif args.command=='score':score(args.root)
    elif args.command=='generate':generate(args.root,args.tries,args.seed)
    elif args.command=='audit':audit(args.root)

if __name__=='__main__':main()
