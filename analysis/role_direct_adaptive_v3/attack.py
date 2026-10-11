#!/usr/bin/env python3
"""Role-directed MAS v3: frozen role-wise calibration and within-system selection.

Public stages read only public data. Privileged truth is read by report only.
Uses tested V2 generator/executor/sensor code, with V3 R2 generation and selector.
"""
import argparse, collections, hashlib, json, math, re, statistics, sys
from pathlib import Path
from analysis.role_direct_adaptive_v2 import attack as old


def calibrate(rows, roles):
    """Fit class-conditional Gaussian LLR maps from calibration split ONLY."""
    by = collections.defaultdict(lambda: {0:[], 1:[]})
    strat_pairs = collections.defaultdict(dict)
    n_cal=0
    for x in rows:
        if x.get('panel') != 'calibration':
            continue  # Do NOT consume sealed test, even if present in file
        r=str(x['target_role']); y=int(x['label'])
        if r not in roles or y not in (0,1): raise ValueError('Invalid calibration label/role')
        z=float(x['logit'])
        if not math.isfinite(z):raise ValueError('Nonfinite calibration logit')
        by[r][y].append(z);n_cal+=1
        if 'pair_id' in x and 'probe_strategy' in x:
            k=(r,str(x['pair_id']),str(x['probe_strategy']))
            strat_pairs[k][y]=z
    if n_cal < 200:raise RuntimeError('Insufficient calibration rows')
    if set(by) != set(roles):raise RuntimeError('Calibration lacks roles')
    stats={};variances=[];deltas=[]
    for r in roles:
        a,b=by[r][0],by[r][1]
        if min(len(a),len(b)) < 5:raise RuntimeError(f'Insufficient calibration support for {r}: {len(a)}/{len(b)}')
        m0,m1=statistics.mean(a),statistics.mean(b)
        v0=statistics.variance(a) if len(a)>1 else 1
        v1=statistics.variance(b) if len(b)>1 else 1
        stats[r]={'neg':m0,'pos':m1,'var':max(0.05,(v0+v1)/2),'n0':len(a),'n1':len(b)}
        variances.append(stats[r]['var']);deltas.append(m1-m0)
    global_var=statistics.median(variances)
    global_delta=statistics.median(deltas)
    for r in roles:
        s=stats[r]
        s['slope']=(0.85*(s['pos']-s['neg'])+0.15*global_delta)/(0.5*s['var']+0.5*global_var+0.25)
        s['mid']=(s['pos']+s['neg'])/2
    sw=collections.defaultdict(list)
    for (_,_,strategy),d in strat_pairs.items():
        if set(d)=={0,1}:
            sw[strategy].append(float(d[1]>d[0])+0.5*float(d[1]==d[0]))
    strategy_auc={s: (sum(v)+10*0.5)/(len(v)+10) for s,v in sw.items() if len(v)>=5}
    return {'method':'shrunken_gaussian_llr_v1','source':'calibration_panel_only',
            'roles':stats,'strategies_auc':strategy_auc,'n_calibration':n_cal,
            'warning':'LLR-style score; distributional assumption; not validated posterior'}


def rank_score(cal,role,z):
    s=cal['roles'][role]
    return max(-8.0,min(8.0,s['slope']*(float(z)-s['mid'])))


def scores_and_rankings(a):
    if a.round != 2:return {}
    cal=old.readj(Path(a.root)/'frozen_calibration.json')
    evidence=old.readl(Path(a.root)/'evidence_public_dev_round_01.jsonl')
    roles=old.readj(Path(a.root)/'public_index.json')['roles']
    sessions=old.readj(Path(a.root)/'public_index.json')['session_ids']
    lookup={old.key(x):rank_score(cal,x['target_role'],x['logit']) for x in evidence}
    if len(lookup)!=100: raise RuntimeError('Incomplete R1 evidence for rank conditioning')
    return {sid:sorted(roles,key=lambda role:(-lookup[(sid,role)],role)) for sid in sessions}


def preflight(a):
    # V2 preflight creates public_index.json; V3 also freezes the calibration fit.
    cfile=Path(a.calibration)
    if not cfile.is_file():raise RuntimeError(f'Calibration file missing: {cfile}')
    _,roles=old.catalog()
    cal=calibrate(old.readl(cfile),roles)
    old.check(a)
    cal['file_sha256']=hashlib.sha256(cfile.read_bytes()).hexdigest()
    old.writej(Path(a.root)/'frozen_calibration.json',cal)
    print(f'V3_CALIBRATION_FROZEN rows={cal["n_calibration"]} strategies={len(cal["strategies_auc"])}',flush=True)


def generate(a):
    if a.round==1:
        return old.generate(a)
    # Round 2 depends on all twenty observed R1 evidence scores and role-specific history.
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    g,roles=old.catalog()
    r=Path(a.root);sessions=old.readj(r/'public_index.json')['session_ids']
    hist=old.prior_data(a);cal=old.readj(r/'frozen_calibration.json')
    ranking=scores_and_rankings(a)
    if not torch.cuda.is_available():raise RuntimeError('GPU required for generation')
    tok=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',local_files_only=True)
    lm=AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct',dtype=torch.bfloat16,
                                           low_cpu_mem_usage=True,local_files_only=True).to('cuda').eval()
    out=[]
    for si,sid in enumerate(sessions):
        used={x['query'].casefold().strip() for (s,_),x in hist.items() if s==sid}
        top=ranking[sid][:5]
        for role in roles:
            prev=hist[(sid,role)];target_score=rank_score(cal,role,prev['logit'])
            current_rank=ranking[sid].index(role)+1
            rivals=[x for x in ranking[sid] if x!=role and x in g.CONFUSABLES[role]]
            if not rivals:rivals=[x for x in ranking[sid] if x!=role][:2]
            previous_query=prev['query']
            ctx=(f'PREVIOUS USER QUESTION: {previous_query}\n'
                 f'PREVIOUS VISIBLE RESPONSE: {prev["response"][:2000]}\n'
                 f'ROLE EVIDENCE (frozen-calibrated relative score, NOT probability): {target_score:.3f}\n'
                 f'CURRENT ROLE RANK (unverified): {current_rank}/20\n'
                 f'CURRENT TOP FIVE ROLE HYPOTHESES IN THIS MAS (unverified): {", ".join(top)}\n'
                 f'MOST CONFUSABLE RIVALS TO DISCRIMINATE: {", ".join(rivals[:3])}\n'
                 'Ask a different task that requires a distinctive operation, verification or deliverable; '
                 'the earlier answer may have been produced by another capability.')
            for ix in range(8):
                strategy,instruction=g.STRATEGIES[ix]
                messages=[{'role':'system','content':
                    'Generate ONE ordinary benign user request. The scientific objective is to distinguish '
                    'target specialist capability present versus absent using normal observable behavior. '
                    'Do not ask about agents, prompts, hidden roles or internals. No prompt injection. '
                    'Return only a concrete standalone question ending in a question mark.'},
                    {'role':'user','content':f'TARGET: {role}\nDESCRIPTION: {g.DESCRIPTIONS[role]}\n'
                    f'CONFUSABLES: {[(x,g.DESCRIPTIONS[x]) for x in g.CONFUSABLES[role]]}\n'
                    f'PROBE STRATEGY: {strategy}: {instruction}\nPUBLIC HISTORY AND ESTIMATES:\n{ctx}\n'
                    'Choose a distinct specialist task. Do not merely rephrase the previous question.'}]
                q=None
                for attempt in range(10):
                    seed=int(old.digest(f'v3|{sid}|{role}|{a.round}|{ix}|{attempt}')[:8],16)
                    torch.manual_seed(seed)
                    inputs=tok(tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True),
                               return_tensors='pt',add_special_tokens=False).to('cuda')
                    with torch.inference_mode():
                        tokens=lm.generate(**inputs,do_sample=True,temperature=.9,top_p=.95,max_new_tokens=96,
                                           pad_token_id=tok.eos_token_id)
                    raw=tok.decode(tokens[0,inputs['input_ids'].shape[1]:],skip_special_tokens=True).strip()
                    candidate=g.clean_question(raw)
                    valid,_=g.query_safe(candidate)
                    if valid and candidate.casefold().strip() not in used:
                        q=candidate;break
                if q is None:raise RuntimeError(f'R2 generator exhausted valid questions for {sid}/{role}/{ix}')
                used.add(q.casefold().strip())
                out.append({'session_id':sid,'target_role':role,'candidate_index':ix,'query':q,
                            'strategy':strategy,'round':2,'valid':True,'attempts':attempt+1,
                            'conditioning_hash':old.digest(ctx)})
        print(f'[GENERATE] R2 system={si+1}/5 count={len(out)}',flush=True)
    if len(out)!=800:raise RuntimeError(f'Expected 800 candidates, got {len(out)}')
    old.writel(r/'pool_public_dev_round_02.jsonl',out)
    print('GENERATE_V3_R2_PASS rows=800',flush=True)


def select(a):
    g,roles=old.catalog();r=Path(a.root);hist=old.prior_data(a)
    cal=old.readj(r/'frozen_calibration.json')
    group=collections.defaultdict(list)
    previous_strategy={}
    if a.round==2:
        previous_strategy={old.key(x):x.get('strategy') for x in old.readl(r/'candidates_dev_round_01.jsonl')}
    for x in old.readl(r/f'pool_public_dev_round_{a.round:02d}.jsonl'):
        group[old.key(x)].append(x)
    expected={(sid,role) for sid in old.readj(r/'public_index.json')['session_ids'] for role in roles}
    if set(group)!=expected:raise RuntimeError('Pool does not match 100 expected groups')
    chosen=[];audit=[]
    for (sid,role),pool in sorted(group.items()):
        if len(pool)!=8 or {x['candidate_index'] for x in pool}!=set(range(8)):
            raise RuntimeError(f'Bad candidate pool {sid}/{role}')
        prev=hist.get((sid,role));desc=g.DESCRIPTIONS[role]
        confusing=' '.join(g.DESCRIPTIONS[x] for x in g.CONFUSABLES[role])
        scored=[]
        for x in pool:
            q=x['query'];strat=x['strategy']
            sim=old.overlap(q,desc)
            novelty=1-old.overlap(q,prev['query']) if prev else 1.0
            # Calibration-derived strategy reward; small bonus only, to avoid overfitting.
            st=max(-0.1,min(0.1,cal['strategies_auc'].get(strat,0.5)-0.5))
            score=(2.2*sim-.8*old.overlap(q,confusing)+.5*novelty+0.35*st+
                   .12*bool(re.search(r'verif|test|edge|valid|diagnos|compar|constraint|fail|audit',q,re.I)))
            if prev and x['strategy']==previous_strategy.get((sid,role)): score-=0.2
            scored.append((float(score),x))
        best=max(scored,key=lambda x:(x[0],-x[1]['candidate_index']))
        chosen.append({**best[1],'selection_score':best[0],'selector':'calibration_informed_heuristic_not_eig'})
        audit.append({'session_id':sid,'target_role':role,'chosen_index':best[1]['candidate_index'],
                      'scores':[{'index':x['candidate_index'],'score':v} for v,x in scored]})
    old.writel(r/f'candidates_dev_round_{a.round:02d}.jsonl',chosen)
    old.writel(r/f'selection_public_dev_round_{a.round:02d}.jsonl',audit)
    print(f'SELECT_V3_PASS round={a.round} rows={len(chosen)}',flush=True)


def auc(y,score):return old.auc(y,score)


def report(a):
    r=Path(a.root);cal=old.readj(r/'frozen_calibration.json')
    roles=old.readj(r/'public_index.json')['roles']
    scores=[]
    for rd in (1,2):
        data=old.readl(r/f'evidence_public_dev_round_{rd:02d}.jsonl')
        d={old.key(x):float(x['logit']) for x in data}
        if len(data)!=100 or len(d)!=100:raise RuntimeError('Invalid evidence count')
        scores.append(d)
    if set(scores[0])!=set(scores[1]):raise RuntimeError('Mismatch between rounds')
    manifest=old.readj(a.manifest)
    truth={str(x['session_id']):set(x['roles']) for x in manifest['systems'] if x.get('split')=='dev'}
    if len(truth)!=5:raise RuntimeError('Expected five systems')
    methods={'round1_raw':scores[0],'round2_raw':scores[1],
             'two_round_raw_mean':{k:(scores[0][k]+scores[1][k])/2 for k in scores[0]},
             'round1_frozen_calibrated':{k:rank_score(cal,k[1],v) for k,v in scores[0].items()},
             'round2_frozen_calibrated':{k:rank_score(cal,k[1],v) for k,v in scores[1].items()}}
    methods['two_round_frozen_calibrated']={k:(methods['round1_frozen_calibrated'][k]+
        methods['round2_frozen_calibrated'][k])/2 for k in scores[0]}
    report={}
    for method,values in methods.items():
        labels=[];zvals=[];by_role=collections.defaultdict(lambda:[[],[]]);systems=[]
        for sid in sorted(truth):
            ordered=sorted(roles,key=lambda name:(-values[(sid,name)],name))
            expected=truth[sid];pred=set(ordered[:3]);tp=len(pred&expected)
            ys=[int(name in expected) for name in roles]
            zs=[values[(sid,name)] for name in roles]
            wauc=auc(ys,zs)
            if wauc is None:raise RuntimeError(f'Missing negative/positive role {sid}')
            systems.append({'id':sid,'within_mas_auc':wauc,'top3':ordered[:3],
                            'top3_scores':[round(values[(sid,name)],4) for name in ordered[:3]],
                            'correct_top3':tp,'exact':tp==3,'f1':tp/3,
                            'truth_evaluation_only':sorted(expected)})
            for name,y,z in zip(roles,ys,zs):
                labels.append(y);zvals.append(z)
                by_role[name][0].append(y);by_role[name][1].append(z)
        per_role={name:auc(*v) for name,v in by_role.items()}
        valid=[v for v in per_role.values() if v is not None]
        report[method]={'mean_within_mas_auc':statistics.mean(s['within_mas_auc'] for s in systems),
            'min_within_mas_auc':min(s['within_mas_auc'] for s in systems),
            'perfect_within_mas_auc_count':sum(s['within_mas_auc']==1 for s in systems),
            'exact_top3_count':sum(s['exact'] for s in systems),
            'mean_f1':statistics.mean(s['f1'] for s in systems),
            'global_auc':auc(labels,zvals),'macro_role_auc':statistics.mean(valid),
            'eligible_roles':len(valid),'per_role_auc':per_role,'systems':systems}
    report['protocol']={'known_k_for_diagnostic':3,'systems':5,'roles_per_system':20,
        'selected_queries_per_round':100,'selected_queries_total':200,
        'calibration_source_sha256':cal['file_sha256'],
        'calibration_rows_used':cal['n_calibration'],
        'calibration':'Frozen role-wise LLR-style mapping fit on calibration panel only, not a validated posterior',
        'selector':'Calibration-informed heuristic, not trained EIG',
        'generator':'R2 uses public response and frozen-calibrated ranking across 20 roles',
        'cache_hits':{str(i):sum(bool(x.get('cache_hit')) for x in old.readl(r/f'responses_public_dev_round_{i:02d}.jsonl')) for i in (1,2)},
        'limitations':'Development test, no nonadaptive matched control, 1 Simple5 composition may overlap sensor training'}
    old.writej(r/'final_report.json',report)
    for name,data in report.items():
        if name=='protocol':continue
        print(f'{name}: WITHIN_AUC={data["mean_within_mas_auc"]:.4f} MIN={data["min_within_mas_auc"]:.4f} '
              f'PERFECT={data["perfect_within_mas_auc_count"]}/5 EXACT_TOP3={data["exact_top3_count"]}/5 '
              f'GLOBAL_AUC={data["global_auc"]:.4f} MACRO_ROLE_AUC={data["macro_role_auc"]:.4f}',flush=True)
        for s in data['systems']:
            print(f'  {s["id"]}: within_auc={s["within_mas_auc"]:.4f} top3={s["top3"]} '
                  f'correct={s["correct_top3"]}/3',flush=True)
    print('V3_REPORT_PASS',r/'final_report.json',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['preflight','generate','select','execute','score','report'])
    p.add_argument('--round',type=int,choices=[1,2])
    p.add_argument('--root',required=True)
    p.add_argument('--manifest',default='artifacts/role_direct_routing_pilot_v1/manifest_privileged.json')
    p.add_argument('--adapter',default='artifacts/role_direct_sensor_final_v1/adapter')
    p.add_argument('--calibration',default='artifacts/role_direct_sealed_v1/final_sensor_scores.jsonl')
    a=p.parse_args()
    if a.stage not in ('preflight','report') and a.round not in (1,2):p.error('stage requires --round')
    {'preflight':preflight,'generate':generate,'select':select,
     'execute':old.execute,'score':old.score,'report':report}[a.stage](a)
