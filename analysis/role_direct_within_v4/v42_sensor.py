#!/usr/bin/env python3
"""V4.2: system-holdout sensor diagnostic and conservative non-neural benchmark.

Read-only source data. Trains fresh baselines under V4. Never touches earlier weights.
Model selection is based ONLY on TRAIN out-of-fold (grouped by MAS), not DEV.
DEV is an already explored development set, never a sealed generalization claim.
"""
from __future__ import annotations
import argparse, ast, collections, json, math, os, re, statistics
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import joblib

PROJECT=Path(os.environ.get('V42_PROJECT_ROOT','/home/wieseld/mas_rl_detective'))
SOURCE=PROJECT/'artifacts/v137_presence_v1/dataset_v1'
CATALOG=PROJECT/'analysis/role_direct_approach/generate_direct_panel_v1.py'
REF_AUC=0.5745533410732716   # Earlier stage0 query_answer / 38 DEV systems
SEED=20261009
WORD=re.compile(r"[a-z][a-z_'-]{2,}",re.I)


def readl(path):
    with Path(path).open() as f:return [json.loads(s) for s in f if s.strip()]


def writej(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n')
    os.replace(tmp,path)


def descriptions():
    tree=ast.parse(CATALOG.read_text());out={}
    for node in tree.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id=='DESCRIPTIONS':out=ast.literal_eval(node.value)
    assert len(out)==20,'20 candidate roles required'
    return out


def load():
    tr=readl(SOURCE/'train_v137.jsonl');dev=readl(SOURCE/'dev_v137.jsonl')
    for rows in (tr,dev):
        for x in rows:
            for k in ('system_id','target_role','query','final_output','label'):
                assert k in x,f'missing {k}'
            assert int(x['label']) in (0,1)
    ta={str(x['system_id']) for x in tr};da={str(x['system_id']) for x in dev}
    assert not ta&da,'leakage: identical systems across train/dev'
    # Also prevent identical (role,query,response) train-to-dev copies.
    sig=lambda r:(r['target_role'],r['query'].strip().lower(),r['final_output'].strip().lower())
    assert not ({sig(r) for r in tr}&{sig(r) for r in dev}),'duplicate train/dev observations'
    return tr,dev


def role_counts(rows):
    counts=collections.defaultdict(lambda:collections.Counter())
    for r in rows:
        counts[str(r['system_id'])]['total']+=1
        counts[str(r['system_id'])]['positive']+=int(r['label'])
        counts[str(r['system_id'])]['roles__'+str(r['target_role'])]+=1
    sizes=[v['total'] for v in counts.values()]
    mixed=sum(0<v['positive']<v['total'] for v in counts.values())
    return {'systems':len(counts),'rows':len(rows),'rows_per_system_histogram':dict(sorted(collections.Counter(sizes).items())),
      'median_rows_per_system':float(statistics.median(sizes)),
      'mixed_label_systems':mixed,'with_20_roles':sum(len([k for k in v if k.startswith('roles__')])==20 for v in counts.values()),
      'role_label_counts':{role:dict(cnt) for role,cnt in sorted(role_label_counts(rows).items())}}


def role_label_counts(rows):
    c=collections.defaultdict(collections.Counter)
    for r in rows:c[str(r['target_role'])][str(int(r['label']))]+=1
    return c


def raw_features(r,ds):
    answer=str(r['final_output']);query=str(r['query']);role=str(r['target_role']);desc=ds[role]
    a=WORD.findall(answer.lower());q=WORD.findall(query.lower());d=WORD.findall(desc.lower())
    aw=set(a);qw=set(q);dw=set(d)
    n=max(1,len(a));denq=max(1,len(qw));dend=max(1,len(dw))
    # All of these are PUBLIC-OBSERVATION features; there is no role truth/tracing.
    return [min(n/300,3), min(len(query)/1200,3),len(set(a))/n,
       len(aw&qw)/denq, len(aw&dw)/dend, len(qw&dw)/dend,
       min(answer.count('\n')/20,3),min(answer.count('```')/6,3),
       min(answer.count('|')/20,3),min(answer.count(':')/20,3),
       min(len(re.findall(r'\b\d+(?:\.\d+)?\b',answer))/25,3),
       min(answer.count('- ')/15,3),int(bool(re.search(r'(?i)\b(i cannot|unable to|cannot access|as an ai)\b',answer))),
       int(bool(re.search(r'(?i)\b(example|step|criteria|constraint|validation|evidence|trade.off|assumption)\b',answer))),
       int(bool(re.search(r'(?i)\b(test|bug|database|source|citation|translate|incident|stakeholder)\b',answer))),
       min(len(set(WORD.findall(query.lower())))/50,3)]


def render(r,mode,ds):
    role=str(r['target_role']);q=' '.join(str(r['query']).split());a=' '.join(str(r['final_output']).split())
    if mode=='query':return f'TARGET {role} TASK {q}'
    if mode=='answer':return f'TARGET {role} RESPONSE {a}'
    if mode=='qa':return f'TARGET {role} TASK {q} RESPONSE {a}'
    if mode=='desc_qa':return f'TARGET {role} DESCRIPTION {ds[role]} TASK {q} RESPONSE {a}'
    if mode=='role_interaction':
        words=set(WORD.findall(a.lower()))
        # Pooled across MAS without internals; targets learn distinguishable response fingerprints.
        tags=' '.join(f'{role}__{w}' for w in sorted(words)[:100])
        return f'TARGET {role} DESCRIPTION {ds[role]} TASK {q} RESPONSE {a} ROLE_RESPONSE {tags}'
    raise ValueError(mode)


class Ranker:
    def __init__(self,mode='qa',C=1.,pairwise=False, use_char=False, use_num=True):
        self.mode=mode;self.C=C;self.pairwise=pairwise;self.use_char=use_char;self.use_num=use_num
    def make_x(self,rows,ds,train=False):
        strings=[render(r,self.mode,ds) for r in rows]
        if train:
            self.vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=40000,sublinear_tf=True,strip_accents='unicode')
            x=self.vec.fit_transform(strings)
        else:x=self.vec.transform(strings)
        chunks=[x]
        if self.use_char:
            if train:
                self.cvec=TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=3,max_features=40000,sublinear_tf=True)
                xc=self.cvec.fit_transform(strings)
            else:xc=self.cvec.transform(strings)
            chunks.append(xc*.65)
        if self.use_num:
            raw=np.asarray([raw_features(r,ds) for r in rows],dtype=np.float64)
            if train:
                self.mu=raw.mean(0);self.std=np.maximum(raw.std(0),.2)
            chunks.append(csr_matrix(((raw-self.mu)/self.std)*.075))
        return hstack(chunks,format='csr')
    def fit(self,rows,ds):
        x=self.make_x(rows,ds,True);y=np.array([int(r['label']) for r in rows]);assert len(set(y))==2
        self.model=LogisticRegression(C=self.C,solver='liblinear',max_iter=600,class_weight='balanced',random_state=SEED)
        if not self.pairwise:
            self.model.fit(x,y);return self
        grouped=collections.defaultdict(lambda:{0:[],1:[]})
        for i,r in enumerate(rows):grouped[str(r['system_id'])][int(r['label'])].append(i)
        xp=[];yp=[]
        rng=np.random.default_rng(SEED)
        for group in grouped.values():
            if not group[0] or not group[1]:continue
            pairs=[(p,n) for p in group[1] for n in group[0]]
            rng.shuffle(pairs)
            for p,n in pairs[:80]:
                delta=x[p]-x[n]
                xp.extend([delta,-delta]);yp.extend([1,0])
        if len(yp)<80:raise RuntimeError(f'Insufficient within-system training pairs: {len(yp)//2}')
        self.model=LogisticRegression(C=self.C,solver='liblinear',fit_intercept=False,max_iter=800,random_state=SEED)
        self.model.fit(vstack(xp,format='csr'),yp)
        return self
    def score(self,rows,ds):
        x=self.make_x(rows,ds,False)
        return np.asarray(self.model.decision_function(x),dtype=float)


def system_metrics(rows,scores):
    gs=collections.defaultdict(list)
    for r,s in zip(rows,scores):gs[str(r['system_id'])].append((str(r['target_role']),int(r['label']),float(s)))
    per=[]
    for sid,items in sorted(gs.items()):
        # A role can appear multiple times from repeated probes. Aggregate without truth information.
        rol=collections.defaultdict(list)
        for role,label,score in items:rol[role].append((label,score))
        byrole=[]
        for role,v in rol.items():
            assert len({a for a,b in v})==1,'Inconsistent truth for repeated role'
            byrole.append((role,v[0][0],statistics.mean(z for _,z in v)))
        y=[z[1] for z in byrole];s=[z[2] for z in byrole]
        if len(set(y))!=2:continue
        k=sum(y);top=sorted(byrole,key=lambda z:(-z[2],z[0]))[:k]
        per.append({'system_id':sid,'n_roles':len(byrole),'k':k,'auc':float(roc_auc_score(y,s)),
                    'exact':{r for r,l,_ in top}=={r for r,l,_ in byrole if l},
                    'topk_role_f1':len({r for r,l,_ in top}&{r for r,l,_ in byrole if l})/k,
                    'top_roles':[r for r,_,_ in top]})
    if not per:return {'evaluable_systems':0}
    aucs=[r['auc'] for r in per]
    k3=[r for r in per if r['k']==3];k4=[r for r in per if r['k']==4]
    return {'evaluable_systems':len(per),'within_auc':statistics.mean(aucs),
            'exact_topk':sum(x['exact'] for x in per),'exact_total':len(per),
            'mean_topk_role_f1':statistics.mean(x['topk_role_f1'] for x in per),
            'frequent_top_roles':dict(collections.Counter(r for x in per for r in x['top_roles']).most_common(12)),
            'k3_systems':len(k3),'k3_within_auc':statistics.mean(r['auc'] for r in k3) if k3 else None,
            'k3_exact':sum(r['exact'] for r in k3),'k4_systems':len(k4),
            'k4_within_auc':statistics.mean(r['auc'] for r in k4) if k4 else None,
            'k4_exact':sum(r['exact'] for r in k4),
            'min_within_auc':min(aucs),'per_system':per}


def coverage(rows):
    cnt=collections.Counter(str(r['target_role']) for r in rows)
    return dict(sorted(cnt.items()))


def bootstrap_auc(rows,scores,n=1000):
    metrics=system_metrics(rows,scores);data=metrics.get('per_system',[])
    if not data:return None
    vals=np.asarray([r['auc'] for r in data]);rng=np.random.default_rng(SEED)
    means=np.mean(rng.choice(vals,size=(n,len(vals)),replace=True),axis=1)
    return [float(x) for x in np.quantile(means,[.025,.975])]


def run(root):
    train,dev=load();ds=descriptions()
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    assert root.is_relative_to(PROJECT/'artifacts/role_direct_within_v4'),'V4 output only'
    query_signatures=lambda rows:{(str(r['target_role']),str(r['query']).strip().lower()) for r in rows}
    diagnostic={'train':role_counts(train),'dev':role_counts(dev),
      'query_signature_overlap_train_dev':len(query_signatures(train)&query_signatures(dev)),
      'note':'Train groups may not be complete 20-role rankings. DEV repeatedly inspected; not a sealed test.'}
    writej(root/'data_diagnostic.json',diagnostic)
    print('DATASET TRAIN',diagnostic['train']['systems'],'systems; role rows histogram',diagnostic['train']['rows_per_system_histogram'],flush=True)
    print('DATASET DEV',diagnostic['dev']['systems'],'systems; role rows histogram',diagnostic['dev']['rows_per_system_histogram'],flush=True)
    config=[('query_control','query',.5,False,False),('answer_control','answer',1.,False,False),
            ('qa_word','qa',1.,False,False),('qa_char','qa',.7,False,True),
            ('desc_qa','desc_qa',.5,False,False),('role_fingerprint','role_interaction',.25,False,False),
            ('within_pair_ranking','desc_qa',.5,True,False)]
    split=GroupKFold(n_splits=4)
    groups=np.asarray([str(r['system_id']) for r in train]);y=np.asarray([int(r['label']) for r in train]);
    oof_reports=[]
    for name,mode,C,pairwise,chars in config:
        oof=np.full(len(train),np.nan);failure=None
        try:
            for f,(tr,te) in enumerate(split.split(np.zeros(len(train)),y,groups),1):
                fit=[train[i] for i in tr];test=[train[i] for i in te]
                model=Ranker(mode,C,pairwise,chars).fit(fit,ds)
                oof[te]=model.score(test,ds)
            if not np.all(np.isfinite(oof)):raise RuntimeError('Missing OOF scores')
            auc=float(roc_auc_score(y,oof));within=system_metrics(train,oof)
            oof_reports.append({'name':name,'mode':mode,'C':C,'pairwise':pairwise,'chars':chars,
               'oof_global_auc':auc,'oof_within_auc':within.get('within_auc'),
               'oof_evaluable_systems':within.get('evaluable_systems',0),'error':None})
            print('OOF',name,'GLOBAL_AUC',round(auc,4),'WITHIN_AUC',within.get('within_auc'),flush=True)
        except Exception as e:
            failure=repr(e);oof_reports.append({'name':name,'error':failure})
            print('SKIP',name,failure,flush=True)
    viable=[x for x in oof_reports if not x.get('error')]
    if not viable:raise RuntimeError('No fitted benchmark methods; see oof report')
    # Internal TRAIN OOF selection only, no DEV-driven hyperparameter search.
    enough=min([x['oof_evaluable_systems'] for x in viable])>=12
    score_key='oof_within_auc' if enough else 'oof_global_auc'
    # Controls remain controls, not promoted without response signals.
    eligible=[x for x in viable if x['name'] not in ('query_control','answer_control')]
    selected=max(eligible,key=lambda x:(x[score_key],x['oof_global_auc']))
    print('SELECTED_FROM_TRAIN_OOF',selected['name'],'metric',score_key,flush=True)
    results={}
    for name in ('query_control','answer_control','qa_word',selected['name']):
        if name in results:continue
        record=next((x for x in viable if x['name']==name),None)
        if record is None:continue
        m=Ranker(record['mode'],record['C'],record['pairwise'],record['chars']).fit(train,ds)
        pred=m.score(dev,ds)
        metrics=system_metrics(dev,pred)
        results[name]={'metrics':metrics,'ci95_by_system':bootstrap_auc(dev,pred)}
        if name==selected['name']:
            dest=root/'sensor_candidate.joblib';joblib.dump({'ranker':m,'catalog':ds,'trained_on':'V137 TRAIN ONLY',
                'cross_validation':'GROUPED_BY_MAS','model_name':selected['name']},dest)
            writej(root/'sensor_candidate_scores.json',[
                {'system_id':r['system_id'],'target_role':r['target_role'],'label_dev_audit_only':int(r['label']),'score':float(s)}
                 for r,s in zip(dev,pred)])
        print('DEV',name,'AUC',round(metrics['within_auc'],5),'EXACT',metrics['exact_topk'],'/',metrics['exact_total'],flush=True)
    winning=results[selected['name']]['metrics']['within_auc']
    ref=max(REF_AUC,results.get('qa_word',{}).get('metrics',{}).get('within_auc',float('-inf')))
    # 'PASS' does NOT mean prospective/real-world accuracy validated.
    gate='PROMISING_DEV_ONLY' if winning>=ref+.02 else 'NO_VALIDATED_GAIN_KEEP_PRIORS'
    out={'status':'COMPLETED_OFFLINE','train_dev_diagnostic':diagnostic,'model_selection_metric':score_key,
         'train_oof_models':oof_reports,'selected_from_train_oof':selected['name'],
         'dev_results':results,'previous_stage0_query_answer_auc':REF_AUC,
         'promotion_gate':gate,'warnings':['DEV has been explored repeatedly: no sealed claim',
         'Full role-ranking and matched PRESENT/ABSENT still require fresh data',
         'No Generator/Selector reward validation occurs in this CPU task']}
    writej(root/'sensor_benchmark.json',out)
    print('V42_SUMMARY',gate,'selected_auc',winning,'reference',ref,flush=True)


def score_external(root,path,out):
    data=joblib.load(Path(root)/'sensor_candidate.joblib')
    rows=readl(path);sc=data['ranker'].score(rows,data['catalog'])
    with Path(out).open('w') as f:
        for r,s in zip(rows,sc):
            f.write(json.dumps({'system_id':r.get('system_id'),'target_role':r['target_role'],
                                'query':r['query'],'sensor_score':float(s)})+'\n')
    print('SCORED',len(rows),out)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['benchmark','predict']);ap.add_argument('--root',required=True)
    ap.add_argument('--input');ap.add_argument('--output');a=ap.parse_args()
    if a.stage=='benchmark':run(Path(a.root).resolve())
    else:
        if not a.input or not a.output:ap.error('--input and --output required for predict')
        score_external(a.root,a.input,a.output)
