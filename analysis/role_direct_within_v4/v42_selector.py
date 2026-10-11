#!/usr/bin/env python3
"""Only train a query-selector after FRESH matched Present/Absent evidence exists.

Uses out-of-target-role folds and comparison against random; avoids exploiting
previously reused 3-query/role DEV proxy. Runtime never needs routing labels.
"""
from __future__ import annotations
import argparse, collections, json, math, random, re, statistics
from pathlib import Path
import numpy as np
from scipy.sparse import hstack,csr_matrix,vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
import joblib
from analysis.role_direct_within_v4 import v42_sensor

SEED=20261009

def query_features(q,desc):
    text=q.lower();words=v42_sensor.WORD.findall(text)
    ds=set(v42_sensor.WORD.findall(desc.lower()))
    overlap=len(ds&set(words))/max(1,len(ds))
    return [min(len(q)/1400,3),min(len(words)/180,3),len(set(words))/max(1,len(words)),
       overlap,min(len(re.findall(r'\d',q))/15,3),int(bool(re.search(r'(?i)\b(schema|audit|verify|test|trade.off|compare|design|incident|analyze|experiment)\b',q))),
       int('?' in q),int('if you were' in text or 'what would you do' in text),
       min(q.count(';')/4,3),min(q.count(':')/4,3)]

class Selector:
    def fit_features(self,rows,descs,train=False):
        txt=[f"QUERY {r['query']} TARGET_DESCRIPTION {descs[r['role']]}" for r in rows]
        num=np.array([query_features(r['query'],descs[r['role']]) for r in rows],dtype=float)
        if train:
            self.vec=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=20000,sublinear_tf=True)
            self.mu=num.mean(0);self.std=np.maximum(num.std(0),.2)
            X=self.vec.fit_transform(txt)
        else:X=self.vec.transform(txt)
        return hstack([X,csr_matrix(((num-self.mu)/self.std)*.15)],format='csr')
    def fit(self,rows,descs):
        X=self.fit_features(rows,descs,True);comparisons=[];classes=[]
        groups=collections.defaultdict(list)
        for i,r in enumerate(rows):groups[r['role']].append(i)
        for indices in groups.values():
            for ai in range(len(indices)):
                for bi in range(ai+1,len(indices)):
                    a,b=indices[ai],indices[bi];gap=rows[a]['utility']-rows[b]['utility']
                    if abs(gap)<.08:continue
                    dx=X[a]-X[b]
                    comparisons.extend([dx,-dx]);classes.extend([int(gap>0),int(gap<0)])
        if len(classes)<80:raise RuntimeError('Not enough useful selector preference comparisons')
        self.model=LogisticRegression(C=.5,solver='liblinear',fit_intercept=False,max_iter=900)
        self.model.fit(vstack(comparisons,format='csr'),classes)
        return self
    def scores(self,rows,descs):return self.model.decision_function(self.fit_features(rows,descs,False))


def train(input_file,out_dir):
    inp=Path(input_file);out=Path(out_dir)
    assert out.is_relative_to(v42_sensor.PROJECT/'artifacts/role_direct_within_v4'), 'Only V4 output permitted'
    assert not out.exists(),'No overwriting existing selector results'
    ds=v42_sensor.descriptions();rows=[]
    for r in v42_sensor.readl(inp):
        if not all(k in r for k in ('role','query','background_pairs','target_activation_rate','paired_win_rate','mean_signed_score_delta')):
            raise RuntimeError('Missing fresh matched-evidence fields')
        if r['background_pairs']<3:continue
        # Conditions: high utility demands BOTH activation and observable discrimination.
        gap=max(-15.,min(15.,float(r['mean_signed_score_delta'])))
        utility=float(r['target_activation_rate'])*float(r['paired_win_rate'])*max(0.,math.tanh(gap/2))
        rows.append({'role':r['role'],'query':r['query'],'utility':utility})
    roles=sorted({r['role'] for r in rows})
    byrole=collections.Counter(r['role'] for r in rows)
    if len(rows)<120 or len(roles)<15 or min(byrole.values())<5:
        raise RuntimeError(f'Insufficient FRESH query variety for selector: {len(rows)} queries, {len(roles)} roles, min/role {min(byrole.values(),default=0)}')
    groups=np.array([r['role'] for r in rows]);preds=np.full(len(rows),np.nan)
    for train_i,test_i in GroupKFold(n_splits=5).split(np.zeros(len(rows)),groups=groups):
        model=Selector().fit([rows[i] for i in train_i],ds)
        preds[test_i]=model.scores([rows[i] for i in test_i],ds)
        assert not set(groups[train_i])&set(groups[test_i])
    if not np.isfinite(preds).all():raise RuntimeError('Incomplete selector OOF')
    gains=[];per_role={}
    for role in roles:
        idx=[i for i,r in enumerate(rows) if r['role']==role]
        choice=rows[max(idx,key=lambda i:preds[i])]['utility']
        random_utility=statistics.mean(rows[i]['utility'] for i in idx)
        gains.append(choice-random_utility)
        per_role[role]={'selected':choice,'random_expected':random_utility,'gain':choice-random_utility}
    rng=np.random.default_rng(SEED)
    boots=np.mean(rng.choice(gains,size=(4000,len(gains)),replace=True),axis=1)
    ci=np.quantile(boots,[.025,.975]);mean=float(statistics.mean(gains))
    verdict='LEARNED_HELDOUT_ROLE_PASS' if float(ci[0])>0 and mean>.015 else 'FALLBACK_TO_FIXED_OR_RANDOM'
    final=Selector().fit(rows,ds)
    out.mkdir(parents=True)
    joblib.dump({'selector':final,'descriptions':ds,'verdict':verdict,
        'input':str(inp),'supports_public_R2_context':False},out/'selector_candidate.joblib')
    report={'roles':len(roles),'n_unique_probes':len(rows),'mean_vs_random':mean,
        'ci95_role_bootstrap':[float(ci[0]),float(ci[1])],'verdict':verdict,'per_role':per_role,
        'limitation':'R2-conditioned inputs are not in the training data; no prospective MAS test yet.'}
    v42_sensor.writej(out/'selector_report.json',report)
    print('V42_SELECTOR',verdict,'GAIN',mean,'CI95',ci,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();train(a.evidence,Path(a.out).resolve())
