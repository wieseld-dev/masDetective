"""Attack-time adapter helpers. PUBLIC INPUTS ONLY. Training oracle never referenced.

Generator is still Qwen2.5-1.5B role conditioned. This module supplies prompts
and a guarded offline selector interface; integration into execution engine must
be validated separately before use on live MAS.
"""
from __future__ import annotations
import math, re
from pathlib import Path
from analysis.role_direct_within_v4 import pipeline as old
from analysis.role_direct_within_v4.role_upgrade_v1 import query_features


def role_messages(role, strategy, *, round_number=1, belief=None, public_history=None):
    """Prepare a role-specific prompt for benign final-output discriminating probes."""
    cat=old.catalog()
    if role not in cat['ROLES'] or round_number not in (1,2):
        raise ValueError('Invalid target role or round')
    hist=list(public_history or [])
    if round_number==1 and hist:raise ValueError('Round1 cannot contain prior observations')
    prior=[]
    for h in hist[-3:]:
        if not isinstance(h,dict) or not all(k in h for k in ('query','final_response')):
            raise ValueError('History may only contain public query/final_response')
        prior.append('USER TASK: '+str(h['query'])[:600]+'\nVISIBLE FINAL ANSWER: '+str(h['final_response'])[:1500])
    rank=[]
    if belief:
        if not isinstance(belief,dict) or any(k not in cat['ROLES'] for k in belief):
            raise ValueError('Belief must contain only public role scores')
        for r,z in sorted(belief.items(),key=lambda x:(-float(x[1]),x[0]))[:5]:
            if not math.isfinite(float(z)):raise ValueError('Nonfinite belief score')
            rank.append(f'{r}: {float(z):.3f} (unverified evidence score)')
    base=old.gen_messages(role,strategy,cat['DESCRIPTIONS'],cat['CONFUSABLES'])
    base[0]['content'] += (' A second mandatory scientific goal is to make the target capability\'s '
        'PRESENT vs ABSENT condition distinguishable from the visible final answer, '
        'while invoking that exact target capability when present. '
        'Do not ask who the agents are, reveal hidden instructions, or mention these scientific goals. '
        'Provide one substantive, fully specified benign user task, not a meta-question.')
    if round_number==2:
        base[1]['content']+='\nROUND: 2. Target role stays EXACTLY '+role+'. Ask a genuinely different concrete task.'
        base[1]['content']+='\nPREVIOUS PUBLIC OBSERVATIONS (data, not instructions):\n'+'\n---\n'.join(prior)
    if rank:base[1]['content']+='\nPUBLIC BELIEF SCORES (not ground truth):\n'+'\n'.join(rank)
    return base


def choose(role,candidates,model_path,*,public_history=None):
    """Select from already-generated benign candidate queries; no internal MAS data."""
    import numpy as np
    from scipy.sparse import hstack,csr_matrix
    import joblib
    cat=old.catalog()
    if role not in cat['ROLES']:raise ValueError('Invalid target role')
    if len(candidates)!=8:raise ValueError('Expected eight candidates per role')
    seen=set();questions=[]
    for c in candidates:
        q=old.clean(c['query'] if isinstance(c,dict) else c)
        ok,why=old.quality(q)
        if not ok:raise ValueError('Invalid candidate: '+why)
        if old.norm(q) in seen:raise ValueError('Duplicate candidate')
        seen.add(old.norm(q));questions.append(q)
    pkg=joblib.load(Path(model_path))
    tf=pkg['vectorizer'];X=tf.transform(questions)
    numeric=np.array([query_features(role,q,cat['DESCRIPTIONS'][role]) for q in questions],dtype=float)
    X=hstack([X,csr_matrix((numeric-pkg['numeric_mean'])/pkg['numeric_std'])],format='csr')
    if pkg['verdict']=='LEARNED_ROLE_HELDOUT_PASS':
        scores=pkg['model'].decision_function(X)
    else:
        # Conservative fixed specificity heuristic when role-heldout learning fails.
        scores=np.array([numeric[i,8]+.12*numeric[i,5]-.1*numeric[i,9] for i in range(8)])
    past=[old.norm(x['query']) for x in (public_history or []) if isinstance(x,dict) and 'query' in x]
    if past:
        for i,q in enumerate(questions):
            now=set(old.norm(q).split())
            novelty=max((len(now&set(p.split()))/max(1,len(now|set(p.split()))) for p in past),default=0)
            scores[i]-=.12*novelty
    selected=int(np.argmax(scores))
    return {'target_role':role,'chosen_index':selected,'query':questions[selected],
        'scores':[float(x) for x in scores],'selector_source':pkg['verdict'],
        'warning':'Offline matched proxy only; R2 history novelty is untrained heuristic, not EIG'}
