from __future__ import annotations

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import binary_entropy,total_entropy
from .schema import Action,EpisodeState


def action_features(state:EpisodeState,action:Action,roles):
    b=state.belief
    pt=float(b.get(action.target,.5)); pc=float(b.get(action.competitor,.5)) if action.competitor else .5
    counts=state.target_counts(); used=sum(1 for x in state.history if x.action.key==action.key)
    vals=[
        pt,binary_entropy(pt),pc,binary_entropy(pc),abs(pt-pc),
        float(counts.get(action.target,0)),float(counts.get(action.competitor,0)) if action.competitor else 0.0,float(used),
        float(state.remaining_budget),float(total_entropy(b,roles)),float(max(binary_entropy(b.get(r,.5)) for r in roles)),float(sum(1 for r in roles if b.get(r,.5)>=.5)),
    ]
    one=np.zeros(len(roles)*2,dtype=np.float32); one[roles.index(action.target)]=1.0
    if action.competitor in roles: one[len(roles)+roles.index(action.competitor)]=1.0
    return np.concatenate([np.asarray(vals,dtype=np.float32),one])


class OracleImitationScorer:
    VERSION="oracle_imitation_action_value_v1"
    def __init__(self,roles):
        self.roles=tuple(roles); self.model=HistGradientBoostingRegressor(max_depth=4,learning_rate=.06,max_iter=250,l2_regularization=1.0,random_state=17); self.fitted=False
    def fit(self,X,y):
        X=np.asarray(X,dtype=np.float32); y=np.asarray(y,dtype=np.float32); self.model.fit(X,y); self.fitted=True; return {"n":len(y),"target_mean":float(np.mean(y)),"target_std":float(np.std(y))}
    def choose(self,state,available):
        if not available:return "STOP"
        X=np.vstack([action_features(state,a,self.roles) for a in available]); pred=self.model.predict(X); i=int(np.argmax(pred)); return available[i] if float(pred[i])>0 else "STOP"
    def save(self,path):joblib.dump({"version":self.VERSION,"roles":self.roles,"model":self.model,"fitted":self.fitted},path)
    @classmethod
    def load(cls,path):
        d=joblib.load(path); o=cls(d["roles"]); o.model=d["model"]; o.fitted=d.get("fitted",True); return o
