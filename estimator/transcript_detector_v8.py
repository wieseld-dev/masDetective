from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class TranscriptDetectorV8:
    """Post-RL black-box detector. Not used anywhere in the RL reward loop."""
    def __init__(self, roles, role_descriptions, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self.roles=tuple(roles);self.role_descriptions=dict(role_descriptions)
        self.embedder=SentenceTransformer(embedding_model,device="cpu")
        self.scaler=StandardScaler();self.clf=LogisticRegression(max_iter=3000,class_weight="balanced",C=0.7)
    def _emb(self,texts):
        return np.asarray(self.embedder.encode(list(texts),normalize_embeddings=True,show_progress_bar=False),dtype=np.float32)
    def featurize(self,rows):
        role=self._emb([self.role_descriptions.get(r["role"],r["role"]) for r in rows])
        seed=self._emb([r.get("seed_output","") for r in rows]);follow=self._emb([r.get("follow_output","") for r in rows])
        delta=follow-seed;prod=follow*role;seedprod=seed*role
        meta=np.asarray([[float(bool(r.get("follow_allowed",False))),len(r.get("seed_output","") or "")/1000.0,len(r.get("follow_output","") or "")/1000.0] for r in rows],dtype=np.float32)
        return np.concatenate([role,seed,follow,delta,prod,seedprod,meta],axis=1)
    def fit(self,rows,y):
        X=self.featurize(rows);X=self.scaler.fit_transform(X);self.clf.fit(X,np.asarray(y,dtype=int));return self
    def predict_proba(self,rows): return self.clf.predict_proba(self.scaler.transform(self.featurize(rows)))[:,1]
    def save(self,path):
        emb=self.embedder;self.embedder=None;joblib.dump(self,path);self.embedder=emb
    @classmethod
    def load(cls,path,embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        obj=joblib.load(path);obj.embedder=SentenceTransformer(embedding_model,device="cpu");return obj


def load_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
