from __future__ import annotations
import math
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

N_HASH = 1024

def h01(p: float) -> float:
    p = min(1.0 - 1e-12, max(1e-12, float(p)))
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))

def total_entropy(role_probs: dict[str,float], roles: list[str]) -> float:
    return float(sum(h01(role_probs[r]) for r in roles))

def numeric_features(state: dict, query: str, target_role: str, roles: list[str], round_index: int) -> np.ndarray:
    rp = {r: float(state["role_probs"][r]) for r in roles}
    ps = np.asarray([rp[r] for r in roles], dtype=np.float64)
    hs = np.asarray([h01(x) for x in ps], dtype=np.float64)
    tp = float(rp.get(target_role, np.mean(ps))) if target_role else float(np.mean(ps))
    th = h01(tp)
    words = query.split()
    base = np.concatenate([
        ps,
        hs,
        np.asarray([
            float(hs.sum()), float(hs.mean()), float(hs.max()), float(hs.std()),
            float(round_index) / 6.0,
            tp, th,
            min(len(query), 1000) / 1000.0,
            min(len(words), 200) / 200.0,
        ], dtype=np.float64),
    ])
    onehot = np.zeros(len(roles) + 1, dtype=np.float64)
    if target_role in roles:
        onehot[roles.index(target_role)] = 1.0
    else:
        onehot[-1] = 1.0
    return np.concatenate([base, onehot])

def vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=N_HASH,
        alternate_sign=False,
        norm="l2",
        analyzer="word",
        ngram_range=(1, 2),
        lowercase=True,
    )

def build_matrix(rows: list[dict], roles: list[str], num_mean=None, num_std=None):
    hv = vectorizer()
    texts = [str(x["query"]) for x in rows]
    Xtxt = hv.transform(texts)
    Xnum = np.vstack([
        numeric_features(x["state"], str(x["query"]), str(x.get("target_role", "")), roles, int(x["round"]))
        for x in rows
    ])
    if num_mean is None:
        num_mean = Xnum.mean(axis=0)
    if num_std is None:
        num_std = Xnum.std(axis=0)
    num_std = np.where(np.asarray(num_std) < 1e-8, 1.0, num_std)
    Xnum = (Xnum - num_mean) / num_std
    X = sparse.hstack([Xtxt, sparse.csr_matrix(Xnum)], format="csr")
    return X, np.asarray(num_mean), np.asarray(num_std)

def predict_one(bundle: dict, state: dict, query: str, target_role: str, round_index: int) -> float:
    roles = list(bundle["roles"])
    row = {"state": state, "query": query, "target_role": target_role, "round": round_index}
    X, _, _ = build_matrix([row], roles, np.asarray(bundle["num_mean"]), np.asarray(bundle["num_std"]))
    return float(bundle["model"].predict(X)[0])
