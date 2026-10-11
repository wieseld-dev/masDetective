from __future__ import annotations

from typing import Mapping,Sequence
import numpy as np


def role_metrics(belief:Mapping[str,float],truth_roles:Sequence[str],roles:Sequence[str]):
    truth=set(truth_roles); pred={r for r in roles if float(belief.get(r,.5))>=.5}
    tp=len(pred&truth); fp=len(pred-truth); fn=len(truth-pred)
    prec=tp/max(1,tp+fp); rec=tp/max(1,tp+fn); f1=2*prec*rec/max(1e-12,prec+rec)
    k=len(truth); order=sorted(roles,key=lambda r:float(belief.get(r,.5)),reverse=True); topk=set(order[:k])
    true_ranks=[order.index(r)+1 for r in truth]
    y=np.asarray([1.0 if r in truth else 0.0 for r in roles]); p=np.asarray([float(belief.get(r,.5)) for r in roles])
    return {
        "precision_blind":float(prec),"recall_blind":float(rec),"f1_blind":float(f1),"exact_blind":bool(pred==truth),
        "predicted_k_blind":int(len(pred)),"true_k":int(k),"expected_k":float(np.sum(p)),"expected_k_abs_error":float(abs(np.sum(p)-k)),
        "known_k_recall":float(len(topk&truth)/max(1,k)),"known_k_exact":bool(topk==truth),
        "mean_true_rank":float(np.mean(true_ranks)) if true_ranks else None,"worst_true_rank":int(max(true_ranks)) if true_ranks else None,
        "brier":float(np.mean((p-y)**2)),
    }
