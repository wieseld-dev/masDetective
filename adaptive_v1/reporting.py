from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def summarize_episode_rows(rows):
    by=defaultdict(list)
    for r in rows:by[(r["controller"],int(r["budget"]))].append(r)
    out={}
    for (c,b),xs in sorted(by.items()):
        def m(k):
            vv=[float(x[k]) for x in xs if x.get(k) is not None];return float(np.mean(vv)) if vv else None
        out.setdefault(c,{})[str(b)]={"n_systems":len(xs),"queries":m("queries"),"f1_blind":m("f1_blind"),"exact_blind":m("exact_blind"),"known_k_recall":m("known_k_recall"),"known_k_exact":m("known_k_exact"),"mean_true_rank":m("mean_true_rank"),"brier":m("brier"),"expected_k_abs_error":m("expected_k_abs_error"),"stop_rate":m("stopped_early")}
    return out


def write_report(path,rows,metadata=None):
    obj={"version":"adaptive_eval_v1","summary":summarize_episode_rows(rows),"metadata":metadata or {}}
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,sort_keys=True),encoding="utf-8");return obj
