from __future__ import annotations

import json
from pathlib import Path

from .confusers import default_competitor


def convert_cross_role_matrices(paths,roles,output):
    """Development-only converter for old V13.2 cross-role matrices.

    This exists only to smoke-test controller mechanics while V11.2/V13.5 finish.
    It must not be used for final adaptive claims because V13.2 is known to have
    systematic semantic self-confirmation and the four systems were heavily inspected.
    """
    out=[]
    for path in paths:
        obj=json.loads(Path(path).read_text(encoding="utf-8"))
        systems=obj.get("systems") if isinstance(obj,dict) else None
        if systems is None:
            systems=[obj]
        for sys in systems:
            sid=str(sys.get("system_id") or sys.get("system") or "legacy")
            truth=list(sys.get("truth_roles_eval_only") or sys.get("truth") or sys.get("true_roles") or [])
            for ev in sys.get("events",[]):
                scores=ev.get("all_role_scores",{});e={r:float(scores[r].get("evidence_delta",0.0)) for r in roles if r in scores}
                if len(e)!=len(roles):continue
                target=str(ev.get("target_role"));out.append({"system_id":sid,"split":"legacy_dev","truth_roles":truth,"action":{"target":target,"competitor":None,"variant":0},"query":str(ev.get("query","")),"evidence":e,"response_id":f"{sid}:{ev.get('step',len(out)+1)}","method":"v13_2_legacy","metadata":{"warning":"development-only V13.2 smoke data"}})
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",encoding="utf-8") as f:
        for r in out:f.write(json.dumps(r,ensure_ascii=False)+"\n")
    return len(out)
