from __future__ import annotations
import argparse, json
from pathlib import Path
from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import write_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_curriculum.belief_utils import role_only_predict, dummy_k_probs
from v11.role_universe import TRAIN_ROLES

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',required=True); ap.add_argument('--belief',required=True); ap.add_argument('--root',required=True); ap.add_argument('--split',required=True)
    a=ap.parse_args(); root=Path(a.root); root.mkdir(parents=True,exist_ok=True)
    obj=json.loads(Path(a.manifest).read_text()); bundle=FrozenBeliefBundle.load(a.belief)
    rp=role_only_predict(bundle,[]); kp=dummy_k_probs(bundle)
    rows=[]
    for s in obj['systems']:
        if s['split']!=a.split: continue
        st=PublicBeliefState(session_id=s['session_id'],split=a.split,round_index=0,k_probs=dict(kp),role_probs=dict(rp),history=[])
        st.validate(TRAIN_ROLES,bundle.ks); rows.append(st.to_public_dict())
    out=root/f"states_{a.split}_round_00.jsonl"; write_jsonl(out,rows); print(f"[init] {out} rows={len(rows)}")
if __name__=='__main__': main()
