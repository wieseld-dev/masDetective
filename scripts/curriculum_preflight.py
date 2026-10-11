from __future__ import annotations
import argparse, json
from pathlib import Path
from adaptive_v11_curriculum.config import STAGES, estimated_target_queries
from adaptive_v11_curriculum.hashguard import tree_sha256
from adaptive_v11_active.belief import FrozenBeliefBundle
from v11.role_universe import TRAIN_ROLES

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='artifacts/adaptive_v11_curriculum_v1'); ap.add_argument('--v11',default='artifacts/v11_policy'); ap.add_argument('--belief',default='artifacts/adaptive_v11_active_v1/frozen_belief.joblib'); a=ap.parse_args()
    root=Path(a.root); v11=Path(a.v11)
    if not v11.is_dir(): raise SystemExit(f"missing {v11}")
    h=tree_sha256(v11); (root/'v11_source_sha256.txt').write_text(h+'\n')
    b=FrozenBeliefBundle.load(a.belief)
    if list(b.roles)!=list(TRAIN_ROLES): raise RuntimeError('belief role universe mismatch')
    report=json.loads((root/'manifest_report.json').read_text())
    for s in STAGES:
        sd=root/f"stage_{s.stage:02d}"
        for c in range(1,s.cycles+1):
            p=sd/f"manifest_train_cycle_{c:02d}.json"; assert p.is_file(),p
        assert (sd/'manifest_dev.json').is_file()
    print('PASS')
    print('frozen V11 =',v11)
    print('frozen V11 sha256 =',h)
    print('all continuation checkpoints are under',root/'checkpoints')
    print('stages =',[(s.stage,s.name,s.horizon,s.cycles,s.group_size) for s in STAGES])
    print('estimated real target queries (train candidate evaluations + stage dev) =',estimated_target_queries())
    print('NOTE: each target query can invoke multiple router/worker/aggregator model calls.')
    print('final_test_manifest_UNTOUCHED.json is created but never used by the training submitter.')
if __name__=='__main__': main()
