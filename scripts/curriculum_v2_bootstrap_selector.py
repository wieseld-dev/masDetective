from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_curriculum_v2.selector import feature_vector, update_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source-root', default='artifacts/adaptive_v11_curriculum_v1')
    ap.add_argument('--selector-model', required=True)
    ap.add_argument('--max-stage', type=int, default=2)
    ap.add_argument('--reset', action='store_true')
    a = ap.parse_args()

    model = Path(a.selector_model)
    if a.reset and model.exists():
        model.unlink()

    source = Path(a.source_root)
    X, y = [], []
    groups = 0

    for stage in range(1, a.max_stage + 1):
        sdir = source / f'stage_{stage:02d}'
        if not sdir.exists():
            continue
        for croot in sorted(sdir.glob('cycle_*')):
            rounds = sorted(croot.glob('train_rows_round_*.jsonl'))
            for train_path in rounds:
                rr = int(train_path.stem.rsplit('_', 1)[-1])
                state_path = croot / f'states_train_round_{rr-1:02d}.jsonl'
                cand_path = croot / f'candidates_train_round_{rr:02d}.jsonl'
                if not state_path.exists() or not cand_path.exists():
                    continue
                states = {
                    z['session_id']: PublicBeliefState.from_public_dict(z)
                    for z in read_jsonl(state_path)
                }
                candidates = {
                    (z['session_id'], int(z['candidate_index'])): z
                    for z in read_jsonl(cand_path)
                }
                rows = read_jsonl(train_path)
                for z in rows:
                    sid = str(z['session_id'])
                    idx = int(z['candidate_index'])
                    st = states.get(sid)
                    c = candidates.get((sid, idx))
                    if st is None or c is None or not c.get('valid'):
                        continue
                    reward = float(z['reward'])
                    X.append(feature_vector(st, str(c.get('query', '')), idx))
                    y.append(reward)
                groups += 1

    if not X:
        raise SystemExit('no bootstrap selector examples found')
    summary = update_model(model, X, y)
    print(json.dumps({
        'source_root': str(source),
        'max_stage': a.max_stage,
        'round_files': groups,
        'examples_added': len(y),
        'selector_model': str(model),
        'summary': summary,
    }, indent=2))


if __name__ == '__main__':
    main()
