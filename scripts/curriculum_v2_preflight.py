from __future__ import annotations

import importlib
import json
import py_compile
from pathlib import Path

from adaptive_v11_curriculum_v2.config import STAGES, planned_train_candidate_executions, validate_curriculum
from adaptive_v11_curriculum_v2.selector import FEATURE_DIM, load_stats


def main() -> None:
    validate_curriculum()
    required = [
        'adaptive_v11_curriculum_v2/config.py',
        'adaptive_v11_curriculum_v2/selector.py',
        'scripts/curriculum_v2_generate.py',
        'scripts/curriculum_v2_advance.py',
        'scripts/curriculum_v2_plan.py',
        'scripts/curriculum_v2_bootstrap_selector.py',
        'cluster/submit_curriculum_adaptive_v11_v2.sh',
        'cluster/curriculum_v2_generate_4090.sbatch',
        'cluster/curriculum_v2_advance_cpu.sbatch',
    ]
    missing = [p for p in required if not Path(p).exists()]
    if missing:
        raise SystemExit('missing V2 files: ' + ', '.join(missing))

    for p in required:
        if p.endswith('.py'):
            py_compile.compile(p, doraise=True)

    for mod in [
        'adaptive_v11_curriculum_v2.config',
        'adaptive_v11_curriculum_v2.selector',
    ]:
        importlib.import_module(mod)

    print(json.dumps({
        'status': 'ok',
        'stages': [
            {
                'stage': s.stage,
                'name': s.name,
                'horizon': s.horizon,
                'train_systems_per_cycle': s.train_systems_per_cycle,
                'dev_systems': s.dev_systems,
                'cycles': s.cycles,
                'group_size': s.group_size,
                'lr': s.lr,
                'kl_beta': s.kl_beta,
                'retention_weight': s.retention_weight,
                'mc_weight': s.mc_weight,
            }
            for s in STAGES
        ],
        'planned_train_candidate_executions': planned_train_candidate_executions(),
        'selector_feature_dim': FEATURE_DIM,
    }, indent=2))


if __name__ == '__main__':
    main()
