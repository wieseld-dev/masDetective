"""Run the existing manifest builder with V2 stages and a V2 root, without editing V1."""
from __future__ import annotations

import os
from pathlib import Path

import adaptive_v11_curriculum.config as base_config
from adaptive_v11_curriculum_v2.config import STAGES

base_config.STAGES = STAGES
root = os.environ.get('CURR_EXP_ROOT', 'artifacts/adaptive_v11_curriculum_v2')
source_path = Path('scripts/curriculum_make_manifests.py')
source = source_path.read_text()
# Preserve the proven manifest-generation logic while redirecting any fixed V1 default path.
source = source.replace('artifacts/adaptive_v11_curriculum_v1', root)
namespace = {
    '__name__': '__main__',
    '__file__': str(source_path),
    '__package__': None,
}
exec(compile(source, str(source_path), 'exec'), namespace, namespace)
