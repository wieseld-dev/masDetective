#!/bin/bash
set -euo pipefail
ROOT="${1:-$HOME/mas_rl_detective}"
cd "$ROOT"

mkdir -p config probes estimator rl detective training cluster tests

# Extend the training-only trace with the complete true role set. This field is never exposed
# by the target's public query() interface and is used only by the V6 white-box reward/logging.
python - <<'PY'
from pathlib import Path
p = Path("targets/hierarchical_mas.py")
s = p.read_text()
if "true_roles: Tuple[str, ...]" not in s:
    s = s.replace(
        "    all_edges: Tuple[Tuple[str, str], ...]\n",
        "    all_edges: Tuple[Tuple[str, str], ...]\n    true_roles: Tuple[str, ...] = ()\n",
    )
if "true_roles=tuple(self.system.worker_roles)" not in s:
    s = s.replace(
        "            selected_role=role,\n            path_nodes=tuple(path),\n",
        "            selected_role=role,\n            true_roles=tuple(self.system.worker_roles),\n            path_nodes=tuple(path),\n",
    )
p.write_text(s)
print("patched privileged trace with training-only true_roles")
PY

echo "V6 role-adaptive patch applied."
echo "Next: python -m pytest -q"
echo "Then: python -m training.bootstrap_role_estimator --limit-systems 8"
