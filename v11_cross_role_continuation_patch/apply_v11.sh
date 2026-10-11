#!/usr/bin/env bash
set -euo pipefail
ROOT="${1:-$HOME/mas_rl_detective}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

mkdir -p v11 targets training cluster tests scripts config artifacts logs

# Never overwrite or modify the frozen V10 artifacts.
for p in artifacts/v10_policy artifacts/v10_summary.json; do
  if [ ! -e "$p" ]; then
    echo "Missing required frozen V10.1 input: $p" >&2
    exit 1
  fi
done

cp "$PATCH_DIR/v11/role_universe.py" v11/role_universe.py
cp "$PATCH_DIR/v11/hashcheck.py" v11/hashcheck.py
cp "$PATCH_DIR/v11/__init__.py" v11/__init__.py
cp "$PATCH_DIR/targets/single_router_zoo_v11.py" targets/single_router_zoo_v11.py
cp "$PATCH_DIR/training/train_cross_role_probe_discovery_v11.py" training/train_cross_role_probe_discovery_v11.py
cp "$PATCH_DIR/config/application_profile_v11.json" config/application_profile_v11.json
cp "$PATCH_DIR/tests/test_v11_role_holdout.py" tests/test_v11_role_holdout.py
cp "$PATCH_DIR/tests/test_v11_hashcheck.py" tests/test_v11_hashcheck.py
cp "$PATCH_DIR/scripts/verify_v11_preflight.py" scripts/verify_v11_preflight.py
cp "$PATCH_DIR/scripts/diagnose_v11.py" scripts/diagnose_v11.py
cp "$PATCH_DIR/cluster/v11_preflight_cpu.sbatch" cluster/v11_preflight_cpu.sbatch
cp "$PATCH_DIR/cluster/v11_cross_role_gpu.sbatch" cluster/v11_cross_role_gpu.sbatch
cp "$PATCH_DIR/cluster/v11_cross_role_resume_gpu.sbatch" cluster/v11_cross_role_resume_gpu.sbatch
chmod +x cluster/v11_preflight_cpu.sbatch cluster/v11_cross_role_gpu.sbatch cluster/v11_cross_role_resume_gpu.sbatch

echo "V11 patch installed."
echo "Frozen V10 inputs preserved: artifacts/v10_policy, artifacts/v10_summary.json"
echo "Run: sbatch cluster/v11_cross_role_gpu.sbatch"
