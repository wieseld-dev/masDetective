One-system end-to-end K=3 smoke demo.

Ground truth: code, planning, research.
Universe: the existing 20 TRAIN_ROLES.
Inference update sees only query + final textual output.
Internal selected_roles trace is printed/stored evaluation-only.

This is a sanity/demo run, not benchmark evidence.

Run:
  unzip -o one_system_k3_demo_bundle.zip
  mv one_system_k3_demo.py analysis/
  mv one_system_k3_demo.sbatch .
  J=$(sbatch --parsable one_system_k3_demo.sbatch)
  echo $J
  tail -f logs/k3demo-${J}.out

Result:
  cat artifacts/one_system_k3_demo/summary.json
