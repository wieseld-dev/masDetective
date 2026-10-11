# V10.1 Post-hoc Confirmed-Elite Diagnostic

Purpose: distinguish two failure modes after the already-revealed V10.1 final result:

1. the frozen policy learned useful probes but sampled weak fresh probes at final time; versus
2. even the exact training-confirmed elites fail to generalize to the fresh compositions.

This patch performs **NO training** and **does not change the original final result**. It reuses the already-consumed V10 final systems as a development diagnostic and evaluates the exact confirmed V10 elites stored in `artifacts/v10_summary.json`.

Expected confirmed elites: 11 total (Writing 3, Math 1, Planning 3, Research 2, Data 1, Design 1; Code/Operations 0).

Run:

```bash
tar -xf ~/v10_posthoc_elite_diagnostic_patch.tar
bash v10_posthoc_elite_diagnostic_patch/apply_v10_elite_diag.sh
cd ~/mas_rl_detective
sbatch cluster/v10_final_elites_diag_cpu.sbatch
```

Monitor:

```bash
tail -f "$(ls -t logs/v10-elite-diag-*.out | head -1)"
```

Outputs:

- `artifacts/v10_final_elite_diagnostic_rows.jsonl`
- `artifacts/v10_final_elite_diagnostic_summary.json`

These outputs are post-hoc diagnostics only and must not be reported as untouched final-test results.
