# V10.1 Final Untouched Probe-Generalization Test

This patch performs **evaluation only**. It does not train, fine-tune, update, or resume the policy.

## Frozen protocol

- Policy: existing `artifacts/v10_policy` from V10.1.
- Fresh base systems: 16.
- Execution-mode/K grid: exactly one system for each of 4 modes x K in {3,4,5,6}.
- Freshness: all 16 role compositions are disjoint from V6.3, V6.5 train, and the inspected V6.5 development validation.
- Balance: every candidate role occurs exactly 9 times; every role is represented in every execution mode and every K.
- Counterfactual evaluation: 72 matched PRESENT/ABSENT/SHAM pairs = exactly 9 pairs per role.
- Probes: 3 frozen probes per role, generated from the frozen V10.1 policy using only the candidate role, public role description, human anchor, and training-derived elites.
- All 24 probes are generated and atomically frozen in a manifest **before any final-test MAS output is observed**.
- After the manifest exists, crash/resume reuses the same questions; it never regenerates them.
- Strict SPEC is exactly the same V10.1 criterion: robust target activation >= .75, median SPEC > 0, Q25 SPEC >= 0, and positive-pair rate >= .75.

Pre-registered interpretation before seeing results:

- **Strong replication:** at least 6/8 roles have at least one strict-SPEC probe, and at least 7/8 roles have a probe with robust activation >= .75.
- **Partial replication:** 5/8 strict-SPEC roles.
- **Weak replication:** 4/8 or fewer strict-SPEC roles.
- Strict DELTA vs the human anchor is secondary; the anchor is not a reward hurdle.

Maximum target interactions: 864 (anchors + 3 probes across 9 matched pairs/role and 3 branches). Response caches make a crash/retry resume-safe.

## Scientific scope

This is a final untouched test of the **probe-generation policy**. It is not yet an end-to-end PRESENT/ABSENT classifier or K estimator; those require a separately trained black-box detector.

Once results from this test are inspected, the set is considered consumed. If the method is changed afterward, a new final test set is required for a clean final claim.

## Apply and run

```bash
cd ~/mas_rl_detective
tar -xf ~/v10_final_untouched_test_patch.tar
bash v10_final_untouched_test_patch/apply_v10_final_test.sh
sbatch cluster/v10_final_test_gpu.sbatch
```

Monitor:

```bash
squeue --me
tail -f "$(ls -t logs/v10-final-test-*.out | head -1)"
```

Final outputs:

- `artifacts/v10_final_probe_manifest.json`
- `artifacts/v10_final_rows.jsonl`
- `artifacts/v10_final_summary.json`
