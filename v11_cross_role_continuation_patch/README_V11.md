# V11 — Cross-Role Continuation from Frozen V10.1

## Scientific question

V10 showed that exact confirmed elite probes generalized across fresh MAS compositions for every role that produced an elite (6/6), but that does **not** prove transfer to a role type the Detective never saw during training.

V11 tests the missing question:

> Can one role-conditioned Detective policy learn a general probe-discovery strategy and generate useful benign probes for completely unseen candidate role types?

## What is preserved

- The run **starts from the exact trained V10.1 QLoRA adapter**, not base Qwen.
- Required frozen start hash:
  `9abbb8cdfa04df70ab8f1dadba23fad15a82e098002df8c6186cf2befaf9a7f9`
- Required V10 summary hash:
  `b965cb9041d2fe323538f6c43591e6d9d82a5ddec95d642f4423d2430087973c`
- `artifacts/v10_policy` and `artifacts/v10_summary.json` are read-only inputs. V11 writes only `artifacts/v11_*`.
- Same V10 reward family: matched PRESENT / ABSENT / SHAM, robust median + Q25 specificity, activation gate, disjoint confirmation, benign gate, GRPO/PPO-style updates.
- Fresh optimizer is intentional; learned capability remains in the V10 LoRA weights.

## Role split

### Training roles: 20

The original eight plus twelve additional role types:

`code, writing, math, planning, research, data, design, operations, qa_testing, translation, financial_analysis, database_design, accessibility_review, fact_checking, requirements_analysis, incident_management, experiment_design, risk_assessment, documentation, customer_support_triage`

### Completely unseen development roles: 6

`legal_contract_review, procurement_analysis, travel_itinerary, spreadsheet_audit, scientific_peer_review, inventory_forecasting`

These six role types are excluded from:

- positive training systems,
- training candidate targets,
- training negative replacements,
- training SHAM systems,
- V10 elite replay.

They are introduced only after training is complete for development evaluation. This is **role-level holdout**, not merely composition holdout.

## MAS diversity

- 80 unique training base systems: 20 each for `single`, `fanout2`, `fanout3`, `mixed`.
- K varies from 3 through 6.
- Every training role appears in every mode and across at least three K values.
- 32 unseen-role development systems: 8 each mode.
- Every unseen role appears repeatedly in all four modes, mixed with familiar distractor roles.

The current SLURM job uses 2 rounds × 20 train roles × 5 generated candidates, with 4-pair primary bundles and 2-pair disjoint confirmation. The schedule prioritizes under-used pairs, mode diversity, K diversity, replacement diversity, and composition diversity.

## What the policy sees

Only public role-conditioning information:

- candidate role name,
- natural-language capability description,
- a mechanically generated benign anchor,
- public prior elite/avoid strings for training roles.

It never observes true roles, K, routing, selected workers, topology, counterfactual branch identity, internal messages, or traces. Privileged truth is used only by the training/evaluation environment to compute scalar reward and diagnostics.

## Critical interpretation

A successful unseen-role development result means the **trained policy itself** can generate a probe for a role type absent from all training episodes. It is therefore different from storing a bank of Math/Planning/etc. elite questions.

The six unseen roles are development validation. If V11 is changed after looking at them, a separate fresh role set is required for the final untouched test.

## Install

```bash
cd ~/mas_rl_detective
tar -xf ~/v11_cross_role_continuation_patch.tar
bash v11_cross_role_continuation_patch/apply_v11.sh
```

## Verify first (recommended; CPU-only, no API calls)

```bash
sbatch cluster/v11_preflight_cpu.sbatch
```

Inspect:

```bash
tail -n 120 "$(ls -t logs/v11-preflight-*.out | head -1)"
```

Only if it prints `EXACT V10.1 START: PASS` and `ROLE-TYPE LEAKAGE CHECK: PASS`, run the training job.

## Run

```bash
sbatch cluster/v11_cross_role_gpu.sbatch
```

Monitor:

```bash
squeue --me
tail -f "$(ls -t logs/v11-xrole-*.out | head -1)"
```

Resume after an interrupted job:

```bash
sbatch cluster/v11_cross_role_resume_gpu.sbatch
```

Final development diagnostic:

```bash
PYTHONPATH="$PWD" ~/.conda/envs/masdetective/bin/python scripts/diagnose_v11.py \
  --summary artifacts/v11_summary.json
```

## Main outputs

- `artifacts/v11_policy/`
- `artifacts/v11_probe_discovery.jsonl`
- `artifacts/v11_summary.json`
- `artifacts/v11_response_public.jsonl`
- `artifacts/v11_response_truth.jsonl`
- `artifacts/v11_checkpoints/`
- `artifacts/v11_checkpoint.json`

## Primary development metric

For the six **unseen role types**:

1. role with any generated probe achieving robust activation >= 0.75;
2. role with any strict target-specific SPEC probe;
3. strict DELTA versus mechanically generated anchor is secondary.

Do not compare an unseen-role percentage directly with the old V10 6/8 composition-dev score as if they were the same task: V11 is deliberately harder because the role type itself is held out.
