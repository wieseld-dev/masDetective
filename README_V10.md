# V10.1 — Robust Cross-Composition Benign Probe Discovery

This patch continues from the trained V9-FULL QLoRA adapter and is designed specifically around the V9-FULL forensic findings and the exhaustive 36-pair development-validation analysis.

The objective remains strictly benign black-box probing: the policy sees only public candidate-role descriptions, the human anchor, and public probe text/history. Privileged routing truth is used only inside the controlled training/evaluation environment to compute scalar reward and diagnostics.

## Why V10.1

The exhaustive V9-FULL analysis showed that the method had real signal, but the original training loop had several avoidable weaknesses:

- reward bundles contained only two randomly sampled matched pairs;
- pair reuse was uneven across roles/compositions;
- reward checked target activation only in PRESENT, not under SHAM perturbation;
- strong probes could become elites after one training bundle only;
- shared-LoRA updates could over-specialize toward the most recent roles;
- the default KL reference pulled toward base Qwen rather than the V9-FULL starting policy;
- semantic paraphrases could bypass lexical duplicate checks and cause elite/mode collapse;
- development validation could generate duplicate probes;
- crash timing could leave adapter/optimizer/checkpoint state out of sync;
- resume summaries could count only the post-resume portion of the run.

## What V10.1 changes

### 1. Pairwise causal reward

Every matched composition is scored independently:

`SPEC_i = d(PRESENT_i, ABSENT_i) - d(PRESENT_i, SHAM_i)`

The robust score is:

`ROBUST_SPEC = 0.5 * median(SPEC_i) + 0.5 * Q25(SPEC_i)`

A single lucky composition therefore cannot dominate the reward.

### 2. Robust activation in PRESENT and SHAM

The target role must route reliably both in the normal PRESENT branch and in the SHAM branch, where an unrelated role is changed while the target remains present.

`activation = min(PRESENT activation rate, SHAM activation rate)`

Positive reward requires robust activation >= 0.75.

### 3. Broad primary pair coverage

Each role gets four matched pairs per primary reward bundle. The deterministic schedule prioritizes:

- previously unseen primary pairs;
- execution-mode diversity;
- K diversity;
- negative-replacement diversity;
- role-composition diversity;
- low pair reuse.

On the actual V6.5 RL split used by this project, the schedule verification covers every available training pair as PRIMARY evidence across three rounds: 12/12 for code/writing/math/planning/operations and 11/11 for research/data/design.

### 4. Disjoint confirmation before elite imitation

A candidate cannot become an elite merely because it wins its primary bundle.

For the strongest primary strict candidates, the same probe is evaluated on two additional matched pairs that are disjoint from the primary bundle. Elite imitation is enabled only if the candidate also passes strict robust SPEC on this confirmation set.

This makes elite imitation much harder to exploit through composition-specific luck.

### 5. Balanced multi-role optimization

All role groups in a round are generated from the same policy snapshot. The samples are then divided into three optimizer minibatches, with each minibatch containing evidence from every role whenever possible.

With the default three balanced minibatches and two PPO epochs per minibatch, this yields six AdamW steps per round (18 across the default three rounds) without reverting to the V9 pattern where updates for one role could immediately interfere with another role.

### 6. KL-to-base disabled by default

V10.1 starts from `artifacts/v9_full_policy`, but `disable_adapter()` would use base Qwen as the reference distribution. Penalizing KL to that distribution can pull the policy back toward the base model rather than preserve V9-FULL behavior.

The default is therefore:

`--kl-beta 0.0`

PPO clipping still constrains movement relative to the sampled-policy snapshot.

### 7. Stronger semantic anti-collapse

Before any semantic-guard or MAS/API call, a local embedding check rejects near-duplicate probes at similarity >= 0.97.

A softer novelty penalty still applies above similarity 0.90. The policy prompt also explicitly distinguishes semantic strategy reuse from mere wording reuse.

### 8. More permissive benign-task gate, same hard boundary

The hard benign gate is unchanged. The semantic layer explicitly treats ordinary questions about the user's own code, logs, data, calculations, workflow, product behavior, reliability, debugging, or validation as benign task language unless they actually request hidden prompts, agents, routing, architecture, traces, secrets, or related private structure.

### 9. Diverse development-validation generation

Development-validation probes are regenerated locally if they are syntactically invalid, semantically too close to the anchor/elites/already-generated probes, or blocked by the benign gate. Only a distinct allowed question spends MAS evaluation calls.

The final generated probes are still evaluated exhaustively over every V6.5 validation matched pair available for that role.

### 10. Transactional checkpoints

A round is resumable only after all of the following are durably saved together:

- adapter snapshot;
- optimizer state;
- round metadata;
- COMMITTED marker;
- atomically advanced root checkpoint JSON.

Snapshots are stored under:

`artifacts/v10_checkpoints/round_XX/`

The mutable `artifacts/v10_policy/` and `artifacts/v10_optimizer.pt` are convenience mirrors only. Resume trusts the immutable committed round referenced by `artifacts/v10_checkpoint.json`.

If a job crashes mid-round, the partial round is removed from the JSONL on resume and regenerated from the last committed policy. Existing response-cache entries are retained and reused.

### 11. Resume-safe final accounting

The final summary is reconstructed from the complete committed JSONL, rather than only from counters in the current process. A resumed run therefore reports the full experiment rather than only the post-resume portion.

## Important evaluation terminology

The V6.5 validation systems have already been inspected during method development. V10.1 labels them as **development validation**, not an untouched final test set.

After the method is frozen, the paper still needs a fresh final test split with no policy tuning, detector tuning, reward design, or checkpoint selection performed against it.

## Files installed

- `v10/reward.py`
- `v10/policy.py`
- `v10/gate.py`
- `v10/checkpoint.py`
- `training/train_robust_probe_discovery_v10.py`
- `scripts/diagnose_v10.py`
- `tests/test_v10_reward.py`
- `tests/test_v10_checkpoint.py`
- `cluster/v10_robust_gpu.sbatch`
- `cluster/v10_robust_resume_gpu.sbatch`

## Outputs

- `artifacts/v10_policy/`
- `artifacts/v10_optimizer.pt`
- `artifacts/v10_checkpoints/round_01/ ...`
- `artifacts/v10_checkpoint.json`
- `artifacts/v10_probe_discovery.jsonl`
- `artifacts/v10_summary.json`
- `artifacts/v10_response_public.jsonl`
- `artifacts/v10_response_truth.jsonl`

The V10 response cache is initially seeded from V9-FULL so existing identical interactions are reused.

## Apply

From your laptop:

```bash
scp ~/Downloads/v10_1_robust_cross_composition_patch.tar \
  wieseld@slurm.bgu.ac.il:~/
```

On the cluster:

```bash
cd ~/mas_rl_detective

tar -xf ~/v10_1_robust_cross_composition_patch.tar
bash v10_1_robust_cross_composition_patch/apply_v10.sh
```

Do not run the training Python directly on the login node.

## Start

```bash
sbatch cluster/v10_robust_gpu.sbatch
```

Monitor:

```bash
squeue --me

tail -f "$(ls -t logs/v10-1-robust-*.out | head -1)"
```

The job prints each candidate, confirmation result, balanced minibatch update, and committed checkpoint.

## Resume after interruption

```bash
sbatch cluster/v10_robust_resume_gpu.sbatch
```

Resume reads only `artifacts/v10_checkpoint.json` and the immutable round snapshot it references.

If the full-run sbatch is submitted when a committed V10 checkpoint already exists, the script intentionally refuses to overwrite it; use the resume job instead.

## Final diagnosis

After completion:

```bash
PYTHONPATH="$PWD" ~/.conda/envs/masdetective/bin/python scripts/diagnose_v10.py \
  --log artifacts/v10_probe_discovery.jsonl \
  --summary artifacts/v10_summary.json
```

Primary development metrics:

- role with any robust activation >= 0.75;
- role with any strict pairwise-SPEC probe;
- robust SPEC strength per role;
- confirmed elite count/diversity;
- comparison against V9 exhaustive development baseline (5/8 roles with a strict SPEC probe).

Strict DELTA-vs-anchor remains a secondary diagnostic. The anchor is not a reward hurdle.
