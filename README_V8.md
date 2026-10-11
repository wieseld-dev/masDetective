# V8 — Counterfactual Matched-Pair Benign-Probing GRPO

V8 removes the V6.5 detector from the RL reward loop. The policy sees only a candidate capability, one fixed benign seed question, and the seed's final black-box answer. It then generates one new free-text benign follow-up.

During training only, the same generated follow-up is executed on matched counterfactual systems:

- `PLUS`: target role present.
- `MINUS`: same K and same execution mode, but the target role is replaced by an absent distractor.
- optional `SHAM`: target remains present while a different role is swapped.

The policy never sees which branch it is in, the true roles, K, routing, selected roles, traces, or system ID.

## Reward

The privileged environment returns one scalar built from:

1. target activation in the PLUS branch;
2. incremental external-output separability: distance(PLUS,MINUS) after the generated probe minus the same distance under the fixed seed;
3. optional sham specificity: target-swap separation should exceed generic sham-swap separation;
4. query cost;
5. strong invalid/unsafe penalties.

External separability uses a frozen sentence-transformer on final outputs, not an LLM reward model and not the V6.5 detector.

## GRPO-style update

From the exact same black-box seed state, Qwen samples a group of candidate follow-ups. Each candidate gets its counterfactual reward. Rewards are normalized within the group, so the update directly learns which question was better for the same state. A sampled KL penalty to the base model is computed by temporarily disabling the LoRA adapter, without loading a second LLM into GPU memory.

The update is memory-safe on ~11GB GPUs: it backpropagates one completion at a time and performs one optimizer step after gradient accumulation.

## Stage 1 deliberately has no STOP

V8 Stage 1 forces exactly one generated follow-up. This prevents the policy from learning the trivial shortcut "the fixed seed is often already correct, therefore stop." STOP should only be introduced after the policy has learned useful questioning.

## Syntax versus semantics

Formatting mistakes are normalized to the first single question before the safety gate. This does not rewrite the semantics. The independent benign gate still blocks structural/internal requests and prompt injection. The V8 semantic gate explicitly allows normal task questions about assumptions, calculations, methods, evidence, reliability, accuracy, constraints, and required inputs.

## Experimental split

The original V6.5 training systems are split by base system into:

- RL-train base systems, used for privileged counterfactual reward;
- detector-reserved base systems, never used for GRPO.

After GRPO is frozen, collect black-box transcripts on the detector-reserved systems and train `TranscriptDetectorV8`. The V6.5 validation systems remain the final unseen black-box evaluation set.

## Run order

1. Apply and unit-test the patch.
2. Run V8 training in PLAN mode (zero API calls).
3. Run the 8-state smoke job.
4. Audit smoke prompts. Do not run strong training unless safety failures are zero and valid-rate/reward behavior looks sane.
5. Run strong GRPO.
6. Freeze the policy and collect detector-training transcripts on the reserved systems.
7. Train the post-RL detector.
8. Run final evaluation on unseen V6.5 validation compositions.

## Post-RL server jobs

After strong GRPO completes and the policy is frozen:

```bash
sbatch cluster/v8_detector_collect_gpu.sbatch
```

Then fit/check the post-RL detector on a CPU node:

```bash
sbatch cluster/v8_detector_train_cpu.sbatch
```

Only if its system-grouped OOF CV is reasonable, run the untouched final V6.5 validation evaluation:

```bash
sbatch cluster/v8_final_eval_gpu.sbatch
```
