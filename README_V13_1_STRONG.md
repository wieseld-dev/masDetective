# V13.1-STRONG — QLoRA Causal Evidence Judge

This supersedes the earlier frozen-MiniLM V13.1A patch.

## Why this version is stronger

The earlier baseline had two important weaknesses:
1. a frozen MiniLM embedding can wash out subtle behavioral differences;
2. splitting only by probe group can leak a reused counterfactual pair/system
   into both train and validation.

This version fixes both.

### Runtime function learned
One shared role-conditioned judge:

    score(role description, diagnostic question, public MAS answer)

It never sees true roles, K, routing, traces, or branch type at runtime.

### Training signal
V13 PLUS / MINUS / SHAM traces are used only during training.

The model is trained on:
- target PLUS > target MINUS
- target SHAM > target MINUS
- on PLUS/SHAM: target > hard semantic replacement
- on MINUS: selected hard replacement > target

That last "flip" is especially important. It prevents the judge from simply
following the topic of the question. The question is target-conditioned in both
PLUS and MINUS; the correct ranking must reverse when the target is replaced.

### Model
The script automatically discovers the same cached Qwen base model used by the
V13/V11 PEFT adapters. It trains a NEW sequence-classification LoRA adapter in
4-bit. Existing RL adapters are never changed.

### Leakage control
Train/validation components are connected components over BOTH:
- probe group_id
- counterfactual pair_id

If a system pair was reused by another probe group, all linked examples remain
on one side of the split.

### Data audit
Before training, the script cross-checks discovery branch outputs and
selected/true roles against:
- artifacts/v13_train_response_public.jsonl
- artifacts/v13_train_response_truth.jsonl

It aborts on >1% semantic mismatches.

## Install

From the project root:

    tar -xzf ~/v13_1_strong_qwen_judge_patch.tar.gz -C .
    python -m py_compile analysis/train_v13_1_strong_judge.py
    bash -n cluster/v13_1_strong_judge.sbatch

Submit:

    sbatch cluster/v13_1_strong_judge.sbatch

Monitor:

    squeue -u $USER
    tail -f logs/v13-1-strong-<JOBID>.out

After completion, only send:

    cat artifacts/v13_1_strong_judge/metrics.json
    cat artifacts/v13_1_strong_judge/decision.json

## One GPU
Designed for one GPU. Qwen 1.5B is loaded in NF4 4-bit and only a LoRA adapter
plus the sequence-classification head are trained.

## Important
This does NOT yet retrain the RL question policy. First we require the evidence
judge to prove it can interpret the causal fingerprints the RL already exposes.
If it passes, the next step is to use this judge as an additional
discrimination/information-gain reward for the existing RL policy.
