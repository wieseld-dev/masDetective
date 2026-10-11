MAS5 V4.7.5 - stabilize CUDA inference and resume without discarding 150/160 questions

- Uses same V4.6 SFT LoRA adapter, but replaces NF4 quantized GENERATION inference with unquantized BF16 and eager attention on RTX 4090.
- Does not change generator prompts, role-slot assignments, quality filter, adapter, training data, or retry/resume logic.
- `v475_preflight.py` checks SHA of approved_102 and combines independent checkpoints without modifying files.
- Existing v473_regenerate.py V474 safely merges approved, previously merged and progress prefixes into a new resume on each new process. Generator keeps atomically checkpointing each accepted question.
- Slurm starts a fresh CUDA process; GPU misaligned errors are not safely recoverable within a running CUDA process.
- No paid MAS/API calls. Candidate bank still requires independent human quality review.
- Untested on BGU RTX4090; tested compilation and static resume logic locally.
