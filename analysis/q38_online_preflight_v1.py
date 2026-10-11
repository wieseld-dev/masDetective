from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from adaptive_v11_active.schema import (
    PublicBeliefState,
)
from adaptive_v11_active.io import (
    read_jsonl,
)
from q38.adaptive_belief_policy import (
    Q38AdaptiveBeliefPolicy,
)


MODEL = "Qwen/Qwen3.8-27B"
ROOT = Path(
    "artifacts/q38_online_r7match_v1"
)
REF = Path(
    "artifacts/q38_utility_lora_v1/policy"
)


print("=" * 100)
print("Q38 ONLINE GRPO PREFLIGHT")
print("=" * 100)


raw = read_jsonl(
    ROOT / "states_train_round_00.jsonl"
)[0]

state = PublicBeliefState.from_public_dict(
    raw
)

obs = {
    "belief_state":
        state.to_public_dict(),
    "candidate_index": 0,
    "max_history_turns": 2,
    "max_answer_chars": 0,
}


print("\n[1/2] dual-adapter inference")

p = Q38AdaptiveBeliefPolicy(
    MODEL,
    adapter_path=str(REF),
    reference_adapter_path=str(REF),
    trainable=False,
    use_4bit=False,
    max_seq_length=3072,
)

torch.manual_seed(12345)
torch.cuda.manual_seed_all(12345)

sample = p.sample_probe(
    obs,
    temperature=0.85,
    top_p=0.95,
    max_new_tokens=32,
)

print("QUERY =", sample.query)

with torch.no_grad():
    a = float(
        p.mean_logprob(
            sample.prompt,
            sample.completion,
            adapter="current",
        )
        .detach()
        .cpu()
    )

    b = float(
        p.mean_logprob(
            sample.prompt,
            sample.completion,
            adapter="reference",
        )
        .detach()
        .cpu()
    )

print("CURRENT LP   =", a)
print("REFERENCE LP =", b)

del p
gc.collect()
torch.cuda.empty_cache()


print("\n[2/2] trainable backward")

p = Q38AdaptiveBeliefPolicy(
    MODEL,
    adapter_path=str(REF),
    trainable=True,
    use_4bit=False,
    max_seq_length=3072,
)

p.model.train()

completion = (
    str(sample.completion).strip()
    or "What information should I provide first?"
)

lp = p.mean_logprob(
    sample.prompt,
    completion,
)

loss = -lp

loss.backward()

ntrain = sum(
    x.numel()
    for x in p.model.parameters()
    if x.requires_grad
)

peak = (
    torch.cuda.max_memory_allocated()
    / 1024**3
)

print("TRAINABLE PARAMS =", ntrain)
print("LOSS             =", float(loss.detach().cpu()))
print("PEAK VRAM GiB    =", round(peak, 2))

if not torch.isfinite(loss):
    raise RuntimeError(
        "non-finite training loss"
    )

print("\nQ38_ONLINE_PREFLIGHT_OK")
