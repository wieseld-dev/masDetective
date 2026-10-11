from pathlib import Path
import json
import torch

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.role_policy import RoleAdaptiveProbePolicyV11
from adaptive_v11_active.schema import PublicBeliefState


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "artifacts/v14_benign_policy"

# Use an existing public state only as a formatting/generation smoke test.
candidates = [
    Path("artifacts/adaptive_v11_role_only_v1/states_train_round_00.jsonl"),
    Path("artifacts/adaptive_v11_active_v1/states_train_round_00.jsonl"),
]

state_path = next((p for p in candidates if p.exists()), None)
if state_path is None:
    raise SystemExit("No existing public belief state file found")

rows = read_jsonl(state_path)
if not rows:
    raise SystemExit(f"Empty state file: {state_path}")

state = PublicBeliefState.from_public_dict(rows[0])

print("=" * 100)
print("V14 ADAPTIVE SMOKE")
print("=" * 100)
print("adapter :", ADAPTER)
print("state   :", state.session_id)
print("round   :", state.round_index)

print()
print("ROLE BELIEF:")
for r, p in sorted(
    state.role_probs.items(),
    key=lambda kv: -kv[1],
):
    print(f"{r:<30} {p:.4f}")

print()
print("HISTORY TURNS:", len(state.history))

policy = RoleAdaptiveProbePolicyV11(
    MODEL,
    adapter_path=ADAPTER,
    trainable=False,
    use_4bit=False,
    max_seq_length=4096,
)

print()
print("=" * 100)
print("GENERATED PROBES")
print("=" * 100)

for j in range(8):
    seed = 20260930 + j
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    sample = policy.sample_probe(
        {
            "belief_state": state.to_public_dict(),
            "candidate_index": j,
            "max_history_turns": 3,
        },
        temperature=0.85,
        top_p=0.95,
        max_new_tokens=96,
    )

    print()
    print(f"[{j}] valid={sample.valid}")
    print(sample.query)

    if j == 0:
        print()
        print("=" * 100)
        print("PROMPT RECEIVED BY V14")
        print("=" * 100)
        print(sample.prompt)

print()
print("SMOKE PASS")
