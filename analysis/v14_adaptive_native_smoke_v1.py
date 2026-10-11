from pathlib import Path

import torch

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import (
    PublicBeliefState,
)

from v14.adaptive_belief_policy import (
    V14AdaptiveBeliefPolicy,
)


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "artifacts/v14_benign_policy"


STATE_CANDIDATES = [
    Path(
        "artifacts/"
        "adaptive_v11_role_only_v1/"
        "states_train_round_01.jsonl"
    ),
    Path(
        "artifacts/"
        "adaptive_v11_role_only_v1/"
        "states_dev_round_01.jsonl"
    ),
    Path(
        "artifacts/"
        "adaptive_v11_active_v1/"
        "states_train_round_01.jsonl"
    ),
]


state_path = next(
    (
        p
        for p in STATE_CANDIDATES
        if p.exists()
    ),
    None,
)

if state_path is None:
    raise SystemExit(
        "No round-01 belief state found"
    )


rows = read_jsonl(
    state_path
)

if not rows:
    raise SystemExit(
        f"Empty state file: {state_path}"
    )


# Prefer a state with the largest belief spread.
def spread(x):
    rp = dict(
        x.get(
            "role_probs",
            {},
        )
    )

    vals = [
        float(v)
        for v in rp.values()
    ]

    if not vals:
        return -1.0

    return max(vals) - min(vals)


raw = max(
    rows,
    key=spread,
)

state = (
    PublicBeliefState
    .from_public_dict(
        raw
    )
)


print("=" * 100)
print("V14 ADAPTIVE NATIVE SMOKE")
print("=" * 100)

print(
    "adapter    :",
    ADAPTER,
)

print(
    "state file :",
    state_path,
)

print(
    "session    :",
    state.session_id,
)

print(
    "round      :",
    state.round_index,
)

print(
    "history    :",
    len(state.history),
)


print()
print("FULL ROLE BELIEF:")

for r, p in sorted(
    state.role_probs.items(),
    key=lambda kv: -kv[1],
):
    print(
        f"{r:<30} "
        f"{p:.4f}"
    )


policy = V14AdaptiveBeliefPolicy(
    MODEL,
    adapter_path=ADAPTER,
    trainable=False,
    use_4bit=False,
    max_seq_length=4096,
)


print()
print("=" * 100)
print("8 V14-ADAPTIVE PROBES")
print("=" * 100)


for j in range(8):

    seed = (
        2026093000
        + j
    )

    torch.manual_seed(
        seed
    )

    torch.cuda.manual_seed_all(
        seed
    )

    obs = {
        "belief_state":
            state.to_public_dict(),

        "candidate_index":
            j,

        "max_history_turns":
            2,
    }

    sample = policy.sample_probe(
        obs,
        temperature=0.85,
        top_p=0.95,
        max_new_tokens=192,
    )

    native = (
        policy
        ._adaptive_observation(
            obs
        )
    )

    print()
    print(
        f"[{j}] "
        f"FOCUS={native['candidate_role']} "
        f"valid={sample.valid}"
    )

    print(
        "COMPETITORS=",
        [
            x["role"]
            for x in native[
                "confusable_roles"
            ]
        ],
    )

    print(
        "QUERY:",
        sample.query,
    )

    if j == 0:
        print()
        print("=" * 100)
        print(
            "EXACT PROMPT RECEIVED "
            "BY V14 WEIGHTS"
        )
        print("=" * 100)
        print(
            sample.prompt
        )


print()
print("SMOKE PASS")
