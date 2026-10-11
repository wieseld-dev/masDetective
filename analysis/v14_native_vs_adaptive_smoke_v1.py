from pathlib import Path
import torch

from adaptive_v11_active.io import read_jsonl
from adaptive_v11_active.schema import PublicBeliefState

from v14.adaptive_belief_policy import (
    V14AdaptiveBeliefPolicy,
)
from v13.policy import (
    DisambiguationProbePolicyV13,
)


MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER = "artifacts/v14_benign_policy"

STATE = Path(
    "artifacts/"
    "adaptive_v11_role_only_v1/"
    "states_train_round_01.jsonl"
)


rows = read_jsonl(STATE)


def spread(x):
    vals = [
        float(v)
        for v in dict(
            x["role_probs"]
        ).values()
    ]
    return max(vals) - min(vals)


raw = max(
    rows,
    key=spread,
)

state = PublicBeliefState.from_public_dict(
    raw
)


native_policy = DisambiguationProbePolicyV13(
    MODEL,
    adapter_path=ADAPTER,
    trainable=False,
    use_4bit=False,
    max_seq_length=4096,
)

adaptive_policy = V14AdaptiveBeliefPolicy(
    MODEL,
    adapter_path=ADAPTER,
    trainable=False,
    use_4bit=False,
    max_seq_length=4096,
)


print("=" * 110)
print("V14 NATIVE vs V14 ADAPTIVE — EXACT A/B")
print("=" * 110)

print("session :", state.session_id)
print("round   :", state.round_index)
print()


for j in range(8):

    outer_obs = {
        "belief_state":
            state.to_public_dict(),

        "candidate_index":
            j,
    }

    # Let adaptive wrapper choose the exact same
    # focus/competitors/history for both branches.
    native_obs = (
        adaptive_policy
        ._adaptive_observation(
            outer_obs
        )
    )

    seed = 2026093000 + j

    # --------------------------------------------------
    # Native V14:
    # exact V13/V14 prompt, no new belief block
    # --------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    a = native_policy.sample_probe(
        native_obs,
        temperature=0.85,
        top_p=0.95,
        max_new_tokens=192,
    )

    # --------------------------------------------------
    # Adaptive V14:
    # same native observation + full belief block
    # --------------------------------------------------

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    b = adaptive_policy.sample_probe(
        outer_obs,
        temperature=0.85,
        top_p=0.95,
        max_new_tokens=192,
    )

    print("-" * 110)

    print(
        f"[{j}] ROLE = "
        f"{native_obs['candidate_role']}"
    )

    print(
        "COMPETITORS =",
        [
            x["role"]
            for x in native_obs[
                "confusable_roles"
            ]
        ],
    )

    print()

    print("NATIVE V14:")
    print(a.query)

    print()

    print("ADAPTIVE V14:")
    print(b.query)

    print()

print("=" * 110)
print("DONE")
print("=" * 110)
