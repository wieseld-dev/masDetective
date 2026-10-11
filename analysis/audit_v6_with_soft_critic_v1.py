#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import joblib
import numpy as np
import torch

from sentence_transformers import SentenceTransformer


sys.path.insert(
    0,
    str(
        Path(".").resolve()
    ),
)


from adaptive_v11_active.schema import (
    PublicBeliefState,
)

from adaptive_v11_curriculum.policy import (
    CurriculumAdaptiveProbePolicyV11,
)

from v11.role_universe import (
    TRAIN_ROLES,
)


OUT = Path(
    "artifacts/v6_soft_critic_audit_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)


SEED = 20260927

N_CONTEXTS = 20
CANDIDATES_PER_CONTEXT = 5

ROLES = list(
    TRAIN_ROLES
)

KS = [
    3,
    4,
    5,
    6,
]


critic_blob = joblib.load(
    "artifacts/"
    "soft_benign_critic_v1/"
    "critic.joblib"
)


critic_model = (
    critic_blob[
        "model"
    ]
)


critic_embedder = (
    SentenceTransformer(
        critic_blob[
            "embedder_name"
        ],
        device="cpu",
    )
)


print(
    "Loading V6-DPO free-form generator..."
)


policy = (
    CurriculumAdaptiveProbePolicyV11(
        "Qwen/Qwen2.5-1.5B-Instruct",

        adapter_path=
            "artifacts/"
            "v6_matched_sft_dpo/"
            "policy",

        trainable=False,

        use_4bit=False,

        max_seq_length=4096,
    )
)


rng = np.random.default_rng(
    SEED
)


rows = []


for context_index in range(
    N_CONTEXTS
):
    # ------------------------------
    # Construct a plausible PUBLIC
    # belief state only.
    #
    # No truth / installed roles.
    # ------------------------------

    karr = rng.dirichlet(
        np.ones(
            len(KS)
        )
        * 3.0
    )


    kp = {
        str(k):
            float(
                karr[i]
            )
        for i, k
        in enumerate(
            KS
        )
    }


    expected_k = sum(
        k
        * kp[
            str(k)
        ]
        for k in KS
    )


    # Smooth distribution over roles whose
    # marginals sum approximately to E[K].
    raw = rng.dirichlet(
        np.ones(
            len(ROLES)
        )
        * 5.0
    )

    rp_arr = (
        raw
        * expected_k
    )


    rp = {
        role:
            float(
                np.clip(
                    rp_arr[i],
                    0.001,
                    0.999,
                )
            )
        for i, role
        in enumerate(
            ROLES
        )
    }


    round_index = int(
        rng.integers(
            0,
            8,
        )
    )


    public = {
        "session_id":
            f"softaudit_{context_index:03d}",

        "split":
            "dev",

        "round_index":
            round_index,

        "k_probs":
            kp,

        "role_probs":
            rp,

        "history":
            [],
    }


    state = (
        PublicBeliefState
        .from_public_dict(
            public
        )
    )


    for candidate_index in range(
        CANDIDATES_PER_CONTEXT
    ):
        seed = (
            SEED
            + context_index
            * 1000
            + candidate_index
        )


        random.seed(
            seed
        )

        torch.manual_seed(
            seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                seed
            )


        sample = policy.sample_probe(
            {
                "belief_state":
                    state.to_public_dict(),

                "candidate_index":
                    candidate_index,

                "max_history_turns":
                    5,
            },

            temperature=
                0.90,

            top_p=
                0.95,

            max_new_tokens=
                192,
        )


        q = str(
            sample.query
            or ""
        ).strip()


        # IMPORTANT:
        # NO benign filter here.
        # Every generated query is retained.
        rows.append({
            "context_index":
                context_index,

            "candidate_index":
                candidate_index,

            "round_index":
                round_index,

            "generator_parser_valid":
                bool(
                    sample.valid
                ),

            "query":
                q,
        })


queries = [
    x["query"]
    for x in rows
]


E = np.asarray(
    critic_embedder.encode(
        queries,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ),
    dtype=np.float32,
)


scores = (
    critic_model
    .predict_proba(
        E
    )[:, 1]
)


for x, score in zip(
    rows,
    scores,
):
    x[
        "soft_benign_score"
    ] = float(
        score
    )


with (
    OUT
    / "queries.jsonl"
).open(
    "w",
    encoding="utf-8",
) as f:

    for x in rows:
        f.write(
            json.dumps(
                x,
                ensure_ascii=False,
            )
            + "\n"
        )


scores = np.asarray(
    scores,
    dtype=float,
)


summary = {
    "queries":
        len(
            rows
        ),

    "generator":
        (
            "artifacts/"
            "v6_matched_sft_dpo/"
            "policy"
        ),

    "hard_filter_used":
        False,

    "mean_soft_benign_score":
        float(
            scores.mean()
        ),

    "median_soft_benign_score":
        float(
            np.median(
                scores
            )
        ),

    "q10":
        float(
            np.quantile(
                scores,
                0.10,
            )
        ),

    "q25":
        float(
            np.quantile(
                scores,
                0.25,
            )
        ),

    "q75":
        float(
            np.quantile(
                scores,
                0.75,
            )
        ),

    "q90":
        float(
            np.quantile(
                scores,
                0.90,
            )
        ),
}


(
    OUT
    / "summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)


print()
print("=" * 100)
print(
    "V6 FREE-FORM GENERATOR — "
    "SOFT BENIGN AUDIT"
)
print("=" * 100)

print(
    json.dumps(
        summary,
        indent=2,
    )
)


print()
print("=" * 100)
print(
    "ALL 100 QUERIES — "
    "LOWEST SOFT-BENIGN SCORE FIRST"
)
print("=" * 100)


for rank, x in enumerate(
    sorted(
        rows,
        key=lambda z:
            z[
                "soft_benign_score"
            ],
    ),
    1,
):
    print()
    print(
        f"[{rank:03d}] "
        f"B={x['soft_benign_score']:.4f} "
        f"context={x['context_index']:02d} "
        f"candidate={x['candidate_index']}"
    )

    print(
        x["query"]
    )


print()
print(
    "Saved:",
    OUT
    / "queries.jsonl",
)
