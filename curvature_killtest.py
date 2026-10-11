#!/usr/bin/env python3

# ============================================================
# HIGHER-ORDER RESPONSE CURVATURE — KILL TEST
#
# Scientific question:
#
#   Does second-order interaction between ordinary benign
#   prompt modifications expose hidden execution order AB vs BA
#   better than:
#
#       1. base response alone,
#       2. first-order response changes,
#       3. the raw responses to combined prompts?
#
# This is deliberately a SMALL falsification experiment.
#
# NO latent execution model.
# NO learned agent operators.
# NO eigenvector assumption.
#
# Second-order interaction is measured directly from black-box
# observable output embeddings:
#
#   Δ_ab y = y_ab - y_a - y_b + y_0
#
# ============================================================

import os
import sys
import json
import time
import math
import random
import asyncio
import hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

from openai import AsyncOpenAI

from sentence_transformers import SentenceTransformer

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)

# ------------------------------------------------------------
# Existing project roles
# ------------------------------------------------------------

try:
    from router_r1 import ROLES
except Exception as e:
    print("\nFATAL: could not import ROLES from router_r1.py")
    print("ERROR:", repr(e))
    sys.exit(1)


# ============================================================
# CONFIG
# ============================================================

SEED = 20260916

MODEL = os.environ.get(
    "CURVATURE_MODEL",
    "gpt-4o-mini"
)

TEMPERATURE = 0.0
MAX_TOKENS = 700

N_BASE = 10
CONCURRENCY = 16

MAX_RETRIES = 8

SOURCE_CACHE = Path(
    "artifacts/execution_operator_big/api_runs.jsonl"
)

OUT = Path(
    "artifacts/curvature_killtest"
)

OUT.mkdir(
    parents=True,
    exist_ok=True
)

RUN_CACHE = OUT / "api_runs.jsonl"
BASE_FILE = OUT / "base_queries.json"
RESULT_CSV = OUT / "cv_results.csv"
OOF_CSV = OUT / "oof_predictions.csv"
INCREMENT_CSV = OUT / "increment_analysis.csv"
SANITY_CSV = OUT / "curvature_sanity.csv"

ALPHAS = [
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
]

BOOTSTRAPS = 5000


# ============================================================
# BENIGN INTERVENTIONS
#
# These alter answer requirements only.
# They do NOT mention:
#   - hidden agents
#   - system prompts
#   - topology
#   - routing
#   - internal reasoning
#
# We want ordinary-looking task refinements.
# ============================================================

MODIFIERS = {
    "A":
        "Include one important edge case or failure mode that "
        "should be considered.",

    "B":
        "Compare two plausible alternatives before settling on "
        "the final answer.",

    "C":
        "State one concrete assumption that your answer depends on.",
}

VARIANT_FACTORS = {
    "0":  (),
    "A":  ("A",),
    "B":  ("B",),
    "C":  ("C",),
    "AB": ("A", "B"),
    "AC": ("A", "C"),
    "BC": ("B", "C"),
}

TOPOLOGIES = [
    "AB",
    "BA",
]


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# UTILS
# ============================================================

def sha1_text(x):
    return hashlib.sha1(
        x.encode("utf-8")
    ).hexdigest()[:12]


def extract_role_text(name):
    """
    Robustly extract a system prompt/description from ROLES,
    whether the project stores roles as strings or dictionaries.
    """

    if not isinstance(ROLES, dict):
        raise RuntimeError(
            "router_r1.ROLES is not a dict"
        )

    key = None

    if name in ROLES:
        key = name
    else:
        for k in ROLES:
            if str(k).lower() == name.lower():
                key = k
                break

    if key is None:
        raise RuntimeError(
            f"Role {name!r} not found. "
            f"Available role keys: {list(ROLES.keys())}"
        )

    value = ROLES[key]

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):

        preferred = [
            "system_prompt",
            "system",
            "prompt",
            "description",
            "instructions",
            "role",
        ]

        for p in preferred:
            v = value.get(p)

            if isinstance(v, str) and v.strip():
                return v.strip()

        # Last resort: concatenate useful string fields.
        vals = [
            str(v).strip()
            for v in value.values()
            if isinstance(v, str)
            and str(v).strip()
        ]

        if vals:
            return "\n".join(vals)

    return str(value)


ROLE_PROMPTS = {
    "A": extract_role_text("A"),
    "B": extract_role_text("B"),
}

print("=" * 100)
print("HIGHER-ORDER RESPONSE CURVATURE KILL TEST")
print("=" * 100)
print("MODEL:", MODEL)
print("N_BASE:", N_BASE)
print("CONCURRENCY:", CONCURRENCY)
print("ROLE A hash:", sha1_text(ROLE_PROMPTS["A"]))
print("ROLE B hash:", sha1_text(ROLE_PROMPTS["B"]))
print()


# ============================================================
# LOAD BASE QUERIES
# ============================================================

def load_source_queries():

    if not SOURCE_CACHE.exists():
        raise FileNotFoundError(
            f"Missing source cache: {SOURCE_CACHE}"
        )

    by_pos = {}

    with SOURCE_CACHE.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            r = json.loads(line)

            if "task" not in r:
                continue

            task = str(
                r["task"]
            ).strip()

            if not task:
                continue

            pos = r.get(
                "position",
                len(by_pos)
            )

            try:
                pos = int(pos)
            except Exception:
                pos = len(by_pos)

            by_pos[pos] = task

    items = [
        (p, by_pos[p])
        for p in sorted(by_pos)
    ]

    # Deduplicate exact tasks.
    seen = set()
    unique = []

    for p, q in items:

        if q in seen:
            continue

        seen.add(q)
        unique.append(
            (p, q)
        )

    if len(unique) < N_BASE:
        raise RuntimeError(
            f"Only {len(unique)} unique source queries found; "
            f"need {N_BASE}."
        )

    return unique


def select_or_load_base_queries():

    # Important for resume:
    # once chosen, NEVER silently choose a new set.

    if BASE_FILE.exists():

        with BASE_FILE.open(
            encoding="utf-8"
        ) as f:
            x = json.load(f)

        if len(x) != N_BASE:
            raise RuntimeError(
                f"{BASE_FILE} already exists with "
                f"{len(x)} queries, but N_BASE={N_BASE}. "
                "Delete the curvature_killtest directory if "
                "you intentionally want a completely new run."
            )

        return x

    source = load_source_queries()

    rng = random.Random(
        SEED
    )

    chosen = rng.sample(
        source,
        N_BASE
    )

    chosen = [
        {
            "base_id": i,
            "source_position": int(pos),
            "task": task,
        }
        for i, (pos, task)
        in enumerate(chosen)
    ]

    with BASE_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            chosen,
            f,
            indent=2,
            ensure_ascii=False
        )

    return chosen


BASES = select_or_load_base_queries()

print("Selected base queries:")

for x in BASES:
    print(
        f"  base={x['base_id']:02d} "
        f"source_pos={x['source_position']} "
        f"{x['task'][:80]!r}"
    )

print()


# ============================================================
# BUILD NATURAL VARIANTS
# ============================================================

def build_variant(
    base_query,
    factors
):

    if not factors:
        return base_query

    requirements = [
        MODIFIERS[x]
        for x in factors
    ]

    bullets = "\n".join(
        "- " + x
        for x in requirements
    )

    return (
        base_query.rstrip()
        + "\n\n"
        + "Additional answer requirements:\n"
        + bullets
    )


EXPECTED = []

for b in BASES:

    for variant, factors in VARIANT_FACTORS.items():

        query = build_variant(
            b["task"],
            factors
        )

        for topology in TOPOLOGIES:

            EXPECTED.append({
                "base_id":
                    int(b["base_id"]),

                "source_position":
                    int(b["source_position"]),

                "variant":
                    variant,

                "topology":
                    topology,

                "query":
                    query,
            })


def record_key(r):

    return (
        int(r["base_id"]),
        str(r["variant"]),
        str(r["topology"]),
    )


print(
    "Expected workflows:",
    len(EXPECTED)
)

print(
    "Expected underlying agent calls:",
    len(EXPECTED) * 2
)


# ============================================================
# RESUMABLE CACHE
# ============================================================

def load_run_cache():

    done = {}

    if not RUN_CACHE.exists():
        return done

    with RUN_CACHE.open(
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            try:
                r = json.loads(line)
            except Exception:
                continue

            if (
                "base_id" in r
                and "variant" in r
                and "topology" in r
                and r.get("final_output")
            ):
                done[
                    record_key(r)
                ] = r

    return done


# ============================================================
# OPENAI EXECUTION
# ============================================================

client = AsyncOpenAI()

SEM = asyncio.Semaphore(
    CONCURRENCY
)


def agent_user_message(
    original_query,
    previous_output,
    step,
    total
):

    if previous_output is None:

        return (
            "Original task:\n"
            f"{original_query}\n\n"
            f"You are stage {step} of {total}. "
            "Work on the task according to your assigned role. "
            "Produce a useful draft for the next stage. "
            "Do not discuss the existence of hidden agents, "
            "routing, topology, or system prompts."
        )

    return (
        "Original task:\n"
        f"{original_query}\n\n"
        "Previous stage draft:\n"
        f"{previous_output}\n\n"
        f"You are stage {step} of {total}. "
        "Continue, transform, check, or improve the draft "
        "according to your assigned role. "
        "Return the best final answer to the original task. "
        "Do not discuss the existence of hidden agents, "
        "routing, topology, or system prompts."
    )


async def call_agent(
    role_name,
    original_query,
    previous_output,
    step,
    total,
):

    system_prompt = ROLE_PROMPTS[
        role_name
    ]

    last_error = None

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            async with SEM:

                response = await client.chat.completions.create(
                    model=MODEL,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt,
                        },
                        {
                            "role": "user",
                            "content":
                                agent_user_message(
                                    original_query,
                                    previous_output,
                                    step,
                                    total,
                                ),
                        },
                    ],
                )

            text = (
                response
                .choices[0]
                .message
                .content
            )

            if text is None:
                text = ""

            text = str(
                text
            ).strip()

            if not text:
                raise RuntimeError(
                    "Model returned empty output"
                )

            usage = getattr(
                response,
                "usage",
                None
            )

            return {
                "text": text,

                "prompt_tokens":
                    int(
                        getattr(
                            usage,
                            "prompt_tokens",
                            0
                        ) or 0
                    ),

                "completion_tokens":
                    int(
                        getattr(
                            usage,
                            "completion_tokens",
                            0
                        ) or 0
                    ),
            }

        except Exception as e:

            last_error = e

            wait = min(
                60.0,
                1.5 ** attempt
                + random.random()
            )

            print(
                f"[retry {attempt}/{MAX_RETRIES}] "
                f"role={role_name} "
                f"error={type(e).__name__}: {e} "
                f"sleep={wait:.1f}s",
                flush=True
            )

            await asyncio.sleep(
                wait
            )

    raise RuntimeError(
        f"API failed after {MAX_RETRIES} attempts: "
        f"{last_error!r}"
    )


async def run_workflow(spec):

    topology = spec[
        "topology"
    ]

    query = spec[
        "query"
    ]

    previous = None
    intermediate = []

    total_prompt_tokens = 0
    total_completion_tokens = 0

    t0 = time.perf_counter()

    for step, role_name in enumerate(
        topology,
        start=1
    ):

        out = await call_agent(
            role_name=role_name,
            original_query=query,
            previous_output=previous,
            step=step,
            total=len(topology),
        )

        previous = out[
            "text"
        ]

        intermediate.append(
            previous
        )

        total_prompt_tokens += out[
            "prompt_tokens"
        ]

        total_completion_tokens += out[
            "completion_tokens"
        ]

    elapsed = (
        time.perf_counter()
        - t0
    )

    return {
        **spec,

        "model":
            MODEL,

        "temperature":
            TEMPERATURE,

        "intermediate_outputs":
            intermediate,

        "final_output":
            previous,

        "prompt_tokens":
            total_prompt_tokens,

        "completion_tokens":
            total_completion_tokens,

        "workflow_seconds":
            elapsed,

        "timestamp":
            time.time(),
    }


async def collect():

    done = load_run_cache()

    missing = [
        x
        for x in EXPECTED
        if record_key(x)
        not in done
    ]

    print()
    print(
        f"Cache already complete: "
        f"{len(done)}/{len(EXPECTED)}"
    )

    print(
        f"Missing workflows: {len(missing)}"
    )

    if not missing:
        return done

    tasks = [
        asyncio.create_task(
            run_workflow(spec)
        )
        for spec in missing
    ]

    completed_new = 0

    with RUN_CACHE.open(
        "a",
        encoding="utf-8"
    ) as f:

        for future in asyncio.as_completed(
            tasks
        ):

            r = await future

            f.write(
                json.dumps(
                    r,
                    ensure_ascii=False
                )
                + "\n"
            )

            f.flush()

            done[
                record_key(r)
            ] = r

            completed_new += 1

            if (
                completed_new % 10 == 0
                or completed_new
                == len(missing)
            ):

                print(
                    f"COLLECTED "
                    f"{completed_new}/{len(missing)} new "
                    f"| total "
                    f"{len(done)}/{len(EXPECTED)}",
                    flush=True
                )

    return done


t_collect_start = time.perf_counter()

DONE = asyncio.run(
    collect()
)

collect_seconds = (
    time.perf_counter()
    - t_collect_start
)


# ============================================================
# COMPLETENESS CHECK
# ============================================================

missing_keys = []

for x in EXPECTED:

    k = record_key(x)

    if k not in DONE:
        missing_keys.append(
            k
        )

if missing_keys:

    print()
    print("FATAL: collection incomplete.")
    print(
        "Missing:",
        missing_keys
    )

    print(
        "Re-run the same sbatch job. "
        "The cache will resume automatically."
    )

    sys.exit(2)


print()
print("=" * 100)
print("COLLECTION COMPLETE")
print("=" * 100)
print(
    f"Collection wall time: "
    f"{collect_seconds / 60:.2f} min"
)


records = [
    DONE[
        record_key(x)
    ]
    for x in EXPECTED
]


prompt_tokens = sum(
    int(
        r.get(
            "prompt_tokens",
            0
        )
    )
    for r in records
)

completion_tokens = sum(
    int(
        r.get(
            "completion_tokens",
            0
        )
    )
    for r in records
)

print(
    "Prompt tokens:",
    prompt_tokens
)

print(
    "Completion tokens:",
    completion_tokens
)


# ============================================================
# EMBED OBSERVABLE FINAL OUTPUTS
# ============================================================

print()
print("Loading MiniLM...")

encoder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# Stable order.
embed_keys = []
texts = []

for b in range(
    N_BASE
):

    for topology in TOPOLOGIES:

        for variant in VARIANT_FACTORS:

            r = DONE[
                (
                    b,
                    variant,
                    topology
                )
            ]

            embed_keys.append(
                (
                    b,
                    topology,
                    variant
                )
            )

            texts.append(
                r["final_output"]
            )


print(
    f"Encoding {len(texts)} final outputs..."
)

E = encoder.encode(
    texts,
    batch_size=64,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True,
)

E = np.asarray(
    E,
    dtype=np.float64
)

EMB = {
    k: E[i]
    for i, k in enumerate(
        embed_keys
    )
}

D_EMB = E.shape[1]

print(
    "Embedding dimension:",
    D_EMB
)


# ============================================================
# BUILD RESPONSE DIFFERENCES
#
# First-order:
#
#   dA = y_A - y_0
#
# Second-order:
#
#   dAB = y_AB - y_A - y_B + y_0
#
# ============================================================

samples = []

sanity_rows = []


def norm(x):
    return float(
        np.linalg.norm(x)
    )


def cosine(a, b):

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)

    if (
        na < 1e-12
        or nb < 1e-12
    ):
        return 0.0

    return float(
        np.dot(a, b)
        /
        (na * nb)
    )


for b in range(
    N_BASE
):

    for topology in TOPOLOGIES:

        y0 = EMB[
            (b, topology, "0")
        ]

        yA = EMB[
            (b, topology, "A")
        ]

        yB = EMB[
            (b, topology, "B")
        ]

        yC = EMB[
            (b, topology, "C")
        ]

        yAB = EMB[
            (b, topology, "AB")
        ]

        yAC = EMB[
            (b, topology, "AC")
        ]

        yBC = EMB[
            (b, topology, "BC")
        ]

        # First-order.
        dA = yA - y0
        dB = yB - y0
        dC = yC - y0

        # Second-order interactions.
        dAB = (
            yAB
            - yA
            - yB
            + y0
        )

        dAC = (
            yAC
            - yA
            - yC
            + y0
        )

        dBC = (
            yBC
            - yB
            - yC
            + y0
        )

        # Interpretable scalar invariants.
        invariants = np.array([
            norm(dA),
            norm(dB),
            norm(dC),

            norm(dAB),
            norm(dAC),
            norm(dBC),

            norm(dAB)
            / (
                norm(dA)
                + norm(dB)
                + 1e-12
            ),

            norm(dAC)
            / (
                norm(dA)
                + norm(dC)
                + 1e-12
            ),

            norm(dBC)
            / (
                norm(dB)
                + norm(dC)
                + 1e-12
            ),

            cosine(
                dA,
                dB
            ),

            cosine(
                dA,
                dC
            ),

            cosine(
                dB,
                dC
            ),

            cosine(
                dAB,
                dA
            ),

            cosine(
                dAB,
                dB
            ),

            cosine(
                dAC,
                dA
            ),

            cosine(
                dAC,
                dC
            ),

            cosine(
                dBC,
                dB
            ),

            cosine(
                dBC,
                dC
            ),
        ], dtype=np.float64)

        reps = {
            # One ordinary response.
            "BASE":
                y0,

            # Raw single-intervention responses.
            "SINGLES_RAW":
                np.concatenate([
                    yA,
                    yB,
                    yC,
                ]),

            # MATCHED CONTROL:
            # exactly the three combined-prompt responses.
            "PAIR_RAW":
                np.concatenate([
                    yAB,
                    yAC,
                    yBC,
                ]),

            # Every observable response.
            "RAW_ALL":
                np.concatenate([
                    y0,
                    yA,
                    yB,
                    yC,
                    yAB,
                    yAC,
                    yBC,
                ]),

            # Pure first-order finite differences.
            "FIRST":
                np.concatenate([
                    dA,
                    dB,
                    dC,
                ]),

            # Pure second-order curvature.
            "SECOND":
                np.concatenate([
                    dAB,
                    dAC,
                    dBC,
                ]),

            # What we know before pair interactions.
            "BASE_FIRST":
                np.concatenate([
                    y0,
                    dA,
                    dB,
                    dC,
                ]),

            # Main higher-order representation.
            "BASE_FIRST_SECOND":
                np.concatenate([
                    y0,
                    dA,
                    dB,
                    dC,
                    dAB,
                    dAC,
                    dBC,
                ]),

            # Low-dimensional / coordinate-light diagnostic.
            "SECOND_INVARIANTS":
                invariants,
        }

        samples.append({
            "base_id":
                b,

            "topology":
                topology,

            "label":
                1
                if topology == "AB"
                else 0,

            "reps":
                reps,
        })

        sanity_rows.append({
            "base_id":
                b,

            "topology":
                topology,

            "dA_norm":
                norm(dA),

            "dB_norm":
                norm(dB),

            "dC_norm":
                norm(dC),

            "dAB_norm":
                norm(dAB),

            "dAC_norm":
                norm(dAC),

            "dBC_norm":
                norm(dBC),

            "AB_ratio":
                norm(dAB)
                / (
                    norm(dA)
                    + norm(dB)
                    + 1e-12
                ),

            "AC_ratio":
                norm(dAC)
                / (
                    norm(dA)
                    + norm(dC)
                    + 1e-12
                ),

            "BC_ratio":
                norm(dBC)
                / (
                    norm(dB)
                    + norm(dC)
                    + 1e-12
                ),
        })


sanity = pd.DataFrame(
    sanity_rows
)

sanity.to_csv(
    SANITY_CSV,
    index=False
)


REP_NAMES = list(
    samples[0]["reps"].keys()
)

X = {
    rep: np.stack([
        s["reps"][rep]
        for s in samples
    ])
    for rep in REP_NAMES
}

y = np.array([
    s["label"]
    for s in samples
])

groups = np.array([
    s["base_id"]
    for s in samples
])


# ============================================================
# SANITY ASSERTIONS
# ============================================================

assert len(samples) == (
    N_BASE * 2
)

assert np.sum(
    y == 1
) == N_BASE

assert np.sum(
    y == 0
) == N_BASE

for rep, xx in X.items():

    if not np.all(
        np.isfinite(xx)
    ):
        raise RuntimeError(
            f"Non-finite values in representation {rep}"
        )

    if np.std(xx) < 1e-12:
        raise RuntimeError(
            f"Representation {rep} is constant"
        )


print()
print("=" * 100)
print("FEATURES")
print("=" * 100)

for rep in REP_NAMES:

    print(
        f"{rep:22s} "
        f"shape={X[rep].shape}"
    )


# ============================================================
# NESTED GROUPED CV
#
# OUTER:
#   Leave ONE BASE QUERY out completely.
#
# Thus both AB and BA versions of that base query are held out.
#
# INNER:
#   GroupKFold only on remaining base queries selects ridge alpha.
#
# No prompt from a test base query can influence:
#   - scaler
#   - hyperparameter choice
#   - classifier fit
# ============================================================

def choose_alpha(
    Xtrain,
    ytrain,
    gtrain
):

    unique_groups = np.unique(
        gtrain
    )

    n_splits = min(
        4,
        len(unique_groups)
    )

    inner = GroupKFold(
        n_splits=n_splits
    )

    rows = []

    for alpha in ALPHAS:

        fold_auc = []

        for tr, va in inner.split(
            Xtrain,
            ytrain,
            gtrain
        ):

            pipe = Pipeline([
                (
                    "scale",
                    StandardScaler()
                ),
                (
                    "clf",
                    RidgeClassifier(
                        alpha=alpha
                    )
                ),
            ])

            pipe.fit(
                Xtrain[tr],
                ytrain[tr]
            )

            score = (
                pipe.decision_function(
                    Xtrain[va]
                )
            )

            fold_auc.append(
                roc_auc_score(
                    ytrain[va],
                    score
                )
            )

        rows.append(
            (
                alpha,
                float(
                    np.mean(
                        fold_auc
                    )
                )
            )
        )

    rows.sort(
        key=lambda z: (
            z[1],
            -math.log10(
                z[0] + 1e-12
            )
        ),
        reverse=True
    )

    return rows[0][0]


def nested_logo(
    rep_name
):

    xx = X[
        rep_name
    ]

    logo = LeaveOneGroupOut()

    oof_score = np.zeros(
        len(y),
        dtype=np.float64
    )

    oof_pred = np.zeros(
        len(y),
        dtype=np.int64
    )

    selected_alphas = []

    for train_idx, test_idx in logo.split(
        xx,
        y,
        groups
    ):

        alpha = choose_alpha(
            xx[train_idx],
            y[train_idx],
            groups[train_idx]
        )

        selected_alphas.append(
            alpha
        )

        pipe = Pipeline([
            (
                "scale",
                StandardScaler()
            ),
            (
                "clf",
                RidgeClassifier(
                    alpha=alpha
                )
            ),
        ])

        pipe.fit(
            xx[train_idx],
            y[train_idx]
        )

        score = pipe.decision_function(
            xx[test_idx]
        )

        pred = (
            score > 0
        ).astype(
            np.int64
        )

        oof_score[
            test_idx
        ] = score

        oof_pred[
            test_idx
        ] = pred

    auc = roc_auc_score(
        y,
        oof_score
    )

    acc = accuracy_score(
        y,
        oof_pred
    )

    # Pairwise orientation accuracy:
    #
    # For each unseen base query,
    # was AB scored higher than BA?
    #
    # This is a very clean order-identification diagnostic.
    pair_correct = []
    pair_margin = []

    for g in sorted(
        np.unique(groups)
    ):

        ids = np.where(
            groups == g
        )[0]

        assert len(ids) == 2

        pos = ids[
            y[ids] == 1
        ][0]

        neg = ids[
            y[ids] == 0
        ][0]

        margin = (
            oof_score[pos]
            -
            oof_score[neg]
        )

        pair_margin.append(
            margin
        )

        pair_correct.append(
            margin > 0
        )

    return {
        "rep":
            rep_name,

        "auc":
            float(auc),

        "accuracy":
            float(acc),

        "pair_accuracy":
            float(
                np.mean(
                    pair_correct
                )
            ),

        "mean_pair_margin":
            float(
                np.mean(
                    pair_margin
                )
            ),

        "median_selected_alpha":
            float(
                np.median(
                    selected_alphas
                )
            ),

        "oof_score":
            oof_score,

        "oof_pred":
            oof_pred,

        "pair_correct":
            np.asarray(
                pair_correct,
                dtype=np.float64
            ),

        "pair_margin":
            np.asarray(
                pair_margin,
                dtype=np.float64
            ),
    }


print()
print("=" * 100)
print("RUNNING NESTED LEAVE-ONE-QUERY-OUT CV")
print("=" * 100)

CV = {}

for rep in REP_NAMES:

    print(
        "Evaluating",
        rep,
        flush=True
    )

    CV[
        rep
    ] = nested_logo(
        rep
    )


# ============================================================
# GROUP BOOTSTRAP ON HELD-OUT PREDICTIONS
#
# Resample BASE QUERIES, not individual samples.
# This preserves the AB/BA pairing.
# ============================================================

rng = np.random.default_rng(
    SEED + 177
)

group_ids = np.array(
    sorted(
        np.unique(
            groups
        )
    )
)


def auc_for_group_sample(
    rep,
    sampled_groups
):

    scores = CV[
        rep
    ]["oof_score"]

    yy = []
    ss = []

    for g in sampled_groups:

        ids = np.where(
            groups == g
        )[0]

        yy.extend(
            y[ids].tolist()
        )

        ss.extend(
            scores[ids].tolist()
        )

    return roc_auc_score(
        yy,
        ss
    )


boot_auc = {
    rep: []
    for rep in REP_NAMES
}

boot_pair = {
    rep: []
    for rep in REP_NAMES
}

for _ in range(
    BOOTSTRAPS
):

    sampled = rng.choice(
        group_ids,
        size=len(group_ids),
        replace=True
    )

    for rep in REP_NAMES:

        boot_auc[
            rep
        ].append(
            auc_for_group_sample(
                rep,
                sampled
            )
        )

        pc = CV[
            rep
        ][
            "pair_correct"
        ]

        boot_pair[
            rep
        ].append(
            float(
                np.mean(
                    pc[
                        sampled.astype(int)
                    ]
                )
            )
        )


# ============================================================
# RESULT TABLE
# ============================================================

result_rows = []

for rep in REP_NAMES:

    ba = np.asarray(
        boot_auc[
            rep
        ]
    )

    bp = np.asarray(
        boot_pair[
            rep
        ]
    )

    result_rows.append({
        "representation":
            rep,

        "dim":
            X[rep].shape[1],

        "auc":
            CV[rep]["auc"],

        "auc_ci_low":
            np.quantile(
                ba,
                .025
            ),

        "auc_ci_high":
            np.quantile(
                ba,
                .975
            ),

        "accuracy":
            CV[rep]["accuracy"],

        "pair_accuracy":
            CV[rep]["pair_accuracy"],

        "pair_ci_low":
            np.quantile(
                bp,
                .025
            ),

        "pair_ci_high":
            np.quantile(
                bp,
                .975
            ),

        "mean_pair_margin":
            CV[
                rep
            ][
                "mean_pair_margin"
            ],

        "median_selected_alpha":
            CV[
                rep
            ][
                "median_selected_alpha"
            ],
    })


result_df = pd.DataFrame(
    result_rows
)

result_df.to_csv(
    RESULT_CSV,
    index=False
)


# ============================================================
# OOF PREDICTIONS
# ============================================================

oof_rows = []

for i, s in enumerate(
    samples
):

    row = {
        "base_id":
            s["base_id"],

        "topology":
            s["topology"],

        "label":
            s["label"],
    }

    for rep in REP_NAMES:

        row[
            rep + "_score"
        ] = float(
            CV[
                rep
            ][
                "oof_score"
            ][i]
        )

    oof_rows.append(
        row
    )


pd.DataFrame(
    oof_rows
).to_csv(
    OOF_CSV,
    index=False
)


# ============================================================
# PAIRED INCREMENT ANALYSIS
#
# Crucial comparisons:
#
# SECOND vs FIRST
#   Does curvature beat first order?
#
# BASE_FIRST_SECOND vs BASE_FIRST
#   Does curvature add information beyond what first-order
#   observations already contain?
#
# SECOND vs PAIR_RAW
#   Does the finite-difference interaction help beyond simply
#   seeing outputs to combined prompts?
#
# BASE_FIRST_SECOND vs RAW_ALL
#   Is the curvature coordinate system doing anything beyond
#   all raw observations?
# ============================================================

COMPARISONS = [
    (
        "SECOND",
        "FIRST",
        "SECOND_minus_FIRST"
    ),

    (
        "BASE_FIRST_SECOND",
        "BASE_FIRST",
        "ADD_SECOND_TO_BASE_FIRST"
    ),

    (
        "SECOND",
        "PAIR_RAW",
        "SECOND_minus_PAIR_RAW"
    ),

    (
        "BASE_FIRST_SECOND",
        "RAW_ALL",
        "CURVATURE_COORDS_minus_RAW_ALL"
    ),

    (
        "SECOND_INVARIANTS",
        "BASE",
        "INVARIANTS_minus_BASE"
    ),
]


increment_rows = []

for better, baseline, name in COMPARISONS:

    d_auc = (
        np.asarray(
            boot_auc[
                better
            ]
        )
        -
        np.asarray(
            boot_auc[
                baseline
            ]
        )
    )

    d_pair = (
        np.asarray(
            boot_pair[
                better
            ]
        )
        -
        np.asarray(
            boot_pair[
                baseline
            ]
        )
    )

    observed_auc_delta = (
        CV[better]["auc"]
        -
        CV[baseline]["auc"]
    )

    observed_pair_delta = (
        CV[better]["pair_accuracy"]
        -
        CV[baseline]["pair_accuracy"]
    )

    increment_rows.append({
        "comparison":
            name,

        "better":
            better,

        "baseline":
            baseline,

        "auc_delta":
            observed_auc_delta,

        "auc_delta_ci_low":
            np.quantile(
                d_auc,
                .025
            ),

        "auc_delta_ci_high":
            np.quantile(
                d_auc,
                .975
            ),

        "bootstrap_prob_auc_delta_gt_0":
            np.mean(
                d_auc > 0
            ),

        "pair_accuracy_delta":
            observed_pair_delta,

        "pair_delta_ci_low":
            np.quantile(
                d_pair,
                .025
            ),

        "pair_delta_ci_high":
            np.quantile(
                d_pair,
                .975
            ),

        "bootstrap_prob_pair_delta_gt_0":
            np.mean(
                d_pair > 0
            ),
    })


increment_df = pd.DataFrame(
    increment_rows
)

increment_df.to_csv(
    INCREMENT_CSV,
    index=False
)


# ============================================================
# SANITY SUMMARY
# ============================================================

print()
print("=" * 110)
print("CURVATURE MAGNITUDE SANITY")
print("=" * 110)

sanity_summary = (
    sanity.groupby(
        "topology"
    )[
        [
            "dA_norm",
            "dB_norm",
            "dC_norm",
            "dAB_norm",
            "dAC_norm",
            "dBC_norm",
            "AB_ratio",
            "AC_ratio",
            "BC_ratio",
        ]
    ]
    .mean()
)

print(
    sanity_summary
    .round(5)
    .to_string()
)


# ============================================================
# MAIN RESULTS
# ============================================================

print()
print("=" * 110)
print("HELD-OUT BASE-QUERY IDENTIFICATION")
print("=" * 110)

print(
    result_df[
        [
            "representation",
            "dim",
            "auc",
            "auc_ci_low",
            "auc_ci_high",
            "accuracy",
            "pair_accuracy",
            "pair_ci_low",
            "pair_ci_high",
            "mean_pair_margin",
        ]
    ]
    .sort_values(
        "auc",
        ascending=False
    )
    .round(5)
    .to_string(
        index=False
    )
)


print()
print("=" * 110)
print("DOES SECOND ORDER ADD INFORMATION?")
print("=" * 110)

print(
    increment_df
    .round(5)
    .to_string(
        index=False
    )
)


# ============================================================
# PER-QUERY ORIENTATION
# ============================================================

print()
print("=" * 110)
print("PER-QUERY AB-vs-BA ORIENTATION")
print("=" * 110)

pair_rows = []

for g in range(
    N_BASE
):

    row = {
        "base_id": g
    }

    for rep in [
        "BASE",
        "FIRST",
        "SECOND",
        "PAIR_RAW",
        "BASE_FIRST",
        "BASE_FIRST_SECOND",
        "RAW_ALL",
    ]:

        ids = np.where(
            groups == g
        )[0]

        pos = ids[
            y[ids] == 1
        ][0]

        neg = ids[
            y[ids] == 0
        ][0]

        margin = (
            CV[rep][
                "oof_score"
            ][pos]
            -
            CV[rep][
                "oof_score"
            ][neg]
        )

        row[
            rep
        ] = margin

    pair_rows.append(
        row
    )


pair_df = pd.DataFrame(
    pair_rows
)

print(
    pair_df
    .round(4)
    .to_string(
        index=False
    )
)


# ============================================================
# PRE-SPECIFIED KILL GATE
# ============================================================

second_auc = CV[
    "SECOND"
]["auc"]

first_auc = CV[
    "FIRST"
]["auc"]

combined_auc = CV[
    "BASE_FIRST_SECOND"
]["auc"]

base_first_auc = CV[
    "BASE_FIRST"
]["auc"]

pairraw_auc = CV[
    "PAIR_RAW"
]["auc"]

rawall_auc = CV[
    "RAW_ALL"
]["auc"]


inc_main = increment_df[
    increment_df[
        "comparison"
    ]
    ==
    "ADD_SECOND_TO_BASE_FIRST"
].iloc[0]


print()
print("=" * 110)
print("PRE-SPECIFIED DECISION")
print("=" * 110)

strong_go = (
    combined_auc >= 0.70
    and
    (
        combined_auc
        -
        base_first_auc
    ) >= 0.10
    and
    inc_main[
        "bootstrap_prob_auc_delta_gt_0"
    ] >= 0.90
)

weak_go = (
    max(
        second_auc,
        combined_auc
    ) >= 0.65
    and
    (
        (
            second_auc
            -
            first_auc
        ) >= 0.05
        or
        (
            combined_auc
            -
            base_first_auc
        ) >= 0.05
    )
)

curvature_specific = (
    second_auc
    >
    pairraw_auc
)


if strong_go:

    verdict = (
        "STRONG GO — second-order response interaction appears "
        "to add held-out execution-order information."
    )

elif weak_go:

    verdict = (
        "WEAK GO — there is enough signal to justify one larger "
        "fresh-query replication, but NOT a scientific claim yet."
    )

else:

    verdict = (
        "NO-GO — current second-order construction does not "
        "justify a larger curvature experiment."
    )


print(
    "BASE AUC              :",
    round(
        CV["BASE"]["auc"],
        4
    )
)

print(
    "FIRST AUC             :",
    round(
        first_auc,
        4
    )
)

print(
    "SECOND AUC            :",
    round(
        second_auc,
        4
    )
)

print(
    "PAIR_RAW AUC          :",
    round(
        pairraw_auc,
        4
    )
)

print(
    "BASE_FIRST AUC        :",
    round(
        base_first_auc,
        4
    )
)

print(
    "BASE_FIRST_SECOND AUC :",
    round(
        combined_auc,
        4
    )
)

print(
    "RAW_ALL AUC           :",
    round(
        rawall_auc,
        4
    )
)

print()

print(
    "Second-order increment:",
    round(
        combined_auc
        -
        base_first_auc,
        4
    )
)

print(
    "P_boot(delta > 0)     :",
    round(
        float(
            inc_main[
                "bootstrap_prob_auc_delta_gt_0"
            ]
        ),
        4
    )
)

print(
    "SECOND > PAIR_RAW?    :",
    curvature_specific
)

print()
print("VERDICT:")
print(verdict)


print()
print("=" * 110)
print("HOW TO INTERPRET")
print("=" * 110)

print("""
The important comparisons are NOT just raw accuracy.

1. SECOND vs FIRST

   If SECOND > FIRST:
       nonlinear interaction carries more order information than
       independent first-order responses.

2. BASE_FIRST_SECOND vs BASE_FIRST

   This is the central test.

   If adding Δ_ij consistently improves held-out identification,
   then second-order response geometry contributes information
   not already captured by the base response and first-order
   perturbations.

3. SECOND vs PAIR_RAW

   This is the matched control.

   PAIR_RAW sees the same combined benign prompts but does not
   perform the finite-difference cancellation.

   If PAIR_RAW is just as good or better, then the result may
   simply be "combined prompts provide more observations", not
   evidence for curvature as the useful representation.

4. BASE_FIRST_SECOND vs RAW_ALL

   These contain almost the same observational information under
   a different coordinate system.

   A large advantage for curvature coordinates would suggest a
   useful inductive bias, not new information.

5. SECOND_INVARIANTS

   This asks whether low-dimensional, coordinate-light curvature
   properties generalize better than the full embedding vectors.


CRITICAL LIMITATION:

This is a KILL TEST with only 10 base queries.
A positive result means "replicate on fresh queries".
It does NOT establish a paper-level claim.

A negative result means we should NOT spend days building
higher-order tensors around this exact construction.
""")


print()
print("Saved:")
for p in [
    RUN_CACHE,
    BASE_FILE,
    SANITY_CSV,
    RESULT_CSV,
    OOF_CSV,
    INCREMENT_CSV,
]:
    print(" ", p)

print()
print(
    "END:",
    time.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
)
