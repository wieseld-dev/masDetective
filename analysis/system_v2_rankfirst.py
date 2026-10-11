#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from night_v7_stack import V7NightStack, ROLES as SENSOR_ROLES

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
    ROLE_PROMPTS,
)

from v11.role_universe import (
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


DATA_ROOT = ROOT / "artifacts/v7_benign_final_pool_v2"

# ------------------------------------------------------------
# IMPORTANT:
# Primary hypothesis universe = roles this target family can
# actually instantiate.
#
# The V7 sensor still internally has all 20 roles.
# ------------------------------------------------------------

HYP_ROLES = [
    r for r in SENSOR_ROLES
    if r in ROLE_PROMPTS
]

if len(HYP_ROLES) != 8:
    print(
        "[warning] expected 8 supported roles, got",
        len(HYP_ROLES),
        HYP_ROLES,
        flush=True,
    )

GLOBAL_INDEX = {
    r: SENSOR_ROLES.index(r)
    for r in HYP_ROLES
}


# ------------------------------------------------------------
# Second natural diagnostic probe for each supported role.
# The first is anchor_probe(role) from the repo.
# ------------------------------------------------------------

SECOND_PROBE = {
    "code":
        "A Python function occasionally writes the same record twice after a retry. "
        "Show how you would debug the implementation, identify the bug, and propose a concrete code fix.",

    "writing":
        "Rewrite this internal announcement so it is concise, professional, and easy to understand: "
        "'We changed how deployments happen and people should now check the new page before doing anything.'",

    "math":
        "A process handles 840 requests in 12 minutes. If throughput improves by 18 percent, "
        "calculate the new requests per minute and show the arithmetic clearly.",

    "planning":
        "Create a practical plan for migrating a small internal service next week. "
        "Break the work into milestones, dependencies, owners, risks, and completion checks.",

    "research":
        "I need to compare two technical approaches for a short report. "
        "Explain how to gather reliable evidence, compare the alternatives, "
        "and reach a conclusion while accounting for uncertainty.",

    "data":
        "I received a table with duplicate rows, missing values, inconsistent category names, "
        "and a numeric column with suspicious outliers. Explain how you would clean and analyze it.",

    "design":
        "Design a simple onboarding flow for a new web application. "
        "Describe the screen structure, information hierarchy, interactions, "
        "and the main usability tradeoffs.",

    "operations":
        "A production service starts timing out after a deployment. "
        "Give an operational runbook for triage, mitigation, rollback, monitoring, "
        "handoff, and confirming recovery.",
}


def read_jsonl(path):
    out = []

    with Path(path).open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                out.append(
                    json.loads(line)
                )

    return out


def atomic_json(path, obj):
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        dir=str(path.parent),
    )

    os.close(fd)

    try:
        Path(tmp).write_text(
            json.dumps(
                obj,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            tmp,
            path,
        )

    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def sigmoid(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    return 1.0 / (
        1.0 + np.exp(
            -np.clip(x, -40, 40)
        )
    )


def logsumexp(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    m = float(
        np.max(x)
    )

    return (
        m
        + math.log(
            float(
                np.exp(
                    x - m
                ).sum()
            )
        )
    )


# ============================================================
# QUERY-CONDITIONED ACTIVATION MODEL
#
# a_r(q) = P(E_r=1 | Z_r=1, q)
#
# Fit ONLY on TRAIN.
# ============================================================

class ActivationModel:
    def __init__(
        self,
        embedder,
        roles,
        coef,
        intercept,
        base_rate,
    ):
        self.embedder = embedder
        self.roles = list(roles)

        self.coef = np.asarray(
            coef,
            dtype=np.float32,
        )

        self.intercept = np.asarray(
            intercept,
            dtype=np.float32,
        )

        self.base_rate = np.asarray(
            base_rate,
            dtype=np.float32,
        )

    def predict_many(
        self,
        queries,
    ):
        X = self.embedder.encode(
            list(queries),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(
            np.float32
        )

        logits = (
            X @ self.coef.T
            + self.intercept.reshape(
                1,
                -1,
            )
        )

        p = sigmoid(
            logits
        )

        return np.clip(
            p,
            0.03,
            0.97,
        )

    def predict(
        self,
        query,
    ):
        return self.predict_many(
            [query]
        )[0]


def fit_or_load_activation_model(
    train_rows,
    out,
):
    path = (
        out
        / "activation_model.pkl"
    )

    print(
        "[activation] loading MiniLM",
        flush=True,
    )

    embedder = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    if path.exists():
        print(
            "[activation] loading cached model:",
            path,
            flush=True,
        )

        obj = pickle.loads(
            path.read_bytes()
        )

        return ActivationModel(
            embedder=embedder,
            roles=obj["roles"],
            coef=obj["coef"],
            intercept=obj["intercept"],
            base_rate=obj["base_rate"],
        )

    print(
        "[activation] embedding TRAIN queries:",
        len(train_rows),
        flush=True,
    )

    queries = [
        str(
            row.get(
                "query",
                "",
            )
        )
        for row in train_rows
    ]

    X = embedder.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=True,
    ).astype(
        np.float32
    )

    coef = []
    intercept = []
    base_rate = []

    for role in HYP_ROLES:
        installed_idx = [
            i
            for i, row in enumerate(
                train_rows
            )
            if role in set(
                row.get(
                    "truth_roles",
                    [],
                )
            )
        ]

        if len(installed_idx) < 20:
            raise RuntimeError(
                f"too few installed examples for {role}: "
                f"{len(installed_idx)}"
            )

        y = np.asarray(
            [
                int(
                    role in set(
                        train_rows[i].get(
                            "selected_roles",
                            [],
                        )
                    )
                )
                for i in installed_idx
            ],
            dtype=np.int64,
        )

        rate = float(
            y.mean()
        )

        base_rate.append(
            rate
        )

        print(
            f"[activation] {role:<12} "
            f"installed={len(installed_idx)} "
            f"active_rate={rate:.3f}",
            flush=True,
        )

        if len(
            np.unique(y)
        ) < 2:
            # Fallback constant probability.
            coef.append(
                np.zeros(
                    X.shape[1],
                    dtype=np.float32,
                )
            )

            p = float(
                np.clip(
                    rate,
                    1e-4,
                    1 - 1e-4,
                )
            )

            intercept.append(
                math.log(
                    p / (1 - p)
                )
            )

            continue

        clf = LogisticRegression(
            C=1.0,
            max_iter=1500,
            solver="liblinear",
            class_weight="balanced",
            random_state=20260927,
        )

        clf.fit(
            X[installed_idx],
            y,
        )

        coef.append(
            clf.coef_[0].astype(
                np.float32
            )
        )

        intercept.append(
            float(
                clf.intercept_[0]
            )
        )

    obj = {
        "roles":
            HYP_ROLES,
        "coef":
            np.asarray(
                coef,
                dtype=np.float32,
            ),
        "intercept":
            np.asarray(
                intercept,
                dtype=np.float32,
            ),
        "base_rate":
            np.asarray(
                base_rate,
                dtype=np.float32,
            ),
    }

    path.write_bytes(
        pickle.dumps(
            obj,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )

    print(
        "[activation] saved:",
        path,
        flush=True,
    )

    return ActivationModel(
        embedder=embedder,
        roles=HYP_ROLES,
        coef=obj["coef"],
        intercept=obj["intercept"],
        base_rate=obj["base_rate"],
    )


# ============================================================
# THREE-STATE SENSOR DENSITIES
#
# For each role:
#
#   Z=0                 -> ABSENT
#   Z=1 and E=0         -> INSTALLED_INACTIVE
#   Z=1 and E=1         -> ACTIVE
#
# This is the key repair.
# ============================================================

def state_of_row(
    row,
    role,
):
    truth = set(
        row.get(
            "truth_roles",
            [],
        )
    )

    active = set(
        row.get(
            "selected_roles",
            [],
        )
    )

    if role not in truth:
        return "absent"

    if role in active:
        return "active"

    return "installed_inactive"


def stratified_dev_indices(
    rows,
    per_state=22,
    max_rows=560,
    seed=20260927,
):
    rng = random.Random(
        seed
    )

    chosen = set()

    for role in HYP_ROLES:
        for state in (
            "absent",
            "installed_inactive",
            "active",
        ):
            hits = [
                i
                for i, row in enumerate(
                    rows
                )
                if state_of_row(
                    row,
                    role,
                ) == state
            ]

            rng.shuffle(
                hits
            )

            chosen.update(
                hits[
                    :per_state
                ]
            )

    ids = list(
        chosen
    )

    rng.shuffle(
        ids
    )

    if len(ids) < max_rows:
        rest = [
            i
            for i in range(
                len(rows)
            )
            if i not in chosen
        ]

        rng.shuffle(
            rest
        )

        ids.extend(
            rest[
                :max_rows
                - len(ids)
            ]
        )

    if len(ids) > max_rows:
        ids = ids[
            :max_rows
        ]

    return ids


class KDE1D:
    def __init__(
        self,
        samples,
    ):
        samples = np.asarray(
            samples,
            dtype=np.float64,
        )

        if len(samples) == 0:
            raise RuntimeError(
                "empty KDE samples"
            )

        self.samples = samples

        if len(samples) >= 2:
            sd = float(
                np.std(
                    samples,
                    ddof=1,
                )
            )
        else:
            sd = 0.5

        if not math.isfinite(sd):
            sd = 0.5

        bw = (
            1.06
            * max(
                sd,
                0.15,
            )
            * (
                max(
                    len(samples),
                    2,
                )
                ** (-0.2)
            )
        )

        self.bw = float(
            np.clip(
                bw,
                0.12,
                1.25,
            )
        )

    def logpdf(
        self,
        x,
    ):
        z = (
            float(x)
            - self.samples
        ) / self.bw

        vals = (
            -0.5 * z * z
        )

        return (
            logsumexp(vals)
            - math.log(
                len(
                    self.samples
                )
            )
            - math.log(
                self.bw
            )
            - 0.5
            * math.log(
                2
                * math.pi
            )
        )


class SensorDensityModel:
    def __init__(
        self,
        samples,
    ):
        self.samples = samples

        self.models = {}

        for role in HYP_ROLES:
            self.models[
                role
            ] = {}

            absent = np.asarray(
                samples[
                    role
                ][
                    "absent"
                ],
                dtype=np.float64,
            )

            inactive = np.asarray(
                samples[
                    role
                ][
                    "installed_inactive"
                ],
                dtype=np.float64,
            )

            active = np.asarray(
                samples[
                    role
                ][
                    "active"
                ],
                dtype=np.float64,
            )

            # Installed-inactive can be rarer.
            # If too small, gently pool with absent rather than
            # pretending we know a separate distribution.
            if len(
                inactive
            ) < 8:
                inactive = np.concatenate(
                    [
                        inactive,
                        absent[
                            :min(
                                20,
                                len(
                                    absent
                                ),
                            )
                        ],
                    ]
                )

            self.models[
                role
            ][
                "absent"
            ] = KDE1D(
                absent
            )

            self.models[
                role
            ][
                "installed_inactive"
            ] = KDE1D(
                inactive
            )

            self.models[
                role
            ][
                "active"
            ] = KDE1D(
                active
            )

    def logbf(
        self,
        role,
        score,
        activation_p,
    ):
        a = float(
            np.clip(
                activation_p,
                1e-4,
                1 - 1e-4,
            )
        )

        m = self.models[
            role
        ]

        lp_absent = m[
            "absent"
        ].logpdf(
            score
        )

        lp_inactive = m[
            "installed_inactive"
        ].logpdf(
            score
        )

        lp_active = m[
            "active"
        ].logpdf(
            score
        )

        # p(s | Z=1,q)
        t1 = (
            math.log(a)
            + lp_active
        )

        t0 = (
            math.log(
                1 - a
            )
            + lp_inactive
        )

        lp_installed = np.logaddexp(
            t1,
            t0,
        )

        # p(s | Z=0)
        return float(
            lp_installed
            - lp_absent
        )


def fit_or_load_density_model(
    stack,
    dev_rows,
    out,
):
    path = (
        out
        / "sensor_density_v2.pkl"
    )

    if path.exists():
        print(
            "[density] loading cached:",
            path,
            flush=True,
        )

        obj = pickle.loads(
            path.read_bytes()
        )

        return SensorDensityModel(
            obj["samples"]
        )

    ids = stratified_dev_indices(
        dev_rows,
        per_state=22,
        max_rows=560,
    )

    rows = [
        dev_rows[i]
        for i in ids
    ]

    outputs = [
        str(
            row.get(
                "final_output",
                "",
            )
        )
        for row in rows
    ]

    print(
        "[density] scoring DEV rows:",
        len(rows),
        flush=True,
    )

    stages = stack.score_outputs(
        outputs
    )

    # IMPORTANT:
    # Structured + calibrated had the strongest clean
    # within-output ranking. We intentionally do NOT use
    # targeted_final here.
    score_matrix = stages[
        "calibrated"
    ]

    samples = {
        role: {
            "absent": [],
            "installed_inactive": [],
            "active": [],
        }
        for role in HYP_ROLES
    }

    for i, row in enumerate(
        rows
    ):
        for role in HYP_ROLES:
            gi = GLOBAL_INDEX[
                role
            ]

            state = state_of_row(
                row,
                role,
            )

            samples[
                role
            ][
                state
            ].append(
                float(
                    score_matrix[
                        i,
                        gi,
                    ]
                )
            )

    report = {}

    for role in HYP_ROLES:
        report[
            role
        ] = {}

        for state in (
            "absent",
            "installed_inactive",
            "active",
        ):
            arr = np.asarray(
                samples[
                    role
                ][
                    state
                ],
                dtype=np.float64,
            )

            report[
                role
            ][
                state
            ] = {
                "n":
                    int(
                        len(arr)
                    ),
                "mean":
                    float(
                        arr.mean()
                    )
                    if len(arr)
                    else None,
                "std":
                    float(
                        arr.std()
                    )
                    if len(arr)
                    else None,
            }

    atomic_json(
        out
        / "density_report.json",
        report,
    )

    path.write_bytes(
        pickle.dumps(
            {
                "samples":
                    samples,
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    )

    print(
        "[density] saved:",
        path,
        flush=True,
    )

    return SensorDensityModel(
        samples
    )


# ============================================================
# PROBES
# ============================================================

def build_probe_bank():
    probes = []

    for role in HYP_ROLES:
        q1 = str(
            anchor_probe(
                role
            )
        ).strip()

        q2 = SECOND_PROBE[
            role
        ].strip()

        probes.append({
            "id":
                f"{role}_anchor",
            "focus_role":
                role,
            "query":
                q1,
            "pass":
                1,
        })

        probes.append({
            "id":
                f"{role}_confirm",
            "focus_role":
                role,
            "query":
                q2,
            "pass":
                2,
        })

    # Pass 1 across all roles first, then pass 2.
    probes.sort(
        key=lambda x: (
            x["pass"],
            HYP_ROLES.index(
                x[
                    "focus_role"
                ]
            ),
        )
    )

    return probes


# ============================================================
# FRESH SYSTEM GENERATION
# ============================================================

def historical_compositions():
    old = set()

    for split in (
        "train",
        "dev",
        "test",
    ):
        p = (
            DATA_ROOT
            / f"{split}.jsonl"
        )

        if not p.exists():
            continue

        for row in read_jsonl(
            p
        ):
            truth = tuple(
                sorted(
                    map(
                        str,
                        row.get(
                            "truth_roles",
                            [],
                        ),
                    )
                )
            )

            if (
                len(truth) == 3
                and set(truth)
                <= set(
                    HYP_ROLES
                )
            ):
                old.add(
                    truth
                )

    # Also exclude previous night benchmark.
    manifest = (
        ROOT
        / "artifacts/night_20260926/"
        "test10_current_stack/"
        "manifest_eval_only.json"
    )

    if manifest.exists():
        try:
            xs = json.loads(
                manifest.read_text()
            )

            for x in xs:
                truth = tuple(
                    sorted(
                        x.get(
                            "true_roles_eval_only",
                            [],
                        )
                    )
                )

                if truth:
                    old.add(
                        truth
                    )

        except Exception:
            pass

    return old


def build_fresh_systems(
    n,
    seed,
):
    old = historical_compositions()

    all_combos = list(
        itertools.combinations(
            HYP_ROLES,
            3,
        )
    )

    rng = random.Random(
        seed
    )

    rng.shuffle(
        all_combos
    )

    fresh = []

    for combo in all_combos:
        key = tuple(
            sorted(
                combo
            )
        )

        if key in old:
            continue

        fresh.append(
            tuple(
                combo
            )
        )

        if len(
            fresh
        ) == n:
            break

    if len(
        fresh
    ) != n:
        raise RuntimeError(
            f"only found {len(fresh)} "
            f"fresh systems"
        )

    return fresh


# ============================================================
# RANK-FIRST INFERENCE
# ============================================================

def role_dict(
    values,
):
    return {
        role:
            float(
                values[i]
            )
        for i, role in enumerate(
            HYP_ROLES
        )
    }


def rank_roles(
    cumulative_logbf,
):
    order = np.argsort(
        -np.asarray(
            cumulative_logbf
        )
    )

    return [
        HYP_ROLES[
            i
        ]
        for i in order
    ]


def infer_one_step(
    *,
    query,
    focus_role,
    output,
    stack,
    activation_model,
    density_model,
    cumulative_logbf,
    focus_floor,
    min_activation,
    bf_clip,
):
    stages = stack.score_output(
        output
    )

    # Best ranking stage from prior experiments.
    sensor_all = stages[
        "calibrated"
    ]

    sensor_scores = np.asarray(
        [
            sensor_all[
                GLOBAL_INDEX[
                    r
                ]
            ]
            for r in HYP_ROLES
        ],
        dtype=np.float64,
    )

    activation = (
        activation_model
        .predict(
            query
        )
        .astype(
            np.float64
        )
    )

    # We know the intent of a probe we ourselves designed.
    # This is NOT hidden-system truth.
    fi = HYP_ROLES.index(
        focus_role
    )

    activation[
        fi
    ] = max(
        float(
            activation[
                fi
            ]
        ),
        focus_floor,
    )

    step_bf = np.zeros(
        len(
            HYP_ROLES
        ),
        dtype=np.float64,
    )

    for i, role in enumerate(
        HYP_ROLES
    ):
        a = float(
            activation[
                i
            ]
        )

        # Low-a queries should not slowly accumulate noise
        # against unrelated roles.
        if a < min_activation:
            step_bf[
                i
            ] = 0.0
            continue

        bf = density_model.logbf(
            role=role,
            score=float(
                sensor_scores[
                    i
                ]
            ),
            activation_p=a,
        )

        step_bf[
            i
        ] = float(
            np.clip(
                bf,
                -bf_clip,
                bf_clip,
            )
        )

    cumulative_logbf = (
        np.asarray(
            cumulative_logbf,
            dtype=np.float64,
        )
        + step_bf
    )

    # Equal prior across roles.
    # These are rank-beliefs, not final calibrated install
    # probabilities.
    belief = sigmoid(
        cumulative_logbf
    )

    ranking = rank_roles(
        cumulative_logbf
    )

    return {
        "sensor_scores":
            sensor_scores,
        "activation":
            activation,
        "step_logbf":
            step_bf,
        "cumulative_logbf":
            cumulative_logbf,
        "belief":
            belief,
        "ranking":
            ranking,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--systems",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=20260927,
    )

    ap.add_argument(
        "--model",
        default=None,
    )

    ap.add_argument(
        "--focus-floor",
        type=float,
        default=0.72,
    )

    ap.add_argument(
        "--min-activation",
        type=float,
        default=0.12,
    )

    ap.add_argument(
        "--bf-clip",
        type=float,
        default=1.75,
    )

    ap.add_argument(
        "--out",
        default=(
            "artifacts/"
            "system_v2_rankfirst/"
            "dev3"
        ),
    )

    args = ap.parse_args()

    if not os.getenv(
        "OPENAI_API_KEY"
    ):
        raise RuntimeError(
            "OPENAI_API_KEY missing"
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; "
            "run on GPU compute node"
        )

    out = (
        ROOT
        / args.out
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "==================================================",
        flush=True,
    )
    print(
        "SYSTEM V2 — RANK FIRST",
        flush=True,
    )
    print(
        "==================================================",
        flush=True,
    )
    print(
        "hypothesis roles:",
        HYP_ROLES,
        flush=True,
    )
    print(
        "K IS NOT USED IN PRIMARY BELIEF UPDATE",
        flush=True,
    )

    train_rows = read_jsonl(
        DATA_ROOT
        / "train.jsonl"
    )

    dev_rows = read_jsonl(
        DATA_ROOT
        / "dev.jsonl"
    )

    stack = V7NightStack(
        sensor_batch_size=8
    )

    activation_model = (
        fit_or_load_activation_model(
            train_rows,
            out.parent,
        )
    )

    density_model = (
        fit_or_load_density_model(
            stack,
            dev_rows,
            out.parent,
        )
    )

    systems = build_fresh_systems(
        args.systems,
        args.seed,
    )

    manifest = [
        {
            "system_id":
                f"rankv2_{i:02d}",
            "true_roles_eval_only":
                list(
                    truth
                ),
        }
        for i, truth in enumerate(
            systems,
            1,
        )
    ]

    atomic_json(
        out
        / "manifest_eval_only.json",
        manifest,
    )

    probe_bank = (
        build_probe_bank()
    )

    results = []

    for si, true_roles in enumerate(
        systems,
        1,
    ):
        sid = (
            f"rankv2_{si:02d}"
        )

        print(
            "\n"
            + "=" * 100,
            flush=True,
        )
        print(
            f"SYSTEM {si}/{len(systems)} "
            f"{sid}",
            flush=True,
        )
        print(
            "truth remains evaluation-only",
            flush=True,
        )

        spec = (
            SingleRouterSystemSpec(
                system_id=sid,
                split="rank_v2",
                execution_mode="fanout2",
                style="s1",
                worker_roles=true_roles,
            )
        )

        mas = (
            SingleRouterFanoutMAS(
                spec,
                model=args.model,
            )
        )

        cumulative_logbf = (
            np.zeros(
                len(
                    HYP_ROLES
                ),
                dtype=np.float64,
            )
        )

        trajectory = []

        for step, probe in enumerate(
            probe_bank,
            1,
        ):
            query = probe[
                "query"
            ]

            focus_role = probe[
                "focus_role"
            ]

            print(
                f"\n[{sid}] "
                f"step={step:02d}/"
                f"{len(probe_bank)} "
                f"focus={focus_role}",
                flush=True,
            )

            print(
                "Q:",
                query,
                flush=True,
            )

            output = str(
                mas.query(
                    query
                )
            )

            update = infer_one_step(
                query=query,
                focus_role=focus_role,
                output=output,
                stack=stack,
                activation_model=activation_model,
                density_model=density_model,
                cumulative_logbf=cumulative_logbf,
                focus_floor=args.focus_floor,
                min_activation=args.min_activation,
                bf_clip=args.bf_clip,
            )

            cumulative_logbf = (
                update[
                    "cumulative_logbf"
                ]
            )

            top = update[
                "ranking"
            ]

            print(
                "TOP BELIEF:",
                [
                    (
                        r,
                        round(
                            float(
                                update[
                                    "belief"
                                ][
                                    HYP_ROLES.index(
                                        r
                                    )
                                ]
                            ),
                            4,
                        ),
                    )
                    for r in top[
                        :6
                    ]
                ],
                flush=True,
            )

            trajectory.append({
                "step":
                    step,
                "probe_id":
                    probe[
                        "id"
                    ],
                "focus_role":
                    focus_role,
                "query":
                    query,
                "final_output":
                    output,

                "sensor_score":
                    role_dict(
                        update[
                            "sensor_scores"
                        ]
                    ),

                "activation_probability":
                    role_dict(
                        update[
                            "activation"
                        ]
                    ),

                "step_log_bayes_factor":
                    role_dict(
                        update[
                            "step_logbf"
                        ]
                    ),

                "cumulative_log_bayes_factor":
                    role_dict(
                        update[
                            "cumulative_logbf"
                        ]
                    ),

                "belief":
                    role_dict(
                        update[
                            "belief"
                        ]
                    ),

                "ranking":
                    update[
                        "ranking"
                    ],
            })

            atomic_json(
                out
                / f"{sid}_trajectory.json",
                trajectory,
            )

        # ----------------------------------------------------
        # Evaluation only AFTER inference complete.
        # ----------------------------------------------------

        final_ranking = (
            rank_roles(
                cumulative_logbf
            )
        )

        true = set(
            true_roles
        )

        top3 = set(
            final_ranking[
                :3
            ]
        )

        ranks = {
            role:
                final_ranking.index(
                    role
                )
                + 1
            for role in true
        }

        exact_top3 = (
            top3
            == true
        )

        result = {
            "system_id":
                sid,

            "true_roles_eval_only":
                sorted(
                    true
                ),

            "final_ranking":
                final_ranking,

            "top3":
                final_ranking[
                    :3
                ],

            "top3_exact_eval_only":
                exact_top3,

            "true_role_ranks_eval_only":
                ranks,

            "worst_true_rank_eval_only":
                max(
                    ranks.values()
                ),

            "mean_true_rank_eval_only":
                float(
                    np.mean(
                        list(
                            ranks.values()
                        )
                    )
                ),

            "final_belief":
                role_dict(
                    sigmoid(
                        cumulative_logbf
                    )
                ),

            "final_logbf":
                role_dict(
                    cumulative_logbf
                ),
        }

        results.append(
            result
        )

        atomic_json(
            out
            / "summary_running.json",
            {
                "completed":
                    len(
                        results
                    ),
                "top3_exact":
                    sum(
                        int(
                            x[
                                "top3_exact_eval_only"
                            ]
                        )
                        for x in results
                    ),
                "systems":
                    results,
            },
        )

        print(
            "\nEVAL:",
            json.dumps(
                result,
                indent=2,
            ),
            flush=True,
        )

    exact = sum(
        int(
            x[
                "top3_exact_eval_only"
            ]
        )
        for x in results
    )

    summary = {
        "experiment":
            "system_v2_rankfirst",

        "primary_goal":
            "installed-role ranking",

        "K_used_in_belief_update":
            False,

        "threshold_used":
            False,

        "hypothesis_roles":
            HYP_ROLES,

        "queries_per_system":
            len(
                probe_bank
            ),

        "systems":
            len(
                results
            ),

        "top3_exact":
            exact,

        "top3_exact_rate":
            exact
            / max(
                len(
                    results
                ),
                1,
            ),

        "mean_worst_true_rank":
            float(
                np.mean([
                    x[
                        "worst_true_rank_eval_only"
                    ]
                    for x in results
                ])
            ),

        "mean_true_rank":
            float(
                np.mean([
                    x[
                        "mean_true_rank_eval_only"
                    ]
                    for x in results
                ])
            ),

        "results":
            results,
    }

    atomic_json(
        out
        / "summary.json",
        summary,
    )

    print(
        "\n"
        + "#" * 100,
        flush=True,
    )
    print(
        "FINAL RANK-FIRST SUMMARY",
        flush=True,
    )
    print(
        "#" * 100,
        flush=True,
    )

    print(
        json.dumps(
            summary,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
