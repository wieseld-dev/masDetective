#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from night_v7_stack import V7NightStack, ROLES
from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
    ROLE_PROMPTS,
)

# The V7 sensor still evaluates all 20 TRAIN_ROLES.
# Fresh synthetic systems, however, may only contain roles that
# the target implementation can actually instantiate.
TARGET_SUPPORTED_ROLES = [
    r for r in ROLES
    if r in ROLE_PROMPTS
]

TARGET_UNSUPPORTED_ROLES = [
    r for r in ROLES
    if r not in ROLE_PROMPTS
]

if len(TARGET_SUPPORTED_ROLES) < 3:
    raise RuntimeError(
        f"target supports too few roles: {TARGET_SUPPORTED_ROLES}"
    )


DATA_ROOT = ROOT / "artifacts/v7_benign_final_pool_v2"


PROBES = [
    {
        "id": "code_qa_docs",
        "tags": ["code", "qa_testing", "documentation"],
        "query":
            "A Python service intermittently returns duplicate records after retries. "
            "Give me a debugging approach, concrete tests to reproduce the issue, "
            "and what should be documented once the fix is verified.",
    },
    {
        "id": "research_fact_writing",
        "tags": ["research", "fact_checking", "writing"],
        "query":
            "I am preparing a short technical brief comparing two approaches. "
            "Explain how to gather reliable evidence, verify conflicting claims, "
            "and present a concise conclusion with appropriate uncertainty.",
    },
    {
        "id": "planning_risk_requirements",
        "tags": ["planning", "risk_assessment", "requirements_analysis"],
        "query":
            "Help me turn a vague request for an internal dashboard into an implementation plan. "
            "Clarify requirements, dependencies, risks, milestones, and acceptance criteria.",
    },
    {
        "id": "data_math_experiment",
        "tags": ["data", "math", "experiment_design"],
        "query":
            "I changed a ranking algorithm and want to determine whether it improved results. "
            "Propose an experiment, useful metrics, sampling considerations, "
            "and how the resulting data should be analyzed.",
    },
    {
        "id": "design_accessibility_writing",
        "tags": ["design", "accessibility_review", "writing"],
        "query":
            "Review the concept of an online checkout form for usability and accessibility. "
            "Suggest field organization, labels, keyboard behavior, error messages, "
            "and clearer microcopy.",
    },
    {
        "id": "operations_incident_support",
        "tags": ["operations", "incident_management", "customer_support_triage"],
        "query":
            "A production service became slow immediately after a deployment and users report timeouts. "
            "Give me a triage and recovery procedure, including what diagnostic information "
            "support should collect and how the incident should be handed off.",
    },
    {
        "id": "finance_data_planning",
        "tags": ["financial_analysis", "data", "planning"],
        "query":
            "I have three vendor options with different fixed costs, per-user prices, "
            "and expected growth. Explain how to compare total cost over the next year "
            "and structure a simple decision scenario.",
    },
    {
        "id": "database_code_requirements",
        "tags": ["database_design", "code", "requirements_analysis"],
        "query":
            "Design a data model for projects, users, tasks, comments, and audit history. "
            "Include keys, useful indexes, constraints, and the main access patterns "
            "the implementation should support.",
    },
    {
        "id": "translation_writing_docs",
        "tags": ["translation", "writing", "documentation"],
        "query":
            "Translate this short onboarding text into natural Hebrew while preserving technical terms, "
            "then suggest one clearer version for new users: "
            "'Create an API token, store it securely, and restart the service after updating the configuration.'",
    },
    {
        "id": "risk_fact_research",
        "tags": ["risk_assessment", "fact_checking", "research"],
        "query":
            "I need to decide whether to adopt a new dependency for a security-sensitive service. "
            "Give me a practical method to verify its maintenance and security claims "
            "and assess the risks of adopting it.",
    },
    {
        "id": "math_finance_data",
        "tags": ["math", "financial_analysis", "data"],
        "query":
            "Monthly revenue is 120,000, variable costs are 35%, and fixed costs are 52,000. "
            "A proposed 15,000 campaign is expected to increase revenue by 12%. "
            "Compare baseline and campaign profit and state the assumptions clearly.",
    },
    {
        "id": "experiment_qa_code",
        "tags": ["experiment_design", "qa_testing", "code"],
        "query":
            "A new cache is intended to reduce API latency without changing returned results. "
            "Design a test plan that checks correctness, performance, edge cases, and regressions.",
    },
    {
        "id": "support_requirements_docs",
        "tags": ["customer_support_triage", "requirements_analysis", "documentation"],
        "query":
            "Users say account setup is confusing. Turn that feedback into diagnostic questions, "
            "likely issue categories, acceptance criteria for an improved flow, "
            "and a short help-center outline.",
    },
    {
        "id": "planning_operations_risk",
        "tags": ["planning", "operations", "risk_assessment"],
        "query":
            "Plan a migration of a small internal service to a new host with minimal downtime. "
            "Include prerequisites, rollback, monitoring, handoffs, and post-migration checks.",
    },
]


def read_jsonl(path):
    out = []
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp = tempfile.mkstemp(
        prefix=path.name + ".",
        dir=str(path.parent),
    )
    os.close(fd)

    try:
        Path(tmp).write_text(
            json.dumps(obj, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(obj, ensure_ascii=False)
            + "\n"
        )
        f.flush()
        os.fsync(f.fileno())


def logsumexp(xs):
    xs = np.asarray(xs, dtype=np.float64)
    m = float(np.max(xs))
    return m + math.log(
        float(np.exp(xs - m).sum())
    )


def entropy01(p):
    p = float(np.clip(p, 1e-8, 1 - 1e-8))
    return -p * math.log(p) - (1-p) * math.log(1-p)


class CardinalityPosterior:
    """
    Exact posterior over all subsets with K in {3,4,5,6}.

    Prior:
        P(K)=1/4
        P(Z | K)=uniform over all size-K subsets

    Evidence:
        cumulative log Bayes factor independently per role.
    """
    def __init__(self):
        self.entries = []

        n = len(ROLES)

        for k in (3, 4, 5, 6):
            base = (
                math.log(0.25)
                - math.log(math.comb(n, k))
            )

            for combo in itertools.combinations(
                range(n),
                k,
            ):
                self.entries.append(
                    (k, combo, base)
                )

    def infer(self, cumulative_bf):
        cumulative_bf = np.asarray(
            cumulative_bf,
            dtype=np.float64,
        )

        logw = np.empty(
            len(self.entries),
            dtype=np.float64,
        )

        for i, (k, combo, base) in enumerate(
            self.entries
        ):
            logw[i] = (
                base
                + float(cumulative_bf[list(combo)].sum())
            )

        z = logsumexp(logw)
        probs = np.exp(logw - z)

        role_p = np.zeros(
            len(ROLES),
            dtype=np.float64,
        )

        k_p = {
            3: 0.0,
            4: 0.0,
            5: 0.0,
            6: 0.0,
        }

        for p, (k, combo, _) in zip(
            probs,
            self.entries,
        ):
            k_p[k] += float(p)
            role_p[list(combo)] += float(p)

        mi = int(np.argmax(logw))
        map_k, map_combo, _ = self.entries[mi]

        return {
            "role_p": role_p,
            "k_p": k_p,
            "map_k": int(map_k),
            "map_roles": [
                ROLES[i]
                for i in map_combo
            ],
            "map_posterior":
                float(probs[mi]),
        }


def build_fresh_compositions(n_systems, seed):
    historical = set()

    for split in ("train", "dev", "test"):
        p = DATA_ROOT / f"{split}.jsonl"
        if not p.exists():
            continue

        for row in read_jsonl(p):
            truth = row.get("truth_roles") or []
            if truth:
                historical.add(
                    tuple(sorted(map(str, truth)))
                )

    rng = random.Random(seed)
    candidates = list(
        itertools.combinations(
            TARGET_SUPPORTED_ROLES,
            3,
        )
    )
    rng.shuffle(candidates)

    fresh = []

    for combo in candidates:
        key = tuple(sorted(combo))
        if key in historical:
            continue

        fresh.append(tuple(combo))

        if len(fresh) >= n_systems:
            break

    if len(fresh) != n_systems:
        raise RuntimeError(
            f"could only construct {len(fresh)} fresh systems"
        )

    return fresh


def choose_calibration_rows(rows, n_rows, seed):
    rng = random.Random(seed)
    chosen = set()

    # Ensure positive examples for every role where possible.
    for role in ROLES:
        hits = [
            i for i, row in enumerate(rows)
            if role in set(row.get("selected_roles") or [])
        ]

        rng.shuffle(hits)

        for i in hits[:12]:
            chosen.add(i)

    rest = [
        i for i in range(len(rows))
        if i not in chosen
    ]
    rng.shuffle(rest)

    for i in rest:
        if len(chosen) >= n_rows:
            break
        chosen.add(i)

    ids = sorted(chosen)

    if len(ids) > n_rows:
        rng.shuffle(ids)
        ids = ids[:n_rows]

    return [rows[i] for i in ids]


def estimate_activation_prior(train_rows):
    installed = {
        r: 0
        for r in ROLES
    }
    activated = {
        r: 0
        for r in ROLES
    }

    for row in train_rows:
        truth = set(
            map(str, row.get("truth_roles") or [])
        )
        selected = set(
            map(str, row.get("selected_roles") or [])
        )

        for role in ROLES:
            if role in truth:
                installed[role] += 1

                if role in selected:
                    activated[role] += 1

    pi = []

    for role in ROLES:
        den = installed[role]

        p = (
            activated[role] / den
            if den
            else 0.30
        )

        pi.append(
            float(np.clip(p, 0.08, 0.85))
        )

    return np.asarray(
        pi,
        dtype=np.float64,
    )


def fit_or_load_score_model(
    stack,
    out,
    calibration_rows,
    seed,
):
    npz_path = out / "activation_score_model.npz"
    json_path = out / "activation_score_model.json"

    if npz_path.exists() and json_path.exists():
        z = np.load(npz_path)

        if list(z["roles"]) == ROLES:
            print(
                "[cal] using cached score model",
                npz_path,
                flush=True,
            )

            return {
                "mu1": z["mu1"].astype(np.float64),
                "sd1": z["sd1"].astype(np.float64),
                "mu0": z["mu0"].astype(np.float64),
                "sd0": z["sd0"].astype(np.float64),
            }

    dev_rows = read_jsonl(
        DATA_ROOT / "dev.jsonl"
    )

    selected_rows = choose_calibration_rows(
        dev_rows,
        calibration_rows,
        seed,
    )

    print(
        f"[cal] scoring {len(selected_rows)} DEV rows "
        f"through the full current V7 stack",
        flush=True,
    )

    outputs = [
        str(r["final_output"])
        for r in selected_rows
    ]

    matrices = stack.score_outputs(
        outputs
    )

    scores = matrices["final"].astype(
        np.float64
    )

    y = np.zeros_like(
        scores,
        dtype=np.int64,
    )

    for i, row in enumerate(selected_rows):
        active = set(
            map(str, row.get("selected_roles") or [])
        )

        for j, role in enumerate(ROLES):
            y[i, j] = int(
                role in active
            )

    global_sd = max(
        float(scores.std()),
        0.35,
    )

    mu1 = []
    sd1 = []
    mu0 = []
    sd0 = []

    report = {}

    for j, role in enumerate(ROLES):
        pos = scores[
            y[:, j] == 1,
            j,
        ]

        neg = scores[
            y[:, j] == 0,
            j,
        ]

        if len(pos) >= 4:
            m1 = float(pos.mean())
            s1 = max(
                float(pos.std(ddof=1))
                if len(pos) > 1
                else global_sd,
                0.35,
            )
        else:
            m1 = float(scores[:, j].mean())
            s1 = global_sd

        if len(neg) >= 4:
            m0 = float(neg.mean())
            s0 = max(
                float(neg.std(ddof=1))
                if len(neg) > 1
                else global_sd,
                0.35,
            )
        else:
            m0 = float(scores[:, j].mean())
            s0 = global_sd

        mu1.append(m1)
        sd1.append(s1)
        mu0.append(m0)
        sd0.append(s0)

        report[role] = {
            "n_pos": int(len(pos)),
            "n_neg": int(len(neg)),
            "active_mean": m1,
            "active_std": s1,
            "inactive_mean": m0,
            "inactive_std": s0,
        }

    np.savez(
        npz_path,
        roles=np.asarray(ROLES),
        mu1=np.asarray(mu1),
        sd1=np.asarray(sd1),
        mu0=np.asarray(mu0),
        sd0=np.asarray(sd0),
    )

    atomic_json(
        json_path,
        {
            "split": "dev_only",
            "rows": len(selected_rows),
            "report": report,
        },
    )

    print(
        "[cal] saved activation score model",
        flush=True,
    )

    return {
        "mu1": np.asarray(mu1, np.float64),
        "sd1": np.asarray(sd1, np.float64),
        "mu0": np.asarray(mu0, np.float64),
        "sd0": np.asarray(sd0, np.float64),
    }


def normal_logpdf(x, mu, sd):
    sd = np.maximum(
        np.asarray(sd, dtype=np.float64),
        1e-6,
    )

    z = (
        np.asarray(x, dtype=np.float64)
        - np.asarray(mu, dtype=np.float64)
    ) / sd

    return (
        -0.5 * z * z
        - np.log(sd)
        - 0.5 * math.log(2 * math.pi)
    )


def installation_evidence(
    score,
    model,
    activation_pi,
):
    lp1 = normal_logpdf(
        score,
        model["mu1"],
        model["sd1"],
    )

    lp0 = normal_logpdf(
        score,
        model["mu0"],
        model["sd0"],
    )

    pi = np.clip(
        activation_pi,
        1e-4,
        1 - 1e-4,
    )

    # H1 installed:
    # role is activated with probability pi;
    # otherwise it behaves like inactive.
    a = (
        np.log(pi)
        + lp1
    )

    b = (
        np.log(1.0 - pi)
        + lp0
    )

    m = np.maximum(a, b)

    mix = (
        m
        + np.log(
            np.exp(a - m)
            + np.exp(b - m)
        )
    )

    bf = mix - lp0

    # Non-activation is weak evidence against installation:
    # an installed specialist may simply not be needed by this query.
    bf = np.where(
        bf < 0,
        0.20 * bf,
        bf,
    )

    return np.clip(
        bf,
        -0.30,
        2.00,
    )


def select_probe(
    unused,
    role_p,
    focus_count,
):
    best = None

    for p in unused:
        idx = [
            ROLES.index(r)
            for r in p["tags"]
        ]

        h = np.mean([
            entropy01(role_p[i])
            for i in idx
        ])

        explore = np.mean([
            1.0 / (1.0 + focus_count[r])
            for r in p["tags"]
        ])

        score = (
            h
            + 0.12 * explore
        )

        candidate = (
            float(score),
            p["id"],
            p,
        )

        if best is None or candidate[:2] > best[:2]:
            best = candidate

    return best[2]


def role_dict(x):
    return {
        r: float(x[i])
        for i, r in enumerate(ROLES)
    }


def set_metrics(true, pred):
    true = set(true)
    pred = set(pred)

    tp = len(true & pred)
    fp = len(pred - true)
    fn = len(true - pred)

    precision = (
        tp / (tp + fp)
        if tp + fp
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn
        else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    union = true | pred

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard":
            len(true & pred) / len(union)
            if union
            else 1.0,
    }


def run_smoke(args):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY missing. "
            "SBATCH must source .env inside masdetective environment."
        )

    stack = V7NightStack(
        sensor_batch_size=args.sensor_batch_size
    )

    roles = (
        "code",
        "planning",
        "research",
    )

    spec = SingleRouterSystemSpec(
        system_id="night_smoke_k3",
        split="night_smoke",
        execution_mode="fanout2",
        style="s1",
        worker_roles=roles,
    )

    mas = SingleRouterFanoutMAS(
        spec,
        model=args.model,
    )

    q = PROBES[0]["query"]

    print("[smoke] query:", q, flush=True)

    output = str(
        mas.query(q)
    )

    # PUBLIC inference first.
    scores = stack.score_output(
        output
    )

    order = np.argsort(
        -scores["final"]
    )

    print(
        "[smoke] top sensor roles:",
        [
            (
                ROLES[i],
                float(scores["final"][i]),
            )
            for i in order[:6]
        ],
        flush=True,
    )

    # Truth only after public inference has completed.
    trace = mas.privileged_last_trace()

    print(
        "[smoke] EVAL-ONLY activated:",
        list(trace.selected_roles),
        flush=True,
    )

    print(
        "[smoke] PASS",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--systems",
        type=int,
        default=10,
    )
    ap.add_argument(
        "--max-probes",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--min-probes",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=20260926,
    )
    ap.add_argument(
        "--calibration-rows",
        type=int,
        default=320,
    )
    ap.add_argument(
        "--sensor-batch-size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--model",
        default=None,
    )
    ap.add_argument(
        "--out",
        default="artifacts/night_20260926/test10_current_stack",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
    )

    args = ap.parse_args()

    if args.smoke:
        run_smoke(args)
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY missing. "
            "source .env before running."
        )

    out = ROOT / args.out
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    stack = V7NightStack(
        sensor_batch_size=args.sensor_batch_size
    )

    train_rows = read_jsonl(
        DATA_ROOT / "train.jsonl"
    )

    activation_pi = estimate_activation_prior(
        train_rows
    )

    score_model = fit_or_load_score_model(
        stack,
        out,
        args.calibration_rows,
        args.seed + 7,
    )

    posterior = CardinalityPosterior()

    print(
        "[benchmark] sensor candidate universe:",
        ROLES,
        flush=True,
    )
    print(
        "[benchmark] target-supported generation pool:",
        TARGET_SUPPORTED_ROLES,
        flush=True,
    )
    print(
        "[benchmark] target-unsupported roles:",
        TARGET_UNSUPPORTED_ROLES,
        flush=True,
    )

    compositions = build_fresh_compositions(
        args.systems,
        args.seed,
    )

    manifest = []

    for i, roles in enumerate(
        compositions,
        1,
    ):
        manifest.append({
            "system_id":
                f"night_fresh_{i:02d}",
            "k_generation_eval_only":
                len(roles),
            "true_roles_eval_only":
                list(roles),
        })

    atomic_json(
        out / "manifest_eval_only.json",
        manifest,
    )

    results = []

    global_tp = 0
    global_fp = 0
    global_fn = 0

    result_jsonl = out / "system_results.jsonl"

    if result_jsonl.exists():
        result_jsonl.unlink()

    for sys_i, true_roles in enumerate(
        compositions,
        1,
    ):
        sid = f"night_fresh_{sys_i:02d}"

        print(
            "\n"
            + "=" * 100,
            flush=True,
        )
        print(
            f"SYSTEM {sys_i}/{args.systems}: {sid}",
            flush=True,
        )
        print(
            "Inference starts with NO access to role truth.",
            flush=True,
        )

        spec = SingleRouterSystemSpec(
            system_id=sid,
            split="night_eval",
            execution_mode="fanout2",
            style="s1",
            worker_roles=true_roles,
        )

        mas = SingleRouterFanoutMAS(
            spec,
            model=args.model,
        )

        cumulative_bf = np.zeros(
            len(ROLES),
            dtype=np.float64,
        )

        state = posterior.infer(
            cumulative_bf
        )

        unused = list(PROBES)

        focus_count = {
            r: 0
            for r in ROLES
        }

        trajectory = []
        last_map = None
        stable = 0

        for step in range(
            1,
            args.max_probes + 1,
        ):
            before = state

            probe = select_probe(
                unused,
                before["role_p"],
                focus_count,
            )

            unused = [
                x for x in unused
                if x["id"] != probe["id"]
            ]

            for role in probe["tags"]:
                focus_count[role] += 1

            q = probe["query"]

            print(
                f"\n[{sid}] STEP {step}",
                flush=True,
            )
            print(
                "probe:",
                probe["id"],
                "tags:",
                probe["tags"],
                flush=True,
            )
            print(
                "query:",
                q,
                flush=True,
            )

            output = str(
                mas.query(q)
            )

            # IMPORTANT:
            # public inference happens BEFORE any trace/truth access.
            stages = stack.score_output(
                output
            )

            step_bf = installation_evidence(
                stages["final"],
                score_model,
                activation_pi,
            )

            cumulative_bf += step_bf

            state = posterior.infer(
                cumulative_bf
            )

            public_record = {
                "step": step,
                "probe_id": probe["id"],
                "probe_tags_selector_only":
                    probe["tags"],
                "query": q,
                "final_output": output,
                "scores": {
                    name: role_dict(stages[name])
                    for name in (
                        "raw",
                        "structured",
                        "calibrated",
                        "targeted_raw",
                        "final",
                    )
                },
                "belief_before": {
                    "role_p":
                        role_dict(before["role_p"]),
                    "k_p":
                        before["k_p"],
                    "map_k":
                        before["map_k"],
                    "map_roles":
                        before["map_roles"],
                },
                "installation_log_bf_step":
                    role_dict(step_bf),
                "installation_log_bf_cumulative":
                    role_dict(cumulative_bf),
                "belief_after": {
                    "role_p":
                        role_dict(state["role_p"]),
                    "k_p":
                        state["k_p"],
                    "map_k":
                        state["map_k"],
                    "map_roles":
                        state["map_roles"],
                    "map_posterior":
                        state["map_posterior"],
                },
            }

            # Only now reveal privileged trace for evaluation.
            trace = mas.privileged_last_trace()

            public_record[
                "selected_roles_eval_only"
            ] = list(
                trace.selected_roles
            )

            trajectory.append(
                public_record
            )

            atomic_json(
                out / f"{sid}_trajectory.json",
                trajectory,
            )

            print(
                "pred:",
                state["map_roles"],
                "K=",
                state["map_k"],
                flush=True,
            )

            print(
                "P(K):",
                {
                    k: round(v, 4)
                    for k, v in state["k_p"].items()
                },
                flush=True,
            )

            print(
                "top role beliefs:",
                sorted(
                    role_dict(
                        state["role_p"]
                    ).items(),
                    key=lambda x: -x[1],
                )[:7],
                flush=True,
            )

            current_map = tuple(
                sorted(state["map_roles"])
            )

            if current_map == last_map:
                stable += 1
            else:
                stable = 0

            last_map = current_map

            pred_idx = [
                ROLES.index(r)
                for r in state["map_roles"]
            ]

            nonpred_idx = [
                i for i in range(len(ROLES))
                if i not in pred_idx
            ]

            pred_min = min(
                state["role_p"][pred_idx]
            )

            nonpred_max = max(
                state["role_p"][nonpred_idx]
            )

            k_conf = max(
                state["k_p"].values()
            )

            if (
                step >= args.min_probes
                and stable >= 2
                and k_conf >= 0.80
                and pred_min >= 0.70
                and nonpred_max <= 0.35
            ):
                print(
                    f"[{sid}] confident adaptive stop at step {step}",
                    flush=True,
                )
                break

        # Evaluation truth is read only after inference session is over.
        true = set(
            map(str, spec.worker_roles)
        )
        pred = set(
            map(str, state["map_roles"])
        )

        m = set_metrics(
            true,
            pred,
        )

        tp = len(true & pred)
        fp = len(pred - true)
        fn = len(true - pred)

        global_tp += tp
        global_fp += fp
        global_fn += fn

        result = {
            "system_id": sid,
            "true_roles_eval_only":
                sorted(true),
            "predicted_roles":
                sorted(pred),
            "true_k_eval_only":
                len(true),
            "predicted_k":
                len(pred),
            "missing":
                sorted(true - pred),
            "extra":
                sorted(pred - true),
            "exact":
                bool(true == pred),
            "queries":
                len(trajectory),
            **m,
            "final_k_p":
                state["k_p"],
            "final_role_p":
                role_dict(state["role_p"]),
        }

        results.append(
            result
        )

        append_jsonl(
            result_jsonl,
            result,
        )

        print(
            "\nEVAL RESULT:",
            json.dumps(
                result,
                indent=2,
            ),
            flush=True,
        )

        denom = (
            2 * global_tp
            + global_fp
            + global_fn
        )

        running_micro_f1 = (
            2 * global_tp / denom
            if denom
            else 0.0
        )

        running_summary = {
            "completed_systems":
                len(results),
            "exact":
                sum(int(x["exact"]) for x in results),
            "exact_rate":
                float(np.mean([x["exact"] for x in results])),
            "mean_jaccard":
                float(np.mean([x["jaccard"] for x in results])),
            "micro_f1":
                running_micro_f1,
            "mean_queries":
                float(np.mean([x["queries"] for x in results])),
            "systems":
                results,
        }

        atomic_json(
            out / "summary_running.json",
            running_summary,
        )

    denom = (
        2 * global_tp
        + global_fp
        + global_fn
    )

    micro_f1 = (
        2 * global_tp / denom
        if denom
        else 0.0
    )

    summary = {
        "experiment":
            "night_10mas_current_v7_stack",
        "systems":
            args.systems,
        "generation_k_eval_only":
            3,
        "inference_k_prior":
            [3, 4, 5, 6],
        "oracle_k_used":
            False,
        "truth_used_during_inference":
            False,
        "sensor_candidate_universe":
            ROLES,
        "target_supported_generation_pool":
            TARGET_SUPPORTED_ROLES,
        "target_unsupported_roles":
            TARGET_UNSUPPORTED_ROLES,
        "execution_mode":
            "fanout2",
        "exact_system_reconstruction":
            sum(int(x["exact"]) for x in results),
        "exact_over":
            len(results),
        "exact_rate":
            float(np.mean([x["exact"] for x in results])),
        "mean_jaccard":
            float(np.mean([x["jaccard"] for x in results])),
        "micro_f1":
            micro_f1,
        "mean_queries":
            float(np.mean([x["queries"] for x in results])),
        "results":
            results,
    }

    atomic_json(
        out / "summary.json",
        summary,
    )

    print(
        "\n"
        + "#" * 100,
        flush=True,
    )
    print(
        "FINAL 10-MAS SUMMARY",
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
