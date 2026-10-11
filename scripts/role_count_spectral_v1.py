from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from openai import AsyncOpenAI
from scipy.stats import spearmanr
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import dag_operator_active_poc as base

# Side effect: registers all V11 role prompts into ROLE_PROMPTS.
import targets.single_router_zoo_v11  # noqa: F401
from targets.hierarchical_mas import ROLE_PROMPTS
from v11.role_universe import TRAIN_ROLES


ROOT = Path("artifacts/role_count_spectral_v1")
CACHE = ROOT / "api_runs.jsonl"
MANIFEST = ROOT / "manifest.json"
OUT = ROOT / "analysis"

SEED = 73
DIM = 16
TRAIN_FRAC = 0.70

FIXED_R = "requirements_analysis"
FIXED_M = "documentation"

# Semantically/confusion-related pairs.
PAIRS = [
    ("p01", "research", "fact_checking"),
    ("p02", "code", "qa_testing"),
    ("p03", "data", "database_design"),
    ("p04", "operations", "incident_management"),
]

HORDER = ["00", "10", "01", "11"]

K_OF = {
    "00": 2,
    "10": 3,
    "01": 3,
    "11": 4,
}

A_PRESENT = {
    "00": 0,
    "10": 1,
    "01": 0,
    "11": 1,
}

B_PRESENT = {
    "00": 0,
    "10": 0,
    "01": 1,
    "11": 1,
}

TYPE_CONTRASTS = [
    ("10", "01"),
]

COUNT_CONTRASTS = [
    ("00", "10"),
    ("00", "01"),
    ("10", "11"),
    ("01", "11"),
]

N_SHUFFLE = 5000


def seq_for(A: str, B: str, h: str) -> list[str]:
    if h == "00":
        return [FIXED_R, FIXED_M]
    if h == "10":
        return [FIXED_R, A, FIXED_M]
    if h == "01":
        return [FIXED_R, B, FIXED_M]
    if h == "11":
        return [FIXED_R, A, B, FIXED_M]
    raise ValueError(h)


def safe_rho(x, y):
    r = float(spearmanr(x, y).statistic)
    return 0.0 if not np.isfinite(r) else r


def l2_rows(x, eps=1e-12):
    x = np.asarray(x, float)
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        eps,
    )


def make_manifest(n_probes: int):
    ROOT.mkdir(parents=True, exist_ok=True)

    if MANIFEST.exists():
        m = json.loads(MANIFEST.read_text())
        if int(m["n_probes"]) != int(n_probes):
            raise RuntimeError(
                f"Existing manifest has n_probes={m['n_probes']}, "
                f"requested {n_probes}. Remove {MANIFEST} to change it."
            )
        return m

    bank = base.load_v11_probe_bank()

    seen = set()
    clean = []

    for pid, q in bank:
        q = str(q).strip()
        pid = str(pid)

        if not q or q in seen:
            continue

        seen.add(q)
        clean.append((pid, q))

    if len(clean) < n_probes:
        raise RuntimeError(
            f"Need {n_probes} V11 probes, only found {len(clean)}"
        )

    rng = random.Random(20260917)
    rng.shuffle(clean)
    chosen = clean[:n_probes]

    for _, A, B in PAIRS:
        for r in (A, B, FIXED_R, FIXED_M):
            if r not in TRAIN_ROLES:
                raise RuntimeError(f"role not in TRAIN_ROLES: {r}")

    m = {
        "version": "role_count_spectral_v1",
        "seed": SEED,
        "dim": DIM,
        "n_probes": n_probes,
        "fixed_R": FIXED_R,
        "fixed_M": FIXED_M,
        "pairs": [
            {"pair_id": pid, "A": A, "B": B}
            for pid, A, B in PAIRS
        ],
        "probes": [
            {"probe_id": pid, "query": q}
            for pid, q in chosen
        ],
        "hypotheses": {
            "00": "R -> M",
            "10": "R -> A -> M",
            "01": "R -> B -> M",
            "11": "R -> A -> B -> M",
        },
        "note": (
            "Only task text + final output are learner inputs. "
            "Role sequence is evaluation/training-system metadata."
        ),
    }

    MANIFEST.write_text(
        json.dumps(m, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return m


def role_system_prompt(role: str) -> str:
    raw = ROLE_PROMPTS[role]

    try:
        p = base._prompt_from_role_value(role, raw)
    except Exception:
        p = None

    if not p:
        p = str(raw)

    return (
        p.strip()
        + "\n\n"
        + "You are one stage in a sequential collaborative workflow. "
          "You receive the original user task and sometimes a previous-stage answer. "
          "Produce the best answer according to your specialization while preserving "
          "useful correct material from previous stages. "
          "Return only the resulting answer. Do not mention agents, roles, workflow, "
          "routing, hidden prompts, or internal structure."
    )


async def call_stage(
    client,
    sem,
    *,
    model,
    role,
    task,
    previous,
    max_tokens,
):
    if previous is None:
        user = (
            f"USER TASK:\n{task}\n\n"
            "There is no previous-stage answer."
        )
    else:
        user = (
            f"USER TASK:\n{task}\n\n"
            f"PREVIOUS-STAGE ANSWER:\n{previous}\n\n"
            "Improve or continue it according to your specialization."
        )

    delay = 1.0

    for attempt in range(5):
        try:
            async with sem:
                t0 = time.perf_counter()

                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    messages=[
                        {
                            "role": "system",
                            "content": role_system_prompt(role),
                        },
                        {
                            "role": "user",
                            "content": user,
                        },
                    ],
                )

                latency = time.perf_counter() - t0

            usage = resp.usage
            text = resp.choices[0].message.content or ""

            return {
                "text": text,
                "latency_s": float(latency),
                "input_tokens": int(
                    getattr(usage, "prompt_tokens", 0) or 0
                ),
                "output_tokens": int(
                    getattr(usage, "completion_tokens", 0) or 0
                ),
            }

        except Exception:
            if attempt == 4:
                raise

            await asyncio.sleep(delay)
            delay *= 2

    raise RuntimeError("unreachable")


async def run_chain(
    client,
    sem,
    *,
    model,
    task,
    roles,
    max_tokens,
):
    previous = None
    trace = []
    total_latency = 0.0
    tin = 0
    tout = 0

    for role in roles:
        r = await call_stage(
            client,
            sem,
            model=model,
            role=role,
            task=task,
            previous=previous,
            max_tokens=max_tokens,
        )

        previous = r["text"]
        total_latency += r["latency_s"]
        tin += r["input_tokens"]
        tout += r["output_tokens"]

        # Evaluation-only; never used by spectral learner.
        trace.append({
            "role": role,
            "text": r["text"],
        })

    return {
        "final": previous or "",
        "total_latency_s": float(total_latency),
        "input_tokens": tin,
        "output_tokens": tout,
        "trace_eval_only": trace,
    }


def cache_key(pair_id, probe_id, h):
    return json.dumps(
        [pair_id, probe_id, h],
        separators=(",", ":"),
    )


def load_cache():
    out = {}

    if not CACHE.exists():
        return out

    for line in CACHE.read_text(
        encoding="utf-8"
    ).splitlines():
        if not line.strip():
            continue

        r = json.loads(line)
        out[r["cache_key"]] = r

    return out


async def collect(args):
    m = make_manifest(args.n_probes)

    done = load_cache()

    pair_map = {
        x["pair_id"]: (x["A"], x["B"])
        for x in m["pairs"]
    }

    wanted = []

    for p in m["probes"]:
        for pair_id, (A, B) in pair_map.items():
            for h in HORDER:
                key = cache_key(
                    pair_id,
                    p["probe_id"],
                    h,
                )

                if key not in done:
                    wanted.append(
                        (
                            pair_id,
                            A,
                            B,
                            p["probe_id"],
                            p["query"],
                            h,
                        )
                    )

    total_systems = (
        len(m["probes"])
        * len(PAIRS)
        * len(HORDER)
    )

    total_agent_calls_per_probe_pair = 2 + 3 + 3 + 4

    print("=" * 90)
    print("ROLE + COUNT SPECTRAL COLLECTION")
    print("=" * 90)
    print("pairs:", len(PAIRS))
    print("probes:", len(m["probes"]))
    print("systems total:", total_systems)
    print(
        "approx agent calls total:",
        len(m["probes"])
        * len(PAIRS)
        * total_agent_calls_per_probe_pair,
    )
    print("missing systems:", len(wanted))

    if not wanted:
        print("[data] Complete cache already exists.")
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    counter = 0

    async def one(
        pair_id,
        A,
        B,
        probe_id,
        query,
        h,
    ):
        nonlocal counter

        seq = seq_for(A, B, h)

        r = await run_chain(
            client,
            sem,
            model=args.model,
            task=query,
            roles=seq,
            max_tokens=args.max_tokens,
        )

        row = {
            "cache_key": cache_key(
                pair_id,
                probe_id,
                h,
            ),
            "pair_id": pair_id,
            "role_A_eval_only": A,
            "role_B_eval_only": B,
            "hypothesis_eval_only": h,
            "k_eval_only": len(seq),
            "roles_eval_only": seq,
            "probe_id": probe_id,
            "query": query,
            **r,
        }

        async with write_lock:
            with CACHE.open(
                "a",
                encoding="utf-8",
            ) as f:
                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                f.flush()

            counter += 1

            if counter % 20 == 0 or counter == len(wanted):
                print(
                    f"[api] {counter}/{len(wanted)} "
                    "missing systems complete",
                    flush=True,
                )

    # Avoid creating thousands of live coroutines at once.
    batch_size = max(
        24,
        args.concurrency * 4,
    )

    for i in range(
        0,
        len(wanted),
        batch_size,
    ):
        chunk = wanted[
            i:i + batch_size
        ]

        await asyncio.gather(
            *(
                one(*x)
                for x in chunk
            )
        )

    print("[data] collection complete")


class GlobalRoleOperators(nn.Module):
    """
    One primitive W_role per actual role.

    For a role sequence [r0,r1,...]:
        F = W_last ... W_first

    This is the direct role/count analogue of the chain operator model.
    """

    def __init__(
        self,
        roles,
        d,
    ):
        super().__init__()

        self.roles = list(roles)
        self.index = {
            r: i
            for i, r in enumerate(self.roles)
        }

        self.d = d

        self.delta = nn.Parameter(
            torch.zeros(
                len(self.roles),
                d,
                d,
            )
        )

        nn.init.normal_(
            self.delta,
            mean=0.0,
            std=0.02,
        )

    def matrices(self):
        I = torch.eye(
            self.d,
            device=self.delta.device,
        ).unsqueeze(0)

        return I + self.delta

    def composition(self, seq):
        W = self.matrices()

        M = torch.eye(
            self.d,
            device=W.device,
        )

        for role in seq:
            M = W[
                self.index[role]
            ] @ M

        return M

    def predict(self, q, seq):
        F = self.composition(seq)
        return q @ F.T


def fit_role_ops(
    qtr,
    ztr,
    pair_defs,
    *,
    device,
    steps=3000,
    lr=0.003,
    weight_decay=0.001,
    identity_reg=0.002,
):
    all_roles = {
        FIXED_R,
        FIXED_M,
    }

    for _, A, B in pair_defs:
        all_roles.add(A)
        all_roles.add(B)

    model = GlobalRoleOperators(
        sorted(all_roles),
        qtr.shape[1],
    ).to(device)

    tq = torch.tensor(
        qtr,
        dtype=torch.float32,
        device=device,
    )

    tz = {
        k: torch.tensor(
            v,
            dtype=torch.float32,
            device=device,
        )
        for k, v in ztr.items()
    }

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best = math.inf
    best_state = None

    for step in range(
        1,
        steps + 1,
    ):
        opt.zero_grad(
            set_to_none=True
        )

        losses = []

        for pair_id, A, B in pair_defs:
            for h in HORDER:
                seq = seq_for(
                    A,
                    B,
                    h,
                )

                pred = model.predict(
                    tq,
                    seq,
                )

                losses.append(
                    (
                        (
                            pred
                            - tz[(pair_id, h)]
                        )
                        ** 2
                    ).mean()
                )

        data_loss = torch.stack(
            losses
        ).mean()

        reg = (
            identity_reg
            * (
                model.delta
                ** 2
            ).mean()
        )

        loss = data_loss + reg

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        opt.step()

        val = float(
            loss.detach().cpu()
        )

        if val < best:
            best = val
            best_state = copy.deepcopy(
                model.state_dict()
            )

        if step % 600 == 0:
            print(
                f"[fit] step={step:4d} "
                f"loss={val:.6f} "
                f"data={float(data_loss):.6f}"
            )

    model.load_state_dict(
        best_state
    )

    return model, best


def metric(score, actual):
    score = np.asarray(
        score,
        float,
    )

    actual = np.asarray(
        actual,
        float,
    )

    rho = safe_rho(
        score,
        actual,
    )

    n = len(score)

    k = max(
        1,
        math.ceil(
            0.20 * n
        ),
    )

    top = np.argsort(
        score
    )[::-1][:k]

    oracle = np.argsort(
        actual
    )[::-1][:k]

    base = float(
        np.mean(actual)
    )

    selected = float(
        np.mean(
            actual[top]
        )
    )

    oracle_v = float(
        np.mean(
            actual[oracle]
        )
    )

    return {
        "rho": rho,
        "lift20":
            selected
            / max(
                base,
                1e-12,
            ),
        "oracle_lift20":
            oracle_v
            / max(
                base,
                1e-12,
            ),
    }


def contrast_entries(
    *,
    qte,
    zte,
    mats,
    pred,
    contrasts,
):
    out = []

    for pair_id, A, B in PAIRS:
        for h0, h1 in contrasts:
            K = (
                mats[(pair_id, h1)]
                -
                mats[(pair_id, h0)]
            )

            score = np.einsum(
                "ni,ij,nj->n",
                qte,
                K.T @ K,
                qte,
            )

            actual = np.sum(
                (
                    zte[(pair_id, h1)]
                    -
                    zte[(pair_id, h0)]
                )
                ** 2,
                axis=1,
            )

            pdis = np.sum(
                (
                    pred[(pair_id, h1)]
                    -
                    pred[(pair_id, h0)]
                )
                ** 2,
                axis=1,
            )

            out.append({
                "pair_id": pair_id,
                "A": A,
                "B": B,
                "contrast":
                    f"{h0}_vs_{h1}",
                "score": score,
                "actual": actual,
                "pred_disagreement": pdis,
                "metric":
                    metric(
                        score,
                        actual,
                    ),
                "pred_metric":
                    metric(
                        pdis,
                        actual,
                    ),
            })

    return out


def macro_summary(entries):
    return {
        "rho":
            float(
                np.mean([
                    x["metric"]["rho"]
                    for x in entries
                ])
            ),

        "lift20":
            float(
                np.mean([
                    x["metric"]["lift20"]
                    for x in entries
                ])
            ),

        "oracle_lift20":
            float(
                np.mean([
                    x["metric"]["oracle_lift20"]
                    for x in entries
                ])
            ),

        "positive_rho":
            int(
                sum(
                    x["metric"]["rho"] > 0
                    for x in entries
                )
            ),

        "n":
            len(entries),

        "pred_disagreement_rho":
            float(
                np.mean([
                    x["pred_metric"]["rho"]
                    for x in entries
                ])
            ),

        "pred_disagreement_lift20":
            float(
                np.mean([
                    x["pred_metric"]["lift20"]
                    for x in entries
                ])
            ),
    }


def shuffle_null(
    entries,
    *,
    seed,
):
    rng = np.random.default_rng(
        seed
    )

    rhos = np.empty(
        N_SHUFFLE
    )

    lifts = np.empty(
        N_SHUFFLE
    )

    for b in range(
        N_SHUFFLE
    ):
        rr = []
        ll = []

        for x in entries:
            s = np.asarray(
                x["score"]
            ).copy()

            rng.shuffle(s)

            m = metric(
                s,
                x["actual"],
            )

            rr.append(
                m["rho"]
            )

            ll.append(
                m["lift20"]
            )

        rhos[b] = np.mean(
            rr
        )

        lifts[b] = np.mean(
            ll
        )

    return {
        "rho_mean":
            float(
                np.mean(rhos)
            ),

        "rho_q95":
            float(
                np.quantile(
                    rhos,
                    .95,
                )
            ),

        "lift_mean":
            float(
                np.mean(lifts)
            ),

        "lift_q95":
            float(
                np.quantile(
                    lifts,
                    .95,
                )
            ),
    }


def spectral_scores(
    q,
    mats,
    posterior,
):
    C = np.zeros(
        (
            q.shape[1],
            q.shape[1],
        ),
        float,
    )

    for i, hi in enumerate(
        HORDER
    ):
        for j in range(
            i + 1,
            len(HORDER),
        ):
            hj = HORDER[j]

            K = (
                mats[hj]
                -
                mats[hi]
            )

            C += (
                posterior[i]
                * posterior[j]
                * (
                    K.T
                    @ K
                )
            )

    return np.einsum(
        "ni,ij,nj->n",
        q,
        C,
        q,
    )


def posterior_update(
    posterior,
    obs,
    pred_one,
    sigma2,
):
    ll = []

    denom = max(
        2.0 * sigma2,
        1e-8,
    )

    for h in HORDER:
        d2 = float(
            np.sum(
                (
                    obs
                    -
                    pred_one[h]
                )
                ** 2
            )
        )

        ll.append(
            -d2 / denom
        )

    ll = np.asarray(
        ll,
        float,
    )

    logp = (
        np.log(
            np.clip(
                posterior,
                1e-12,
                1.0,
            )
        )
        + ll
    )

    logp -= logp.max()

    p = np.exp(logp)

    return p / p.sum()


def classification_from_posterior(
    p,
):
    state = HORDER[
        int(
            np.argmax(p)
        )
    ]

    pk = {
        2: p[0],
        3: p[1] + p[2],
        4: p[3],
    }

    pred_k = max(
        pk,
        key=pk.get,
    )

    pred_A = int(
        p[1] + p[3]
        >= 0.5
    )

    pred_B = int(
        p[2] + p[3]
        >= 0.5
    )

    return (
        state,
        pred_k,
        pred_A,
        pred_B,
    )


def run_identification(
    *,
    qte,
    zte,
    pred,
    mats,
    sigma2,
    policy,
    repeats,
    seed,
):
    budgets = [
        b
        for b in (
            1,
            2,
            3,
            5,
            10,
        )
        if b <= len(qte)
    ]

    accum = {
        b: defaultdict(list)
        for b in budgets
    }

    rng = np.random.default_rng(
        seed
    )

    for rep in range(
        repeats
    ):
        for pair_id, A, B in PAIRS:

            pmats = {
                h:
                    mats[
                        (
                            pair_id,
                            h,
                        )
                    ]
                for h in HORDER
            }

            ppred = {
                h:
                    pred[
                        (
                            pair_id,
                            h,
                        )
                    ]
                for h in HORDER
            }

            pactual = {
                h:
                    zte[
                        (
                            pair_id,
                            h,
                        )
                    ]
                for h in HORDER
            }

            for true_h in HORDER:
                posterior = (
                    np.ones(4)
                    / 4.0
                )

                unused = list(
                    range(
                        len(qte)
                    )
                )

                for step in range(
                    1,
                    max(budgets) + 1,
                ):
                    if policy == "spectral":
                        score = spectral_scores(
                            qte[unused],
                            pmats,
                            posterior,
                        )

                        local = int(
                            np.argmax(
                                score
                            )
                        )

                        chosen = unused[
                            local
                        ]

                    elif policy == "random":
                        chosen = int(
                            rng.choice(
                                unused
                            )
                        )

                    else:
                        raise ValueError(
                            policy
                        )

                    obs = pactual[
                        true_h
                    ][chosen]

                    pred_one = {
                        h:
                            ppred[h][
                                chosen
                            ]
                        for h in HORDER
                    }

                    posterior = posterior_update(
                        posterior,
                        obs,
                        pred_one,
                        sigma2,
                    )

                    unused.remove(
                        chosen
                    )

                    if step in budgets:
                        (
                            state,
                            pred_k,
                            pred_A,
                            pred_B,
                        ) = (
                            classification_from_posterior(
                                posterior
                            )
                        )

                        accum[
                            step
                        ][
                            "state_acc"
                        ].append(
                            float(
                                state
                                == true_h
                            )
                        )

                        accum[
                            step
                        ][
                            "k_acc"
                        ].append(
                            float(
                                pred_k
                                == K_OF[
                                    true_h
                                ]
                            )
                        )

                        accum[
                            step
                        ][
                            "k_mae"
                        ].append(
                            abs(
                                pred_k
                                -
                                K_OF[
                                    true_h
                                ]
                            )
                        )

                        accum[
                            step
                        ][
                            "A_acc"
                        ].append(
                            float(
                                pred_A
                                == A_PRESENT[
                                    true_h
                                ]
                            )
                        )

                        accum[
                            step
                        ][
                            "B_acc"
                        ].append(
                            float(
                                pred_B
                                == B_PRESENT[
                                    true_h
                                ]
                            )
                        )

    return {
        b: {
            k:
                float(
                    np.mean(v)
                )
            for k, v
            in accum[b].items()
        }
        for b in budgets
    }


def analyze(args):
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    m = make_manifest(
        args.n_probes
    )

    rows = list(
        load_cache().values()
    )

    expected = (
        len(m["probes"])
        * len(PAIRS)
        * len(HORDER)
    )

    if len(rows) < expected:
        raise RuntimeError(
            f"Cache incomplete: "
            f"{len(rows)}/{expected}"
        )

    probe_ids = [
        x["probe_id"]
        for x in m["probes"]
    ]

    q_text = [
        x["query"]
        for x in m["probes"]
    ]

    pos = {
        pid: i
        for i, pid
        in enumerate(
            probe_ids
        )
    }

    outputs = {
        (
            pair_id,
            h,
        ):
        [""] * len(probe_ids)
        for pair_id, _, _
        in PAIRS
        for h in HORDER
    }

    for r in rows:
        key = (
            r["pair_id"],
            r[
                "hypothesis_eval_only"
            ],
        )

        pid = r["probe_id"]

        if (
            key in outputs
            and pid in pos
        ):
            outputs[
                key
            ][
                pos[pid]
            ] = r["final"]

    for key, vals in outputs.items():
        if any(
            not x
            for x in vals
        ):
            raise RuntimeError(
                f"Incomplete outputs for {key}"
            )

    idx = np.arange(
        len(probe_ids)
    )

    tr_idx, te_idx = train_test_split(
        idx,
        train_size=TRAIN_FRAC,
        random_state=SEED,
        shuffle=True,
    )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 100)
    print(
        "ROLE + COUNT SPECTRAL ANALYSIS"
    )
    print("=" * 100)
    print("device:", device)
    print(
        f"probes={len(idx)} "
        f"train={len(tr_idx)} "
        f"heldout={len(te_idx)}"
    )

    enc = SentenceTransformer(
        args.embed_model,
        device=device,
    )

    print("[embed] queries")

    q_full = enc.encode(
        q_text,
        batch_size=64,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    z_full = {}

    for key, texts in outputs.items():
        print(
            "[embed]",
            key,
        )

        z_full[key] = enc.encode(
            texts,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    # Shared PCA, TRAIN ONLY.
    pca_train = [
        q_full[tr_idx]
    ] + [
        z_full[key][tr_idx]
        for key in sorted(
            z_full
        )
    ]

    pca_train = np.vstack(
        pca_train
    )

    d = min(
        DIM,
        pca_train.shape[0] - 1,
        pca_train.shape[1],
    )

    pca = PCA(
        n_components=d,
        random_state=SEED,
    ).fit(
        pca_train
    )

    q = l2_rows(
        pca.transform(
            q_full
        )
    )

    z = {
        key:
            l2_rows(
                pca.transform(
                    val
                )
            )
        for key, val
        in z_full.items()
    }

    qtr = q[tr_idx]
    qte = q[te_idx]

    ztr = {
        key:
            val[tr_idx]
        for key, val
        in z.items()
    }

    zte = {
        key:
            val[te_idx]
        for key, val
        in z.items()
    }

    # --------------------------------------------------------
    # DIRECT MAPS
    # --------------------------------------------------------

    direct_mats = {}
    direct_pred_tr = {}
    direct_pred_te = {}

    for key in sorted(
        ztr
    ):
        rr = Ridge(
            alpha=args.ridge_alpha,
            fit_intercept=False,
        ).fit(
            qtr,
            ztr[key],
        )

        F = rr.coef_.copy()

        direct_mats[key] = F
        direct_pred_tr[key] = (
            qtr @ F.T
        )
        direct_pred_te[key] = (
            qte @ F.T
        )

    direct_sigma2 = float(
        np.mean([
            np.mean(
                np.sum(
                    (
                        direct_pred_tr[key]
                        -
                        ztr[key]
                    )
                    ** 2,
                    axis=1,
                )
            )
            for key in ztr
        ])
    )

    # --------------------------------------------------------
    # SHARED ROLE OPERATORS
    # --------------------------------------------------------

    model, train_loss = fit_role_ops(
        qtr,
        ztr,
        PAIRS,
        device=device,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        identity_reg=args.identity_reg,
    )

    factor_mats = {}
    factor_pred_tr = {}
    factor_pred_te = {}

    with torch.no_grad():
        tqtr = torch.tensor(
            qtr,
            dtype=torch.float32,
            device=device,
        )

        tqte = torch.tensor(
            qte,
            dtype=torch.float32,
            device=device,
        )

        for pair_id, A, B in PAIRS:
            for h in HORDER:
                seq = seq_for(
                    A,
                    B,
                    h,
                )

                F = (
                    model.composition(
                        seq
                    )
                    .cpu()
                    .numpy()
                )

                factor_mats[
                    (
                        pair_id,
                        h,
                    )
                ] = F

                factor_pred_tr[
                    (
                        pair_id,
                        h,
                    )
                ] = (
                    model.predict(
                        tqtr,
                        seq,
                    )
                    .cpu()
                    .numpy()
                )

                factor_pred_te[
                    (
                        pair_id,
                        h,
                    )
                ] = (
                    model.predict(
                        tqte,
                        seq,
                    )
                    .cpu()
                    .numpy()
                )

    factor_sigma2 = float(
        np.mean([
            np.mean(
                np.sum(
                    (
                        factor_pred_tr[key]
                        -
                        ztr[key]
                    )
                    ** 2,
                    axis=1,
                )
            )
            for key in ztr
        ])
    )

    # --------------------------------------------------------
    # HELD-OUT RANKING: ROLE TYPE
    # --------------------------------------------------------

    f_type = contrast_entries(
        qte=qte,
        zte=zte,
        mats=factor_mats,
        pred=factor_pred_te,
        contrasts=TYPE_CONTRASTS,
    )

    d_type = contrast_entries(
        qte=qte,
        zte=zte,
        mats=direct_mats,
        pred=direct_pred_te,
        contrasts=TYPE_CONTRASTS,
    )

    # --------------------------------------------------------
    # HELD-OUT RANKING: COUNT / INSERTION
    # --------------------------------------------------------

    f_count = contrast_entries(
        qte=qte,
        zte=zte,
        mats=factor_mats,
        pred=factor_pred_te,
        contrasts=COUNT_CONTRASTS,
    )

    d_count = contrast_entries(
        qte=qte,
        zte=zte,
        mats=direct_mats,
        pred=direct_pred_te,
        contrasts=COUNT_CONTRASTS,
    )

    f_type_sum = macro_summary(
        f_type
    )

    d_type_sum = macro_summary(
        d_type
    )

    f_count_sum = macro_summary(
        f_count
    )

    d_count_sum = macro_summary(
        d_count
    )

    f_type_null = shuffle_null(
        f_type,
        seed=SEED + 10,
    )

    f_count_null = shuffle_null(
        f_count,
        seed=SEED + 20,
    )

    d_type_null = shuffle_null(
        d_type,
        seed=SEED + 30,
    )

    d_count_null = shuffle_null(
        d_count,
        seed=SEED + 40,
    )

    # --------------------------------------------------------
    # ACTUAL IDENTIFICATION:
    # none / A / B / both
    # and therefore K=2/3/4.
    # --------------------------------------------------------

    factor_spectral = run_identification(
        qte=qte,
        zte=zte,
        pred=factor_pred_te,
        mats=factor_mats,
        sigma2=factor_sigma2,
        policy="spectral",
        repeats=1,
        seed=SEED + 100,
    )

    factor_random = run_identification(
        qte=qte,
        zte=zte,
        pred=factor_pred_te,
        mats=factor_mats,
        sigma2=factor_sigma2,
        policy="random",
        repeats=300,
        seed=SEED + 200,
    )

    direct_spectral = run_identification(
        qte=qte,
        zte=zte,
        pred=direct_pred_te,
        mats=direct_mats,
        sigma2=direct_sigma2,
        policy="spectral",
        repeats=1,
        seed=SEED + 300,
    )

    direct_random = run_identification(
        qte=qte,
        zte=zte,
        pred=direct_pred_te,
        mats=direct_mats,
        sigma2=direct_sigma2,
        policy="random",
        repeats=300,
        seed=SEED + 400,
    )

    def clean_entries(xs):
        return [
            {
                "pair_id":
                    x["pair_id"],
                "A":
                    x["A"],
                "B":
                    x["B"],
                "contrast":
                    x["contrast"],
                "metric":
                    x["metric"],
                "pred_metric":
                    x["pred_metric"],
            }
            for x in xs
        ]

    result = {
        "experiment":
            "role_count_spectral_v1",

        "latent_dim":
            d,

        "train_probe_ids": [
            probe_ids[i]
            for i in tr_idx
        ],

        "heldout_probe_ids": [
            probe_ids[i]
            for i in te_idx
        ],

        "factorized": {
            "train_loss":
                float(
                    train_loss
                ),

            "sigma2":
                factor_sigma2,

            "type":
                f_type_sum,

            "type_null":
                f_type_null,

            "count":
                f_count_sum,

            "count_null":
                f_count_null,

            "type_details":
                clean_entries(
                    f_type
                ),

            "count_details":
                clean_entries(
                    f_count
                ),

            "spectral_identification":
                factor_spectral,

            "random_identification":
                factor_random,
        },

        "direct": {
            "sigma2":
                direct_sigma2,

            "type":
                d_type_sum,

            "type_null":
                d_type_null,

            "count":
                d_count_sum,

            "count_null":
                d_count_null,

            "type_details":
                clean_entries(
                    d_type
                ),

            "count_details":
                clean_entries(
                    d_count
                ),

            "spectral_identification":
                direct_spectral,

            "random_identification":
                direct_random,
        },
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    L = []

    L.append("=" * 106)
    L.append(
        "ROLE + AGENT-COUNT SPECTRAL ACTIVE PROBING V1"
    )
    L.append("=" * 106)

    L.append(
        f"pairs={len(PAIRS)} | "
        f"probes={len(probe_ids)} | "
        f"train={len(tr_idx)} | "
        f"heldout={len(te_idx)} | "
        f"latent_dim={d}"
    )

    L.append(
        "Local hypotheses: "
        "H00=R-M (K2), "
        "H10=R-A-M (K3), "
        "H01=R-B-M (K3), "
        "H11=R-A-B-M (K4)"
    )

    def block(
        name,
        t,
        tn,
        c,
        cn,
    ):
        L.append("")
        L.append("-" * 106)
        L.append(name)
        L.append("-" * 106)

        L.append(
            "ROLE TYPE — H10 vs H01 "
            "(same K=3)"
        )

        L.append(
            f"  macro rho      = "
            f"{t['rho']:+.4f}"
        )

        L.append(
            f"  lift@20        = "
            f"{t['lift20']:.4f}x"
        )

        L.append(
            f"  oracle lift    = "
            f"{t['oracle_lift20']:.4f}x"
        )

        L.append(
            f"  positive pairs = "
            f"{t['positive_rho']}/{t['n']}"
        )

        L.append(
            f"  shuffle 95th   = "
            f"rho {tn['rho_q95']:+.4f}, "
            f"lift {tn['lift_q95']:.4f}x"
        )

        L.append("")
        L.append(
            "AGENT COUNT / INSERTION — "
            "all one-agent insertion contrasts"
        )

        L.append(
            f"  macro rho      = "
            f"{c['rho']:+.4f}"
        )

        L.append(
            f"  lift@20        = "
            f"{c['lift20']:.4f}x"
        )

        L.append(
            f"  oracle lift    = "
            f"{c['oracle_lift20']:.4f}x"
        )

        L.append(
            f"  positive       = "
            f"{c['positive_rho']}/{c['n']}"
        )

        L.append(
            f"  shuffle 95th   = "
            f"rho {cn['rho_q95']:+.4f}, "
            f"lift {cn['lift_q95']:.4f}x"
        )

    block(
        "SHARED FACTORIZED ROLE OPERATORS",
        f_type_sum,
        f_type_null,
        f_count_sum,
        f_count_null,
    )

    block(
        "DIRECT COMPOSITION-SPECIFIC MAPS",
        d_type_sum,
        d_type_null,
        d_count_sum,
        d_count_null,
    )

    def ident_table(
        title,
        spec,
        rnd,
    ):
        L.append("")
        L.append("-" * 106)
        L.append(title)
        L.append("-" * 106)

        L.append(
            "budget | exact H acc | random H | "
            "K acc | random K | "
            "A-pres | B-pres | K MAE"
        )

        for b in sorted(
            spec
        ):
            s = spec[b]
            r = rnd[b]

            L.append(
                f"{b:6d} | "
                f"{s['state_acc']:.3f}       | "
                f"{r['state_acc']:.3f}    | "
                f"{s['k_acc']:.3f} | "
                f"{r['k_acc']:.3f}    | "
                f"{s['A_acc']:.3f}  | "
                f"{s['B_acc']:.3f}  | "
                f"{s['k_mae']:.3f}"
            )

    ident_table(
        "ACTUAL IDENTIFICATION — FACTORIZED SPECTRAL vs RANDOM",
        factor_spectral,
        factor_random,
    )

    ident_table(
        "ACTUAL IDENTIFICATION — DIRECT SPECTRAL vs RANDOM",
        direct_spectral,
        direct_random,
    )

    factor_type_go = (
        f_type_sum["rho"]
        > f_type_null["rho_q95"]
        and
        f_type_sum["lift20"]
        > f_type_null["lift_q95"]
    )

    factor_count_go = (
        f_count_sum["rho"]
        > f_count_null["rho_q95"]
        and
        f_count_sum["lift20"]
        > f_count_null["lift_q95"]
    )

    direct_type_go = (
        d_type_sum["rho"]
        > d_type_null["rho_q95"]
        and
        d_type_sum["lift20"]
        > d_type_null["lift_q95"]
    )

    direct_count_go = (
        d_count_sum["rho"]
        > d_count_null["rho_q95"]
        and
        d_count_sum["lift20"]
        > d_count_null["lift_q95"]
    )

    if (
        factor_type_go
        and factor_count_go
    ):
        verdict = (
            "GO_ROLE_AND_COUNT_SPECTRAL"
        )

    elif (
        direct_type_go
        or direct_count_go
    ):
        verdict = (
            "SIGNAL_EXISTS_FACTORIZATION_WEAK"
        )

    else:
        verdict = (
            "NO_GO_CURRENT_ROLE_COUNT_MODEL"
        )

    L.append("")
    L.append("=" * 106)
    L.append(
        f"VERDICT: {verdict}"
    )

    if verdict == "GO_ROLE_AND_COUNT_SPECTRAL":
        L.append(
            "Shared primitive role operators predict which held-out benign probes "
            "separate both role identity and agent insertion/count."
        )

    elif verdict == "SIGNAL_EXISTS_FACTORIZATION_WEAK":
        L.append(
            "The black-box outputs contain role/count-selective probe signal, "
            "but the shared primitive operator factorization is not yet capturing all of it."
        )

    else:
        L.append(
            "Current latent/operator formulation does not reliably predict "
            "held-out role/count-separating probes."
        )

    L.append("=" * 106)

    report = "\n".join(
        L
    )

    (
        OUT / "REPORT.txt"
    ).write_text(
        report + "\n",
        encoding="utf-8",
    )

    print()
    print(report)


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--collect",
        action="store_true",
    )

    p.add_argument(
        "--analyze",
        action="store_true",
    )

    p.add_argument(
        "--n-probes",
        type=int,
        default=int(
            os.getenv(
                "ROLECOUNT_N_PROBES",
                "40",
            )
        ),
    )

    p.add_argument(
        "--model",
        default=os.getenv(
            "ROLECOUNT_MODEL",
            "gpt-4o-mini-2024-07-18",
        ),
    )

    p.add_argument(
        "--concurrency",
        type=int,
        default=10,
    )

    p.add_argument(
        "--max-tokens",
        type=int,
        default=180,
    )

    p.add_argument(
        "--embed-model",
        default=(
            "sentence-transformers/"
            "all-MiniLM-L6-v2"
        ),
    )

    p.add_argument(
        "--ridge-alpha",
        type=float,
        default=3.0,
    )

    p.add_argument(
        "--steps",
        type=int,
        default=3000,
    )

    p.add_argument(
        "--lr",
        type=float,
        default=0.003,
    )

    p.add_argument(
        "--weight-decay",
        type=float,
        default=0.001,
    )

    p.add_argument(
        "--identity-reg",
        type=float,
        default=0.002,
    )

    return p.parse_args()


def main():
    args = parse_args()

    if not args.collect and not args.analyze:
        raise SystemExit(
            "Use --collect or --analyze"
        )

    if args.collect:
        asyncio.run(
            collect(args)
        )

    if args.analyze:
        analyze(args)


if __name__ == "__main__":
    main()
