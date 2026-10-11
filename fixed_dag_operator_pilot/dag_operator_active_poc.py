#!/usr/bin/env python3
"""
dag_operator_active_poc.py

Focused feasibility test for adaptive black-box identification of FIXED DAG MAS.

Core hypothesis
---------------
For a fixed DAG G over the same latent agents, approximate the final black-box
response embedding with a graph polynomial built from per-agent linear operators:

    F_G(W) = sum_{p in source->sink paths(G)}  prod_{i in p} W_i

and

    z_G(q) ~= normalize(F_G(W) q)

For a chain, changing execution order reduces to the familiar commutator-style
case. For a branching DAG, the graph is represented by a sum of path products.

The primary active probe score is the spectral DAG-disagreement operator:

    K_gh = F_g - F_h
    C_t  = sum_{g<h} p_t(g)p_t(h) K_gh^T K_gh
    A_spec(q) = q^T C_t q

The dominant eigenvector v1(C_t) is the continuous latent query direction with
maximum predicted graph separation; lambda1(C_t) is the maximum achievable
quadratic separation on the unit sphere. Because we have a finite V11 probe
bank, the deployed selector chooses the unused query maximizing q^T C_t q.

We retain normalized predictive-disagreement selection as an ablation/control.

Important scope
---------------
- Fixed DAG only; no dynamic router.
- Same agents across candidate topologies.
- Learner sees only query + FINAL output at evaluation time.
- Ground-truth topology is used only to build/train surrogate systems and score.
- V11 is used as a benign probe BANK, not as topology truth.

Pilot topology family (4 agents, all structurally distinct):
  chain:        0->1->2->3
  star:         0->1,0->2,0->3,1->3,2->3
  tree/diamond: 0->1,0->2,1->3,2->3
  complete_dag: all i<j
  random_dag:   0->1,1->2,0->2,2->3

Usage
-----
  # Collect/cache API outputs + analyze
  python dag_operator_active_poc.py --n-prompts 60 --concurrency 12

  # Analyze an existing cache without API calls
  python dag_operator_active_poc.py --n-prompts 60 --analysis-only

The script first tries operator3_poc.load_v11_probes(). If unavailable, it
falls back to artifacts/v11_probe_discovery.jsonl, then to the built-in benign
probe list from blackbox_commutator_poc.py.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import math
import os
import random
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Fixed-DAG hypothesis family
# -----------------------------------------------------------------------------

TOPOLOGIES: dict[str, list[tuple[int, int]]] = {
    "chain": [(0, 1), (1, 2), (2, 3)],
    "star": [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)],
    "tree": [(0, 1), (0, 2), (1, 3), (2, 3)],
    "complete_dag": [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)],
    "random_dag": [(0, 1), (0, 2), (1, 2), (2, 3)],
}
TOPOLOGY_NAMES = list(TOPOLOGIES)
N_AGENTS = 4
SOURCE = 0
SINK = 3


# Fallback roles only if router_r1.ROLES cannot be loaded.
FALLBACK_ROLES = [
    (
        "requirements",
        "You are a requirements-focused analyst. Identify constraints, assumptions, edge cases, "
        "and exact success criteria. Produce a useful answer, not commentary about your role.",
    ),
    (
        "reasoning",
        "You are a rigorous problem-solving specialist. Work through the task carefully, correct "
        "logical mistakes, and produce a technically sound draft. Do not discuss your role.",
    ),
    (
        "review",
        "You are a critical reviewer. Given the user task and predecessor drafts, detect errors, "
        "missing requirements, and inconsistencies, then improve the answer. Do not discuss your role.",
    ),
    (
        "synthesis",
        "You are a final synthesis specialist. Combine useful predecessor information into one "
        "clear, concise, correct answer to the user. Return only the final answer.",
    ),
]


def _prompt_from_role_value(name: str, value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for k in (
            "system_prompt", "system", "prompt", "instruction", "instructions",
            "description", "role_description", "desc",
        ):
            v = value.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    for k in (
        "system_prompt", "system", "prompt", "instruction", "instructions",
        "description", "role_description", "desc",
    ):
        v = getattr(value, k, None)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def load_role_pool() -> list[tuple[str, str]]:
    try:
        from router_r1 import ROLES  # type: ignore

        pool: list[tuple[str, str]] = []
        if isinstance(ROLES, dict):
            for name, val in ROLES.items():
                p = _prompt_from_role_value(str(name), val)
                if p:
                    pool.append((str(name), p))
        elif isinstance(ROLES, (list, tuple)):
            for i, val in enumerate(ROLES):
                if isinstance(val, dict):
                    name = str(val.get("name") or val.get("role") or val.get("id") or f"role_{i}")
                else:
                    name = str(getattr(val, "name", f"role_{i}"))
                p = _prompt_from_role_value(name, val)
                if p:
                    pool.append((name, p))
        if len(pool) >= N_AGENTS:
            print(f"[roles] Loaded {len(pool)} roles from router_r1.ROLES")
            return pool
    except Exception as e:
        print(f"[roles] router_r1.ROLES unavailable ({e!r}); using fallback roles")

    return list(FALLBACK_ROLES)


# -----------------------------------------------------------------------------
# V11 probe bank
# -----------------------------------------------------------------------------

def _normalize_probe_item(x: Any, idx: int) -> tuple[str, str] | None:
    if isinstance(x, str):
        q = x.strip()
        return (f"q{idx:04d}", q) if q else None
    if isinstance(x, dict):
        q = str(x.get("query") or x.get("query_text") or x.get("text") or "").strip()
        if not q:
            return None
        qid = str(x.get("id") or x.get("query_id") or x.get("probe_id") or f"q{idx:04d}")
        return qid, q
    if isinstance(x, (tuple, list)) and len(x) >= 2:
        if isinstance(x[1], str) and x[1].strip():
            return str(x[0]), x[1].strip()
    return None


def load_v11_probe_bank() -> list[tuple[str, str]]:
    # Preferred: exact loader already used in the operator experiments.
    try:
        from operator3_poc import load_v11_probes  # type: ignore

        raw = load_v11_probes()
        out = []
        for i, x in enumerate(raw):
            z = _normalize_probe_item(x, i)
            if z:
                out.append(z)
        if out:
            print(f"[probes] Loaded {len(out)} probes via operator3_poc.load_v11_probes")
            return out
    except Exception as e:
        print(f"[probes] operator3_poc loader unavailable ({e!r}); trying V11 JSONL")

    # Fallback: discovery file. Keep unique benign query text.
    p = Path("artifacts/v11_probe_discovery.jsonl")
    if p.exists():
        out = []
        seen = set()
        for ln, line in enumerate(p.open(encoding="utf-8"), 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            q = str(r.get("query") or r.get("query_text") or "").strip()
            if not q or q in seen:
                continue
            seen.add(q)
            out.append((str(r.get("probe_id") or r.get("id") or f"v11_{ln}"), q))
        if out:
            print(f"[probes] Loaded {len(out)} unique probes from {p}")
            return out

    # Last fallback: the original commutator PoC's benign tasks.
    try:
        import blackbox_commutator_poc as base  # type: ignore
        out = [(f"base_{i:03d}", str(q)) for i, q in enumerate(base.DEFAULT_TASKS)]
        print(f"[probes] V11 bank unavailable; using {len(out)} blackbox_commutator_poc.DEFAULT_TASKS")
        return out
    except Exception as e:
        raise RuntimeError("Could not load V11 probes or fallback benign tasks") from e


# -----------------------------------------------------------------------------
# DAG utilities
# -----------------------------------------------------------------------------

def validate_dag(edges: Sequence[tuple[int, int]], n: int = N_AGENTS) -> None:
    indeg = [0] * n
    children = [[] for _ in range(n)]
    for a, b in edges:
        if not (0 <= a < n and 0 <= b < n) or a == b:
            raise ValueError(f"Bad edge {(a, b)}")
        children[a].append(b)
        indeg[b] += 1
    dq = deque(i for i, d in enumerate(indeg) if d == 0)
    seen = []
    while dq:
        u = dq.popleft()
        seen.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    if len(seen) != n:
        raise ValueError(f"Not a DAG: {edges}")

    # Require every node to lie on some source->sink path.
    par = [[] for _ in range(n)]
    ch = [[] for _ in range(n)]
    for a, b in edges:
        ch[a].append(b)
        par[b].append(a)

    reach_from_source = set([SOURCE])
    dq = deque([SOURCE])
    while dq:
        u = dq.popleft()
        for v in ch[u]:
            if v not in reach_from_source:
                reach_from_source.add(v)
                dq.append(v)

    reach_sink_back = set([SINK])
    dq = deque([SINK])
    while dq:
        u = dq.popleft()
        for v in par[u]:
            if v not in reach_sink_back:
                reach_sink_back.add(v)
                dq.append(v)

    if len(reach_from_source & reach_sink_back) != n:
        raise ValueError(f"Every node must lie on source->sink path: {edges}")


def topo_order(edges: Sequence[tuple[int, int]], n: int = N_AGENTS) -> list[int]:
    indeg = [0] * n
    children = [[] for _ in range(n)]
    for a, b in edges:
        indeg[b] += 1
        children[a].append(b)
    dq = deque(i for i in range(n) if indeg[i] == 0)
    out = []
    while dq:
        u = min(dq)
        dq.remove(u)
        out.append(u)
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    return out


def parents_of(edges: Sequence[tuple[int, int]], node: int) -> list[int]:
    return sorted(a for a, b in edges if b == node)


def enumerate_paths(
    edges: Sequence[tuple[int, int]], source: int = SOURCE, sink: int = SINK
) -> list[list[int]]:
    children = defaultdict(list)
    for a, b in edges:
        children[a].append(b)
    for a in children:
        children[a].sort()

    paths: list[list[int]] = []

    def dfs(u: int, cur: list[int]) -> None:
        if u == sink:
            paths.append(cur.copy())
            return
        for v in children[u]:
            dfs(v, cur + [v])

    dfs(source, [source])
    return paths


for _name, _edges in TOPOLOGIES.items():
    validate_dag(_edges)


# -----------------------------------------------------------------------------
# Controlled fixed-DAG MAS
# -----------------------------------------------------------------------------

def build_agent_message(task: str, parent_outputs: list[tuple[int, str]]) -> str:
    if not parent_outputs:
        return (
            "USER TASK:\n"
            f"{task}\n\n"
            "You are the first processing stage. Produce the best useful answer you can."
        )

    blocks = []
    for pid, text in parent_outputs:
        blocks.append(f"PREDECESSOR {pid} DRAFT:\n{text}")
    joined = "\n\n".join(blocks)
    return (
        "USER TASK:\n"
        f"{task}\n\n"
        f"{joined}\n\n"
        "Use the predecessor drafts as working material. Integrate, correct, or improve them "
        "according to your system role. Return only the answer/draft for the next stage."
    )


async def call_agent(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    *,
    model: str,
    system_prompt: str,
    user_message: str,
    max_tokens: int,
    retries: int = 5,
) -> dict:
    delay = 1.0
    for attempt in range(retries):
        try:
            async with semaphore:
                t0 = time.perf_counter()
                resp = await client.chat.completions.create(
                    model=model,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                )
                latency = time.perf_counter() - t0
            usage = resp.usage
            return {
                "text": resp.choices[0].message.content or "",
                "latency_s": latency,
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        except Exception:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("unreachable")


async def run_fixed_dag(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    *,
    task: str,
    edges: Sequence[tuple[int, int]],
    roles: Sequence[tuple[str, str]],
    model: str,
    max_tokens: int,
) -> dict:
    outputs: dict[int, dict] = {}
    order = topo_order(edges)

    for node in order:
        ps = parents_of(edges, node)
        pouts = [(p, outputs[p]["text"]) for p in ps]
        msg = build_agent_message(task, pouts)
        name, sys_prompt = roles[node]
        r = await call_agent(
            client,
            semaphore,
            model=model,
            system_prompt=sys_prompt,
            user_message=msg,
            max_tokens=max_tokens,
        )
        outputs[node] = r

    return {
        "final": outputs[SINK]["text"],
        "total_latency_s": float(sum(x["latency_s"] for x in outputs.values())),
        "input_tokens": int(sum(x["input_tokens"] for x in outputs.values())),
        "output_tokens": int(sum(x["output_tokens"] for x in outputs.values())),
        # Privileged trace is cached only for sanity/evaluation, never used as learner input.
        "trace_eval_only": {str(k): outputs[k]["text"] for k in order},
    }


def cache_key(probe_id: str, topology: str, role_names: Sequence[str]) -> str:
    return json.dumps([probe_id, topology, list(role_names)], separators=(",", ":"))


def load_cache(path: Path) -> dict[str, dict]:
    out = {}
    if not path.exists():
        return out
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
            out[str(r["cache_key"])] = r
        except Exception:
            continue
    return out


async def collect_data(args, probes: list[tuple[str, str]], roles: list[tuple[str, str]]) -> list[dict]:
    path = Path(args.cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_cache(path)
    role_names = [r[0] for r in roles]

    wanted: list[tuple[str, str, str]] = []
    for pid, q in probes:
        for topo in TOPOLOGY_NAMES:
            k = cache_key(pid, topo, role_names)
            if k not in done:
                wanted.append((pid, q, topo))

    if wanted and args.analysis_only:
        raise RuntimeError(
            f"analysis-only requested but cache is missing {len(wanted)} probe/topology rows"
        )

    if wanted:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set and cache is incomplete")
        print(f"[data] Need {len(wanted)} topology runs ({len(probes)} probes x {len(TOPOLOGIES)} graphs total)")
        client = AsyncOpenAI()
        semaphore = asyncio.Semaphore(args.concurrency)
        write_lock = asyncio.Lock()
        counter = 0
        total = len(wanted)

        async def one(pid: str, q: str, topo: str):
            nonlocal counter
            r = await run_fixed_dag(
                client,
                semaphore,
                task=q,
                edges=TOPOLOGIES[topo],
                roles=roles,
                model=args.model,
                max_tokens=args.max_tokens,
            )
            row = {
                "cache_key": cache_key(pid, topo, role_names),
                "probe_id": pid,
                "query": q,
                "topology": topo,
                "edges_eval_only": TOPOLOGIES[topo],
                "roles_eval_only": role_names,
                **r,
            }
            async with write_lock:
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                done[row["cache_key"]] = row
                counter += 1
                if counter % 10 == 0 or counter == total:
                    print(f"[api] {counter}/{total} new topology runs complete")
            return row

        # Bound task creation as well as provider concurrency.
        batch = max(args.concurrency * 3, 12)
        for i in range(0, len(wanted), batch):
            chunk = wanted[i:i + batch]
            await asyncio.gather(*(one(*x) for x in chunk))
    else:
        print(f"[data] Reusing complete cache: {path}")

    rows = []
    for pid, q in probes:
        for topo in TOPOLOGY_NAMES:
            k = cache_key(pid, topo, role_names)
            rows.append(done[k])
    return rows


# -----------------------------------------------------------------------------
# Operator model: DAG = path polynomial in agent matrices
# -----------------------------------------------------------------------------

def l2_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


def torch_l2_rows(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


PATHS = {name: enumerate_paths(edges) for name, edges in TOPOLOGIES.items()}


class DAGPathOperators(nn.Module):
    def __init__(self, n_agents: int, d: int):
        super().__init__()
        self.n_agents = n_agents
        self.d = d
        self.delta = nn.Parameter(torch.zeros(n_agents, d, d))
        nn.init.normal_(self.delta, mean=0.0, std=0.02)
        # Per-agent residual gain, initialized near 1; shared across all topologies.
        self.log_gain = nn.Parameter(torch.zeros(n_agents))

    def matrices(self) -> torch.Tensor:
        I = torch.eye(self.d, device=self.delta.device).unsqueeze(0)
        gain = torch.exp(torch.clamp(self.log_gain, -1.5, 1.5)).view(-1, 1, 1)
        return I + gain * self.delta

    def graph_matrix(self, topology: str) -> torch.Tensor:
        W = self.matrices()
        mats = []
        for path in PATHS[topology]:
            M = torch.eye(self.d, device=W.device)
            for node in path:
                M = W[node] @ M
            mats.append(M)
        # Sum of path products = fixed-DAG path polynomial.
        F = torch.stack(mats, dim=0).sum(dim=0)
        return F

    def predict(self, q: torch.Tensor, topology: str) -> torch.Tensor:
        F = self.graph_matrix(topology)
        z = q @ F.T
        return torch_l2_rows(z)

    def forward_all(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        return {g: self.predict(q, g) for g in TOPOLOGY_NAMES}


def fit_operator_model(
    q_train: np.ndarray,
    z_train: dict[str, np.ndarray],
    *,
    device: str,
    steps: int,
    lr: float,
    weight_decay: float,
    identity_reg: float,
    seed: int,
) -> tuple[DAGPathOperators, float]:
    torch.manual_seed(seed)
    q = torch.tensor(q_train, dtype=torch.float32, device=device)
    z = {
        g: torch.tensor(z_train[g], dtype=torch.float32, device=device)
        for g in TOPOLOGY_NAMES
    }
    model = DAGPathOperators(N_AGENTS, q.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = math.inf
    best_state = None
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        losses = []
        for g in TOPOLOGY_NAMES:
            pred = model.predict(q, g)
            losses.append(((pred - z[g]) ** 2).mean())
        data_loss = torch.stack(losses).mean()
        reg = identity_reg * (model.delta ** 2).mean()
        loss = data_loss + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        val = float(loss.detach().cpu())
        if val < best:
            best = val
            best_state = copy.deepcopy(model.state_dict())
        if step % max(1, steps // 5) == 0:
            print(f"[fit-op] step={step:5d} loss={val:.6f} data={float(data_loss):.6f}")

    assert best_state is not None
    model.load_state_dict(best_state)
    return model, best


# -----------------------------------------------------------------------------
# Direct graph-specific ridge baseline
# -----------------------------------------------------------------------------

def fit_direct_maps(qtr: np.ndarray, ztr: dict[str, np.ndarray], alpha: float) -> dict[str, np.ndarray]:
    out = {}
    for g in TOPOLOGY_NAMES:
        rr = Ridge(alpha=alpha, fit_intercept=False).fit(qtr, ztr[g])
        out[g] = rr.coef_.copy()
    return out


def direct_predict(q: np.ndarray, F: np.ndarray) -> np.ndarray:
    return l2_rows(q @ F.T)


# -----------------------------------------------------------------------------
# Adaptive identification simulation from cached black-box outputs
# -----------------------------------------------------------------------------

def entropy(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def pairwise_disagreement(pred_by_graph: dict[str, np.ndarray], posterior: np.ndarray) -> np.ndarray:
    """Ablation: normalized prediction-space disagreement for each candidate query."""
    n = next(iter(pred_by_graph.values())).shape[0]
    score = np.zeros(n, dtype=float)
    for i, gi in enumerate(TOPOLOGY_NAMES):
        for j in range(i + 1, len(TOPOLOGY_NAMES)):
            gj = TOPOLOGY_NAMES[j]
            d2 = np.sum((pred_by_graph[gi] - pred_by_graph[gj]) ** 2, axis=1)
            score += posterior[i] * posterior[j] * d2
    return score


def spectral_disagreement_matrix(
    graph_mats: dict[str, np.ndarray], posterior: np.ndarray
) -> np.ndarray:
    """Posterior-weighted DAG analogue of K^T K.

    For only two order hypotheses AB vs BA this reduces (up to the posterior
    scalar) to [W_B,W_A]^T [W_B,W_A], i.e. the original commutator spectral
    matrix. For a DAG family, K_gh = F_g-F_h where F_g is the path polynomial.
    """
    d = next(iter(graph_mats.values())).shape[0]
    C = np.zeros((d, d), dtype=float)
    for i, gi in enumerate(TOPOLOGY_NAMES):
        for j in range(i + 1, len(TOPOLOGY_NAMES)):
            gj = TOPOLOGY_NAMES[j]
            K = np.asarray(graph_mats[gi], dtype=float) - np.asarray(graph_mats[gj], dtype=float)
            C += float(posterior[i] * posterior[j]) * (K.T @ K)
    # Numerical symmetrization.
    return 0.5 * (C + C.T)


def spectral_probe_scores(
    q_candidates: np.ndarray, graph_mats: dict[str, np.ndarray], posterior: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    C = spectral_disagreement_matrix(graph_mats, posterior)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    v1 = vecs[:, 0]
    scores = np.einsum('ni,ij,nj->n', q_candidates, C, q_candidates)
    align = np.abs(q_candidates @ v1)
    gap = float(vals[0] / max(vals[1], 1e-12)) if len(vals) > 1 else float('inf')
    return scores, {
        'lambda1': float(vals[0]),
        'lambda2': float(vals[1]) if len(vals) > 1 else 0.0,
        'spectral_gap': gap,
        'v1': v1,
        'alignment': align,
    }


def update_posterior(
    posterior: np.ndarray,
    obs: np.ndarray,
    pred_by_graph_one: dict[str, np.ndarray],
    sigma2: float,
) -> np.ndarray:
    ll = []
    denom = max(2.0 * sigma2, 1e-8)
    for g in TOPOLOGY_NAMES:
        d2 = float(np.sum((obs - pred_by_graph_one[g]) ** 2))
        ll.append(-d2 / denom)
    ll = np.asarray(ll, dtype=float)
    logp = np.log(np.clip(posterior, 1e-12, 1.0)) + ll
    logp -= logp.max()
    p = np.exp(logp)
    return p / p.sum()


def simulate_policy(
    *,
    pred_by_graph: dict[str, np.ndarray],
    actual_by_graph: dict[str, np.ndarray],
    sigma2: float,
    budgets: Sequence[int],
    policy: str,
    rng: np.random.Generator,
    q_candidates: np.ndarray | None = None,
    graph_mats: dict[str, np.ndarray] | None = None,
) -> dict[int, dict[str, float]]:
    max_budget = max(budgets)
    per_budget_acc = {b: [] for b in budgets}
    per_budget_truep = {b: [] for b in budgets}
    per_budget_ent = {b: [] for b in budgets}
    per_budget_lambda1 = {b: [] for b in budgets}
    per_budget_gap = {b: [] for b in budgets}

    for true_idx, true_g in enumerate(TOPOLOGY_NAMES):
        n = actual_by_graph[true_g].shape[0]
        posterior = np.ones(len(TOPOLOGY_NAMES), dtype=float) / len(TOPOLOGY_NAMES)
        unused = list(range(n))
        last_spec = {'lambda1': float('nan'), 'spectral_gap': float('nan')}

        for step in range(1, max_budget + 1):
            if policy == "spectral":
                if q_candidates is None or graph_mats is None:
                    raise ValueError('spectral policy requires q_candidates and graph_mats')
                scores, spec = spectral_probe_scores(q_candidates[unused], graph_mats, posterior)
                local = int(np.argmax(scores))
                chosen = unused[local]
                last_spec = spec
            elif policy == "disagreement":
                subpred = {g: pred_by_graph[g][unused] for g in TOPOLOGY_NAMES}
                scores = pairwise_disagreement(subpred, posterior)
                local = int(np.argmax(scores))
                chosen = unused[local]
            elif policy == "random":
                chosen = int(rng.choice(unused))
            else:
                raise ValueError(policy)

            obs = actual_by_graph[true_g][chosen]
            p1 = {g: pred_by_graph[g][chosen] for g in TOPOLOGY_NAMES}
            posterior = update_posterior(posterior, obs, p1, sigma2)
            unused.remove(chosen)

            if step in budgets:
                pred_idx = int(np.argmax(posterior))
                per_budget_acc[step].append(float(pred_idx == true_idx))
                per_budget_truep[step].append(float(posterior[true_idx]))
                per_budget_ent[step].append(entropy(posterior))
                if policy == 'spectral':
                    per_budget_lambda1[step].append(float(last_spec['lambda1']))
                    per_budget_gap[step].append(float(last_spec['spectral_gap']))

    return {
        b: {
            "acc": float(np.mean(per_budget_acc[b])),
            "true_p": float(np.mean(per_budget_truep[b])),
            "entropy": float(np.mean(per_budget_ent[b])),
            **({
                "lambda1": float(np.mean(per_budget_lambda1[b])),
                "spectral_gap": float(np.mean(per_budget_gap[b])),
            } if policy == 'spectral' else {}),
        }
        for b in budgets
    }


def repeated_random_policy(
    *,
    pred_by_graph: dict[str, np.ndarray],
    actual_by_graph: dict[str, np.ndarray],
    sigma2: float,
    budgets: Sequence[int],
    repeats: int,
    seed: int,
) -> dict[int, dict[str, float]]:
    agg = {b: defaultdict(list) for b in budgets}
    for r in range(repeats):
        res = simulate_policy(
            pred_by_graph=pred_by_graph,
            actual_by_graph=actual_by_graph,
            sigma2=sigma2,
            budgets=budgets,
            policy="random",
            rng=np.random.default_rng(seed + 1000 + r),
        )
        for b in budgets:
            for k, v in res[b].items():
                agg[b][k].append(v)
    return {
        b: {
            "acc": float(np.mean(agg[b]["acc"])),
            "acc_std": float(np.std(agg[b]["acc"])),
            "true_p": float(np.mean(agg[b]["true_p"])),
            "entropy": float(np.mean(agg[b]["entropy"])),
        }
        for b in budgets
    }


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def analyze(args, probes: list[tuple[str, str]], rows: list[dict]) -> dict:
    rng = np.random.default_rng(args.seed)
    q_text = [q for _, q in probes]
    id_to_pos = {pid: i for i, (pid, _) in enumerate(probes)}

    by_topo: dict[str, list[str]] = {g: [""] * len(probes) for g in TOPOLOGY_NAMES}
    for r in rows:
        by_topo[str(r["topology"])][id_to_pos[str(r["probe_id"])]] = str(r["final"])

    if any(not x for g in TOPOLOGY_NAMES for x in by_topo[g]):
        raise RuntimeError("Incomplete cache after collection")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")
    enc = SentenceTransformer(args.embed_model, device=device)
    q_full = enc.encode(q_text, batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    z_full = {
        g: enc.encode(by_topo[g], batch_size=64, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
        for g in TOPOLOGY_NAMES
    }

    idx = np.arange(len(probes))
    tr_idx, te_idx = train_test_split(idx, train_size=args.train_frac, random_state=args.seed, shuffle=True)

    pca_train = [q_full[tr_idx]] + [z_full[g][tr_idx] for g in TOPOLOGY_NAMES]
    pca_train = np.vstack(pca_train)
    d = min(args.dim, pca_train.shape[0] - 1, pca_train.shape[1])
    pca = PCA(n_components=d, random_state=args.seed).fit(pca_train)

    q = l2_rows(pca.transform(q_full))
    z = {g: l2_rows(pca.transform(z_full[g])) for g in TOPOLOGY_NAMES}
    qtr, qte = q[tr_idx], q[te_idx]
    ztr = {g: z[g][tr_idx] for g in TOPOLOGY_NAMES}
    zte = {g: z[g][te_idx] for g in TOPOLOGY_NAMES}

    print(f"[split] train_probes={len(tr_idx)} heldout_probes={len(te_idx)} latent_dim={d}")
    print("[paths]")
    for g in TOPOLOGY_NAMES:
        print(f"  {g:12s}: {PATHS[g]}")

    # Direct graph-specific maps = observability upper baseline.
    direct_maps = fit_direct_maps(qtr, ztr, args.ridge_alpha)
    direct_pred_tr = {g: direct_predict(qtr, direct_maps[g]) for g in TOPOLOGY_NAMES}
    direct_pred_te = {g: direct_predict(qte, direct_maps[g]) for g in TOPOLOGY_NAMES}
    direct_mse = float(np.mean([
        np.mean((direct_pred_te[g] - zte[g]) ** 2) for g in TOPOLOGY_NAMES
    ]))

    # Shared per-agent path-polynomial operators.
    op_model, train_loss = fit_operator_model(
        qtr,
        ztr,
        device=device,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        identity_reg=args.identity_reg,
        seed=args.seed,
    )
    with torch.no_grad():
        tqtr = torch.tensor(qtr, dtype=torch.float32, device=device)
        tqte = torch.tensor(qte, dtype=torch.float32, device=device)
        op_pred_tr = {g: op_model.predict(tqtr, g).cpu().numpy() for g in TOPOLOGY_NAMES}
        op_pred_te = {g: op_model.predict(tqte, g).cpu().numpy() for g in TOPOLOGY_NAMES}
        W = op_model.matrices().cpu().numpy()
        FG = {g: op_model.graph_matrix(g).cpu().numpy() for g in TOPOLOGY_NAMES}

    op_mse = float(np.mean([
        np.mean((op_pred_te[g] - zte[g]) ** 2) for g in TOPOLOGY_NAMES
    ]))

    # Train residual scale used only to calibrate the Bayesian-ish update.
    op_sigma2 = float(np.mean([
        np.sum((op_pred_tr[g] - ztr[g]) ** 2, axis=1).mean() for g in TOPOLOGY_NAMES
    ]))
    direct_sigma2 = float(np.mean([
        np.sum((direct_pred_tr[g] - ztr[g]) ** 2, axis=1).mean() for g in TOPOLOGY_NAMES
    ]))

    budgets = [b for b in (1, 2, 3, 5, 10, 20) if b <= len(te_idx)]

    # Primary policy: posterior-adaptive spectral selection.
    op_spectral = simulate_policy(
        pred_by_graph=op_pred_te,
        actual_by_graph=zte,
        sigma2=op_sigma2,
        budgets=budgets,
        policy="spectral",
        rng=rng,
        q_candidates=qte,
        graph_mats=FG,
    )
    # Control: the older normalized predictive-disagreement selector.
    op_disagreement = simulate_policy(
        pred_by_graph=op_pred_te,
        actual_by_graph=zte,
        sigma2=op_sigma2,
        budgets=budgets,
        policy="disagreement",
        rng=rng,
    )
    op_random = repeated_random_policy(
        pred_by_graph=op_pred_te,
        actual_by_graph=zte,
        sigma2=op_sigma2,
        budgets=budgets,
        repeats=args.random_repeats,
        seed=args.seed,
    )

    direct_spectral = simulate_policy(
        pred_by_graph=direct_pred_te,
        actual_by_graph=zte,
        sigma2=direct_sigma2,
        budgets=budgets,
        policy="spectral",
        rng=rng,
        q_candidates=qte,
        graph_mats=direct_maps,
    )
    direct_disagreement = simulate_policy(
        pred_by_graph=direct_pred_te,
        actual_by_graph=zte,
        sigma2=direct_sigma2,
        budgets=budgets,
        policy="disagreement",
        rng=rng,
    )
    direct_random = repeated_random_policy(
        pred_by_graph=direct_pred_te,
        actual_by_graph=zte,
        sigma2=direct_sigma2,
        budgets=budgets,
        repeats=args.random_repeats,
        seed=args.seed + 50000,
    )

    # Pairwise graph-polynomial norms: DAG analogue of commutator strength.
    pair_norms = {}
    for i, gi in enumerate(TOPOLOGY_NAMES):
        for j in range(i + 1, len(TOPOLOGY_NAMES)):
            gj = TOPOLOGY_NAMES[j]
            K = FG[gi] - FG[gj]
            pair_norms[f"{gi}__vs__{gj}"] = float(np.linalg.norm(K, ord="fro"))

    # Agent commutators retained as a diagnostic; not the whole DAG score.
    comm = {}
    for i in range(N_AGENTS):
        for j in range(i + 1, N_AGENTS):
            K = W[j] @ W[i] - W[i] @ W[j]
            comm[f"W{j}W{i}-W{i}W{j}"] = float(np.linalg.norm(K, ord="fro"))

    print("\n" + "=" * 82)
    print("FIXED-DAG OPERATOR PILOT")
    print("=" * 82)
    print(f"Direct held-out MSE        : {direct_mse:.6f}")
    print(f"Path-operator held-out MSE : {op_mse:.6f}")
    print(f"Path-operator train loss   : {train_loss:.6f}")
    print(f"Topology classes           : {len(TOPOLOGY_NAMES)} (chance={1/len(TOPOLOGY_NAMES):.3f})")

    print("\nACTIVE IDENTIFICATION — PATH OPERATOR / SPECTRAL EIGEN POLICY")
    print("budget | spectral | disagree | random ± sd | trueP(spec) | lambda1 | gap")
    for b in budgets:
        print(
            f"{b:6d} | {op_spectral[b]['acc']:.3f}    | {op_disagreement[b]['acc']:.3f}    | "
            f"{op_random[b]['acc']:.3f} ± {op_random[b]['acc_std']:.3f} | "
            f"{op_spectral[b]['true_p']:.3f}       | {op_spectral[b]['lambda1']:.4f} | "
            f"{op_spectral[b]['spectral_gap']:.2f}x"
        )

    print("\nACTIVE IDENTIFICATION — DIRECT GRAPH MAP / SPECTRAL UPPER BASELINE")
    print("budget | spectral | disagree | random ± sd | trueP(spec) | lambda1 | gap")
    for b in budgets:
        print(
            f"{b:6d} | {direct_spectral[b]['acc']:.3f}    | {direct_disagreement[b]['acc']:.3f}    | "
            f"{direct_random[b]['acc']:.3f} ± {direct_random[b]['acc_std']:.3f} | "
            f"{direct_spectral[b]['true_p']:.3f}       | {direct_spectral[b]['lambda1']:.4f} | "
            f"{direct_spectral[b]['spectral_gap']:.2f}x"
        )

    # Uniform-prior spectral diagnostic: what latent direction would we ask first?
    prior = np.ones(len(TOPOLOGY_NAMES), dtype=float) / len(TOPOLOGY_NAMES)
    initial_scores, initial_spec = spectral_probe_scores(qte, FG, prior)
    first_local = int(np.argmax(initial_scores))
    first_global = int(te_idx[first_local])
    print("\nINITIAL PRINCIPAL EIGENMODE — PATH OPERATOR")
    print(f"lambda1={initial_spec['lambda1']:.6f} gap={initial_spec['spectral_gap']:.3f}x")
    print(f"best discrete V11 probe q^T C q={initial_scores[first_local]:.6f}")
    print(f"probe_id={probes[first_global][0]}")
    print(f"probe={probes[first_global][1]}")
    print(f"|cos(q,v1)|={initial_spec['alignment'][first_local]:.4f}")

    print("\nDAG DIFFERENCE OPERATOR NORMS ||F_G-F_H||_F")
    for k, v in sorted(pair_norms.items(), key=lambda kv: -kv[1]):
        print(f"  {k:34s} {v:.5f}")

    print("\nAGENT COMMUTATOR NORMS")
    for k, v in comm.items():
        print(f"  {k:20s} {v:.5f}")

    # Clear falsification-oriented interpretation.
    b1 = budgets[0]
    blast = budgets[-1]
    spectral_gain = op_spectral[blast]["acc"] - op_random[blast]["acc"]
    spectral_vs_disagreement = op_spectral[blast]["acc"] - op_disagreement[blast]["acc"]
    print("\nINTERPRETATION GATE")
    if direct_spectral[blast]["acc"] <= 0.30:
        print("NO-GO A: even graph-specific spectral maps cannot identify fixed DAGs reliably; observability is weak.")
    elif op_spectral[blast]["acc"] <= 0.30:
        print("NO-GO B: fixed DAG is observable, but the shared path-operator algebra is not capturing it.")
    elif spectral_gain <= 0.05:
        print("PARTIAL: operator algebra carries topology, but eigen/spectral probe selection adds little over random.")
    elif spectral_vs_disagreement <= 0.0:
        print("GO-ish: active probing helps, but the principal-eigen spectral rule does not beat normalized disagreement.")
    else:
        print("GO: fixed-DAG path operators + principal-eigen spectral probing show a useful signal.")

    result = {
        "n_probes": len(probes),
        "train_probes": len(tr_idx),
        "test_probes": len(te_idx),
        "latent_dim": d,
        "topologies": TOPOLOGIES,
        "paths": PATHS,
        "direct_mse": direct_mse,
        "operator_mse": op_mse,
        "operator_train_loss": train_loss,
        "operator_sigma2": op_sigma2,
        "direct_sigma2": direct_sigma2,
        "operator_spectral": op_spectral,
        "operator_disagreement": op_disagreement,
        "operator_random": op_random,
        "direct_spectral": direct_spectral,
        "direct_disagreement": direct_disagreement,
        "direct_random": direct_random,
        "initial_spectral": {
            "lambda1": initial_spec["lambda1"],
            "lambda2": initial_spec["lambda2"],
            "spectral_gap": initial_spec["spectral_gap"],
            "best_probe_id": probes[first_global][0],
            "best_probe_text": probes[first_global][1],
            "best_qCq": float(initial_scores[first_local]),
            "best_alignment_v1": float(initial_spec["alignment"][first_local]),
        },
        "pair_operator_norms": pair_norms,
        "agent_commutator_norms": comm,
        "heldout_probe_ids": [probes[i][0] for i in te_idx],
        "heldout_probe_text": [probes[i][1] for i in te_idx],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved: {out}")
    return result


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--n-prompts", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--max-tokens", type=int, default=180)
    p.add_argument("--cache", default="artifacts/dag_operator_active_poc/api_runs.jsonl")
    p.add_argument("--out", default="artifacts/dag_operator_active_poc/report.json")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--ridge-alpha", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--identity-reg", type=float, default=2e-3)
    p.add_argument("--random-repeats", type=int, default=200)
    p.add_argument("--seed", type=int, default=73)
    p.add_argument("--role-offset", type=int, default=0,
                   help="Choose a different consecutive 4-role surrogate set from router_r1.ROLES")
    p.add_argument("--analysis-only", action="store_true")
    p.add_argument("--collect-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bank = load_v11_probe_bank()
    if args.n_prompts > len(bank):
        raise ValueError(f"Requested {args.n_prompts} probes, but bank has only {len(bank)}")

    # Deterministic, diverse-ish subsample across the bank rather than first N only.
    if args.n_prompts == len(bank):
        probes = bank
    else:
        ids = np.linspace(0, len(bank) - 1, args.n_prompts, dtype=int)
        probes = [bank[int(i)] for i in ids]

    role_pool = load_role_pool()
    if len(role_pool) < N_AGENTS:
        raise RuntimeError(f"Need at least {N_AGENTS} roles; got {len(role_pool)}")
    off = args.role_offset % max(1, len(role_pool) - N_AGENTS + 1)
    roles = role_pool[off:off + N_AGENTS]
    print("[roles] Pilot agents:")
    for i, (name, _) in enumerate(roles):
        print(f"  {i}: {name}")

    rows = asyncio.run(collect_data(args, probes, roles))

    inp = sum(int(r.get("input_tokens", 0)) for r in rows)
    out = sum(int(r.get("output_tokens", 0)) for r in rows)
    print(f"[usage] cached total input_tokens={inp:,} output_tokens={out:,}")

    if args.collect_only:
        print("[done] collection complete; skipping analysis (--collect-only)")
        return

    analyze(args, probes, rows)


if __name__ == "__main__":
    main()
