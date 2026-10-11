#!/usr/bin/env python3
"""
blackbox_commutator_poc.py

Minimal black-box proof-of-concept for spectral / commutator-driven active probing.

What it tests
-------------
We construct TWO controlled black-box 2-agent systems with the same agents but
different execution order:

    H_AB: A -> B
    H_BA: B -> A

The learner receives ONLY:
    task text x
    final output y_AB
    final output y_BA
    optional end-to-end timing

It does NOT receive intermediate agent messages during learning.

We then:
1. Embed task + final outputs locally with SentenceTransformers on CUDA.
2. Fit simple black-box linear transfer maps F_AB and F_BA.
3. Define the structural difference / order operator:
       K = F_AB - F_BA
   (for an exact shared-operator model, this is W_B W_A - W_A W_B,
    i.e. a commutator).
4. Define the spectral informativeness of a probe q:
       score(q) = ||K q||^2 = q^T K^T K q
5. On HELD-OUT prompts, test whether score(q) predicts the ACTUAL observed
   separation ||z_AB - z_BA||^2.
6. Extract the dominant eigenvector of K^T K and check whether held-out prompts
   aligned with this direction are empirically more discriminative.

Optional:
7. Fit factorized W_A, W_B using only final I/O and compare:
       W_B W_A q ~= z_AB
       W_A W_B q ~= z_BA

This is a feasibility test, not a novelty proof.

Usage
-----
export OPENAI_API_KEY="..."
python blackbox_commutator_poc.py --n-prompts 60 --concurrency 8

Re-run safely: API outputs are checkpointed in JSONL and reused.

Dependencies
------------
pip install openai sentence-transformers scikit-learn scipy torch numpy
"""

import argparse
import asyncio
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI


# Current public GPT-4o-mini token prices at time this PoC was written.
# Override from CLI if pricing changes.
DEFAULT_INPUT_PRICE_PER_M = 0.15
DEFAULT_OUTPUT_PRICE_PER_M = 0.60


DEFAULT_TASKS = [
    "Explain why binary search requires sorted data, using one concrete example.",
    "Write a Python function that returns the second-largest distinct number in a list.",
    "A shop gives 20% off and then adds 17% tax. Explain how to compute the final price.",
    "Compare TCP and UDP for a real-time multiplayer game in a concise recommendation.",
    "Find the logical flaw: 'All birds have wings. Penguins have wings. Therefore penguins can fly.'",
    "Design three edge cases for a function that parses dates in YYYY-MM-DD format.",
    "Explain the difference between a process and a thread to a second-year CS student.",
    "Given [4, 1, 9, 9, 2], return the distinct values in descending order and explain.",
    "Write pseudocode for breadth-first search and state its time complexity.",
    "Explain what overfitting is and give one practical mitigation.",
    "A server handles 120 requests per second and traffic rises by 35%. Compute the new load.",
    "Review the claim 'a larger neural network always generalizes better' and correct it.",
    "Give a minimal SQL query to find duplicate email addresses in a users table.",
    "Explain why a hash table can have O(n) worst-case lookup despite O(1) average lookup.",
    "Propose a clean API for a key-value cache with get, set, delete, and TTL.",
    "Explain precision versus recall using spam detection.",
    "Write a robust Python snippet that safely reads an integer from a string.",
    "Explain what a race condition is with a two-thread example.",
    "A model has 92 true positives, 8 false positives, and 20 false negatives. Compute precision and recall.",
    "Explain why train/test leakage invalidates an evaluation.",
    "Design a small unit-test suite for a palindrome checker.",
    "Explain the intuition behind gradient descent without calculus-heavy notation.",
    "Compare a stack and a queue using one real-world use case for each.",
    "Given a directed graph, explain how to detect whether it contains a cycle.",
    "Write a concise code-review checklist for a new REST endpoint.",
    "Explain normalization in relational databases and give a tiny example.",
    "A job runs every 15 minutes. How many runs occur in 24 hours?",
    "Explain the difference between authentication and authorization.",
    "Suggest a data structure for implementing an LRU cache and justify it.",
    "Explain why correlation does not imply causation with a concrete example.",
    "Create a short debugging plan for a program that intermittently times out.",
    "Explain idempotency in HTTP APIs and name two idempotent methods.",
    "Write a Python function that groups words by their first letter.",
    "Explain the purpose of a validation set in machine learning.",
    "A dataset is 80% class A and 20% class B. Explain why accuracy can be misleading.",
    "Compare optimistic and pessimistic locking in databases.",
    "Explain the producer-consumer problem in operating systems.",
    "Give a concise strategy for finding a memory leak in a long-running service.",
    "Explain what a confusion matrix tells us.",
    "Write pseudocode for merging two sorted arrays.",
    "Explain why caching can create stale-data bugs and how to mitigate them.",
    "Given latencies 10, 12, 11, 80, 13 ms, explain why the mean may be misleading.",
    "Design an input validation policy for a username field.",
    "Explain the difference between encryption and hashing.",
    "A queue receives 50 jobs/minute and processes 45 jobs/minute. Explain what happens over time.",
    "Write a function specification for checking whether two strings are anagrams.",
    "Explain what backpressure means in a distributed system.",
    "Compare horizontal and vertical scaling.",
    "Explain why retries can accidentally duplicate side effects.",
    "Design two metamorphic tests for a sorting function.",
    "Explain the bias-variance tradeoff in practical terms.",
    "Given a 4-core CPU, explain why 100 threads do not imply 100-way parallel execution.",
    "Write a concise algorithm for finding connected components in an undirected graph.",
    "Explain the difference between latency and throughput.",
    "Propose three monitoring metrics for a web API and justify each.",
    "Explain why a database index can speed up reads but slow down writes.",
    "Given a probability of failure 0.01 per independent request, compute the probability of no failures in 100 requests.",
    "Explain what a deadlock is and list the four Coffman conditions.",
    "Write a short plan to evaluate whether a classifier is calibrated.",
    "Explain the difference between a compiler and an interpreter.",
    "Design a simple schema for users, posts, and comments.",
    "Explain why random seeds matter for reproducible ML experiments.",
    "Compare breadth-first search and depth-first search for shortest paths in an unweighted graph.",
    "Explain what an API rate limit is and how a client should react to HTTP 429.",
    "Write a short Python function that counts word frequencies case-insensitively.",
    "Explain what concept drift means in a deployed ML system.",
    "Propose an experiment to test whether a new ranking model improves click-through rate.",
    "Explain why p95 latency can be more informative than average latency.",
    "Design a minimal rollback plan for a risky production deployment.",
    "Explain the difference between supervised and unsupervised learning.",
]


AGENT_A = """You are Agent A, a constraint-focused technical reviewer.
Given the user's task and, when present, a draft from another agent:
- identify the exact requirements and hidden edge cases,
- correct factual or logical mistakes,
- revise the answer so every requirement is explicitly satisfied.
Return only the revised answer, not commentary about your role.
If no draft is provided, create a strong initial answer yourself.
Keep the response under 180 words."""

AGENT_B = """You are Agent B, a simplicity-focused technical reviewer.
Given the user's task and, when present, a draft from another agent:
- improve clarity, directness, and practical usefulness,
- remove unnecessary material,
- preserve or improve correctness,
- revise the answer into a clean final form.
Return only the revised answer, not commentary about your role.
If no draft is provided, create a strong initial answer yourself.
Keep the response under 180 words."""


def build_user_message(task: str, previous: str | None) -> str:
    if not previous:
        return f"USER TASK:\n{task}\n\nThere is no previous draft."
    return (
        f"USER TASK:\n{task}\n\n"
        "PREVIOUS AGENT DRAFT:\n"
        f"{previous}\n\n"
        "Revise it according to your role."
    )


async def call_agent(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    system_prompt: str,
    task: str,
    previous: str | None,
    max_tokens: int,
    retries: int = 5,
) -> Dict:
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
                        {"role": "user", "content": build_user_message(task, previous)},
                    ],
                )
                latency = time.perf_counter() - t0

            text = resp.choices[0].message.content or ""
            usage = resp.usage
            return {
                "text": text,
                "latency_s": latency,
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "output_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            }
        except Exception as e:
            if attempt == retries - 1:
                raise
            await asyncio.sleep(delay)
            delay *= 2
    raise RuntimeError("Unreachable")


async def run_chain(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model: str,
    task: str,
    order: str,
    max_tokens: int,
) -> Dict:
    roles = {
        "A": AGENT_A,
        "B": AGENT_B,
    }
    previous = None
    total_latency = 0.0
    in_tok = out_tok = 0

    # Intermediate messages are used by the controlled MAS, but are NOT saved
    # into the learner dataset. The learner only receives the final result.
    for role in order:
        r = await call_agent(
            client, semaphore, model, roles[role], task, previous, max_tokens
        )
        previous = r["text"]
        total_latency += r["latency_s"]
        in_tok += r["input_tokens"]
        out_tok += r["output_tokens"]

    return {
        "final": previous,
        "total_latency_s": total_latency,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
    }


def load_checkpoint(path: Path) -> Dict[int, Dict]:
    done = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                done[int(row["id"])] = row
    return done


async def collect_data(args, tasks: List[str]) -> List[Dict]:
    path = Path(args.cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = load_checkpoint(path)

    missing = [(i, task) for i, task in enumerate(tasks) if i not in done]
    if not missing:
        print(f"[data] Reusing all {len(done)} cached prompt pairs from {path}")
        return [done[i] for i in range(len(tasks))]

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set and API data are missing from the cache."
        )

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(args.concurrency)
    write_lock = asyncio.Lock()

    async def one(i: int, task: str):
        # AB and BA are two black-box systems. Run them concurrently.
        ab_task = asyncio.create_task(
            run_chain(client, semaphore, args.model, task, "AB", args.max_tokens)
        )
        ba_task = asyncio.create_task(
            run_chain(client, semaphore, args.model, task, "BA", args.max_tokens)
        )
        ab, ba = await asyncio.gather(ab_task, ba_task)

        row = {
            "id": i,
            "task": task,
            "AB": ab,
            "BA": ba,
        }
        async with write_lock:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()
        print(f"[api] {i+1:03d}/{len(tasks)} complete")
        return row

    results = await asyncio.gather(*(one(i, task) for i, task in missing))
    for row in results:
        done[int(row["id"])] = row

    return [done[i] for i in range(len(tasks))]


def token_cost(rows: List[Dict], input_price: float, output_price: float):
    inp = out = 0
    for row in rows:
        for key in ("AB", "BA"):
            inp += row[key]["input_tokens"]
            out += row[key]["output_tokens"]
    usd = inp / 1_000_000 * input_price + out / 1_000_000 * output_price
    return inp, out, usd


def l2_rows(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, eps)


class FactorizedOperators(nn.Module):
    """Learns W_A and W_B from final black-box I/O only."""
    def __init__(self, d: int):
        super().__init__()
        # Near-identity initialization stabilizes composition.
        self.A_delta = nn.Parameter(torch.zeros(d, d))
        self.B_delta = nn.Parameter(torch.zeros(d, d))
        nn.init.normal_(self.A_delta, mean=0.0, std=0.02)
        nn.init.normal_(self.B_delta, mean=0.0, std=0.02)

    def matrices(self):
        d = self.A_delta.shape[0]
        I = torch.eye(d, device=self.A_delta.device)
        WA = I + self.A_delta
        WB = I + self.B_delta
        return WA, WB

    def forward(self, q):
        WA, WB = self.matrices()
        # Column-vector convention z = Mq; batches are row vectors => q M^T
        M_ab = WB @ WA
        M_ba = WA @ WB
        pred_ab = q @ M_ab.T
        pred_ba = q @ M_ba.T
        return pred_ab, pred_ba


def fit_factorized(
    q_train: np.ndarray,
    z_ab_train: np.ndarray,
    z_ba_train: np.ndarray,
    device: str,
    steps: int,
    lr: float,
    weight_decay: float,
):
    q = torch.tensor(q_train, dtype=torch.float32, device=device)
    zab = torch.tensor(z_ab_train, dtype=torch.float32, device=device)
    zba = torch.tensor(z_ba_train, dtype=torch.float32, device=device)

    model = FactorizedOperators(q.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best = math.inf
    best_state = None
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        pab, pba = model(q)
        loss = ((pab - zab) ** 2).mean() + ((pba - zba) ** 2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        val = float(loss.detach().cpu())
        if val < best:
            best = val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    WA, WB = model.matrices()
    return (
        model,
        WA.detach().cpu().numpy(),
        WB.detach().cpu().numpy(),
        best,
    )


def evaluate_scores(name, K, q_test, z_ab_test, z_ba_test, test_tasks):
    # Predicted spectral informativeness.
    pred = np.sum((q_test @ K.T) ** 2, axis=1)

    # What actually happened in the black box.
    empirical = np.sum((z_ab_test - z_ba_test) ** 2, axis=1)

    rho, pval = spearmanr(pred, empirical)

    n = len(pred)
    k = max(1, math.ceil(0.20 * n))
    top_idx = np.argsort(pred)[-k:]
    mean_all = float(empirical.mean())
    mean_top = float(empirical[top_idx].mean())
    lift = mean_top / max(mean_all, 1e-12)

    # Principal spectral mode.
    D = K.T @ K
    eigvals, eigvecs = np.linalg.eigh(D)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    v1 = eigvecs[:, order[0]]
    alignment = np.abs(q_test @ v1)
    nearest = int(np.argmax(alignment))
    empirical_rank = int(
        np.where(np.argsort(empirical)[::-1] == nearest)[0][0]
    ) + 1

    spectral_gap = (
        float(eigvals[0] / max(eigvals[1], 1e-12))
        if len(eigvals) > 1 else float("inf")
    )

    print(f"\n===== {name} =====")
    print(f"Spearman(predicted spectral score, actual AB-vs-BA separation): "
          f"rho={rho:.3f}, p={pval:.4g}")
    print(f"Top-20% predicted probes: actual separation lift vs test mean = {lift:.3f}x")
    print(f"lambda_1={eigvals[0]:.6f}, spectral gap lambda1/lambda2={spectral_gap:.3f}x")
    print("Held-out prompt most aligned with principal eigenmode:")
    print(f"  {test_tasks[nearest]}")
    print(f"  Actual separation rank among held-out prompts: {empirical_rank}/{n}")

    return {
        "rho": float(rho),
        "pval": float(pval),
        "lift": lift,
        "lambda1": float(eigvals[0]),
        "spectral_gap": spectral_gap,
        "principal_prompt": test_tasks[nearest],
        "principal_prompt_actual_rank": empirical_rank,
        "pred": pred,
        "empirical": empirical,
    }


def main_analysis(args, rows: List[Dict]):
    rng = np.random.default_rng(args.seed)

    tasks = [r["task"] for r in rows]
    out_ab = [r["AB"]["final"] for r in rows]
    out_ba = [r["BA"]["final"] for r in rows]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[device] torch device: {device}")
    if torch.cuda.is_available():
        print(f"[device] GPU: {torch.cuda.get_device_name(0)}")

    print(f"[embed] Loading {args.embed_model}")
    encoder = SentenceTransformer(args.embed_model, device=device)

    # Embeddings are local. No embedding API cost.
    q_full = encoder.encode(
        tasks, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )
    zab_full = encoder.encode(
        out_ab, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )
    zba_full = encoder.encode(
        out_ba, batch_size=64, convert_to_numpy=True,
        normalize_embeddings=True, show_progress_bar=True
    )

    idx = np.arange(len(rows))
    tr_idx, te_idx = train_test_split(
        idx, train_size=args.train_frac, random_state=args.seed, shuffle=True
    )

    # Fit a shared PCA only on TRAIN embeddings, avoiding held-out leakage.
    pca_train = np.vstack([
        q_full[tr_idx],
        zab_full[tr_idx],
        zba_full[tr_idx],
    ])
    max_dim = min(args.dim, pca_train.shape[0] - 1, pca_train.shape[1])
    pca = PCA(n_components=max_dim, random_state=args.seed)
    pca.fit(pca_train)

    q = l2_rows(pca.transform(q_full))
    zab = l2_rows(pca.transform(zab_full))
    zba = l2_rows(pca.transform(zba_full))

    qtr, qte = q[tr_idx], q[te_idx]
    zabtr, zabte = zab[tr_idx], zab[te_idx]
    zbatr, zbate = zba[tr_idx], zba[te_idx]
    test_tasks = [tasks[i] for i in te_idx]

    print(f"[split] train={len(tr_idx)}, test={len(te_idx)}, latent_dim={max_dim}")

    # ------------------------------------------------------------
    # Baseline 1: direct black-box transfer maps.
    # ------------------------------------------------------------
    ridge_ab = Ridge(alpha=args.ridge_alpha, fit_intercept=False).fit(qtr, zabtr)
    ridge_ba = Ridge(alpha=args.ridge_alpha, fit_intercept=False).fit(qtr, zbatr)

    # sklearn predicts z = q @ coef_.T, so F = coef_.
    F_ab = ridge_ab.coef_
    F_ba = ridge_ba.coef_
    K_direct = F_ab - F_ba

    direct = evaluate_scores(
        "DIRECT BLACK-BOX SPECTRAL TEST",
        K_direct, qte, zabte, zbate, test_tasks
    )

    # Held-out prediction MSE for sanity.
    pred_ab = ridge_ab.predict(qte)
    pred_ba = ridge_ba.predict(qte)
    mse_direct = float(
        np.mean((pred_ab - zabte) ** 2) + np.mean((pred_ba - zbate) ** 2)
    )
    print(f"Held-out transfer prediction MSE (AB+BA): {mse_direct:.6f}")

    # ------------------------------------------------------------
    # Baseline 2 / research prototype: shared factorized operators W_A, W_B.
    # Only final I/O is used.
    # ------------------------------------------------------------
    model, WA, WB, train_loss = fit_factorized(
        qtr, zabtr, zbatr,
        device=device,
        steps=args.steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # The commutator corresponding to AB-vs-BA.
    K_comm = WB @ WA - WA @ WB

    factorized = evaluate_scores(
        "FACTORIZED COMMUTATOR TEST",
        K_comm, qte, zabte, zbate, test_tasks
    )

    with torch.no_grad():
        tq = torch.tensor(qte, dtype=torch.float32, device=device)
        pab, pba = model(tq)
        pab = pab.cpu().numpy()
        pba = pba.cpu().numpy()
    mse_factorized = float(
        np.mean((pab - zabte) ** 2) + np.mean((pba - zbate) ** 2)
    )
    print(f"Factorized train loss: {train_loss:.6f}")
    print(f"Held-out factorized prediction MSE (AB+BA): {mse_factorized:.6f}")
    print(f"||[W_B,W_A]||_F = {np.linalg.norm(K_comm, ord='fro'):.6f}")

    # Random-direction sanity baseline.
    empirical = direct["empirical"]
    random_scores = rng.random(len(empirical))
    random_rho = spearmanr(random_scores, empirical).statistic
    print(f"\nRandom-score Spearman sanity baseline: rho={random_rho:.3f}")

    # Heuristic signal only; not a statistical conclusion.
    go_direct = direct["rho"] >= args.go_rho and direct["lift"] >= args.go_lift
    go_comm = factorized["rho"] >= args.go_rho and factorized["lift"] >= args.go_lift

    print("\n================ SUMMARY ================")
    print(f"Direct spectral:     rho={direct['rho']:.3f}, top20 lift={direct['lift']:.3f}x")
    print(f"Factorized commutator: rho={factorized['rho']:.3f}, top20 lift={factorized['lift']:.3f}x")
    print(f"Direct held-out MSE:     {mse_direct:.6f}")
    print(f"Factorized held-out MSE: {mse_factorized:.6f}")
    print("-----------------------------------------")
    if go_direct:
        print("GO SIGNAL A: black-box spectral geometry predicts useful probes.")
    else:
        print("NO GO SIGNAL A yet: spectral ranking did not clear the heuristic threshold.")

    if go_comm:
        print("GO SIGNAL B: a shared factorized commutator also predicts useful probes.")
    else:
        print("NO GO SIGNAL B yet: the factorized operator version needs work / more data.")

    print(
        "\nImportant: GO/NO-GO thresholds are engineering heuristics, not a paper claim. "
        "Repeat across random splits/seeds before believing the result."
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gpt-4o-mini-2024-07-18")
    p.add_argument("--n-prompts", type=int, default=60)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=180)
    p.add_argument("--cache", default="artifacts/commutator_poc/api_runs.jsonl")
    p.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--dim", type=int, default=16)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--ridge-alpha", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-price-per-m", type=float, default=DEFAULT_INPUT_PRICE_PER_M)
    p.add_argument("--output-price-per-m", type=float, default=DEFAULT_OUTPUT_PRICE_PER_M)
    p.add_argument("--go-rho", type=float, default=0.25)
    p.add_argument("--go-lift", type=float, default=1.20)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.n_prompts > len(DEFAULT_TASKS):
        raise ValueError(
            f"--n-prompts={args.n_prompts}, but only {len(DEFAULT_TASKS)} built-in tasks exist."
        )
    tasks = DEFAULT_TASKS[: args.n_prompts]

    rows = asyncio.run(collect_data(args, tasks))

    inp, out, usd = token_cost(
        rows, args.input_price_per_m, args.output_price_per_m
    )
    print("\n===== API USAGE =====")
    print(f"Input tokens:  {inp:,}")
    print(f"Output tokens: {out:,}")
    print(f"Estimated API token cost: ${usd:.4f}")
    print("(Uses CLI price constants; update them if provider pricing changes.)")

    main_analysis(args, rows)
