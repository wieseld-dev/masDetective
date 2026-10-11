#!/usr/bin/env python3

import argparse
import asyncio
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from openai import AsyncOpenAI
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer

import blackbox_commutator_poc as base


AGENTS = {
"A": """You are Agent A: a requirements and constraints analyst.
Given the user's task and possibly a previous draft:
- identify exact requirements, assumptions, and edge cases,
- correct missed constraints,
- revise the answer so requirements are explicitly satisfied.
Return only the revised answer.
Keep it under 180 words.""",

"B": """You are Agent B: a technical problem solver.
Given the user's task and possibly a previous draft:
- make the solution concrete and technically correct,
- add useful implementation, algorithmic, or mathematical details,
- correct technical mistakes.
Return only the revised answer.
Keep it under 180 words.""",

"C": """You are Agent C: a critical verifier and finalizer.
Given the user's task and possibly a previous draft:
- test the reasoning for errors and edge cases,
- correct unsupported or incorrect statements,
- make the final answer precise and internally consistent.
Return only the revised answer.
Keep it under 180 words."""
}

PAIRS = ["AB","BA","AC","CA","BC","CB"]
TRIPLES = ["ABC","ACB","BAC","BCA","CAB","CBA"]

# Each pair differs ONLY in the first-two order.
# The final agent is identical.
SAME_TERMINAL = [
    ("ABC","BAC"),  # C last
    ("ACB","CAB"),  # B last
    ("BCA","CBA"),  # A last
]


def load_v11_probes(n):
    p = Path("artifacts/v11_response_public.jsonl")
    probes, seen = [], set()

    for line in p.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        q = r.get("query")
        if isinstance(q, str):
            q = q.strip()
            if len(q) >= 5 and q not in seen:
                seen.add(q)
                probes.append(q)

    return probes[:n]


async def collect(args, tasks):
    cache = Path(args.cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    done = {}
    if cache.exists():
        for line in cache.open(encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done[int(r["id"])] = r

    missing = [(i,q) for i,q in enumerate(tasks) if i not in done]

    print(f"[data] total={len(tasks)} cached={len(done)} missing={len(missing)}")

    if not missing:
        return [done[i] for i in range(len(tasks))]

    client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()

    async def call(role, task, previous):
        return await base.call_agent(
            client,
            sem,
            args.model,
            AGENTS[role],
            task,
            previous,
            args.max_tokens,
        )

    async def one(i, task):
        outputs = {}
        total_in = 0
        total_out = 0

        # Level 1: A, B, C
        rs = await asyncio.gather(
            *[call(r, task, None) for r in "ABC"]
        )

        for role, res in zip("ABC", rs):
            outputs[role] = res["text"]
            total_in += res["input_tokens"]
            total_out += res["output_tokens"]

        # Level 2: all ordered pairs
        async def pair(comp):
            res = await call(comp[1], task, outputs[comp[0]])
            return comp, res

        rs = await asyncio.gather(*[pair(c) for c in PAIRS])

        for comp, res in rs:
            outputs[comp] = res["text"]
            total_in += res["input_tokens"]
            total_out += res["output_tokens"]

        # Level 3: all 6 permutations
        async def triple(comp):
            res = await call(comp[2], task, outputs[comp[:2]])
            return comp, res

        rs = await asyncio.gather(*[triple(c) for c in TRIPLES])

        for comp, res in rs:
            outputs[comp] = res["text"]
            total_in += res["input_tokens"]
            total_out += res["output_tokens"]

        row = {
            "id": i,
            "task": task,
            "outputs": outputs,
            "input_tokens": total_in,
            "output_tokens": total_out,
        }

        async with lock:
            with cache.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[api] {i+1}/{len(tasks)} complete")
        return row

    new = await asyncio.gather(*[one(i,q) for i,q in missing])

    for r in new:
        done[int(r["id"])] = r

    return [done[i] for i in range(len(tasks))]


def l2(x):
    return x / np.maximum(
        np.linalg.norm(x, axis=1, keepdims=True),
        1e-12
    )


class Operator3(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.A = nn.Parameter(torch.randn(d,d) * 0.02)
        self.B = nn.Parameter(torch.randn(d,d) * 0.02)
        self.C = nn.Parameter(torch.randn(d,d) * 0.02)

    def W(self):
        I = torch.eye(self.d, device=self.A.device)
        return {
            "A": I + self.A,
            "B": I + self.B,
            "C": I + self.C,
        }

    def compose(self, comp):
        W = self.W()
        M = torch.eye(self.d, device=self.A.device)

        # "ABC" means A acts first, then B, then C:
        # z = W_C W_B W_A q
        for role in comp:
            M = W[role] @ M

        return M

    def predict(self, q, comp):
        M = self.compose(comp)
        return q @ M.T


def fit_model(q, targets, train_idx, compositions,
              device="cpu", steps=3500):

    qtr = torch.tensor(
        q[train_idx],
        dtype=torch.float32,
        device=device
    )

    z = {
        c: torch.tensor(
            targets[c][train_idx],
            dtype=torch.float32,
            device=device
        )
        for c in compositions
    }

    model = Operator3(q.shape[1]).to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=3e-3,
        weight_decay=1e-3
    )

    best = float("inf")
    best_state = None

    for step in range(steps):
        opt.zero_grad(set_to_none=True)

        loss = 0
        for c in compositions:
            pred = model.predict(qtr, c)
            loss = loss + ((pred - z[c])**2).mean()

        loss = loss / len(compositions)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0
        )

        opt.step()

        lv = float(loss.detach().cpu())

        if lv < best:
            best = lv
            best_state = {
                k:v.detach().cpu().clone()
                for k,v in model.state_dict().items()
            }

    model.load_state_dict(best_state)
    return model, best


def predictions(model, q, comps, device):
    tq = torch.tensor(
        q,
        dtype=torch.float32,
        device=device
    )

    result = {}

    with torch.no_grad():
        for c in comps:
            result[c] = (
                model.predict(tq,c)
                .cpu()
                .numpy()
            )

    return result


def retrieval_accuracy(pred, actual, idx,
                       actual_comps, candidate_comps):

    correct = 0
    total = 0
    ranks = []

    for i in idx:
        for truth in actual_comps:
            z = actual[truth][i]

            ds = []
            for candidate in candidate_comps:
                d = np.sum(
                    (z - pred[candidate][i])**2
                )
                ds.append((d,candidate))

            ds.sort()
            ranking = [x[1] for x in ds]

            rank = ranking.index(truth) + 1
            ranks.append(rank)

            correct += (rank == 1)
            total += 1

    return correct / total, float(np.mean(ranks))


def same_terminal_accuracy(pred, actual, idx):

    correct = 0
    total = 0

    for c1,c2 in SAME_TERMINAL:
        for i in idx:

            for truth in [c1,c2]:

                z = actual[truth][i]

                d1 = np.sum(
                    (z-pred[c1][i])**2
                )
                d2 = np.sum(
                    (z-pred[c2][i])**2
                )

                guess = c1 if d1 < d2 else c2

                correct += (guess == truth)
                total += 1

    return correct / total


def commutator_test(model, q, actual, idx):

    device = next(model.parameters()).device
    W = model.W()

    predicted_scores = []
    empirical_scores = []

    # ABC vs BAC:
    # W_C (W_B W_A - W_A W_B)
    operators = [
        (
            "ABC","BAC",
            W["C"] @ (
                W["B"] @ W["A"]
                - W["A"] @ W["B"]
            )
        ),
        (
            "ACB","CAB",
            W["B"] @ (
                W["C"] @ W["A"]
                - W["A"] @ W["C"]
            )
        ),
        (
            "BCA","CBA",
            W["A"] @ (
                W["C"] @ W["B"]
                - W["B"] @ W["C"]
            )
        ),
    ]

    with torch.no_grad():
        for c1,c2,K in operators:

            K = K.cpu().numpy()

            for i in idx:
                qi = q[i]

                predicted_scores.append(
                    float(np.sum((K @ qi)**2))
                )

                empirical_scores.append(
                    float(np.sum(
                        (actual[c1][i]
                         - actual[c2][i])**2
                    ))
                )

    rho, p = spearmanr(
        predicted_scores,
        empirical_scores
    )

    pred_arr = np.array(predicted_scores)
    emp_arr = np.array(empirical_scores)

    k = max(1, math.ceil(0.20*len(emp_arr)))
    top = np.argsort(pred_arr)[-k:]

    lift = (
        emp_arr[top].mean()
        / max(emp_arr.mean(),1e-12)
    )

    return float(rho), float(p), float(lift)


def mse(pred, actual, idx, comps):
    vals = []
    for c in comps:
        vals.append(
            np.mean(
                (pred[c][idx]
                 - actual[c][idx])**2
            )
        )
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-prompts", type=int, default=60)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--model",
                    default="gpt-4o-mini-2024-07-18")
    ap.add_argument("--max-tokens", type=int, default=180)
    ap.add_argument(
        "--cache",
        default="artifacts/operator3_poc/api_runs.jsonl"
    )
    args = ap.parse_args()

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    tasks = load_v11_probes(args.n_prompts)

    print(f"[V11] using {len(tasks)} probes")

    rows = asyncio.run(
        collect(args,tasks)
    )

    inp = sum(r["input_tokens"] for r in rows)
    out = sum(r["output_tokens"] for r in rows)

    cost = (
        inp/1_000_000*0.15
        + out/1_000_000*0.60
    )

    print("\n===== API USAGE =====")
    print(f"Input tokens:  {inp:,}")
    print(f"Output tokens: {out:,}")
    print(f"Estimated cost: ${cost:.4f}")

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:",device)

    enc = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device=device
    )

    q0 = enc.encode(
        tasks,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=64
    )

    all_comps = ["A","B","C"] + PAIRS + TRIPLES

    z0 = {}

    for comp in all_comps:
        texts = [
            r["outputs"][comp]
            for r in rows
        ]

        z0[comp] = enc.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=64
        )

    ids = np.arange(len(tasks))

    tr,te = train_test_split(
        ids,
        train_size=.70,
        random_state=42,
        shuffle=True
    )

    # PCA sees only task vectors and PAIR outputs
    # from TRAIN tasks. No triple outputs leak into
    # the pure pairwise->triple experiment.
    pca_data = [q0[tr]]
    for c in PAIRS:
        pca_data.append(z0[c][tr])

    pca_data = np.vstack(pca_data)

    pca = PCA(
        n_components=16,
        random_state=42
    )
    pca.fit(pca_data)

    q = l2(pca.transform(q0))

    z = {
        c:l2(pca.transform(z0[c]))
        for c in all_comps
    }

    print(
        f"\n[split] train={len(tr)} "
        f"test={len(te)} dim=16"
    )

    # =====================================================
    # TEST 1:
    # Learn ONLY from all six 2-agent compositions.
    # Predict ALL six unseen 3-agent compositions.
    # =====================================================

    print("\n========================================")
    print("TEST 1: PAIRWISE -> UNSEEN TRIPLES")
    print("Train:", PAIRS)
    print("Test :", TRIPLES)
    print("========================================")

    model1, loss1 = fit_model(
        q,z,tr,PAIRS,device=device
    )

    pred1 = predictions(
        model1,q,TRIPLES,device
    )

    mse1 = mse(
        pred1,z,te,TRIPLES
    )

    acc6,meanrank = retrieval_accuracy(
        pred1,z,te,
        TRIPLES,TRIPLES
    )

    sameacc = same_terminal_accuracy(
        pred1,z,te
    )

    rho,p,lift = commutator_test(
        model1,q,z,te
    )

    print(f"Training loss: {loss1:.6f}")
    print(
        "Unseen triple prediction MSE:",
        f"{mse1:.6f}"
    )
    print(
        "6-way execution-order retrieval:",
        f"{acc6*100:.1f}% "
        "(chance=16.7%)"
    )
    print(
        "Mean correct-order rank:",
        f"{meanrank:.2f}/6"
    )
    print(
        "Same-terminal order accuracy:",
        f"{sameacc*100:.1f}% "
        "(chance=50%)"
    )
    print(
        "Commutator separation:",
        f"rho={rho:+.3f}, "
        f"p={p:.4g}, "
        f"top20 lift={lift:.3f}x"
    )

    # =====================================================
    # ORACLE LINEAR REFERENCE
    #
    # Separate direct map for each triple composition.
    # This HAS seen triple outputs and therefore is NOT
    # compositional. It is only an upper/reference bound.
    # =====================================================

    oracle = {}

    for c in TRIPLES:
        r = Ridge(
            alpha=3.0,
            fit_intercept=False
        )
        r.fit(q[tr],z[c][tr])
        oracle[c] = r.predict(q)

    omse = mse(
        oracle,z,te,TRIPLES
    )

    oacc,orank = retrieval_accuracy(
        oracle,z,te,
        TRIPLES,TRIPLES
    )

    osame = same_terminal_accuracy(
        oracle,z,te
    )

    print("\n--- ORACLE DIRECT-MAP REFERENCE ---")
    print(
        "Triple-specific linear MSE:",
        f"{omse:.6f}"
    )
    print(
        "6-way retrieval:",
        f"{oacc*100:.1f}%"
    )
    print(
        "Same-terminal:",
        f"{osame*100:.1f}%"
    )

    # =====================================================
    # TEST 2:
    # Give operator model a small amount of 3-agent
    # calibration. Train on cyclic permutations and test
    # the reverse cyclic permutations.
    # =====================================================

    seen_triples = ["ABC","BCA","CAB"]
    heldout_triples = ["ACB","BAC","CBA"]

    train_comps = PAIRS + seen_triples

    print("\n========================================")
    print("TEST 2: COMPOSITIONAL CALIBRATION")
    print("Triple train:",seen_triples)
    print("Triple heldout:",heldout_triples)
    print("========================================")

    model2,loss2 = fit_model(
        q,z,tr,train_comps,device=device
    )

    pred2 = predictions(
        model2,q,TRIPLES,device
    )

    mse2 = mse(
        pred2,z,te,heldout_triples
    )

    # Truths are heldout, candidates are ALL 6 orders.
    acc2,rank2 = retrieval_accuracy(
        pred2,z,te,
        heldout_triples,
        TRIPLES
    )

    rho2,p2,lift2 = commutator_test(
        model2,q,z,te
    )

    print(f"Training loss: {loss2:.6f}")
    print(
        "Heldout-composition MSE:",
        f"{mse2:.6f}"
    )
    print(
        "Heldout order retrieval "
        "among all 6:",
        f"{acc2*100:.1f}% "
        "(chance=16.7%)"
    )
    print(
        "Mean heldout rank:",
        f"{rank2:.2f}/6"
    )
    print(
        "Commutator separation:",
        f"rho={rho2:+.3f}, "
        f"p={p2:.4g}, "
        f"top20 lift={lift2:.3f}x"
    )

    print("\n========================================")
    print("INTERPRETATION")
    print("========================================")

    if sameacc > 0.60 and acc6 > 0.25:
        print(
            "PROMISING: pairwise-trained operators "
            "show non-trivial unseen composition signal."
        )
    else:
        print(
            "WEAK/NO SIGNAL: pairwise operator "
            "composition does not yet generalize strongly."
        )

    if rho > 0.15 and lift > 1.10:
        print(
            "PROMISING COMMUTATOR: spectral order "
            "score predicts real same-terminal separation."
        )
    else:
        print(
            "COMMUTATOR NOT ESTABLISHED: "
            "needs a better representation/model."
        )


if __name__ == "__main__":
    main()
