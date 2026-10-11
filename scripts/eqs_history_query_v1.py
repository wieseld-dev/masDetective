import argparse
import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# UTILS
# ==============================================================================

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def entropy(p):
    p = p.clamp_min(1e-9)
    return -(p * p.log()).sum(dim=-1)


# ==============================================================================
# SYNTHETIC HIDDEN-SYSTEM ENVIRONMENT
#
# Hidden hypothesis H
# Query q
# Stochastic binary response y
#
# The model NEVER sees the true latent query geometry.
# It only sees raw query features + interaction history.
# ==============================================================================

class SyntheticWorld:
    def __init__(
        self,
        n_hypotheses=8,
        n_queries=192,
        true_dim=10,
        raw_dim=24,
        scale=3.0,
        seed=0,
    ):
        g = torch.Generator().manual_seed(seed)

        self.n_h = n_hypotheses
        self.n_q = n_queries

        H = F.normalize(
            torch.randn(n_hypotheses, true_dim, generator=g),
            dim=-1
        )

        Q = F.normalize(
            torch.randn(n_queries, true_dim, generator=g),
            dim=-1
        )

        # Hide the true query geometry behind another representation.
        A = torch.randn(raw_dim, true_dim, generator=g) / math.sqrt(true_dim)

        self.raw_q = (
            Q @ A.T
            + 0.03 * torch.randn(n_queries, raw_dim, generator=g)
        )

        bias = 0.20 * torch.randn(n_hypotheses, generator=g)

        logits = scale * (H @ Q.T) + bias[:, None]

        # P(response=1 | hidden hypothesis, query)
        self.probs = torch.sigmoid(logits)

    def observe(self, hidden_h, query_id, generator):
        p = self.probs[hidden_h, query_id]
        return int(torch.bernoulli(p, generator=generator).item())

    def update_belief(self, belief, query_id, response):
        if response == 1:
            likelihood = self.probs[:, query_id]
        else:
            likelihood = 1.0 - self.probs[:, query_id]

        posterior = belief * likelihood
        posterior = posterior / posterior.sum().clamp_min(1e-12)

        return posterior

    def eig(self, belief):
        """
        Exact Expected Information Gain for every possible query.
        Used ONLY as teacher/oracle during synthetic training/testing.
        """

        p1 = (belief[:, None] * self.probs).sum(dim=0)
        p0 = 1.0 - p1

        post1 = belief[:, None] * self.probs
        post1 = post1 / post1.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-12)

        post0 = belief[:, None] * (1.0 - self.probs)
        post0 = post0 / post0.sum(
            dim=0, keepdim=True
        ).clamp_min(1e-12)

        current_entropy = entropy(belief)

        expected_future_entropy = (
            p1 * entropy(post1.T)
            + p0 * entropy(post0.T)
        )

        return current_entropy - expected_future_entropy


# ==============================================================================
# DATA
# ==============================================================================

@dataclass
class StateExample:
    queries: list
    responses: list
    hidden: int
    eig: torch.Tensor


class StateDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def make_examples(
    world,
    candidate_ids,
    n_episodes=2200,
    max_history=6,
    seed=1,
):
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed + 17)

    examples = []

    for _ in range(n_episodes):

        hidden = rng.randrange(world.n_h)

        belief = torch.full(
            (world.n_h,),
            1.0 / world.n_h
        )

        q_hist = []
        y_hist = []

        target_length = rng.randrange(max_history + 1)

        for t in range(target_length + 1):

            exact_eig = world.eig(belief)

            examples.append(
                StateExample(
                    queries=q_hist.copy(),
                    responses=y_hist.copy(),
                    hidden=hidden,
                    eig=exact_eig.clone(),
                )
            )

            if t == target_length:
                break

            available = [
                q for q in candidate_ids
                if q not in q_hist
            ]

            # Mixed behavior policy:
            # often informative, sometimes random.
            # Prevents training only on oracle trajectories.
            if rng.random() < 0.55:

                vals = exact_eig[available]
                k = min(10, len(available))

                top_local = torch.topk(
                    vals, k
                ).indices.tolist()

                query = available[
                    rng.choice(top_local)
                ]

            else:
                query = rng.choice(available)

            response = world.observe(
                hidden,
                query,
                gen
            )

            q_hist.append(query)
            y_hist.append(response)

            belief = world.update_belief(
                belief,
                query,
                response
            )

    return examples


def collate(batch, world, max_history):

    B = len(batch)
    raw_dim = world.raw_q.shape[1]

    # Each historical step =
    # query representation + one-hot binary answer.
    x = torch.zeros(
        B,
        max_history,
        raw_dim + 2
    )

    mask = torch.zeros(
        B,
        max_history,
        dtype=torch.bool
    )

    used = torch.zeros(
        B,
        world.n_q,
        dtype=torch.bool
    )

    hidden = torch.tensor(
        [e.hidden for e in batch],
        dtype=torch.long
    )

    eig = torch.stack(
        [e.eig for e in batch]
    )

    for b, example in enumerate(batch):

        for t, (q, y) in enumerate(
            zip(
                example.queries[:max_history],
                example.responses[:max_history]
            )
        ):
            x[b, t, :raw_dim] = world.raw_q[q]

            # answer position
            x[b, t, raw_dim + y] = 1.0

            mask[b, t] = True
            used[b, q] = True

    return x, mask, hidden, eig, used


# ==============================================================================
# ARCHITECTURE
#
# History -> Information Need Vector d_t
# Query   -> Query Vector z_q
#
# Fast retrieval:
#       score = d_t dot z_q
#
# Then a small nonlinear reranker considers only Top-K.
# ==============================================================================

class EQSModel(nn.Module):

    def __init__(
        self,
        raw_dim,
        n_hypotheses,
        state_dim=128,
        query_dim=64,
    ):
        super().__init__()

        self.h0 = nn.Parameter(
            torch.zeros(state_dim)
        )

        # Encode each historical query-answer pair.
        self.step_encoder = nn.Sequential(
            nn.Linear(raw_dim + 2, state_dim),
            nn.GELU(),
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
        )

        # Sequential history state.
        self.history_gru = nn.GRUCell(
            state_dim,
            state_dim
        )

        # History -> "what information do I need next?"
        self.information_need = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Linear(state_dim, query_dim),
        )

        # Natural/raw query representation -> latent query vector.
        self.query_encoder = nn.Sequential(
            nn.Linear(raw_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, query_dim),
        )

        # Auxiliary belief prediction.
        # Makes history representation preserve hidden-state information.
        self.belief_head = nn.Linear(
            state_dim,
            n_hypotheses
        )

        # Used by nonlinear Top-K reranker.
        self.state_to_query = nn.Linear(
            state_dim,
            query_dim
        )

        self.reranker = nn.Sequential(
            nn.Linear(4 * query_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(10.0))
        )

    def encode_history(self, x, mask):

        B, T, _ = x.shape

        h = self.h0[None, :].expand(B, -1)

        tokens = self.step_encoder(x)

        for t in range(T):

            h_new = self.history_gru(
                tokens[:, t],
                h
            )

            h = torch.where(
                mask[:, t, None],
                h_new,
                h
            )

        return h

    def query_embeddings(self, raw_q):

        return F.normalize(
            self.query_encoder(raw_q),
            dim=-1
        )

    def retrieval_logits(self, history_state, query_embeddings):

        direction = F.normalize(
            self.information_need(history_state),
            dim=-1
        )

        scale = self.logit_scale.exp().clamp(
            max=100
        )

        # Vector retrieval:
        #
        # scores = Z d_t
        #
        return scale * (
            direction @ query_embeddings.T
        )

    def rerank_scores(
        self,
        history_state,
        candidate_embeddings
    ):

        direction = F.normalize(
            self.state_to_query(history_state),
            dim=-1
        )

        direction = (
            direction[:, None, :]
            .expand_as(candidate_embeddings)
        )

        features = torch.cat(
            [
                direction,
                candidate_embeddings,
                direction * candidate_embeddings,
                (direction - candidate_embeddings).abs(),
            ],
            dim=-1
        )

        return self.reranker(
            features
        ).squeeze(-1)


def soft_cross_entropy(logits, target_distribution):

    return -(
        target_distribution
        * F.log_softmax(logits, dim=-1)
    ).sum(dim=-1).mean()


# ==============================================================================
# QUERY SELECTION
# ==============================================================================

@torch.no_grad()
def choose_query(
    model,
    world,
    q_history,
    y_history,
    candidate_ids,
    max_history=6,
    rerank_k=16,
):

    model.eval()

    example = StateExample(
        queries=q_history,
        responses=y_history,
        hidden=0,
        eig=torch.empty(0),
    )

    x, mask, _, _, used = collate(
        [example],
        world,
        max_history
    )

    history_state = model.encode_history(
        x,
        mask
    )

    query_embeddings = model.query_embeddings(
        world.raw_q
    )

    scores = model.retrieval_logits(
        history_state,
        query_embeddings
    )[0]

    candidate_mask = torch.zeros(
        world.n_q,
        dtype=torch.bool
    )

    candidate_mask[candidate_ids] = True

    valid = (
        candidate_mask
        & (~used[0])
    )

    scores = scores.masked_fill(
        ~valid,
        -1e9
    )

    K = min(
        rerank_k,
        int(valid.sum().item())
    )

    top = torch.topk(
        scores,
        K
    ).indices

    rerank = model.rerank_scores(
        history_state,
        query_embeddings[top][None, :, :]
    )[0]

    retrieval_top = scores[top]

    retrieval_norm = (
        retrieval_top - retrieval_top.mean()
    ) / (
        retrieval_top.std() + 1e-6
    )

    rerank_norm = (
        rerank - rerank.mean()
    ) / (
        rerank.std() + 1e-6
    )

    combined = (
        retrieval_norm
        + rerank_norm
    )

    return int(
        top[combined.argmax()]
    )


# ==============================================================================
# EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate_states(
    model,
    world,
    examples,
    candidate_ids,
    max_states=1000,
    max_history=6,
):

    model.eval()

    examples = examples[:max_states]

    loader = DataLoader(
        StateDataset(examples),
        batch_size=192,
        shuffle=False,
        collate_fn=lambda b:
            collate(b, world, max_history)
    )

    global_mask = torch.zeros(
        world.n_q,
        dtype=torch.bool
    )

    global_mask[candidate_ids] = True

    query_embeddings = model.query_embeddings(
        world.raw_q
    )

    ratios = []
    top10_hits = []

    for x, mask, _, eig, used in loader:

        history = model.encode_history(
            x,
            mask
        )

        scores = model.retrieval_logits(
            history,
            query_embeddings
        )

        valid = (
            global_mask[None, :]
            & (~used)
        )

        scores = scores.masked_fill(
            ~valid,
            -1e9
        )

        K = min(
            16,
            len(candidate_ids)
        )

        top = torch.topk(
            scores,
            K,
            dim=-1
        ).indices

        candidate_embeddings = query_embeddings[top]

        rerank = model.rerank_scores(
            history,
            candidate_embeddings
        )

        retrieval = torch.gather(
            scores,
            1,
            top
        )

        retrieval_norm = (
            retrieval
            - retrieval.mean(1, keepdim=True)
        ) / (
            retrieval.std(1, keepdim=True)
            + 1e-6
        )

        rerank_norm = (
            rerank
            - rerank.mean(1, keepdim=True)
        ) / (
            rerank.std(1, keepdim=True)
            + 1e-6
        )

        best_local = (
            retrieval_norm + rerank_norm
        ).argmax(dim=1)

        chosen = top[
            torch.arange(top.shape[0]),
            best_local
        ]

        obtained_utility = torch.gather(
            eig,
            1,
            chosen[:, None]
        ).squeeze(1)

        masked_eig = eig.masked_fill(
            ~valid,
            -1e9
        )

        oracle_utility = masked_eig.max(
            dim=1
        ).values

        ratio = (
            obtained_utility
            / (oracle_utility + 1e-9)
        )

        ratios.extend(
            ratio.tolist()
        )

        k10 = max(
            1,
            math.ceil(
                len(candidate_ids) * 0.10
            )
        )

        threshold = torch.topk(
            masked_eig,
            min(k10, masked_eig.shape[1]),
            dim=1
        ).values[:, -1]

        top10_hits.extend(
            (
                obtained_utility
                >= threshold
            ).float().tolist()
        )

    return {
        "utility_ratio_vs_oracle":
            float(np.mean(ratios)),
        "top10_percent_hit_rate":
            float(np.mean(top10_hits)),
    }


@torch.no_grad()
def evaluate_adaptive_policy(
    model,
    world,
    candidate_ids,
    steps=5,
    n_episodes=400,
    seed=99,
):

    rng = random.Random(seed)

    gen = torch.Generator().manual_seed(
        seed + 9
    )

    modes = [
        "random",
        "learned",
        "oracle"
    ]

    results = {
        m: {
            "accuracy": 0.0,
            "entropy": 0.0,
        }
        for m in modes
    }

    for mode in modes:

        for _ in range(n_episodes):

            hidden = rng.randrange(
                world.n_h
            )

            belief = torch.full(
                (world.n_h,),
                1.0 / world.n_h
            )

            q_hist = []
            y_hist = []

            for _ in range(steps):

                available = [
                    q for q in candidate_ids
                    if q not in q_hist
                ]

                if mode == "random":

                    query = rng.choice(
                        available
                    )

                elif mode == "oracle":

                    exact_eig = world.eig(
                        belief
                    )

                    query = max(
                        available,
                        key=lambda q:
                            float(
                                exact_eig[q]
                            )
                    )

                else:

                    query = choose_query(
                        model,
                        world,
                        q_hist,
                        y_hist,
                        candidate_ids
                    )

                response = world.observe(
                    hidden,
                    query,
                    gen
                )

                q_hist.append(query)
                y_hist.append(response)

                belief = world.update_belief(
                    belief,
                    query,
                    response
                )

            results[mode]["accuracy"] += (
                float(
                    belief.argmax().item()
                    == hidden
                )
                / n_episodes
            )

            results[mode]["entropy"] += (
                float(entropy(belief))
                / n_episodes
            )

    return results


# ==============================================================================
# TRAIN
# ==============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=8
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=2200
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7
    )

    parser.add_argument(
        "--save",
        default="artifacts/eqs_v1/model.pt"
    )

    parser.add_argument(
        "--results",
        default="artifacts/eqs_v1/results.json"
    )

    args = parser.parse_args()

    seed_all(args.seed)

    world = SyntheticWorld(
        seed=args.seed
    )

    # ----------------------------------------------------------
    # Hold out 25% of query actions entirely.
    #
    # This tests whether the learned geometry generalizes to
    # QUERY ACTIONS not used for training.
    # ----------------------------------------------------------

    query_ids = list(
        range(world.n_q)
    )

    random.Random(
        args.seed
    ).shuffle(query_ids)

    split = int(
        0.75 * len(query_ids)
    )

    train_query_ids = query_ids[:split]
    heldout_query_ids = query_ids[split:]

    examples = make_examples(
        world,
        train_query_ids,
        n_episodes=args.episodes,
        max_history=6,
        seed=args.seed + 1,
    )

    random.Random(
        args.seed
    ).shuffle(examples)

    train_split = int(
        0.85 * len(examples)
    )

    train_examples = examples[
        :train_split
    ]

    val_examples = examples[
        train_split:
    ]

    print(
        f"train_states={len(train_examples)} "
        f"val_states={len(val_examples)} "
        f"train_queries={len(train_query_ids)} "
        f"heldout_queries={len(heldout_query_ids)}"
    )

    model = EQSModel(
        raw_dim=world.raw_q.shape[1],
        n_hypotheses=world.n_h,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-3,
        weight_decay=1e-4
    )

    loader = DataLoader(
        StateDataset(train_examples),
        batch_size=96,
        shuffle=True,
        collate_fn=lambda b:
            collate(b, world, 6)
    )

    train_query_mask = torch.zeros(
        world.n_q,
        dtype=torch.bool
    )

    train_query_mask[
        train_query_ids
    ] = True

    best_ratio = -1.0
    best_state = None

    for epoch in range(
        1,
        args.epochs + 1
    ):

        model.train()

        total_loss = 0.0

        for (
            x,
            mask,
            hidden,
            eig,
            used
        ) in loader:

            history_state = model.encode_history(
                x,
                mask
            )

            query_embeddings = model.query_embeddings(
                world.raw_q
            )

            retrieval_logits = model.retrieval_logits(
                history_state,
                query_embeddings
            )

            valid = (
                train_query_mask[None, :]
                & (~used)
            )

            retrieval_logits = (
                retrieval_logits.masked_fill(
                    ~valid,
                    -1e9
                )
            )

            # --------------------------------------------------
            # Soft oracle target.
            #
            # Instead of saying only one query is correct,
            # preserve relative utility among candidate queries.
            # --------------------------------------------------

            oracle_distribution = F.softmax(
                eig.masked_fill(
                    ~valid,
                    -1e9
                ) / 0.035,
                dim=-1
            )

            retrieval_loss = soft_cross_entropy(
                retrieval_logits,
                oracle_distribution
            )

            belief_loss = F.cross_entropy(
                model.belief_head(
                    history_state
                ),
                hidden
            )

            # --------------------------------------------------
            # Candidate shortlist for expensive nonlinear
            # reranker.
            # --------------------------------------------------

            with torch.no_grad():

                shortlist_score = (
                    retrieval_logits
                    + 6.0
                    * eig.masked_fill(
                        ~valid,
                        -1e9
                    )
                )

                candidate_ids = torch.topk(
                    shortlist_score,
                    min(
                        32,
                        len(train_query_ids)
                    ),
                    dim=1
                ).indices

            candidate_embeddings = (
                query_embeddings[
                    candidate_ids
                ]
            )

            rerank_scores = model.rerank_scores(
                history_state,
                candidate_embeddings
            )

            candidate_utility = torch.gather(
                eig,
                1,
                candidate_ids
            )

            rerank_target = F.softmax(
                candidate_utility / 0.025,
                dim=-1
            )

            rerank_loss = soft_cross_entropy(
                rerank_scores,
                rerank_target
            )

            loss = (
                retrieval_loss
                + 0.35 * belief_loss
                + 0.55 * rerank_loss
            )

            optimizer.zero_grad()

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * x.shape[0]
            )

        validation = evaluate_states(
            model,
            world,
            val_examples,
            train_query_ids,
            max_states=1000,
        )

        print(
            f"epoch={epoch:02d} "
            f"loss={total_loss / len(train_examples):.4f} "
            f"val_ratio={validation['utility_ratio_vs_oracle']:.4f} "
            f"top10={validation['top10_percent_hit_rate']:.4f}"
        )

        if (
            validation[
                "utility_ratio_vs_oracle"
            ] > best_ratio
        ):

            best_ratio = validation[
                "utility_ratio_vs_oracle"
            ]

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

    model.load_state_dict(
        best_state
    )

    # ----------------------------------------------------------
    # FINAL TESTS
    # ----------------------------------------------------------

    seen_results = evaluate_states(
        model,
        world,
        val_examples,
        train_query_ids,
        max_states=1200,
    )

    heldout_results = evaluate_states(
        model,
        world,
        val_examples,
        heldout_query_ids,
        max_states=1200,
    )

    adaptive_results = (
        evaluate_adaptive_policy(
            model,
            world,
            train_query_ids,
            steps=5,
            n_episodes=400,
            seed=args.seed + 99,
        )
    )

    results = {
        "seen_query_ranking":
            seen_results,

        "heldout_query_ranking":
            heldout_results,

        "five_question_policy":
            adaptive_results,
    }

    print()
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)
    print(
        json.dumps(
            results,
            indent=2
        )
    )

    os.makedirs(
        os.path.dirname(args.save),
        exist_ok=True
    )

    torch.save(
        {
            "model":
                model.state_dict(),

            "raw_queries":
                world.raw_q,

            "train_query_ids":
                train_query_ids,

            "heldout_query_ids":
                heldout_query_ids,

            "config":
                vars(args),
        },
        args.save
    )

    os.makedirs(
        os.path.dirname(args.results),
        exist_ok=True
    )

    with open(
        args.results,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )

    print()
    print(
        "MODEL SAVED:",
        args.save
    )

    print(
        "RESULTS SAVED:",
        args.results
    )


if __name__ == "__main__":
    main()
