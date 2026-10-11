import argparse, json, math, os, random, statistics
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def entropy(p):
    p = p.clamp_min(1e-9)
    return -(p * p.log()).sum(dim=-1)


class World:
    def __init__(self, probs):
        self.probs = probs.float()
        self.n_h, self.n_q = self.probs.shape

    def observe_with_u(self, hidden, q, u):
        return int(float(u) < float(self.probs[hidden, q]))

    def observe(self, hidden, q, generator):
        p = self.probs[hidden, q]
        return int(torch.bernoulli(p, generator=generator).item())

    def update_belief(self, belief, q, y):
        likelihood = self.probs[:, q] if y == 1 else (1.0 - self.probs[:, q])
        post = belief * likelihood
        return post / post.sum().clamp_min(1e-12)

    def eig(self, belief):
        p1 = (belief[:, None] * self.probs).sum(0)
        p0 = 1.0 - p1
        post1 = belief[:, None] * self.probs
        post1 = post1 / post1.sum(0, keepdim=True).clamp_min(1e-12)
        post0 = belief[:, None] * (1.0 - self.probs)
        post0 = post0 / post0.sum(0, keepdim=True).clamp_min(1e-12)
        return entropy(belief) - (p1 * entropy(post1.T) + p0 * entropy(post0.T))


class MultiWorldFamily:
    def __init__(self, n_h=8, n_q=256, latent_dim=16, raw_dim=32, master_seed=1234):
        g = torch.Generator().manual_seed(master_seed)
        self.n_h = n_h
        self.n_q = n_q
        self.latent_dim = latent_dim
        self.raw_dim = raw_dim
        self.base_h = F.normalize(torch.randn(n_h, latent_dim, generator=g), dim=-1)
        self.base_q = F.normalize(torch.randn(n_q, latent_dim, generator=g), dim=-1)
        A = torch.randn(raw_dim, latent_dim, generator=g) / math.sqrt(latent_dim)
        self.raw_q = self.base_q @ A.T + 0.025 * torch.randn(n_q, raw_dim, generator=g)

    def make_world(self, seed, difficulty="id"):
        g = torch.Generator().manual_seed(seed)
        D = self.latent_dim
        rank = 4
        U = torch.randn(D, rank, generator=g) / math.sqrt(D)
        V = torch.randn(D, rank, generator=g) / math.sqrt(D)
        if difficulty == "ood":
            alpha = 0.35 + 0.30 * float(torch.rand((), generator=g))
            h_noise = 0.10 + 0.10 * float(torch.rand((), generator=g))
            q_noise = 0.08 + 0.08 * float(torch.rand((), generator=g))
            nonlinear_weight = 0.45
        else:
            alpha = 0.10 + 0.25 * float(torch.rand((), generator=g))
            h_noise = 0.04 + 0.08 * float(torch.rand((), generator=g))
            q_noise = 0.03 + 0.06 * float(torch.rand((), generator=g))
            nonlinear_weight = 0.25
        M = torch.eye(D) + alpha * (U @ V.T)
        H = F.normalize(self.base_h + h_noise * torch.randn(self.n_h, D, generator=g), dim=-1)
        Q = F.normalize(self.base_q @ M.T + q_noise * torch.randn(self.n_q, D, generator=g), dim=-1)
        # Nonlinear world-specific low-rank interaction
        R1 = torch.randn(D, rank, generator=g) / math.sqrt(D)
        R2 = torch.randn(D, rank, generator=g) / math.sqrt(D)
        nonlinear = torch.tanh(H @ R1) @ torch.tanh(Q @ R2).T
        scale = 2.4 + 1.8 * float(torch.rand((), generator=g))
        h_bias = 0.20 * torch.randn(self.n_h, generator=g)
        q_bias = 0.10 * torch.randn(self.n_q, generator=g)
        logits = scale * (H @ Q.T) + nonlinear_weight * nonlinear + h_bias[:, None] + q_bias[None, :]
        return World(torch.sigmoid(logits))


@dataclass
class StateExample:
    queries: list
    responses: list
    hidden: int
    eig: torch.Tensor


class StateDataset(Dataset):
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


def make_examples(world, candidate_ids, n_episodes, max_history, seed):
    rng = random.Random(seed)
    gen = torch.Generator().manual_seed(seed + 100003)
    rows = []
    for _ in range(n_episodes):
        hidden = rng.randrange(world.n_h)
        belief = torch.full((world.n_h,), 1.0 / world.n_h)
        qs, ys = [], []
        target_len = rng.randrange(max_history + 1)
        for t in range(target_len + 1):
            eig = world.eig(belief)
            rows.append(StateExample(qs.copy(), ys.copy(), hidden, eig.clone()))
            if t == target_len: break
            available = [q for q in candidate_ids if q not in qs]
            r = rng.random()
            if r < 0.55:
                vals = eig[available]
                topk = min(12, len(available))
                top_local = torch.topk(vals, topk).indices.tolist()
                q = available[rng.choice(top_local)]
            elif r < 0.80:
                # Boltzmann exploration around informative actions.
                vals = eig[available]
                probs = torch.softmax(vals / 0.04, dim=0)
                q = available[int(torch.multinomial(probs, 1, generator=gen))]
            else:
                q = rng.choice(available)
            y = world.observe(hidden, q, gen)
            qs.append(q); ys.append(y)
            belief = world.update_belief(belief, q, y)
    return rows


def collate(batch, family, max_history):
    B = len(batch); raw_dim = family.raw_dim
    x = torch.zeros(B, max_history, raw_dim + 2)
    mask = torch.zeros(B, max_history, dtype=torch.bool)
    used = torch.zeros(B, family.n_q, dtype=torch.bool)
    hidden = torch.tensor([e.hidden for e in batch], dtype=torch.long)
    eig = torch.stack([e.eig for e in batch])
    for b, e in enumerate(batch):
        for t, (q, y) in enumerate(zip(e.queries[:max_history], e.responses[:max_history])):
            x[b, t, :raw_dim] = family.raw_q[q]
            x[b, t, raw_dim + y] = 1.0
            mask[b, t] = True
            used[b, q] = True
    return x, mask, hidden, eig, used


class EQSModel(nn.Module):
    def __init__(self, raw_dim, n_h, state_dim=192, query_dim=96):
        super().__init__()
        self.h0 = nn.Parameter(torch.zeros(state_dim))
        self.step_encoder = nn.Sequential(
            nn.Linear(raw_dim + 2, state_dim), nn.GELU(), nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim), nn.GELU()
        )
        self.history_gru = nn.GRUCell(state_dim, state_dim)
        self.history_norm = nn.LayerNorm(state_dim)
        self.belief_head = nn.Linear(state_dim, n_h)
        self.information_need = nn.Sequential(
            nn.Linear(state_dim + n_h, state_dim), nn.GELU(), nn.Linear(state_dim, query_dim)
        )
        self.query_encoder = nn.Sequential(
            nn.Linear(raw_dim, 192), nn.GELU(), nn.LayerNorm(192),
            nn.Linear(192, query_dim)
        )
        self.state_to_query = nn.Linear(state_dim + n_h, query_dim)
        self.reranker = nn.Sequential(
            nn.Linear(4 * query_dim, 192), nn.GELU(),
            nn.Linear(192, 96), nn.GELU(), nn.Linear(96, 1)
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def encode_history(self, x, mask):
        B, T, _ = x.shape
        h = self.h0[None].expand(B, -1)
        tok = self.step_encoder(x)
        for t in range(T):
            hn = self.history_gru(tok[:, t], h)
            h = torch.where(mask[:, t, None], hn, h)
        return self.history_norm(h)

    def query_embeddings(self, raw_q):
        return F.normalize(self.query_encoder(raw_q), dim=-1)

    def decision_state(self, h):
        b = F.softmax(self.belief_head(h), dim=-1)
        return torch.cat([h, b], dim=-1)

    def retrieval_logits(self, h, zq):
        ds = self.decision_state(h)
        d = F.normalize(self.information_need(ds), dim=-1)
        return self.logit_scale.exp().clamp(max=100) * (d @ zq.T)

    def rerank_scores(self, h, zcand):
        ds = self.decision_state(h)
        d = F.normalize(self.state_to_query(ds), dim=-1)[:, None, :].expand_as(zcand)
        f = torch.cat([d, zcand, d * zcand, (d - zcand).abs()], dim=-1)
        return self.reranker(f).squeeze(-1)


def soft_ce(logits, target):
    return -(target * F.log_softmax(logits, dim=-1)).sum(-1).mean()


@torch.no_grad()
def rank_batch(model, family, batch, candidate_ids, rerank_k=24):
    x, mask, _, eig, used = batch
    h = model.encode_history(x, mask)
    zq = model.query_embeddings(family.raw_q)
    scores = model.retrieval_logits(h, zq)
    allowed = torch.zeros(family.n_q, dtype=torch.bool)
    allowed[candidate_ids] = True
    valid = allowed[None] & (~used)
    scores = scores.masked_fill(~valid, -1e9)
    K = min(rerank_k, len(candidate_ids))
    top = torch.topk(scores, K, dim=1).indices
    zr = zq[top]
    rr = model.rerank_scores(h, zr)
    ret = torch.gather(scores, 1, top)
    ret_n = (ret - ret.mean(1, keepdim=True)) / (ret.std(1, keepdim=True) + 1e-6)
    rr_n = (rr - rr.mean(1, keepdim=True)) / (rr.std(1, keepdim=True) + 1e-6)
    best = (ret_n + rr_n).argmax(1)
    chosen = top[torch.arange(top.shape[0]), best]
    return chosen, eig, valid


@torch.no_grad()
def evaluate_ranking(model, family, examples, candidate_ids, max_history, max_states=5000):
    model.eval()
    rows = examples[:max_states]
    loader = DataLoader(StateDataset(rows), batch_size=384, shuffle=False,
                        collate_fn=lambda b: collate(b, family, max_history))
    ratios, hits = [], []
    for batch in loader:
        chosen, eig, valid = rank_batch(model, family, batch, candidate_ids)
        got = eig.gather(1, chosen[:, None]).squeeze(1)
        masked = eig.masked_fill(~valid, -1e9)
        opt = masked.max(1).values
        ratios.extend((got / (opt + 1e-9)).tolist())
        k10 = max(1, math.ceil(len(candidate_ids) * 0.10))
        threshold = torch.topk(masked, min(k10, masked.shape[1]), dim=1).values[:, -1]
        hits.extend((got >= threshold).float().tolist())
    return {
        "utility_ratio_vs_oracle": float(statistics.fmean(ratios)),
        "top10_percent_hit_rate": float(statistics.fmean(hits)),
        "n_states": len(rows),
    }


@torch.no_grad()
def choose_query(model, family, qs, ys, candidate_ids, zq_cache, max_history, rerank_k=24):
    e = StateExample(qs, ys, 0, torch.empty(0))
    x, mask, _, _, used = collate([e], family, max_history)
    h = model.encode_history(x, mask)
    s = model.retrieval_logits(h, zq_cache)[0]
    allowed = torch.zeros(family.n_q, dtype=torch.bool); allowed[candidate_ids] = True
    valid = allowed & (~used[0])
    s = s.masked_fill(~valid, -1e9)
    K = min(rerank_k, int(valid.sum()))
    top = torch.topk(s, K).indices
    rr = model.rerank_scores(h, zq_cache[top][None])[0]
    ret = s[top]
    ret_n = (ret - ret.mean()) / (ret.std() + 1e-6)
    rr_n = (rr - rr.mean()) / (rr.std() + 1e-6)
    return int(top[(ret_n + rr_n).argmax()])


def init_metrics(modes, checkpoints):
    return {m: {str(k): {"accuracy":0.0,"entropy":0.0,"nll":0.0,"brier":0.0,"count":0}
                for k in checkpoints} for m in modes}


def add_metrics(bucket, belief, hidden):
    K = belief.numel()
    onehot = torch.zeros(K); onehot[hidden] = 1.0
    bucket["accuracy"] += float(belief.argmax() == hidden)
    bucket["entropy"] += float(entropy(belief))
    bucket["nll"] += float(-belief[hidden].clamp_min(1e-9).log())
    bucket["brier"] += float(((belief - onehot) ** 2).mean())
    bucket["count"] += 1


def finalize_metrics(metrics):
    for m in metrics.values():
        for b in m.values():
            n = max(1, b.pop("count"))
            for key in list(b): b[key] /= n
    return metrics


@torch.no_grad()
def evaluate_rollouts(model, family, worlds, candidate_ids, static_scores, max_history,
                      n_episodes_per_world=80, steps=8, include_static=True, seed=7000):
    modes = ["random", "learned", "oracle"]
    if include_static: modes.insert(1, "static")
    checkpoints = sorted(set([1,2,3,5,steps]))
    metrics = init_metrics(modes, checkpoints)
    model.eval(); zq = model.query_embeddings(family.raw_q)
    rng = random.Random(seed)
    for wi, world in enumerate(worlds):
        gen = torch.Generator().manual_seed(seed + 1009 * wi)
        for ep in range(n_episodes_per_world):
            hidden = rng.randrange(world.n_h)
            # Common random numbers across policies.
            u = torch.rand(steps, family.n_q, generator=gen)
            for mode in modes:
                belief = torch.full((world.n_h,), 1.0 / world.n_h)
                qs, ys = [], []
                for t in range(steps):
                    avail = [q for q in candidate_ids if q not in qs]
                    if mode == "random": q = rng.choice(avail)
                    elif mode == "static": q = max(avail, key=lambda j: float(static_scores[j]))
                    elif mode == "oracle":
                        eig = world.eig(belief); q = max(avail, key=lambda j: float(eig[j]))
                    else:
                        q = choose_query(model, family, qs, ys, candidate_ids, zq, max_history)
                    y = world.observe_with_u(hidden, q, u[t, q])
                    qs.append(q); ys.append(y)
                    belief = world.update_belief(belief, q, y)
                    step = t + 1
                    if step in checkpoints: add_metrics(metrics[mode][str(step)], belief, hidden)
    return finalize_metrics(metrics)


def build_worlds(family, start_seed, n, difficulty="id"):
    return [family.make_world(start_seed + i * 7919, difficulty=difficulty) for i in range(n)]


def build_dataset(worlds, candidate_ids, episodes_per_world, max_history, base_seed):
    rows = []
    for i, w in enumerate(worlds):
        rows.extend(make_examples(w, candidate_ids, episodes_per_world, max_history, base_seed + i * 104729))
    return rows


def compute_static_scores(worlds):
    vals = []
    for w in worlds:
        b = torch.full((w.n_h,), 1.0 / w.n_h)
        vals.append(w.eig(b))
    return torch.stack(vals).mean(0)


def train_one(model_seed, args, family, train_worlds, dev_worlds, train_rows, dev_rows,
              train_query_ids, heldout_query_ids):
    seed_all(model_seed)
    model = EQSModel(family.raw_dim, family.n_h)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    loader_gen = torch.Generator().manual_seed(model_seed + 500)
    loader = DataLoader(StateDataset(train_rows), batch_size=args.batch_size, shuffle=True,
                        generator=loader_gen, collate_fn=lambda b: collate(b, family, args.max_history))
    train_mask = torch.zeros(family.n_q, dtype=torch.bool); train_mask[train_query_ids] = True
    best = -1.0; best_state = None
    for epoch in range(1, args.epochs + 1):
        model.train(); total = 0.0; seen = 0
        for x, mask, hidden, eig, used in loader:
            h = model.encode_history(x, mask)
            zq = model.query_embeddings(family.raw_q)
            logits = model.retrieval_logits(h, zq)
            valid = train_mask[None] & (~used)
            logits = logits.masked_fill(~valid, -1e9)
            target = F.softmax(eig.masked_fill(~valid, -1e9) / args.teacher_temp, dim=-1)
            retrieval_loss = soft_ce(logits, target)
            belief_loss = F.cross_entropy(model.belief_head(h), hidden)
            with torch.no_grad():
                # Half learned retrieval, half oracle utility -> hard but useful shortlist.
                shortlist = logits + 5.0 * eig.masked_fill(~valid, -1e9)
                ids = torch.topk(shortlist, min(args.train_rerank_k, len(train_query_ids)), dim=1).indices
            zcand = zq[ids]
            rr = model.rerank_scores(h, zcand)
            util = eig.gather(1, ids)
            rr_target = F.softmax(util / args.rerank_temp, dim=-1)
            rr_loss = soft_ce(rr, rr_target)
            loss = retrieval_loss + args.belief_weight * belief_loss + args.rerank_weight * rr_loss
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += float(loss.detach()) * x.shape[0]; seen += x.shape[0]
        # Unseen-world dev validation.
        dev_seen = evaluate_ranking(model, family, dev_rows, train_query_ids, args.max_history, args.dev_states)
        dev_unseenq = evaluate_ranking(model, family, dev_rows, heldout_query_ids, args.max_history, args.dev_states)
        score = 0.7 * dev_seen["utility_ratio_vs_oracle"] + 0.3 * dev_unseenq["utility_ratio_vs_oracle"]
        print(f"seed={model_seed} epoch={epoch:02d} loss={total/max(1,seen):.4f} "
              f"dev_world_seenQ={dev_seen['utility_ratio_vs_oracle']:.4f} "
              f"dev_world_unseenQ={dev_unseenq['utility_ratio_vs_oracle']:.4f} score={score:.4f}", flush=True)
        if score > best:
            best = score
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-seed", type=int, default=7)
    p.add_argument("--master-seed", type=int, default=1234)
    p.add_argument("--train-worlds", type=int, default=96)
    p.add_argument("--dev-worlds", type=int, default=24)
    p.add_argument("--test-worlds-id", type=int, default=32)
    p.add_argument("--test-worlds-ood", type=int, default=32)
    p.add_argument("--train-episodes", type=int, default=240)
    p.add_argument("--dev-episodes", type=int, default=70)
    p.add_argument("--test-ranking-episodes", type=int, default=80)
    p.add_argument("--rollout-episodes", type=int, default=80)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--max-history", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--teacher-temp", type=float, default=0.035)
    p.add_argument("--rerank-temp", type=float, default=0.025)
    p.add_argument("--belief-weight", type=float, default=0.30)
    p.add_argument("--rerank-weight", type=float, default=0.60)
    p.add_argument("--train-rerank-k", type=int, default=40)
    p.add_argument("--dev-states", type=int, default=5000)
    p.add_argument("--outdir", default="artifacts/eqs_v2_big/seed7")
    args = p.parse_args()

    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    seed_all(args.model_seed)
    os.makedirs(args.outdir, exist_ok=True)
    print("CONFIG", json.dumps(vars(args), sort_keys=True), flush=True)

    family = MultiWorldFamily(master_seed=args.master_seed)
    ids = list(range(family.n_q)); random.Random(args.master_seed).shuffle(ids)
    split = int(0.75 * len(ids)); train_q = ids[:split]; heldout_q = ids[split:]

    # Completely disjoint world seeds.
    train_worlds = build_worlds(family, 100000, args.train_worlds)
    dev_worlds = build_worlds(family, 300000, args.dev_worlds)
    test_worlds_id = build_worlds(family, 500000, args.test_worlds_id, difficulty="id")
    test_worlds_ood = build_worlds(family, 700000, args.test_worlds_ood, difficulty="ood")
    print(f"WORLDS train={len(train_worlds)} dev={len(dev_worlds)} test_id={len(test_worlds_id)} test_ood={len(test_worlds_ood)} "
          f"QUERIES train={len(train_q)} heldout={len(heldout_q)}", flush=True)

    print("BUILDING SYNTHETIC TRAIN STATES IN RAM (NO DOWNLOAD)", flush=True)
    train_rows = build_dataset(train_worlds, train_q, args.train_episodes, args.max_history, 11111)
    print("BUILDING SYNTHETIC DEV STATES IN RAM (NO DOWNLOAD)", flush=True)
    dev_rows = build_dataset(dev_worlds, train_q, args.dev_episodes, args.max_history, 22222)
    print(f"STATES train={len(train_rows)} dev={len(dev_rows)}", flush=True)

    model, best_dev = train_one(args.model_seed, args, family, train_worlds, dev_worlds,
                                train_rows, dev_rows, train_q, heldout_q)
    del train_rows, dev_rows

    print("BUILDING TEST-ID STATES IN RAM (NO DOWNLOAD)", flush=True)
    test_rows_id = build_dataset(test_worlds_id, train_q, args.test_ranking_episodes, args.max_history, 33333)
    id_rank_seen = evaluate_ranking(model, family, test_rows_id, train_q, args.max_history, max_states=12000)
    id_rank_unseenq = evaluate_ranking(model, family, test_rows_id, heldout_q, args.max_history, max_states=12000)
    del test_rows_id

    print("BUILDING TEST-OOD STATES IN RAM (NO DOWNLOAD)", flush=True)
    test_rows_ood = build_dataset(test_worlds_ood, train_q, args.test_ranking_episodes, args.max_history, 43333)
    ood_rank_seen = evaluate_ranking(model, family, test_rows_ood, train_q, args.max_history, max_states=12000)
    ood_rank_unseenq = evaluate_ranking(model, family, test_rows_ood, heldout_q, args.max_history, max_states=12000)
    del test_rows_ood

    static_scores = compute_static_scores(train_worlds)

    print("ROLLOUT ID-A: unseen ID worlds + seen query bank", flush=True)
    id_policy_seen = evaluate_rollouts(model, family, test_worlds_id, train_q, static_scores,
                                       args.max_history, args.rollout_episodes, 8, True, 44444)
    print("ROLLOUT ID-B: unseen ID worlds + ALL queries", flush=True)
    all_q = list(range(family.n_q))
    id_policy_all = evaluate_rollouts(model, family, test_worlds_id, all_q, static_scores,
                                      args.max_history, args.rollout_episodes, 8, False, 55555)

    print("ROLLOUT OOD-A: harder OOD worlds + seen query bank", flush=True)
    ood_policy_seen = evaluate_rollouts(model, family, test_worlds_ood, train_q, static_scores,
                                        args.max_history, args.rollout_episodes, 8, True, 64444)
    print("ROLLOUT OOD-B: harder OOD worlds + ALL queries", flush=True)
    ood_policy_all = evaluate_rollouts(model, family, test_worlds_ood, all_q, static_scores,
                                       args.max_history, args.rollout_episodes, 8, False, 75555)

    result = {
        "model_seed": args.model_seed,
        "best_dev_score": best_dev,
        "test_id": {
            "ranking_seen_queries": id_rank_seen,
            "ranking_heldout_queries": id_rank_unseenq,
            "policy_seen_queries": id_policy_seen,
            "policy_all_queries": id_policy_all,
        },
        "test_ood": {
            "ranking_seen_queries": ood_rank_seen,
            "ranking_heldout_queries": ood_rank_unseenq,
            "policy_seen_queries": ood_policy_seen,
            "policy_all_queries": ood_policy_all,
        },
        "config": vars(args),
    }
    with open(os.path.join(args.outdir, "results.json"), "w") as f: json.dump(result, f, indent=2)
    torch.save({"model":model.state_dict(), "raw_queries":family.raw_q, "train_query_ids":train_q,
                "heldout_query_ids":heldout_q, "config":vars(args)}, os.path.join(args.outdir, "model.pt"))
    print("="*88); print("FINAL RESULTS"); print("="*88)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
