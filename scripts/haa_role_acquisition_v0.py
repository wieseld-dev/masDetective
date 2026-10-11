from __future__ import annotations

import argparse
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


ROLES = list(TRAIN_ROLES)
R = len(ROLES)
RIDX = {r: i for i, r in enumerate(ROLES)}
EPS = 1e-8
SEED = 56173

# 20 choose 3..6 = 60,249. Cheap enough.
COMBOS = {
    k: np.asarray(
        list(itertools.combinations(range(R), k)),
        dtype=np.int16,
    )
    for k in (3, 4, 5, 6)
}


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def read_jsonl(path: Path):
    if not path.exists():
        return []

    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def sid_of(row):
    return str(
        row.get("system_id")
        or row.get("session_id")
        or ""
    )


def find_existing(paths):
    for p in paths:
        p = Path(p)
        if p.exists():
            return p
    raise FileNotFoundError(
        "None of these exist:\n"
        + "\n".join(map(str, paths))
    )


# ---------------------------------------------------------------------
# Role-only belief update.
#
# Phase-1 intentionally uses UNPROJECTED calibrated role probabilities,
# i.e. no K-based mean shift.
# ---------------------------------------------------------------------

def clip01(x):
    return float(np.clip(float(x), 1e-6, 1.0 - 1e-6))


def unprojected_role_probs(bundle, history):
    if not history:
        _, rp = bundle.initial()
        return {
            r: clip01(rp[r])
            for r in ROLES
        }

    raw = bundle._raw_role_vector_with(
        bundle.role_model,
        history,
    )

    pcal = bundle._apply_platt(
        bundle.role_platt,
        raw,
    )

    return {
        r: clip01(pcal[i])
        for i, r in enumerate(ROLES)
    }


def role_log_score(role_probs, true_roles):
    """
    Mean Bernoulli proper log score across all 20 roles.
    Higher is better.
    """
    truth = set(map(str, true_roles))

    vals = []
    for r in ROLES:
        p = clip01(role_probs[r])
        y = float(r in truth)

        vals.append(
            y * math.log(p)
            + (1.0 - y) * math.log(1.0 - p)
        )

    return float(np.mean(vals))


# ---------------------------------------------------------------------
# Load every actually observed candidate from the old role-only run.
# ---------------------------------------------------------------------

def load_observed_rows(root, manifest_path, belief_path):
    manifest = json.loads(
        Path(manifest_path).read_text(encoding="utf-8")
    )

    truth = {
        str(x["session_id"]): x
        for x in manifest["systems"]
        if str(x.get("split")) == "train"
    }

    bundle = FrozenBeliefBundle.load(belief_path)

    rounds = []

    for p in sorted(root.glob("candidates_train_round_*.jsonl")):
        try:
            rr = int(p.stem.rsplit("_", 1)[-1])
            rounds.append(rr)
        except Exception:
            pass

    rounds = sorted(set(rounds))

    if not rounds:
        raise RuntimeError(
            f"No train candidate rounds under {root}"
        )

    rows = []

    for rnd in rounds:
        state_path = (
            root
            / f"states_train_round_{rnd-1:02d}.jsonl"
        )

        cand_path = (
            root
            / f"candidates_train_round_{rnd:02d}.jsonl"
        )

        resp_path = (
            root
            / f"responses_train_round_{rnd:02d}.jsonl"
        )

        score_path = (
            root
            / f"scored_train_round_{rnd:02d}.jsonl"
        )

        if not all(
            p.exists()
            for p in (
                state_path,
                cand_path,
                resp_path,
                score_path,
            )
        ):
            print(
                f"[skip round {rnd}] missing one or more files",
                flush=True,
            )
            continue

        states = {
            str(x["session_id"]):
                PublicBeliefState.from_public_dict(x)
            for x in read_jsonl(state_path)
        }

        candidates = read_jsonl(cand_path)

        responses = {
            (
                str(x["session_id"]),
                int(x["candidate_index"]),
            ): x
            for x in read_jsonl(resp_path)
        }

        scored = {}
        for x in read_jsonl(score_path):
            sid = sid_of(x)
            idx = int(
                dict(x.get("action", {})).get(
                    "variant",
                    x.get("candidate_index", 0),
                )
            )
            scored[(sid, idx)] = x

        for c in candidates:
            sid = str(c.get("session_id") or "")
            idx = int(c.get("candidate_index", 0))

            if sid not in states or sid not in truth:
                continue

            resp = responses.get((sid, idx))
            sc = scored.get((sid, idx))

            if not c.get("valid"):
                continue

            if not resp:
                continue

            if resp.get("execution_status") != "ok":
                continue

            if sc is None:
                continue

            v135 = {
                str(k): float(v)
                for k, v in dict(
                    sc.get("evidence", {})
                ).items()
            }

            v134 = {
                str(k): float(v)
                for k, v in dict(
                    sc.get("metadata", {}).get(
                        "evidence_v134",
                        {},
                    )
                ).items()
            }

            if not all(r in v135 for r in ROLES):
                continue

            if not all(r in v134 for r in ROLES):
                continue

            state = states[sid]

            q = str(c.get("query") or "")
            ans = str(
                resp.get("final_output")
                or resp.get("output")
                or ""
            )

            frame = EvidenceFrame(
                query=q,
                answer=ans,
                v135=v135,
                v134=v134,
            )

            new_history = [
                *state.history,
                frame,
            ]

            after_probs = unprojected_role_probs(
                bundle,
                new_history,
            )

            true_roles = tuple(
                map(str, truth[sid]["roles"])
            )

            before_score = role_log_score(
                state.role_probs,
                true_roles,
            )

            after_score = role_log_score(
                after_probs,
                true_roles,
            )

            utility = after_score - before_score

            truth_mask = np.asarray(
                [
                    float(r in set(true_roles))
                    for r in ROLES
                ],
                dtype=float,
            )

            evidence = np.asarray(
                [v135[r] for r in ROLES]
                + [v134[r] for r in ROLES],
                dtype=float,
            )

            role_probs = np.asarray(
                [
                    float(state.role_probs[r])
                    for r in ROLES
                ],
                dtype=float,
            )

            k_probs = {
                int(k): float(v)
                for k, v in state.k_probs.items()
            }

            rows.append({
                "sid": sid,
                "round": int(rnd),
                "candidate_index": idx,
                "query": q,
                "old_logp": float(
                    c.get("old_logp", -1e9)
                ),
                "utility": float(utility),
                "before_score": float(before_score),
                "after_score": float(after_score),
                "role_probs": role_probs,
                "k_probs": k_probs,
                "truth_mask": truth_mask,
                "evidence": evidence,
            })

    return rows


# ---------------------------------------------------------------------
# Sentence embeddings
# ---------------------------------------------------------------------

def embed_queries(rows):
    model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        device="cpu",
    )

    queries = list(
        dict.fromkeys(r["query"] for r in rows)
    )

    qemb = model.encode(
        queries,
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    qmap = {
        q: np.asarray(e, dtype=float)
        for q, e in zip(queries, qemb)
    }

    desc = [
        str(ROLE_DESCRIPTIONS.get(r, r))
        for r in ROLES
    ]

    remb = np.asarray(
        model.encode(
            desc,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ),
        dtype=float,
    )

    for row in rows:
        row["qemb"] = qmap[row["query"]]

        # semantic affinity to each role description
        row["affinity"] = (
            row["qemb"] @ remb.T
        ).astype(float)

    return rows


# ---------------------------------------------------------------------
# Hypothesis particles
# ---------------------------------------------------------------------

def softmax(x):
    x = np.asarray(x, float)
    x = x - np.max(x)
    z = np.exp(np.clip(x, -60, 60))
    return z / max(z.sum(), EPS)


def particles_from_belief(
    role_probs,
    k_probs,
    top_per_k=16,
):
    """
    Approximate role-set posterior.

    Within fixed K, under the independent-marginal approximation:

      log P(H | |H|=K) = const + sum_{r in H} logit(p_r)

    Keep top role sets for each K. Preserve P(K) mass.
    """
    p = np.clip(
        np.asarray(role_probs, float),
        1e-5,
        1.0 - 1e-5,
    )

    logits = np.log(p) - np.log(1.0 - p)

    masks = []
    weights = []

    valid_k = [
        k for k in (3, 4, 5, 6)
        if k in k_probs
    ]

    kp = np.asarray(
        [max(0.0, k_probs[k]) for k in valid_k],
        dtype=float,
    )

    if kp.sum() <= 0:
        kp[:] = 1.0

    kp /= kp.sum()

    for k, pk in zip(valid_k, kp):
        combos = COMBOS[k]

        scores = logits[combos].sum(axis=1)

        m = min(
            int(top_per_k),
            len(scores),
        )

        if m < len(scores):
            ids = np.argpartition(
                scores,
                -m,
            )[-m:]
        else:
            ids = np.arange(len(scores))

        ids = ids[
            np.argsort(scores[ids])[::-1]
        ]

        local_w = softmax(scores[ids])

        for cid, lw in zip(ids, local_w):
            mask = np.zeros(R, dtype=float)
            mask[combos[cid]] = 1.0

            masks.append(mask)
            weights.append(float(pk * lw))

    H = np.stack(masks)
    w = np.asarray(weights, float)
    w /= w.sum()

    inclusion = w @ H

    return H, w, inclusion


# ---------------------------------------------------------------------
# Feature maps
# ---------------------------------------------------------------------

def obs_features(q_low, affinity, hmask):
    """
    Learned observation model F(q,H) -> expected Judge evidence.

    Deliberately low capacity/compositional:
      query latent
      role-set mask
      query-role semantic affinity
      mask x affinity
    """
    return np.concatenate([
        q_low,
        hmask,
        affinity,
        hmask * affinity,
    ])


def direct_features(
    q_low,
    affinity,
    role_probs,
):
    """
    Strong non-hypothesis baseline:
      directly predict realized utility from q + current belief.
    """
    p = np.asarray(role_probs, float)
    u = p * (1.0 - p)

    return np.concatenate([
        q_low,
        p,
        u,
        affinity,
        p * affinity,
        u * affinity,
    ])


def uncertainty_score(
    affinity,
    role_probs,
):
    p = np.asarray(role_probs, float)
    u = p * (1.0 - p)

    a = np.maximum(
        np.asarray(affinity, float),
        0.0,
    )

    return float(
        np.sum(u * a * a)
    )


# ---------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------

def concordance(scores, utility):
    s = np.asarray(scores, float)
    y = np.asarray(utility, float)

    good = 0.0
    total = 0.0

    for i in range(len(y)):
        for j in range(i + 1, len(y)):
            dy = y[i] - y[j]
            ds = s[i] - s[j]

            if abs(dy) < 1e-12:
                continue

            total += 1.0

            if dy * ds > 0:
                good += 1.0
            elif abs(ds) < 1e-12:
                good += 0.5

    return (
        float(good / total)
        if total
        else np.nan
    )


def rank_info(chosen, utility):
    y = np.asarray(utility, float)

    order = np.argsort(y)[::-1]

    pos = int(
        np.where(order == chosen)[0][0]
    ) + 1

    return {
        "rank": pos,
        "top1": int(pos == 1),
        "top2": int(pos <= min(2, len(y))),
    }


# ---------------------------------------------------------------------
# Clustered statistics
# ---------------------------------------------------------------------

def system_means(records, field):
    by = defaultdict(list)

    for r in records:
        by[r["sid"]].append(
            float(r[field])
        )

    return {
        sid: float(np.mean(v))
        for sid, v in by.items()
    }


def bootstrap_system_ci(
    sysvals,
    B=20000,
):
    ids = sorted(sysvals)

    if not ids:
        return [None, None]

    vals = np.asarray(
        [sysvals[s] for s in ids],
        float,
    )

    rng = np.random.default_rng(SEED)

    boot = np.empty(B)

    for b in range(B):
        boot[b] = np.mean(
            rng.choice(
                vals,
                len(vals),
                replace=True,
            )
        )

    return [
        float(np.quantile(boot, .025)),
        float(np.quantile(boot, .975)),
    ]


def signflip_greater(
    sysvals,
    B=50000,
):
    vals = np.asarray(
        list(sysvals.values()),
        float,
    )

    if len(vals) == 0:
        return None

    obs = float(np.mean(vals))

    # exact when practical
    if len(vals) <= 16:
        null = []
        for bits in itertools.product(
            [-1.0, 1.0],
            repeat=len(vals),
        ):
            null.append(
                np.mean(
                    vals * np.asarray(bits)
                )
            )

        null = np.asarray(null)

    else:
        rng = np.random.default_rng(
            SEED + 1
        )

        signs = rng.choice(
            [-1.0, 1.0],
            size=(B, len(vals)),
        )

        null = np.mean(
            signs * vals[None, :],
            axis=1,
        )

    return float(
        (
            1
            + np.sum(null >= obs - 1e-12)
        )
        / (
            1 + len(null)
        )
    )


# ---------------------------------------------------------------------
# Main LOSO experiment
# ---------------------------------------------------------------------

METHODS = [
    "policy_logp",
    "uncertainty",
    "q_only",
    "direct_belief",
    "haa_full",
    "haa_focus",
]


def fit_fold(rows, train_idx):
    Q = np.stack(
        [rows[i]["qemb"] for i in train_idx]
    )

    ncomp = min(
        12,
        Q.shape[0] - 1,
        Q.shape[1],
    )

    pca = PCA(
        n_components=ncomp,
        random_state=SEED,
    )

    Qlow = pca.fit_transform(Q)

    # -------------------------------------------------------------
    # observation surrogate F(q,H) -> dual Judge evidence
    # -------------------------------------------------------------

    Xobs = []

    for z, i in zip(Qlow, train_idx):
        Xobs.append(
            obs_features(
                z,
                rows[i]["affinity"],
                rows[i]["truth_mask"],
            )
        )

    Xobs = np.stack(Xobs)

    Y = np.stack(
        [rows[i]["evidence"] for i in train_idx]
    )

    y_scaler = StandardScaler()
    Yz = y_scaler.fit_transform(Y)

    obs_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    obs_model.fit(
        Xobs,
        Yz,
    )

    train_pred = obs_model.predict(
        Xobs
    )

    residual_var = np.var(
        Yz - train_pred,
        axis=0,
    )

    # Prevent a nearly-zero train residual from creating
    # absurd acquisition weight.
    residual_var = np.maximum(
        residual_var,
        0.05,
    )

    # -------------------------------------------------------------
    # q-only utility model
    # -------------------------------------------------------------

    U = np.asarray(
        [rows[i]["utility"] for i in train_idx],
        float,
    )

    q_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    q_model.fit(
        Qlow,
        U,
    )

    # -------------------------------------------------------------
    # strongest direct baseline:
    # q + current belief -> realized one-step utility
    # -------------------------------------------------------------

    Xdirect = []

    for z, i in zip(Qlow, train_idx):
        Xdirect.append(
            direct_features(
                z,
                rows[i]["affinity"],
                rows[i]["role_probs"],
            )
        )

    Xdirect = np.stack(Xdirect)

    direct_model = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0),
    )

    direct_model.fit(
        Xdirect,
        U,
    )

    return {
        "pca": pca,
        "obs_model": obs_model,
        "residual_var": residual_var,
        "q_model": q_model,
        "direct_model": direct_model,
    }


def score_candidate(
    row,
    fold,
    H,
    w,
    inclusion,
):
    q_low = fold["pca"].transform(
        row["qemb"][None, :]
    )[0]

    affinity = row["affinity"]

    # q-only predictor
    q_score = float(
        fold["q_model"].predict(
            q_low[None, :]
        )[0]
    )

    # direct q+belief predictor
    xd = direct_features(
        q_low,
        affinity,
        row["role_probs"],
    )

    direct_score = float(
        fold["direct_model"].predict(
            xd[None, :]
        )[0]
    )

    # explicit hypothesis-conditioned observation model
    Xp = np.stack([
        obs_features(
            q_low,
            affinity,
            h,
        )
        for h in H
    ])

    MU = fold["obs_model"].predict(
        Xp
    )

    mean_mu = w @ MU

    predictive_var = (
        w[:, None]
        * (MU - mean_mu[None, :]) ** 2
    ).sum(axis=0)

    normalized_var = (
        predictive_var
        / fold["residual_var"]
    )

    haa_full = float(
        np.sum(normalized_var)
    )

    # Primary acquisition:
    # focus disagreement on roles that the CURRENT H-posterior
    # itself is uncertain about.
    hu = inclusion * (1.0 - inclusion)

    dim_weight = np.concatenate([
        hu,
        hu,
    ])

    if dim_weight.mean() > EPS:
        dim_weight = (
            dim_weight
            / dim_weight.mean()
        )
    else:
        dim_weight = np.ones_like(
            dim_weight
        )

    haa_focus = float(
        np.sum(
            normalized_var
            * dim_weight
        )
    )

    return {
        "policy_logp":
            float(row["old_logp"]),

        "uncertainty":
            uncertainty_score(
                affinity,
                row["role_probs"],
            ),

        "q_only":
            q_score,

        "direct_belief":
            direct_score,

        "haa_full":
            haa_full,

        "haa_focus":
            haa_focus,
    }


def run_loso(rows):
    sids = sorted(
        set(r["sid"] for r in rows)
    )

    records = []

    for hi, held in enumerate(sids):
        print(
            f"[LOSO {hi+1}/{len(sids)}] held={held}",
            flush=True,
        )

        train_idx = [
            i for i, r in enumerate(rows)
            if r["sid"] != held
        ]

        test_idx = [
            i for i, r in enumerate(rows)
            if r["sid"] == held
        ]

        if not train_idx or not test_idx:
            continue

        fold = fit_fold(
            rows,
            train_idx,
        )

        groups = defaultdict(list)

        for i in test_idx:
            groups[
                (
                    rows[i]["sid"],
                    rows[i]["round"],
                )
            ].append(i)

        for (sid, rnd), inds in sorted(
            groups.items()
        ):
            # Need a genuine choice.
            if len(inds) < 2:
                continue

            # All candidates in a group share the same pre-state.
            base = rows[inds[0]]

            H, w, inclusion = (
                particles_from_belief(
                    base["role_probs"],
                    base["k_probs"],
                    top_per_k=16,
                )
            )

            util = np.asarray(
                [
                    rows[i]["utility"]
                    for i in inds
                ],
                float,
            )

            score_dict = {
                m: []
                for m in METHODS
            }

            for i in inds:
                sc = score_candidate(
                    rows[i],
                    fold,
                    H,
                    w,
                    inclusion,
                )

                for m in METHODS:
                    score_dict[m].append(
                        sc[m]
                    )

            random_utility = float(
                np.mean(util)
            )

            oracle = float(
                np.max(util)
            )

            worst = float(
                np.min(util)
            )

            denom = (
                oracle - worst
            )

            rec = {
                "sid": sid,
                "round": int(rnd),
                "n_candidates": len(inds),
                "random_utility":
                    random_utility,
                "oracle_utility":
                    oracle,
                "worst_utility":
                    worst,
                "utility_range":
                    float(denom),
            }

            for m in METHODS:
                scores = np.asarray(
                    score_dict[m],
                    float,
                )

                chosen = int(
                    np.argmax(scores)
                )

                chosen_u = float(
                    util[chosen]
                )

                ri = rank_info(
                    chosen,
                    util,
                )

                regret = (
                    (oracle - chosen_u)
                    / denom
                    if denom > 1e-12
                    else 0.0
                )

                rec[
                    f"{m}_utility"
                ] = chosen_u

                rec[
                    f"{m}_gain_random"
                ] = (
                    chosen_u
                    - random_utility
                )

                rec[
                    f"{m}_regret"
                ] = float(regret)

                rec[
                    f"{m}_top1"
                ] = ri["top1"]

                rec[
                    f"{m}_top2"
                ] = ri["top2"]

                rec[
                    f"{m}_concordance"
                ] = concordance(
                    scores,
                    util,
                )

            records.append(rec)

    return records


# ---------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------

def summarize_method(records, method):
    if not records:
        return {}

    gain_field = (
        f"{method}_gain_random"
    )

    sys_gain = system_means(
        records,
        gain_field,
    )

    return {
        "groups": len(records),
        "systems": len(sys_gain),

        "selected_utility_mean":
            float(np.mean([
                r[f"{method}_utility"]
                for r in records
            ])),

        "gain_over_random_mean":
            float(np.mean([
                r[gain_field]
                for r in records
            ])),

        "gain_over_random_cluster_ci95":
            bootstrap_system_ci(
                sys_gain
            ),

        "gain_over_random_signflip_p":
            signflip_greater(
                sys_gain
            ),

        "normalized_regret_mean":
            float(np.mean([
                r[f"{method}_regret"]
                for r in records
            ])),

        "top1_rate":
            float(np.mean([
                r[f"{method}_top1"]
                for r in records
            ])),

        "top2_rate":
            float(np.mean([
                r[f"{method}_top2"]
                for r in records
            ])),

        "pairwise_concordance":
            float(np.nanmean([
                r[f"{method}_concordance"]
                for r in records
            ])),
    }


def paired_comparison(
    records,
    a,
    b,
):
    by = defaultdict(list)

    for r in records:
        by[r["sid"]].append(
            r[f"{a}_utility"]
            - r[f"{b}_utility"]
        )

    sysvals = {
        sid: float(np.mean(v))
        for sid, v in by.items()
    }

    vals = list(sysvals.values())

    return {
        "a": a,
        "b": b,
        "systems": len(vals),
        "mean_selected_utility_delta":
            float(np.mean(vals))
            if vals else None,

        "cluster_ci95":
            bootstrap_system_ci(
                sysvals
            ),

        "one_sided_signflip_p":
            signflip_greater(
                sysvals
            ),
    }


def make_summary(records):
    result = {}

    subsets = {
        "cold_round1": [
            r for r in records
            if r["round"] == 1
        ],

        "round2": [
            r for r in records
            if r["round"] == 2
        ],

        "round3": [
            r for r in records
            if r["round"] == 3
        ],

        # PRIMARY endpoint:
        "core_rounds2_3": [
            r for r in records
            if r["round"] >= 2
        ],
    }

    for name, rr in subsets.items():
        result[name] = {
            "groups": len(rr),
            "systems": len(
                set(x["sid"] for x in rr)
            ),
            "mean_candidates":
                float(np.mean([
                    x["n_candidates"]
                    for x in rr
                ]))
                if rr else None,

            "random_utility_mean":
                float(np.mean([
                    x["random_utility"]
                    for x in rr
                ]))
                if rr else None,

            "oracle_utility_mean":
                float(np.mean([
                    x["oracle_utility"]
                    for x in rr
                ]))
                if rr else None,

            "methods": {
                m: summarize_method(
                    rr,
                    m,
                )
                for m in METHODS
            },

            "haa_focus_vs": {
                b: paired_comparison(
                    rr,
                    "haa_focus",
                    b,
                )
                for b in (
                    "policy_logp",
                    "uncertainty",
                    "q_only",
                    "direct_belief",
                    "haa_full",
                )
            },
        }

    core = result[
        "core_rounds2_3"
    ]

    hf = core["methods"][
        "haa_focus"
    ]

    qcmp = core["haa_focus_vs"][
        "q_only"
    ]

    dcmp = core["haa_focus_vs"][
        "direct_belief"
    ]

    # Exploratory continuation decision.
    #
    # Strong:
    #   hypothesis-aware score beats random,
    #   q-only, and strong direct q+belief predictor.
    #
    # Partial:
    #   beats random + q-only, but not direct predictor.
    #
    # No-go:
    #   cannot even consistently improve over random/q-only.
    gain_ok = (
        hf
        and hf[
            "gain_over_random_mean"
        ] > 0
        and hf[
            "gain_over_random_signflip_p"
        ] is not None
        and hf[
            "gain_over_random_signflip_p"
        ] <= 0.05
    )

    q_ok = (
        qcmp[
            "mean_selected_utility_delta"
        ] is not None
        and qcmp[
            "mean_selected_utility_delta"
        ] > 0
        and qcmp[
            "one_sided_signflip_p"
        ] is not None
        and qcmp[
            "one_sided_signflip_p"
        ] <= 0.10
    )

    direct_ok = (
        dcmp[
            "mean_selected_utility_delta"
        ] is not None
        and dcmp[
            "mean_selected_utility_delta"
        ] > 0
        and dcmp[
            "one_sided_signflip_p"
        ] is not None
        and dcmp[
            "one_sided_signflip_p"
        ] <= 0.10
    )

    if gain_ok and q_ok and direct_ok:
        verdict = "STRONG_GO"
    elif gain_ok and q_ok:
        verdict = "PARTIAL_GO"
    else:
        verdict = "NO_GO"

    result["decision"] = {
        "verdict": verdict,

        "primary_endpoint":
            "core_rounds2_3",

        "primary_method":
            "haa_focus",

        "interpretation": {
            "STRONG_GO":
                "Explicit hypothesis-conditioned acquisition "
                "adds value beyond random, q-only informativeness, "
                "and a direct q+belief utility predictor.",

            "PARTIAL_GO":
                "Active ranking is useful beyond random/q-only, "
                "but explicit hypothesis modeling has not yet "
                "beaten the strong direct q+belief predictor.",

            "NO_GO":
                "The current hypothesis-aware acquisition does not "
                "reliably improve probe choice; do not build an "
                "online controller from this version.",
        }[verdict],
    }

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        default="artifacts/adaptive_v11_role_only_v1",
    )

    ap.add_argument(
        "--manifest",
        default="artifacts/adaptive_v11_active_v1/manifest_privileged.json",
    )

    ap.add_argument(
        "--belief",
        default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib",
    )

    ap.add_argument(
        "--output",
        default="artifacts/haa_role_acquisition_v0",
    )

    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    belief_path = find_existing([
        args.belief,
        root / "frozen_belief.joblib",
    ])

    manifest_path = find_existing([
        args.manifest,
        root / "manifest_privileged.json",
    ])

    print("=" * 96)
    print("HAA-V0 — HYPOTHESIS-AWARE ACTIVE PROBE ACQUISITION")
    print("=" * 96)
    print("root    :", root)
    print("belief  :", belief_path)
    print("manifest:", manifest_path)
    print("API calls: 0")
    print()

    print("[1/4] loading realized counterfactual candidate groups", flush=True)

    rows = load_observed_rows(
        root,
        manifest_path,
        belief_path,
    )

    print(
        f"[data] usable candidate rows={len(rows)} "
        f"systems={len(set(r['sid'] for r in rows))} "
        f"rounds={sorted(set(r['round'] for r in rows))}",
        flush=True,
    )

    by_group = defaultdict(int)
    for r in rows:
        by_group[
            (r["sid"], r["round"])
        ] += 1

    print(
        "[data] candidate-count histogram:",
        {
            n: sum(v == n for v in by_group.values())
            for n in sorted(set(by_group.values()))
        },
        flush=True,
    )

    if len(rows) < 100:
        raise RuntimeError(
            "Too few observed role-only candidates for HAA-V0"
        )

    print("[2/4] frozen semantic embeddings", flush=True)
    rows = embed_queries(rows)

    print("[3/4] strict leave-one-system-out acquisition test", flush=True)
    records = run_loso(rows)

    if not records:
        raise RuntimeError("No evaluable candidate groups")

    print("[4/4] clustered evaluation", flush=True)
    summary = make_summary(records)

    with (
        out / "group_results.jsonl"
    ).open("w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(r)
                + "\n"
            )

    (
        out / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Human report
    # -------------------------------------------------------------

    L = []

    L.append("=" * 100)
    L.append(
        "HAA-V0 — HYPOTHESIS-AWARE ACTIVE PROBE ACQUISITION"
    )
    L.append("=" * 100)

    L.append(
        f"candidate rows={len(rows)} | "
        f"systems={len(set(r['sid'] for r in rows))} | "
        f"evaluated groups={len(records)}"
    )

    L.append(
        "Primary endpoint: rounds 2-3 "
        "(post-initial-belief / non-symmetric states)"
    )

    L.append("")

    for section in (
        "cold_round1",
        "round2",
        "round3",
        "core_rounds2_3",
    ):
        x = summary[section]

        L.append("-" * 100)
        L.append(
            f"{section.upper()} | "
            f"groups={x['groups']} "
            f"systems={x['systems']} "
            f"mean-candidates={x['mean_candidates']}"
        )

        if not x["groups"]:
            continue

        L.append(
            f"  random expected utility = "
            f"{x['random_utility_mean']:+.6f}"
        )

        L.append(
            f"  oracle best utility     = "
            f"{x['oracle_utility_mean']:+.6f}"
        )

        L.append("")

        for m in METHODS:
            z = x["methods"][m]

            if not z:
                continue

            ci = z[
                "gain_over_random_cluster_ci95"
            ]

            L.append(
                f"  {m:16s} "
                f"U={z['selected_utility_mean']:+.6f} "
                f"gain/random={z['gain_over_random_mean']:+.6f} "
                f"CI=[{ci[0]:+.6f},{ci[1]:+.6f}] "
                f"p={z['gain_over_random_signflip_p']:.5f} "
                f"regret={z['normalized_regret_mean']:.3f} "
                f"top1={z['top1_rate']:.3f} "
                f"top2={z['top2_rate']:.3f} "
                f"conc={z['pairwise_concordance']:.3f}"
            )

        L.append("")
        L.append("  HAA-FOCUS PAIRED COMPARISONS")

        for b, z in x[
            "haa_focus_vs"
        ].items():
            ci = z["cluster_ci95"]

            L.append(
                f"    vs {b:14s}: "
                f"delta-U="
                f"{z['mean_selected_utility_delta']:+.6f} "
                f"CI=[{ci[0]:+.6f},{ci[1]:+.6f}] "
                f"one-sided-p="
                f"{z['one_sided_signflip_p']:.5f}"
            )

        L.append("")

    d = summary["decision"]

    L.append("=" * 100)
    L.append("DECISION")
    L.append("=" * 100)
    L.append(
        f"VERDICT: {d['verdict']}"
    )
    L.append(
        d["interpretation"]
    )
    L.append("")
    L.append(
        "Interpretation rule:"
    )
    L.append(
        "STRONG_GO = HAA-focus improves probe selection beyond "
        "random, q-only, and the direct q+belief utility predictor."
    )
    L.append(
        "PARTIAL_GO = useful active ranking exists, but explicit "
        "hypothesis modeling is not yet justified over direct utility learning."
    )
    L.append(
        "NO_GO = do not spend new black-box queries on this version."
    )
    L.append("=" * 100)

    report = "\n".join(L)

    (
        out / "REPORT.txt"
    ).write_text(
        report + "\n",
        encoding="utf-8",
    )

    print()
    print(report)
    print()
    print(
        "[done]",
        out / "REPORT.txt",
    )


if __name__ == "__main__":
    main()
