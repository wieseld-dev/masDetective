#!/usr/bin/env python3
"""
matrix_selector_oracle_diag.py

Retrospective diagnostic for the V11 curriculum candidate pools.

Goal
----
Test the mathematical selector BEFORE training any new selector.

For every recorded state:
  1) Build a coherent 20-agent covariance Sigma from:
       - k_probs  : posterior over number of agents
       - role_probs: relative evidence over agent identities
  2) Parse each candidate's V11 DIAGNOSTIC FOCUS REGION from its prompt.
     This gives an AVAILABLE-BEFORE-EXECUTION probe direction a_q.
  3) Score each candidate using several covariance-reduction criteria:
       count:
         ((1^T Sigma a)^2) / (a^T Sigma a + noise)
       identity:
         ||P Sigma a||^2 / (a^T Sigma a + noise)
       trace:
         ||Sigma a||^2 / (a^T Sigma a + noise)
       eig1:
         (v1^T Sigma a)^2 / (a^T Sigma a + noise)
         where v1 is the top eigenvector of P Sigma P.
  4) Compare those scores to the EXISTING oracle outcome:
       number of NEW selected hidden roles exposed by that candidate.

Important validity distinction
------------------------------
Round 1 is the cleanest test: every policy starts from the same state.
Rounds >1 are "one-step conditional on the actually recorded history", because
the stored state was produced by the original policy, not by our counterfactual
policy. We therefore report round 1 separately.

No API calls. No new MAS executions. No model training.

Expected repository layout
--------------------------
artifacts/adaptive_v11_curriculum_v2/**/
    states_train_round_XX.jsonl
    responses_train_round_XX.jsonl

Usage
-----
python analysis/matrix_selector_oracle_diag.py \
  --root artifacts/adaptive_v11_curriculum_v2 \
  --out artifacts/matrix_selector_oracle_diag_v1

"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None


FOCUS_MARKER = "V11 DIAGNOSTIC FOCUS REGION FOR THIS CANDIDATE:"
FOCUS_END_MARKERS = (
    "PREVIOUS USER QUESTIONS",
    "Generate ONE",
    "Hard constraints:",
)


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_chosen(v) -> bool:
    if v is True or v == 1:
        return True
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return False


def parse_round_from_name(path: Path) -> int:
    m = re.search(r"round_(\d+)", path.name)
    if not m:
        raise ValueError(f"Cannot parse round from {path}")
    return int(m.group(1))


def parse_focus_roles(prompt: str, valid_roles: set[str]) -> List[str]:
    if not prompt or FOCUS_MARKER not in prompt:
        return []

    block = prompt.split(FOCUS_MARKER, 1)[1]
    for marker in FOCUS_END_MARKERS:
        if marker in block:
            block = block.split(marker, 1)[0]

    roles = []
    # Typical format: "- accessibility_review: p=0.225; ..."
    for line in block.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].split(":", 1)[0].strip()
        if item in valid_roles:
            roles.append(item)

    # preserve order, remove duplicates
    out = []
    seen = set()
    for r in roles:
        if r not in seen:
            out.append(r)
            seen.add(r)
    return out


def poly_coeffs(weights: np.ndarray, kmax: int) -> np.ndarray:
    """
    e[k] = elementary symmetric polynomial of degree k.
    Coefficients of prod_i (1 + w_i t), truncated at kmax.
    """
    e = np.zeros(kmax + 1, dtype=np.float64)
    e[0] = 1.0
    for w in weights:
        for k in range(kmax, 0, -1):
            e[k] += w * e[k - 1]
    return e


def divide_linear(coeff: np.ndarray, w: float) -> np.ndarray:
    """
    Given coeffs e of P(t)=(1+w t)Q(t), recover Q coeffs q recursively:
      q[0]=e[0]
      q[k]=e[k]-w*q[k-1]
    Only needs the same truncated degree.
    """
    q = np.zeros_like(coeff)
    q[0] = coeff[0]
    for k in range(1, len(coeff)):
        q[k] = coeff[k] - w * q[k - 1]
    return q


def coherent_covariance(
    k_probs_dict: Dict[str, float],
    role_probs_dict: Dict[str, float],
    roles: List[str],
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build a coherent distribution over subsets:

      P(S | K=k) proportional to product_{i in S} odds_i

    and mix exactly according to the supplied P(K=k).

    This uses role_probs as RELATIVE inclusion evidence, while k_probs controls
    the exact cardinality distribution. This is intentional because the raw
    role marginals and k_probs need not be mutually consistent.

    Returns:
      mu     [N]
      Sigma  [N,N]
      info   diagnostics
    """
    p_raw = np.array([float(role_probs_dict[r]) for r in roles], dtype=np.float64)
    p = np.clip(p_raw, 1e-5, 1 - 1e-5)
    odds = p / (1.0 - p)

    kpairs = []
    for k, v in k_probs_dict.items():
        kk = int(k)
        vv = float(v)
        if vv > 0:
            kpairs.append((kk, vv))
    if not kpairs:
        raise ValueError("empty k_probs")
    z = sum(v for _, v in kpairs)
    kpairs = [(k, v / z) for k, v in kpairs]
    kmax = max(k for k, _ in kpairs)

    e = poly_coeffs(odds, kmax)

    # Polynomial with item i excluded.
    excl1 = []
    for i in range(len(roles)):
        excl1.append(divide_linear(e, odds[i]))

    n = len(roles)
    mu = np.zeros(n, dtype=np.float64)
    second = np.zeros((n, n), dtype=np.float64)

    # exact first moments under the cardinality mixture
    for k, pk in kpairs:
        if k < 0 or k > n or k >= len(e) or e[k] <= 0:
            continue
        for i in range(n):
            if k == 0:
                pi = 0.0
            else:
                pi = odds[i] * excl1[i][k - 1] / e[k]
            mu[i] += pk * pi

    # diagonal second moments E[Zi^2] = E[Zi]
    np.fill_diagonal(second, mu)

    # pair moments
    for i in range(n):
        for j in range(i + 1, n):
            # Exclude j from polynomial already excluding i.
            excl_ij = divide_linear(excl1[i], odds[j])
            pij = 0.0
            for k, pk in kpairs:
                if k < 2 or k >= len(e) or e[k] <= 0:
                    continue
                pij_k = odds[i] * odds[j] * excl_ij[k - 2] / e[k]
                pij += pk * pij_k
            second[i, j] = second[j, i] = pij

    Sigma = second - np.outer(mu, mu)
    Sigma = 0.5 * (Sigma + Sigma.T)

    # Numerical cleanup only.
    vals, vecs = np.linalg.eigh(Sigma)
    min_eval = float(vals.min())
    if min_eval < -1e-8:
        vals = np.clip(vals, 0, None)
        Sigma = (vecs * vals) @ vecs.T
        Sigma = 0.5 * (Sigma + Sigma.T)

    expected_k_input = sum(k * pk for k, pk in kpairs)
    info = {
        "expected_k_from_k_probs": float(expected_k_input),
        "sum_raw_role_probs": float(p_raw.sum()),
        "sum_coherent_mu": float(mu.sum()),
        "min_cov_eigenvalue_before_cleanup": min_eval,
    }
    return mu.astype(np.float64), Sigma.astype(np.float64), info


def score_candidate(
    Sigma: np.ndarray,
    focus_vec: np.ndarray,
    noise: float,
) -> Dict[str, float]:
    n = Sigma.shape[0]
    a = focus_vec.astype(np.float64)

    if np.linalg.norm(a) <= 1e-12:
        return {
            "count": float("-inf"),
            "identity": float("-inf"),
            "trace": float("-inf"),
            "eig1": float("-inf"),
            "variance": float("-inf"),
        }

    # Normalize so score is not automatically larger for longer focus lists.
    a = a / np.linalg.norm(a)

    one = np.ones(n, dtype=np.float64) / math.sqrt(n)
    P = np.eye(n) - np.outer(one, one)

    Sa = Sigma @ a
    denom = float(a @ Sa + noise)

    count = float((one @ Sa) ** 2 / denom)
    identity = float(np.linalg.norm(P @ Sa) ** 2 / denom)
    trace = float(np.linalg.norm(Sa) ** 2 / denom)
    variance = float(a @ Sa)

    Sid = P @ Sigma @ P
    evals, evecs = np.linalg.eigh(Sid)
    idx = int(np.argmax(evals))
    v1 = evecs[:, idx]
    eig1 = float((v1 @ Sa) ** 2 / denom)

    return {
        "count": count,
        "identity": identity,
        "trace": trace,
        "eig1": eig1,
        "variance": variance,
    }


def rank_corr(scores: List[float], rewards: List[float]) -> Optional[float]:
    x = np.asarray(scores, dtype=float)
    y = np.asarray(rewards, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return None
    if spearmanr is not None:
        val = spearmanr(x, y).correlation
        return None if np.isnan(val) else float(val)

    # fallback: simple ordinal ranking (ties not specially corrected)
    rx = np.empty(len(x), dtype=float)
    ry = np.empty(len(y), dtype=float)
    rx[np.argsort(x)] = np.arange(len(x))
    ry[np.argsort(y)] = np.arange(len(y))
    val = np.corrcoef(rx, ry)[0, 1]
    return None if np.isnan(val) else float(val)


def choose_by_score(rows: List[dict], method: str) -> dict:
    # deterministic tie break: smaller candidate index, matching oracle script spirit
    return max(
        rows,
        key=lambda r: (
            r["scores"][method],
            -int(r.get("candidate_index") or 0),
        ),
    )


def oracle_key(r: dict, seen: set[str]):
    selected = set(r["selected"])
    new = len(selected - seen)
    return (
        new,
        len(selected),
        -int(r.get("candidate_index") or 0),
    )


def load_parent(parent: Path):
    """
    Returns:
      states[(sid, round_index)] -> state row
      responses[(sid, round)] -> candidate rows
    """
    states = {}
    responses = defaultdict(list)

    for p in sorted(parent.glob("states_train_round_*.jsonl")):
        file_round = parse_round_from_name(p)
        for x in read_jsonl(p):
            sid = x.get("session_id") or x.get("system_id")
            if not sid:
                continue
            ridx = int(x.get("round_index", file_round))
            states[(sid, ridx)] = x

    for p in sorted(parent.glob("responses_train_round_*.jsonl")):
        file_round = parse_round_from_name(p)
        for x in read_jsonl(p):
            sid = x.get("system_id") or x.get("session_id")
            if not sid:
                continue
            rnd = int(x.get("round", file_round))
            responses[(sid, rnd)].append(x)

    return states, responses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="artifacts/adaptive_v11_curriculum_v2",
    )
    ap.add_argument(
        "--out",
        default="artifacts/matrix_selector_oracle_diag_v1",
    )
    ap.add_argument(
        "--noise",
        type=float,
        default=0.10,
        help="Observation-noise floor in covariance-reduction score.",
    )
    ap.add_argument(
        "--min-pool",
        type=int,
        default=6,
    )
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    parent_dirs = sorted(
        {p.parent for p in root.rglob("responses_train_round_*.jsonl")}
    )

    methods = ["count", "identity", "trace", "eig1", "variance"]
    per_group = []

    # Aggregate metrics.
    agg = {
        m: {
            "groups": 0,
            "top1_oracle_match": 0,
            "new_exposed_sum": 0.0,
            "oracle_new_sum": 0.0,
            "regret_sum": 0.0,
            "spearman": [],
        }
        for m in methods
    }

    # Separate round-1 metrics (clean counterfactual comparison).
    agg_r1 = {
        m: {
            "groups": 0,
            "top1_oracle_match": 0,
            "new_exposed_sum": 0.0,
            "oracle_new_sum": 0.0,
            "regret_sum": 0.0,
            "spearman": [],
        }
        for m in methods
    }

    parse_fail = 0
    missing_state = 0
    kept_groups = 0
    total_groups = 0

    for parent in parent_dirs:
        states, responses = load_parent(parent)
        if not responses:
            continue

        # Maintain ACTUAL seen roles for each sid so round>1 conditions on the
        # state that was actually observed in the saved trajectory.
        by_sid = defaultdict(dict)
        for (sid, rnd), rows in responses.items():
            by_sid[sid][rnd] = rows

        for sid, rm in by_sid.items():
            seen_actual: set[str] = set()

            for rnd in sorted(rm):
                total_groups += 1
                rows0 = rm[rnd]
                if len(rows0) < args.min_pool:
                    # still update actual seen below if possible
                    chosen_rows = [x for x in rows0 if is_chosen(x.get("chosen_for_continuation"))]
                    if len(chosen_rows) == 1:
                        seen_actual.update(
                            chosen_rows[0].get("selected_roles_training_only") or []
                        )
                    continue

                # State before round r should normally be round r-1.
                state = states.get((sid, rnd - 1))
                if state is None:
                    # fallback in case stored indexing uses same round
                    state = states.get((sid, rnd))
                if state is None:
                    missing_state += 1
                    continue

                role_probs = state.get("role_probs") or {}
                k_probs = state.get("k_probs") or {}
                if not role_probs or not k_probs:
                    missing_state += 1
                    continue

                roles = list(role_probs.keys())
                role_to_i = {r: i for i, r in enumerate(roles)}
                valid_roles = set(roles)

                mu, Sigma, cov_info = coherent_covariance(
                    k_probs, role_probs, roles
                )

                prepared = []
                for x in rows0:
                    selected = tuple(sorted(
                        x.get("selected_roles_training_only") or []
                    ))
                    truth = tuple(sorted(
                        x.get("truth_roles_training_only") or []
                    ))
                    prompt = x.get("prompt") or ""
                    focus_roles = parse_focus_roles(prompt, valid_roles)

                    if not focus_roles:
                        parse_fail += 1
                        continue

                    a = np.zeros(len(roles), dtype=np.float64)
                    for r in focus_roles:
                        a[role_to_i[r]] = 1.0

                    scores = score_candidate(Sigma, a, args.noise)

                    prepared.append({
                        "candidate_index": int(x.get("candidate_index") or 0),
                        "query": x.get("query"),
                        "chosen": is_chosen(x.get("chosen_for_continuation")),
                        "selected": selected,
                        "truth": truth,
                        "focus_roles": focus_roles,
                        "selector_predicted_utility": x.get("selector_predicted_utility"),
                        "scores": scores,
                    })

                if len(prepared) < args.min_pool:
                    chosen_rows = [x for x in rows0 if is_chosen(x.get("chosen_for_continuation"))]
                    if len(chosen_rows) == 1:
                        seen_actual.update(
                            chosen_rows[0].get("selected_roles_training_only") or []
                        )
                    continue

                oracle_row = max(prepared, key=lambda r: oracle_key(r, seen_actual))
                oracle_new = len(set(oracle_row["selected"]) - seen_actual)

                rewards = [
                    len(set(r["selected"]) - seen_actual)
                    for r in prepared
                ]

                group_result = {
                    "parent": str(parent),
                    "session_id": sid,
                    "round": rnd,
                    "pool_size": len(prepared),
                    "seen_actual_before": sorted(seen_actual),
                    "covariance_info": cov_info,
                    "oracle_candidate": oracle_row["candidate_index"],
                    "oracle_new_exposed": oracle_new,
                    "candidates": [],
                }

                for r, reward in zip(prepared, rewards):
                    rr = dict(r)
                    rr["oracle_new_exposed"] = reward
                    group_result["candidates"].append(rr)

                for method in methods:
                    chosen = choose_by_score(prepared, method)
                    chosen_new = len(set(chosen["selected"]) - seen_actual)
                    top1 = int(chosen["candidate_index"] == oracle_row["candidate_index"])
                    regret = oracle_new - chosen_new
                    sp = rank_corr(
                        [r["scores"][method] for r in prepared],
                        rewards,
                    )

                    group_result[f"{method}_chosen_candidate"] = chosen["candidate_index"]
                    group_result[f"{method}_chosen_new_exposed"] = chosen_new
                    group_result[f"{method}_regret"] = regret
                    group_result[f"{method}_spearman"] = sp

                    A = agg[method]
                    A["groups"] += 1
                    A["top1_oracle_match"] += top1
                    A["new_exposed_sum"] += chosen_new
                    A["oracle_new_sum"] += oracle_new
                    A["regret_sum"] += regret
                    if sp is not None:
                        A["spearman"].append(sp)

                    if rnd == 1:
                        B = agg_r1[method]
                        B["groups"] += 1
                        B["top1_oracle_match"] += top1
                        B["new_exposed_sum"] += chosen_new
                        B["oracle_new_sum"] += oracle_new
                        B["regret_sum"] += regret
                        if sp is not None:
                            B["spearman"].append(sp)

                per_group.append(group_result)
                kept_groups += 1

                # Update seen using the ACTUAL chosen row, not our counterfactual choice.
                actual = [r for r in prepared if r["chosen"]]
                if len(actual) == 1:
                    seen_actual.update(actual[0]["selected"])
                else:
                    # fallback from raw rows
                    actual0 = [x for x in rows0 if is_chosen(x.get("chosen_for_continuation"))]
                    if len(actual0) == 1:
                        seen_actual.update(
                            actual0[0].get("selected_roles_training_only") or []
                        )

    def summarize(block):
        outm = {}
        for method, A in block.items():
            n = A["groups"]
            outm[method] = {
                "groups": n,
                "top1_oracle_match_rate":
                    (A["top1_oracle_match"] / n if n else None),
                "mean_new_exposed":
                    (A["new_exposed_sum"] / n if n else None),
                "mean_oracle_new_exposed":
                    (A["oracle_new_sum"] / n if n else None),
                "mean_regret":
                    (A["regret_sum"] / n if n else None),
                "mean_within_group_spearman":
                    (float(np.mean(A["spearman"])) if A["spearman"] else None),
                "spearman_groups":
                    len(A["spearman"]),
            }
        return outm

    summary = {
        "root": str(root),
        "parents": len(parent_dirs),
        "total_groups_seen": total_groups,
        "kept_groups": kept_groups,
        "focus_parse_fail_rows": parse_fail,
        "missing_state_groups": missing_state,
        "noise": args.noise,
        "all_rounds_one_step_conditional_on_actual_history": summarize(agg),
        "round_1_clean_comparison": summarize(agg_r1),
        "notes": {
            "count": "Expected covariance reduction along the all-ones/count direction.",
            "identity": "Expected covariance reduction orthogonal to count.",
            "trace": "Expected total covariance trace reduction.",
            "eig1": "Expected reduction along top identity-ambiguity eigenvector.",
            "variance": "Raw predictive variance a^T Sigma a; no noise-normalized reduction.",
            "validity": "Round 1 is cleanest. Later rounds condition on states produced by the original policy.",
            "probe_vector": "Parsed pre-execution V11 focus-region roles from each candidate prompt; hidden selected_roles are used only as oracle outcomes.",
        },
    }

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with (out / "per_group.jsonl").open("w", encoding="utf-8") as f:
        for row in per_group:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("=" * 100)
    print("MATRIX SELECTOR ORACLE DIAGNOSTIC")
    print("=" * 100)
    print("parents:", len(parent_dirs))
    print("total_groups_seen:", total_groups)
    print("kept_groups:", kept_groups)
    print("focus_parse_fail_rows:", parse_fail)
    print("missing_state_groups:", missing_state)
    print()

    def print_block(title, block):
        print(title)
        print("-" * len(title))
        s = summarize(block)
        for method in methods:
            x = s[method]
            print(
                f"{method:10s} "
                f"n={x['groups']:5d} "
                f"top1={x['top1_oracle_match_rate']!s:>8} "
                f"new={x['mean_new_exposed']!s:>8} "
                f"oracle={x['mean_oracle_new_exposed']!s:>8} "
                f"regret={x['mean_regret']!s:>8} "
                f"rho={x['mean_within_group_spearman']!s:>8}"
            )
        print()

    print_block("ROUND 1 — CLEAN COMPARISON", agg_r1)
    print_block("ALL ROUNDS — ONE-STEP CONDITIONAL", agg)

    print("[saved]", out / "summary.json")
    print("[saved]", out / "per_group.jsonl")


if __name__ == "__main__":
    main()
