from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.io import read_jsonl, write_json, write_jsonl
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from adaptive_v11_curriculum.belief_utils import role_only_predict, dummy_k_probs
from adaptive_v11_curriculum.reward import curriculum_reward, group_relative_advantages, role_log_score, role_brier, margin
from adaptive_v11_curriculum_v2.selector import feature_vector, update_model
from v11.role_universe import TRAIN_ROLES


def auc_simple(probs, true_roles):
    truth = set(true_roles)
    pos = [probs[r] for r in TRAIN_ROLES if r in truth]
    neg = [probs[r] for r in TRAIN_ROLES if r not in truth]
    if not pos or not neg:
        return float('nan')
    wins = sum((a > b) + 0.5 * (a == b) for a in pos for b in neg)
    return wins / (len(pos) * len(neg))


def topk_recall(probs, true_roles):
    k = len(true_roles)
    top = sorted(TRAIN_ROLES, key=lambda r: probs[r], reverse=True)[:k]
    return len(set(top) & set(true_roles)) / max(1, k)


def shaped_group_advantages(rewards: list[float]) -> list[float]:
    """Keep relative ranking, but never call an all-negative group a success."""
    if not rewards:
        return []
    xs = np.asarray(rewards, dtype=np.float64)
    if float(np.max(xs)) <= 0.0:
        # Every candidate hurt inference: push all sampled probes down, with magnitude by actual harm.
        return [float(np.clip(r, -1.5, 0.0)) for r in xs]
    rel = np.asarray(group_relative_advantages(xs.tolist()), dtype=np.float64)
    abs_term = 0.50 * np.clip(xs, -1.0, 1.0)
    return [float(x) for x in np.clip(rel + abs_term, -4.0, 4.0)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--split', choices=['train', 'dev'], required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--belief', required=True)
    ap.add_argument('--selector-model', required=True)
    ap.add_argument('--stage', type=int, default=1)
    a = ap.parse_args()

    root = Path(a.root)
    states = {
        x['session_id']: PublicBeliefState.from_public_dict(x)
        for x in read_jsonl(root / f"states_{a.split}_round_{a.round-1:02d}.jsonl")
    }
    cand = read_jsonl(root / f"candidates_{a.split}_round_{a.round:02d}.jsonl")
    resp = read_jsonl(root / f"responses_{a.split}_round_{a.round:02d}.jsonl")
    scored = read_jsonl(root / f"scored_{a.split}_round_{a.round:02d}.jsonl")
    mani = json.loads(Path(a.manifest).read_text())
    truth = {x['session_id']: x for x in mani['systems'] if x['split'] == a.split}
    bundle = FrozenBeliefBundle.load(a.belief)

    resp_by = {
        (
            str(x.get('session_id', x.get('system_id', ''))),
            int(x.get('candidate_index', dict(x.get('action', {})).get('variant', 0))),
        ): x
        for x in resp
    }
    score_by = {}
    for x in scored:
        sid = str(x.get('system_id', x.get('session_id', '')))
        idx = int(dict(x.get('action', {})).get('variant', x.get('candidate_index', 0)))
        score_by[(sid, idx)] = x

    grouped = defaultdict(list)
    for x in cand:
        grouped[x['session_id']].append(x)

    train_rows = []
    next_states = []
    metrics = []
    selector_X = []
    selector_y = []
    selection_diags = []

    for sid, cs in sorted(grouped.items()):
        st = states[sid]
        tr = truth[sid]
        true_roles = tuple(map(str, tr['roles']))
        evals = []

        for c in sorted(cs, key=lambda x: int(x['candidate_index'])):
            idx = int(c['candidate_index'])
            rr = resp_by.get((sid, idx))
            sc = score_by.get((sid, idx))
            frame = None
            if c.get('valid') and rr and rr.get('execution_status') == 'ok' and sc is not None:
                frame = EvidenceFrame(
                    query=str(c['query']),
                    answer=str(rr.get('final_output', '')),
                    v135={str(k): float(v) for k, v in dict(sc.get('evidence', {})).items()},
                    v134={str(k): float(v) for k, v in dict(sc.get('metadata', {}).get('evidence_v134', {})).items()},
                )
                nh = [*st.history, frame]
                nr = role_only_predict(bundle, nh)
            else:
                nh = list(st.history)
                nr = dict(st.role_probs)

            rb = curriculum_reward(
                old_probs=st.role_probs,
                new_probs=nr,
                true_roles=true_roles,
                roles=TRAIN_ROLES,
                query=str(c.get('query', '')),
                previous_queries=[h.query for h in st.history],
                generator_valid=bool(c.get('valid') and frame is not None),
            )
            evals.append((c, frame, nr, rb))

        rewards = [float(e[3].total) for e in evals]
        advs = shaped_group_advantages(rewards)
        chosen_item = None
        chosen_reward = None
        chosen_idx = None

        for (c, frame, nr, rb), adv in zip(evals, advs):
            chosen = bool(c.get('chosen_for_continuation')) and frame is not None
            if chosen:
                chosen_item = (c, frame, nr, rb)
                chosen_reward = float(rb.total)
                chosen_idx = int(c['candidate_index'])

            if a.split == 'train' and c.get('completion'):
                train_rows.append(
                    {
                        'session_id': sid,
                        'round': a.round,
                        'candidate_index': int(c['candidate_index']),
                        'group_id': c['group_id'],
                        'prompt': c['prompt'],
                        'completion': c['completion'],
                        'old_logp': float(c['old_logp']),
                        'ref_logp': float(c['ref_logp']),
                        'reward': float(rb.total),
                        'local_advantage': float(adv),
                        'chosen_for_continuation': bool(chosen),
                        'selector_predicted_utility': float(c.get('selector_predicted_utility', float('nan'))),
                        'selector_model_samples': int(c.get('selector_model_samples', 0)),
                        'selector_mode': str(c.get('selector_mode', 'unknown')),
                        'reward_breakdown_training_only': rb.to_dict(),
                    }
                )

            if a.split == 'train' and frame is not None and bool(c.get('valid')):
                selector_X.append(feature_vector(st, str(c.get('query', '')), int(c['candidate_index'])))
                selector_y.append(float(rb.total))

        if chosen_item is None:
            ns = PublicBeliefState(
                session_id=sid,
                split=a.split,
                round_index=a.round,
                k_probs=dummy_k_probs(bundle),
                role_probs=dict(st.role_probs),
                history=list(st.history),
            )
            status = 'carry_no_valid_chosen'
        else:
            c, frame, nr, rb = chosen_item
            ns = PublicBeliefState(
                session_id=sid,
                split=a.split,
                round_index=a.round,
                k_probs=dummy_k_probs(bundle),
                role_probs=dict(nr),
                history=[*st.history, frame],
            )
            status = 'ok'

        ns.validate(TRAIN_ROLES, bundle.ks)
        next_states.append(ns.to_public_dict())
        metrics.append(
            {
                'session_id': sid,
                'round': a.round,
                'status': status,
                'role_log_loss': -role_log_score(ns.role_probs, true_roles, TRAIN_ROLES),
                'role_brier': role_brier(ns.role_probs, true_roles, TRAIN_ROLES),
                'role_auc': auc_simple(ns.role_probs, true_roles),
                'true_false_margin': margin(ns.role_probs, true_roles, TRAIN_ROLES),
                'oracle_k_role_recall': topk_recall(ns.role_probs, true_roles),
            }
        )

        finite_rewards = [r for (c, frame, nr, rb), r in zip(evals, rewards) if frame is not None]
        if finite_rewards:
            best_reward = float(max(finite_rewards))
            worst_reward = float(min(finite_rewards))
            if chosen_reward is not None:
                rank = 1 + sum(1 for r in finite_rewards if r > chosen_reward)
                regret = best_reward - chosen_reward
            else:
                rank = None
                regret = None
            selection_diags.append(
                {
                    'session_id': sid,
                    'round': a.round,
                    'chosen_candidate': chosen_idx,
                    'chosen_reward': chosen_reward,
                    'best_reward': best_reward,
                    'worst_reward': worst_reward,
                    'regret': regret,
                    'chosen_rank': rank,
                    'pool_size': len(finite_rewards),
                    'pool_has_positive': bool(best_reward > 0.0),
                }
            )

    write_jsonl(root / f"states_{a.split}_round_{a.round:02d}.jsonl", next_states)
    write_jsonl(root / f"metrics_{a.split}_round_{a.round:02d}.jsonl", metrics)
    write_jsonl(root / f"selector_diag_{a.split}_round_{a.round:02d}.jsonl", selection_diags)

    selector_summary = None
    if a.split == 'train':
        write_jsonl(root / f"train_rows_round_{a.round:02d}.jsonl", train_rows)
        # Idempotent selector update: update id is persisted atomically inside the .npz model.
        update_key = f"{root.resolve()}|stage={a.stage}|round={a.round}"
        update_hash = hashlib.sha256(update_key.encode('utf-8')).hexdigest()
        selector_summary = update_model(
            a.selector_model, selector_X, selector_y, update_id=update_hash
        )
        selector_summary['update_key'] = update_key

    chosen_vals = [x['chosen_reward'] for x in selection_diags if x.get('chosen_reward') is not None]
    regrets = [x['regret'] for x in selection_diags if x.get('regret') is not None]
    summary = {
        'split': a.split,
        'stage': int(a.stage),
        'round': a.round,
        'systems': len(metrics),
        'train_rows': len(train_rows),
        'mean_role_log_loss': float(np.mean([x['role_log_loss'] for x in metrics])) if metrics else float('nan'),
        'valid_state_updates': sum(x['status'] == 'ok' for x in metrics),
        'pool_positive_fraction': float(np.mean([x['pool_has_positive'] for x in selection_diags])) if selection_diags else float('nan'),
        'mean_best_reward': float(np.mean([x['best_reward'] for x in selection_diags])) if selection_diags else float('nan'),
        'mean_chosen_reward': float(np.mean(chosen_vals)) if chosen_vals else float('nan'),
        'mean_selector_regret': float(np.mean(regrets)) if regrets else float('nan'),
        'chosen_is_best_fraction': float(np.mean([x.get('chosen_rank') == 1 for x in selection_diags if x.get('chosen_rank') is not None])) if selection_diags else float('nan'),
        'selector_model': selector_summary,
    }
    write_json(root / f"advance_{a.split}_round_{a.round:02d}_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
