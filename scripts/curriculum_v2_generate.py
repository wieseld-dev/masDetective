from __future__ import annotations

import argparse
import gc
import random
from pathlib import Path

from adaptive_v11_active.io import read_jsonl, stable_int, write_jsonl
from adaptive_v11_active.reward import surface_valid
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe
from adaptive_v11_curriculum.policy import CurriculumAdaptiveProbePolicyV11
from adaptive_v11_curriculum_v2.selector import choose_candidate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True)
    ap.add_argument('--split', choices=['train', 'dev'], required=True)
    ap.add_argument('--root', required=True)
    ap.add_argument('--adapter', required=True)
    ap.add_argument('--v11-reference', default='artifacts/v11_policy')
    ap.add_argument('--group-size', type=int, default=6)
    ap.add_argument('--model-name', default='Qwen/Qwen2.5-1.5B-Instruct')
    ap.add_argument('--seed', type=int, default=20260918)
    ap.add_argument('--temperature', type=float, default=.85)
    ap.add_argument('--top-p', type=float, default=.95)
    ap.add_argument('--max-new-tokens', type=int, default=192)
    ap.add_argument('--max-seq-length', type=int, default=4096)
    ap.add_argument('--local-retries', type=int, default=12)
    ap.add_argument('--selector-model', required=True)
    ap.add_argument('--selector-epsilon', type=float, default=0.20)
    ap.add_argument('--selector-min-samples', type=int, default=96)
    ap.add_argument('--stage', type=int, default=1)
    a = ap.parse_args()

    root = Path(a.root)
    states_path = root / f"states_{a.split}_round_{a.round-1:02d}.jsonl"
    if not states_path.exists():
        raise SystemExit(f"missing {states_path}")
    if a.group_size < 1:
        raise SystemExit('--group-size must be >= 1')

    import torch

    current = CurriculumAdaptiveProbePolicyV11(
        a.model_name,
        adapter_path=a.adapter,
        trainable=False,
        use_4bit=False,
        max_seq_length=a.max_seq_length,
    )
    reference = CurriculumAdaptiveProbePolicyV11(
        a.model_name,
        adapter_path=a.v11_reference,
        trainable=False,
        use_4bit=False,
        max_seq_length=a.max_seq_length,
    )

    states = read_jsonl(states_path)
    rows = []
    pool_rows = []
    G = int(a.group_size)

    for si, raw in enumerate(states):
        assert_policy_safe(raw)
        st = PublicBeliefState.from_public_dict(raw)
        seen = [h.query for h in st.history]
        group = []

        for j in range(G):
            last = None
            for retry in range(max(1, a.local_retries)):
                seed = stable_int(a.seed, a.split, a.stage, a.round, st.session_id, j, retry)
                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                obs = {
                    'belief_state': st.to_public_dict(),
                    'candidate_index': j,
                    'max_history_turns': 5,
                }
                sample = current.sample_probe(
                    obs,
                    temperature=a.temperature,
                    top_p=a.top_p,
                    max_new_tokens=a.max_new_tokens,
                )
                q = str(sample.query or '').strip()
                ok, why = surface_valid(q)
                valid = bool(sample.valid) and ok and q not in seen and all(q != x.get('query') for x in group)
                last = (sample, q, valid, why, seed, retry)
                if valid:
                    break

            if last is None:
                raise RuntimeError('candidate generation loop produced no sample')
            sample, q, valid, why, seed, retry = last
            prompt = str(sample.prompt)
            completion = str(sample.completion)
            old_lp = getattr(sample, 'old_mean_logprob', None)
            if old_lp is None and completion:
                old_lp = float(current.mean_logprob(prompt, completion).detach().cpu())
            if old_lp is None:
                old_lp = -20.0
            ref_lp = float(reference.mean_logprob(prompt, completion).detach().cpu()) if completion else -20.0

            rec = {
                'session_id': st.session_id,
                'split': a.split,
                'round': a.round,
                'candidate_index': j,
                'group_id': f"{st.session_id}::r{a.round:02d}",
                'query': q,
                'valid': bool(valid),
                'invalid_reason': '' if valid else (str(getattr(sample, 'invalid_reason', '')) or why or 'duplicate'),
                'prompt': prompt,
                'completion': completion,
                'old_logp': float(old_lp),
                'ref_logp': float(ref_lp),
                'raw_completion': str(sample.raw_completion),
                'seed': int(seed),
                'local_attempts': int(retry + 1),
                'adapter': str(a.adapter),
            }
            group.append(rec)
            if valid:
                seen.append(q)

        epsilon = float(a.selector_epsilon) if a.split == 'train' else 0.0
        chosen, scored, selector_mode = choose_candidate(
            a.selector_model,
            st,
            group,
            epsilon=epsilon,
            seed_parts=(a.seed, 'selector', a.split, a.stage, a.round, st.session_id),
            min_samples=a.selector_min_samples,
        )
        score_by_idx = {int(x['candidate_index']): x for x in scored}

        for x in group:
            idx = int(x['candidate_index'])
            sel = score_by_idx.get(idx, {})
            x['chosen_for_continuation'] = bool(idx == chosen)
            x['selector_predicted_utility'] = float(sel.get('predicted_utility', float('-inf')))
            x['selector_model_samples'] = int(sel.get('selector_model_samples', 0))
            x['selector_mode'] = selector_mode
            x['selector_epsilon'] = epsilon
            x['curriculum_stage'] = int(a.stage)
            pool_rows.append(dict(x))

        if a.split == 'train':
            rows.extend(group)
        else:
            selected = next((x for x in group if int(x['candidate_index']) == chosen), None)
            if selected is None:
                # Preserve one row so downstream code carries the state instead of silently losing the system.
                selected = dict(group[0])
                selected['chosen_for_continuation'] = False
                selected['valid'] = False
                selected['invalid_reason'] = selected.get('invalid_reason') or 'selector_no_valid_candidate'
            rows.append(selected)

        chosen_pred = score_by_idx.get(chosen, {}).get('predicted_utility', float('nan'))
        print(
            f"[gen-v2] {si+1}/{len(states)} stage={a.stage} r={a.round} sid={st.session_id} "
            f"valid={sum(int(x['valid']) for x in group)}/{G} chosen={chosen} "
            f"pred={chosen_pred} mode={selector_mode} eps={epsilon:.3f}",
            flush=True,
        )

    out = root / f"candidates_{a.split}_round_{a.round:02d}.jsonl"
    write_jsonl(out, rows)
    pool_out = root / f"candidate_pool_{a.split}_round_{a.round:02d}.jsonl"
    write_jsonl(pool_out, pool_rows)
    print(
        f"[done-v2] {out} rows={len(rows)} pool_rows={len(pool_rows)} "
        f"valid_pool={sum(int(x['valid']) for x in pool_rows)}"
    )

    del current, reference
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
