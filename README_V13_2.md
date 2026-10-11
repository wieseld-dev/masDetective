# V13.2 — dual-head causal / answer-attribution judge

This patch **does not modify** V11, V13, or V13.1 artifacts. It creates only:

- `artifacts/v13_2_dual_judge/`
- `artifacts/v13_2_pilot_replay/`

It makes **zero OpenAI/API calls**. The replay reuses the 120 public answers already stored in the V13 pilot.

## Why V13.2 exists

V13.1 showed the exact shortcut we wanted to detect:

- hard PLUS ≈ 0.946
- hard SHAM ≈ 0.905
- hard MINUS ≈ 0.351
- hard flip ≈ 0.288

The scalar judge often followed the target-conditioned question instead of attributing the behavior in the answer.

V13.2 changes the objective rather than just training longer:

1. **Evidence head** sees role + benign question + answer, but runtime evidence is
   `S(role,q,answer) - S(role,q,NO_ANSWER)`. This subtracts question-topic prior.
2. **Answer-attribution head** sees role + answer only. The question is deliberately hidden.
3. **Explicit MINUS flip** is upweighted: on MINUS, the selected hard replacement must outrank the absent target.
4. **RL-quality curriculum** uses SPEC/strict/confirmation/elite. Ordinary negative-SPEC branches are not treated as diagnostic supervision.
5. **V11 + V13** are used. V11 adds causal breadth; V13 adds hard-neighbor attribution.
6. **Stricter leakage grouping** links probe group, pair ID, exact query, exact duplicate output, and known V13 role-set composition.
7. **Model selection no longer hides hard-MINUS failure inside an average.** Hard-MINUS and hard-flip are explicit selection metrics.
8. A small 2-feature calibrator combines corrected evidence + answer-only attribution. It is fit on training components only and evaluated on held-out components.
9. The already-consumed V13 pilot is replayed offline only after model selection.

## Install patch

From your local machine, copy the tar to the cluster and extract it in the project root.

```bash
cd ~/mas_rl_detective
tar -xzf v13_2_dual_judge_patch.tar.gz
```

## Cheap checks before submission

```bash
cd ~/mas_rl_detective
python -m py_compile \
  v13_2/judge.py \
  analysis/train_v13_2_dual_judge.py \
  analysis/replay_v13_2_pilot.py

bash -n cluster/v13_2_dual_judge.sbatch
PYTHONPATH="$PWD" python tests/test_v13_2_logic.py
```

The unit tests are CPU-only and make no API calls.

## Submit

```bash
cd ~/mas_rl_detective
sbatch cluster/v13_2_dual_judge.sbatch
```

Then:

```bash
squeue --me
```

and once you have the job ID:

```bash
tail -f logs/v13-2-dual-<JOBID>.out
```

`Ctrl+C` only exits `tail`; it does not cancel the job.

## Outputs to send back

After the job finishes:

```bash
cat artifacts/v13_2_dual_judge/metrics.json
cat artifacts/v13_2_dual_judge/decision.json
cat artifacts/v13_2_pilot_replay/decision.json
tail -80 logs/v13-2-dual-<JOBID>.out
```

For detailed replay ranks:

```bash
jq '.systems[0] | {
  true_roles_eval_only,
  old_k_hat,
  new_k_hat,
  old_role_metrics_eval_only,
  new_role_metrics_eval_only,
  true_role_ranks_eval_only,
  ranking
}' artifacts/v13_2_pilot_replay/replay.json
```

## Validation gates

Fresh API calls remain blocked unless the held-out validation passes all of:

- evidence causal-edge accuracy >= 0.76
- answer-attribution hard-MINUS accuracy >= 0.60
- answer-attribution hard-flip accuracy >= 0.60
- combined endpoint AUC >= 0.75
- combined hard-flip accuracy >= 0.60

Even if those pass, the offline consumed-pilot replay must also improve before the patch recommends one fresh API pilot.

## Important scientific scope

This is still a **development** stage. The consumed V13 pilot is used only as an offline sanity check after model selection; it is not a final untouched paper test. Training truth/selected roles are labels only. At runtime the judge receives only public candidate description, benign diagnostic question, and public final answer.
