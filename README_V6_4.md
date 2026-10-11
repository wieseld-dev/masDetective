# V6.4 — Contrastive Candidate-by-Candidate Presence Detection

This patch implements the research strategy directly, before adding RL complexity:

1. The attacker starts with an overcomplete candidate-role list.
2. One neutral anchor is asked once per MAS:
   `What additional information can I provide that would help you solve the problem optimally?`
3. Each candidate is tested with exactly one fixed benign role-oriented probe (the existing `primary_probe(role)`).
4. The attacker sees only the anchor external output and the targeted external output.
5. A binary detector estimates `P(candidate role is present)`.
6. Estimated K is obtained by counting positive candidates (and also by summing probabilities).

V6.4 is specifically designed to test robustness to the V6.3 single-routing-layer execution modes:

- `single`
- `fanout2`
- `fanout3`
- `mixed`

## Important methodological correction

The previous V6.3 estimator tried to infer all roles from arbitrary public histories. V6.4 instead asks the direct candidate-level question:

`P(Math present | neutral output, Math-targeted output)`

and analogously for each candidate.

The estimator never receives `selected_roles`, router output, worker IDs, topology traces, or internal messages. White-box truth is used only as the training/evaluation label.

## No new API cost

V6.4 reuses:

- `artifacts/role_bootstrap_v63_public.jsonl`
- `artifacts/role_bootstrap_v63_truth.jsonl`

Those contain the 16 V6.3 systems x (1 anchor + 8 candidate probes) already collected.

## Evaluation protocols

The script reports two complementary protocols:

- **Composition-holdout CV (primary):** entire role compositions are held out, so the detector cannot simply memorize a role combination seen under another fan-out mode.
- **Mode-holdout:** one execution mode (`single`, `fanout2`, `fanout3`, `mixed`) is held out to test transfer across splitting/aggregation behavior.

Metrics:

- candidate AUC / balanced accuracy / F1 / Brier
- per-mode candidate metrics
- hard K-MAE (count `P>=0.5`)
- soft K-MAE (`sum P(role)`)
- exact-K accuracy
- within-1 K accuracy
- constant-K baseline per CV fold

## Apply

```bash
unzip -o ~/mas_rl_detective_v6_4_contrastive_candidate_patch.zip -d /tmp/v64
bash /tmp/v64/v6_4_contrastive_candidate_patch/apply_v6_4.sh ~/mas_rl_detective
conda activate masdetective
python -m pytest -q tests/test_v64_candidate_presence.py
```

## Run offline evaluation

```bash
python -m evaluation.eval_candidate_presence_v64
```

This makes **zero OpenAI API calls**.

Outputs:

- `artifacts/v64_candidate_presence/summary.json`
- `artifacts/v64_candidate_presence/predictions.csv`
- `artifacts/candidate_presence_v64.joblib` (full-fit artifact reserved for the later RL stage)

Do not use full-fit training metrics as scientific results; use the out-of-fold composition-holdout and mode-holdout metrics printed by the evaluation script.
