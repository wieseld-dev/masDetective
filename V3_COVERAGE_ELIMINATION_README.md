# V3 Coverage + Elimination Generator Pilot

## Decision

This pilot changes the optimization target, not merely the number of epochs.

The current failure mode is:
1. the visible detector can recover some activated roles;
2. the exact Bayesian posterior can recover K and the exact composition when probes cover the capability universe;
3. the current generator often reconfirms the same semantic region instead of eliminating plausible extra roles.

V3 therefore trains the generator on **deployment-visible posterior progress** plus an explicit public coverage term.

## Training loop

For each of 16 fresh train MAS systems:

1. exact prior over all 60,249 K=3..6 role subsets;
2. generate G=4 benign candidates from the current generator;
3. execute all four;
4. infer a visible activation fingerprint from query + final output;
5. update the exact Bayesian posterior for each candidate;
6. training-only reward:
   - 50% clipped Δ log P(exact true composition)
   - 25% clipped Δ log P(true K)
   - 15% true-vs-false role-margin improvement
   - 10% public coverage/diagnosticity bonus
   - repetition penalty
7. continuation candidate is chosen **before execution**, uniformly among valid generated candidates;
8. chosen candidate updates the real trajectory;
9. after six rounds, one conservative GRPO/PPO-style update is run with the existing curriculum updater.

Truth never enters the posterior update or the generator state. It is used only for the scalar training reward and metrics.

## Coverage state

For each role i and previous chosen probe q:

d_i(q) =
| E[P(E_i=1)|Z_i=1,q] - E[P(E_i=1)|Z_i=0,q] |

Coverage is the max d_i over previous chosen probes. The prompt prioritizes capabilities with:
- meaningful posterior uncertainty/plausibility, and
- low previous diagnostic coverage.

This explicitly targets the missing "prove there is no fourth/fifth agent" behavior.

## Cost

Default:
- 16 train systems
- 6 rounds
- 4 candidates/round
- max 384 MAS calls
- 8 fresh dev systems are reserved and untouched by training

## Install

```bash
cd ~/mas_rl_detective
unzip -o v3_coverage_elimination_bundle.zip
chmod +x cluster/submit_v3_coverage_elimination.sh
python -m py_compile \
  analysis/v3_common.py \
  analysis/v3_coverage_policy.py \
  analysis/v3_prepare.py \
  analysis/v3_generate.py \
  analysis/v3_execute.py \
  analysis/v3_advance.py \
  analysis/v3_report.py
```

## Zero-API preparation first

Run this before submitting the expensive chain:

```bash
python -u analysis/v3_prepare.py \
  --root artifacts/v3_coverage_elimination_pilot \
  | tee artifacts/v3_coverage_elimination_pilot/PREP.log
```

Check:
```bash
cat artifacts/v3_coverage_elimination_pilot/prepare_summary.json
```

Expected:
- exact_hypotheses = 60249
- train_systems = 16
- dev_systems = 8
- visible bridge API checks pass

## Submit training

```bash
V3_CONFIRM=YES \
bash cluster/submit_v3_coverage_elimination.sh
```

Track:
```bash
cat artifacts/v3_coverage_elimination_pilot/jobs.tsv
squeue --me
```

Report:
```bash
python analysis/v3_report.py \
  --root artifacts/v3_coverage_elimination_pilot \
  --horizon 6
```

## Gate before scaling

Do not run a larger V3 curriculum until this pilot shows:
- coverage rises across rounds;
- mean P(true composition) rises;
- K-MAP or K-GAP accuracy improves rather than collapsing;
- candidate reward has nontrivial within-group spread;
- update KL remains controlled.

The eight dev systems remain untouched for a clean post-training comparison against the epoch-2 generator.
