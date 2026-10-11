# Adaptive Continuing Probing v1

This package is intentionally isolated from the currently queued V11.2 / V13.5 jobs. It adds new files only.

## Core methodological correction

Do **not** update a belief by adding `logit(calibrator_p)` or by treating V13.5 evidence as an independent likelihood ratio. Repeated probes are correlated, and V13.5 produces evidence scores, not calibrated Bayesian LRs.

Instead this package learns a **prefix aggregator** from grouped synthetic systems. For each candidate role it summarizes the entire history (raw evidence, within-answer robust z-scores, ranks, targeted-vs-cross evidence, repetition counts, trends) and learns `P(role present | history)` with group-disjoint train/dev systems.

## Controller layers

1. **Random** — lower baseline.
2. **Max uncertainty** — query the role whose current learned presence probability has highest binary entropy.
3. **Empirical VOI** — for each candidate action, sample plausible future all-role evidence vectors from a training-only empirical outcome model, pass each hypothetical history back through the learned prefix aggregator, and estimate expected entropy reduction. Ask only when expected information gain exceeds query cost; otherwise STOP.
4. **Oracle one-step upper bound** — uses hidden truth and cached future observations; never deployable. It is used only to quantify headroom and generate imitation labels.
5. **Oracle imitation initialization** — learns an action-value scorer from oracle utilities on training systems, without exposing truth at inference.

## Dedicated adaptive replay bank

The strong V11.2 generator benchmark is **not sufficient** for controller training because its PLUS systems only contain actions for targets that are present. That would leak action availability and make controller evaluation artificially easy.

Therefore Adaptive v1 creates a separate locked synthetic bank where **every hidden system receives the same action library for all 20 candidate roles**, regardless of truth. This gives both present-target and absent-target observations under a deployment-like action space.

Default bank:
- 24 controller-train systems
- 8 group-held-out controller-dev systems
- K=3..6 and single/fanout2/fanout3/mixed
- compositions disjoint from V11 training; strong-eval compositions are also excluded if its manifest exists
- 1 V11.2 probe per role by default => 20 black-box queries/system, 640 interactions total
- truth/selected-role traces are stored only for offline labels/evaluation

Reserve a **new seed and new compositions** for final paper testing.

## End-to-end pipeline (after V11.2 and V13.5 freeze)

```text
final V11.2 adapter
  -> generate fixed benign action library (all 20 roles)
  -> lock adaptive train/dev MAS manifest
  -> CPU/API collect every action on every hidden system
  -> frozen V13.5 scores each visible response against all 20 roles
  -> train prefix belief aggregator on train systems only
  -> fit empirical outcome model on train systems only
  -> compare Random / Max-Uncertainty / VOI / Oracle over query budgets
  -> optionally train oracle-imitation action scorer
```

Primary final metric: **role extraction quality as a function of black-box queries**. Report blind F1/exact-set, expected-K error, mean true rank, average queries, STOP rate, and known-K recall/exact only as an explicitly labeled diagnostic upper bound.

## What can run now

`adaptive_legacy_smoke.py` can fit the mechanics on the old V13.2 cross-role matrices while the new models train. This is **only a software smoke test**: those systems are development data and V13.2 is known to be biased.
