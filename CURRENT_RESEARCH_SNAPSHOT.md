# MAS RL Detective — Source Snapshot

Snapshot date: 2026-10-11

Current attack pipeline:

Belief over candidate roles
-> Generator creates benign candidate probes
-> Selector chooses one probe
-> Black-box MAS returns only final response
-> Sensor scores role evidence
-> FrozenBelief updates role probabilities
-> repeat

Current main generator:
- Clean-R7
- Historical adaptive G=50 training at R7
- Runtime generation normally G=4

Hybrid generator:
- Uses Clean-R7 weights
- Different prompt / sampling / retry strategy
- It is not a separately trained checkpoint

Current long-run experiment:
artifacts/HIGHEST_CLEAN_AUC/experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1

Highest observed deployment-style AUC so far:
- 0.823529 at R24
- Long-horizon experiment is being extended toward R50

Current Generator TargetAux experiment:
artifacts/HIGHEST_CLEAN_AUC/experiments/16_CLEAN_R7_TARGETAUX_V1

TargetAux idea:
global belief-improvement reward
+
small target-specific PRESENT-vs-ABSENT auxiliary reward

R7 reconstruction:
- 1200 rows
- 24 systems
- G=50
- 1020 non-zero advantages
- parent checkpoint: Clean-R6
- epochs: 1
- lr: 1.5e-6
- clip: 0.20
- kl_beta: 0.05

TargetAux variants:
- global_sign
- composite_sign
- control
