K decision-rule benchmark on the CURRENT calibrated Bayesian bridge.

Purpose
-------
Test the hypothesis that role marginals already separate into a high-probability
"installed roles" cluster and a low-probability "absent roles" cluster, while
the explicit P(K) posterior MAP may choose the wrong K because probability mass
is spread across many superset hypotheses.

Method
------
- strict system split from artifacts/observation_model_v1/split_systems.json
- validation used only to choose threshold tau
- untouched test systems for reported final metrics
- fixed budget = 5 recorded trajectory probes
- uses ONLY chosen_for_continuation rows, never counterfactual sibling candidates
- exact 60,249 hypotheses for K=3..6
- two parallel tracks:
    1) ORACLE: true selected_roles evidence, beta=1.0 (diagnostic only)
    2) VISIBLE: query+final-output fingerprint, calibrated deployment beta
- no new MAS/API calls

Decision rules compared:
- posterior P(K) MAP
- largest gap in sorted role marginals, k constrained to 3..6
- 1-D two-cluster split
- round(sum role marginals)
- threshold count; tau selected on validation only

Also reports:
- top-true-K role-set exact recovery
- gap-derived role-set exact recovery
- fraction where every true role outranks every false role
- true/false role margin
- gap size

Run
---
unzip -o k_gap_exact_current_v1_bundle.zip
mv -f k_gap_exact_current_v1.py analysis/
mv -f k_gap_exact_current_v1.sbatch cluster/
python -m py_compile analysis/k_gap_exact_current_v1.py

J=$(sbatch --parsable cluster/k_gap_exact_current_v1.sbatch)
echo "$J"
tail -f logs/k-gap-exact-${J}.out

Results:
cat artifacts/k_gap_exact_current_v1/summary.json
