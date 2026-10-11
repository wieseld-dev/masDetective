# V6 — Overcomplete Candidate Roles + Adaptive Benign RL

V6 changes the inference target from direct K-estimation to an overcomplete role-presence problem.

## Training/deployment separation

Policy-visible:
- known application description
- overcomplete candidate-role list
- public Q/O history
- black-box P(role present) beliefs
- remaining query budget

Training-only white box:
- true role set
- selected worker/role
- execution path/edges

The white-box fields are used only by the reward and logs. They are never inserted into the policy prompt.

## Episode

1. Mandatory universal benign anchor probe.
2. Update black-box role beliefs.
3. Policy chooses a candidate hypothesis (`FOCUS`) and generates one benign task-oriented probe (`QUESTION`).
4. Hard benign gate blocks direct structure/agent/role-count questions before they reach the MAS.
5. Role estimator updates P(role present).
6. Reward = role-Brier improvement + expected-K improvement + new worker/edge/path coverage - costs.

## Pilot commands

Plan bootstrap (0 API calls):

```bash
python -m training.bootstrap_role_estimator --limit-systems 8
```

Execute bootstrap:

```bash
python -m training.bootstrap_role_estimator --limit-systems 8 --execute
```

Plan RL (0 API calls):

```bash
python -m training.train_role_ppo --episodes 20 --budget 3
```

For unattended runs use `cluster/train_v6_role20.sbatch`.
