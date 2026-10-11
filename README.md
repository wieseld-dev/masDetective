# MAS RL Detective — Offline Pilot V2

Current goal: validate whether **adaptive benign probe selection has measurable headroom** on the 300 real MAS interactions already collected.

This stage is deliberately **not RL yet**. The previous context-free bandit was removed because it only learned a global probe ranking and could waste budget on multiple probes for the same role.

The pilot now asks a simpler question first:

> Is there a meaningful gap between a strong fixed probe set and a system-specific best probe set under the same query budget?

If there is no gap, RL has little to learn on this dataset. If there is a gap, the next stage adds a state-dependent policy.

Expected data links:

```text
data/real_role_presence/
├── public_outputs.csv
├── truth.csv
├── probe_bank.json
└── compositions.json
```

Run:

```bash
source .venv/bin/activate
python -m pytest -q
python -m scripts.run_value_pilot
```

No API calls are made.

Important limitation: this old dataset assumes a known universe of five semantic roles and one worker per present role, so `K == number of present roles`. This is useful for an offline plumbing pilot, but it is **not** the final general hierarchical-MAS formulation.
