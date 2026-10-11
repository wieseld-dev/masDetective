# V5.2 — Benign RL correction

Fixes observed reward-hacking in the first 20-episode run:
- removes K / task-specialist-count objective from the policy prompt;
- does not expose K-belief in the textual policy prompt;
- hard-blocks direct/indirect hidden-structure language (experts, specialists, agents, roles, routing, architecture, etc.);
- blocks fake transcript labels and malformed/multi-line actions;
- blocked generated actions are **not sent to the MAS** and receive reward -1.0 while remaining PPO-trainable;
- shortens actions to 48 generated tokens;
- keeps low-memory PPO defaults (rollout-batch=2, ppo-epochs=1);
- logs block_reason and raw/clipped K information reward.

IMPORTANT: start the next run from the base Qwen model, not the contaminated V5 checkpoint.
