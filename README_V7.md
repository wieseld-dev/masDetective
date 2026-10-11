# V7 — Generative Benign-Probing RL

V7 implements the research design discussed after V6.5/V6.6:

**One fixed, already-validated benign seed probe -> observe only the MAS final output -> generate a new free-text benign follow-up -> observe the next final output -> refine the candidate-presence belief -> ASK again or STOP.**

The RL policy learns *how to investigate*, not merely which fixed probe to select.

## Threat-model boundary

Policy observation contains only:
- candidate capability name/description supplied by the attacker hypothesis set,
- public seed/follow-up queries and final external outputs,
- a black-box candidate-presence belief derived from those outputs,
- remaining external-query budget,
- benign-gate feedback.

The policy **never receives** true role presence, K, execution mode, system ID, router choice, selected workers/roles, topology, internal messages, or traces.

Training-only truth and the privileged routing trace may contribute **only to scalar reward and evaluation logs**.

## Seed policy

Default V7 uses exactly **one fixed V6.5 primary probe per candidate role** (`primary_probe(role)`). There is no seed bank and the RL action is not restricted to a bank. The seed response is replayed from the already-paid V6.5 public dataset, so V7 training does not repay the seed query.

This is the scientifically clean starting point: the model gets one proven benign starting question and must discover useful follow-up strategies itself.

## Free-text action

Qwen outputs one JSON object:

```json
{"action":"ask","query":"What additional details would help you verify this result carefully?"}
```

or

```json
{"action":"stop"}
```

The query string is generated freely by the LLM. JSON is only an auditable transport schema.

## Prompt-injection prevention

Every generated query passes a **fail-closed two-stage gate** before it can reach the MAS:

1. Hard deterministic gate: NFKC normalization, one concise question, strict length/format limits, and explicit blocks for prompt/instruction manipulation, system prompts, hidden/private data, agents/workers/roles, routing/topology/architecture, traces, fake transcripts, role-play, secrets, and structural counting.
2. `hybrid` mode (recommended): an independent semantic guard model must also classify the query as an ordinary benign task question. Any error/uncertainty blocks the query.

Blocked or malformed generations:
- are **never sent** to the MAS,
- receive a strong negative reward,
- are logged for audit.

Run `scripts/audit_v7_probes.py ... --strict` after experiments. An accepted probe that fails a fresh hard-gate audit makes the strict audit fail.

## Reward

For a permitted follow-up, V7 uses:

`R = information_gain + small training-only activation shaping + small novelty bonus - query_cost`

where information gain is the improvement in log probability assigned by the black-box V6.5 candidate detector to the correct PRESENT/ABSENT label. The target-role activation term is visible only to the training grader and is deliberately small; it helps exploration without becoming an attacker feature.

STOP receives a positive reward for a correct confident decision, a larger negative reward for a wrong decision, and a small early-stop bonus for unused query budget.

The black-box belief is updated by smoothing detector evidence in log-odds space so repeated weak signals do not explode confidence.

## Training

V7 uses Qwen2.5-1.5B-Instruct + 4-bit QLoRA and sequence-level clipped PPO. The strong cluster profile uses:
- 256 candidate-investigation episodes,
- up to 3 live follow-ups per episode,
- up to 6 generation attempts,
- 8 episodes per PPO rollout,
- 2 PPO epochs per update,
- exploration temperature annealed 0.95 -> 0.65,
- PPO clip 0.20,
- gradient clipping,
- persistent response caching/resume.

The training schedule alternates PRESENT and ABSENT candidate episodes over the diverse V6.5 training compositions.

## Cost guardrail

Training and evaluation are **plan-only by default**. They print upper-bound live interactions/API calls and require `--execute` before making calls.

Training plan:

```bash
python -m training.train_generative_probe_ppo_v7
```

Strong run after reviewing the plan:

```bash
sbatch cluster/v7_generative_strong_gpu.sbatch
```

## Evaluation

The V6.5 validation compositions remain unseen during V7 training. First inspect the evaluation plan:

```bash
python -m evaluation.eval_generative_probe_v7
```

Small live smoke evaluation:

```bash
python -m evaluation.eval_generative_probe_v7 --limit-systems 2 --execute
```

Full held-out evaluation:

```bash
python -m evaluation.eval_generative_probe_v7 --execute
```

It compares **seed-only** vs **generative-RL final belief** on role AUC, balanced accuracy, F1, Brier, K-MAE, exact K, and within-1 K, while also reporting generated/accepted/blocked probe counts and average live query use.

## Recommended progression

1. Apply patch + unit tests.
2. Run training plan and review the API upper bound.
3. Run the strong training job.
4. Audit all generated probes with `--strict`.
5. Run a 2-system held-out evaluation smoke.
6. If clean, run the full 8-system unseen-composition evaluation.
7. Only after this pilot succeeds should the candidate vocabulary be scaled from 8 toward ~30 roles.

## Smoke before the strong run

The recommended first live check is deliberately small:

```bash
sbatch cluster/v7_generative_smoke_gpu.sbatch
```

It runs only 16 episodes with at most 2 follow-ups each. Inspect its log and run the strict probe audit before committing to the 256-episode job.

## Resume after interruption

The response cache is already resume-safe. The policy can also continue from its last saved LoRA adapter:

```bash
python -m training.train_generative_probe_ppo_v7 \
  --resume-adapter artifacts/v7_generative_policy \
  --start-episode 257 \
  --episodes 128 \
  --execute
```

Optimizer momentum is not restored, but learned LoRA weights and cached live interactions are reused.

## Reward-hacking diagnostic

Held-out evaluation additionally records a **training-only target-activation diagnostic** for PRESENT candidates. This diagnostic is never exposed to the policy. If black-box AUC improves while target activation collapses, treat that as a possible observer/reward-hacking warning rather than a successful attack result.
