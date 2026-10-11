from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import torch

ROOTDIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOTDIR))
sys.path.insert(0, str(ROOTDIR / "analysis"))

from adaptive_v11_active.schema import PublicBeliefState
from adaptive_v11_active.reward import surface_valid
from v11.role_universe import TRAIN_ROLES
from v3_common import load_matrix_npz
from v3_coverage_policy import CoverageEliminationProbePolicyV3


V3ROOT = Path("artifacts/v3_coverage_elimination_pilot")
GOLDROOT = Path("artifacts/v4_gold_train_v1")
OUT = Path("artifacts/v4_dpo_mining")
ADAPTER = Path("artifacts/robust_visible_generator_v1_epoch2/policy")

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SAMPLES_PER_PROMPT = 8
TEMPERATURE = 1.00
TOP_P = 0.95
MAX_NEW_TOKENS = 128
BASE_SEED = 20260924


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def stable_seed(*parts) -> int:
    s = "||".join(map(str, parts)).encode("utf-8")
    h = hashlib.sha256(s).digest()
    return int.from_bytes(h[:8], "big") % (2**31 - 1)


# ------------------------------------------------------------
# 1. Load the exact 45 manually-audited Gold training examples
# ------------------------------------------------------------

gold = []

for r in range(1, 7):
    p = GOLDROOT / f"train_rows_round_{r:02d}.jsonl"
    if p.exists():
        gold.extend(read_jsonl(p))

assert len(gold) == 45, f"expected exactly 45 Gold rows, got {len(gold)}"

keys = [
    (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )
    for x in gold
]

assert len(set(keys)) == 45, "duplicate Gold context keys"

print("=== GOLD INVENTORY ===", flush=True)
print("gold rows:", len(gold), flush=True)
print("unique prompt-context keys:", len(set(keys)), flush=True)


# ------------------------------------------------------------
# 2. Cache exact historical public states + coverage
# ------------------------------------------------------------

round_cache = {}

for r in sorted(set(int(x["round"]) for x in gold)):
    state_path = V3ROOT / f"states_train_round_{r-1:02d}.jsonl"
    cov_path = V3ROOT / f"coverage_train_round_{r-1:02d}.npz"

    if not state_path.exists():
        raise FileNotFoundError(state_path)

    if not cov_path.exists():
        raise FileNotFoundError(cov_path)

    states = {
        str(x["session_id"]): x
        for x in read_jsonl(state_path)
    }

    cov_ids, cov_mat = load_matrix_npz(cov_path, "coverage")
    cov_by = {
        str(sid): cov_mat[i]
        for i, sid in enumerate(cov_ids)
    }

    round_cache[r] = (states, cov_by)


# ------------------------------------------------------------
# 3. Reconstruct every observation BEFORE model loading.
#    This catches schema/path mistakes without wasting GPU time.
# ------------------------------------------------------------

contexts = []

for idx, g in enumerate(gold):
    r = int(g["round"])
    sid = str(g["session_id"])
    j = int(g["candidate_index"])

    states, cov_by = round_cache[r]

    if sid not in states:
        raise RuntimeError(f"missing state r={r} sid={sid}")
    if sid not in cov_by:
        raise RuntimeError(f"missing coverage r={r} sid={sid}")

    st = PublicBeliefState.from_public_dict(states[sid])
    st.validate(TRAIN_ROLES, sorted(st.k_probs))

    cv = cov_by[sid]

    if len(cv) != len(TRAIN_ROLES):
        raise RuntimeError(
            f"bad coverage width sid={sid}: {len(cv)} != {len(TRAIN_ROLES)}"
        )

    coverage = {
        role: float(cv[i])
        for i, role in enumerate(TRAIN_ROLES)
    }

    obs = {
        "belief_state": st.to_public_dict(),
        "coverage": coverage,
        "candidate_index": j,
        "max_history_turns": 6,
    }

    contexts.append({
        "gold_index": idx,
        "round": r,
        "session_id": sid,
        "candidate_index": j,
        "obs": obs,
        "gold_prompt": str(g["prompt"]),
        "gold_completion": str(g["completion"]),
        "gold_reward": float(g.get("reward", 0.0)),
        "gold_advantage": float(g.get("local_advantage", 0.0)),
    })

assert len(contexts) == 45

print("state/coverage reconstruction: PASS", flush=True)
print(
    "planned generations:",
    len(contexts) * SAMPLES_PER_PROMPT,
    flush=True,
)


# ------------------------------------------------------------
# 4. Load ONLY Epoch2.
# ------------------------------------------------------------

print("\n=== LOADING EPOCH2 ===", flush=True)

policy = CoverageEliminationProbePolicyV3(
    MODEL,
    adapter_path=str(ADAPTER),
    trainable=False,
    use_4bit=False,
    max_seq_length=4096,
)

print("Epoch2 loaded.", flush=True)


# ------------------------------------------------------------
# 5. Same-prompt sampling
# ------------------------------------------------------------

OUT.mkdir(parents=True, exist_ok=True)
out_path = OUT / "same_prompt_samples.jsonl"

all_rows = []
prompt_checks = 0
valid_n = 0
unique_n = 0
changed_n = 0


with out_path.open("w", encoding="utf-8") as fout:

    for ci, c in enumerate(contexts, 1):

        seen_queries = set()

        for sidx in range(SAMPLES_PER_PROMPT):

            seed = stable_seed(
                BASE_SEED,
                c["round"],
                c["session_id"],
                c["candidate_index"],
                sidx,
            )

            random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            sample = policy.sample_probe(
                c["obs"],
                temperature=TEMPERATURE,
                top_p=TOP_P,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            generated_prompt = str(sample.prompt)

            # Absolutely essential for valid DPO pairing.
            if generated_prompt != c["gold_prompt"]:
                raise RuntimeError(
                    "PROMPT MISMATCH\n"
                    f"round={c['round']} sid={c['session_id']} "
                    f"j={c['candidate_index']} sample={sidx}"
                )

            prompt_checks += 1

            query = str(sample.query or "").strip()
            completion = str(sample.completion or "")
            ok_surface, why_surface = surface_valid(query)

            valid = bool(sample.valid) and bool(ok_surface)

            is_unique = query not in seen_queries
            if query:
                seen_queries.add(query)

            gold_text = str(c["gold_completion"]).strip()
            changed_from_gold = query != gold_text

            valid_n += int(valid)
            unique_n += int(is_unique)
            changed_n += int(changed_from_gold)

            row = {
                "gold_index": c["gold_index"],
                "round": c["round"],
                "session_id": c["session_id"],
                "candidate_index": c["candidate_index"],
                "sample_index": sidx,
                "seed": seed,

                # Exact DPO pair context:
                "prompt": generated_prompt,

                # Known manually-audited positive:
                "chosen_completion_gold": c["gold_completion"],
                "gold_reward": c["gold_reward"],
                "gold_advantage": c["gold_advantage"],

                # Candidate rejected completion to be judged:
                "sampled_completion": completion,
                "sampled_query": query,

                "valid": valid,
                "surface_reason": "" if ok_surface else str(why_surface),
                "sample_valid": bool(sample.valid),
                "sample_invalid_reason": str(
                    getattr(sample, "invalid_reason", "")
                ),

                "unique_within_prompt": is_unique,
                "changed_from_gold": changed_from_gold,
                "adapter": str(ADAPTER),
            }

            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
            all_rows.append(row)

        if ci % 5 == 0 or ci == len(contexts):
            print(
                f"[mine] {ci:02d}/45 prompts "
                f"| generated={len(all_rows)}",
                flush=True,
            )


# ------------------------------------------------------------
# 6. Sanity report
# ------------------------------------------------------------

assert len(all_rows) == 45 * SAMPLES_PER_PROMPT
assert prompt_checks == len(all_rows)

valid_rows = [x for x in all_rows if x["valid"]]
distinct_queries = len(set(x["sampled_query"] for x in all_rows if x["sampled_query"]))

summary = {
    "gold_prompts": 45,
    "samples_per_prompt": SAMPLES_PER_PROMPT,
    "total_samples": len(all_rows),
    "prompt_identity_checks_passed": prompt_checks,
    "valid_samples": valid_n,
    "valid_fraction": valid_n / len(all_rows),
    "unique_within_prompt_flags": unique_n,
    "changed_from_gold": changed_n,
    "globally_distinct_nonempty_queries": distinct_queries,
    "adapter": str(ADAPTER),
    "temperature": TEMPERATURE,
    "top_p": TOP_P,
}

(OUT / "mining_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\n======================================", flush=True)
print(" SAME-PROMPT NEGATIVE MINING SUMMARY", flush=True)
print("======================================", flush=True)

print("Gold prompts:             45", flush=True)
print(
    f"samples:                  {len(all_rows)}",
    flush=True,
)
print(
    f"prompt identity:          {prompt_checks}/{len(all_rows)}",
    flush=True,
)
print(
    f"valid:                    {valid_n}/{len(all_rows)} "
    f"= {valid_n/len(all_rows):.3f}",
    flush=True,
)
print(
    f"changed from Gold:        {changed_n}/{len(all_rows)} "
    f"= {changed_n/len(all_rows):.3f}",
    flush=True,
)
print(
    f"distinct nonempty query:  {distinct_queries}",
    flush=True,
)
print("wrote:", out_path, flush=True)
print("wrote:", OUT / "mining_summary.json", flush=True)
