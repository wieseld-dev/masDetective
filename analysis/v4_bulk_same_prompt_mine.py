from __future__ import annotations

import argparse, hashlib, json, random, sys
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


V3 = Path("artifacts/v3_coverage_elimination_pilot")
GOLD = Path("artifacts/v4_userlike_audit/gold_strict.jsonl")
OUT = Path("artifacts/v4_bulk_mining")
ADAPTER = "artifacts/robust_visible_generator_v1_epoch2/policy"

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def read_jsonl(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def seed_for(*xs):
    b = "||".join(map(str, xs)).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], "big") % (2**31-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-prompt", type=int, default=32)
    args = ap.parse_args()

    gold = read_jsonl(GOLD)
    assert len(gold) == 24, f"expected 24 strict Gold, got {len(gold)}"

    cache = {}

    for r in sorted(set(int(x["round"]) for x in gold)):
        states = {
            str(x["session_id"]): x
            for x in read_jsonl(V3 / f"states_train_round_{r-1:02d}.jsonl")
        }

        ids, cov = load_matrix_npz(
            V3 / f"coverage_train_round_{r-1:02d}.npz",
            "coverage",
        )

        cache[r] = (
            states,
            {str(sid): cov[i] for i, sid in enumerate(ids)},
        )

    contexts = []

    for g in gold:
        r = int(g["round"])
        sid = str(g["session_id"])
        j = int(g["candidate_index"])

        states, cov_by = cache[r]

        assert sid in states
        assert sid in cov_by

        st = PublicBeliefState.from_public_dict(states[sid])
        st.validate(TRAIN_ROLES, sorted(st.k_probs))

        cv = cov_by[sid]

        obs = {
            "belief_state": st.to_public_dict(),
            "coverage": {
                role: float(cv[i])
                for i, role in enumerate(TRAIN_ROLES)
            },
            "candidate_index": j,
            "max_history_turns": 6,
        }

        contexts.append({
            "gold_index": int(g["gold_index"]),
            "round": r,
            "session_id": sid,
            "candidate_index": j,
            "prompt": str(g["prompt"]),
            "chosen": str(g["completion"]),
            "chosen_query": str(g["query"]),
            "obs": obs,
        })

    print("strict Gold:", len(contexts), flush=True)
    print(
        "planned generations:",
        len(contexts) * args.samples_per_prompt,
        flush=True,
    )

    print("=== LOADING EPOCH2 ===", flush=True)

    policy = CoverageEliminationProbePolicyV3(
        MODEL,
        adapter_path=ADAPTER,
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "samples.jsonl"

    total = valid_n = prompt_ok = 0

    with out.open("w", encoding="utf-8") as f:

        for ci, c in enumerate(contexts, 1):

            seen = set()

            for sidx in range(args.samples_per_prompt):

                # Half exploitation, half diversity.
                if sidx < args.samples_per_prompt // 2:
                    temp, top_p, regime = .85, .95, "standard"
                else:
                    temp, top_p, regime = 1.10, .98, "explore"

                seed = seed_for(
                    20260924,
                    c["gold_index"],
                    c["round"],
                    c["session_id"],
                    c["candidate_index"],
                    sidx,
                )

                random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                s = policy.sample_probe(
                    c["obs"],
                    temperature=temp,
                    top_p=top_p,
                    max_new_tokens=128,
                )

                p = str(s.prompt)

                if p != c["prompt"]:
                    raise RuntimeError(
                        f"PROMPT MISMATCH G{c['gold_index']} sample={sidx}"
                    )

                prompt_ok += 1

                q = str(s.query or "").strip()
                completion = str(s.completion or "")

                surf, why = surface_valid(q)

                valid = (
                    bool(s.valid)
                    and bool(surf)
                    and bool(q)
                    and q != c["chosen_query"]
                )

                norm = " ".join(q.lower().split())
                unique = bool(norm) and norm not in seen

                if norm:
                    seen.add(norm)

                rec = {
                    "gold_index": c["gold_index"],
                    "round": c["round"],
                    "session_id": c["session_id"],
                    "candidate_index": c["candidate_index"],
                    "sample_index": sidx,

                    "prompt": p,
                    "chosen": c["chosen"],
                    "chosen_query": c["chosen_query"],

                    "completion": completion,
                    "query": q,

                    "seed": seed,
                    "regime": regime,
                    "temperature": temp,
                    "top_p": top_p,

                    "valid": valid,
                    "unique_within_prompt": unique,
                    "invalid_reason": (
                        "" if valid else
                        str(getattr(s, "invalid_reason", "")) or str(why)
                    ),
                }

                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                total += 1
                valid_n += int(valid)

            print(
                f"[mine] {ci:02d}/24 | total={total}",
                flush=True,
            )

    assert total == 24 * args.samples_per_prompt
    assert prompt_ok == total

    rows = read_jsonl(out)

    usable = [
        x for x in rows
        if x["valid"] and x["unique_within_prompt"]
    ]

    coverage = len(set(x["gold_index"] for x in usable))

    print()
    print("================================")
    print(" BULK MINING SUMMARY")
    print("================================")
    print("total            :", total)
    print("prompt identity  :", f"{prompt_ok}/{total}")
    print("valid            :", valid_n)
    print("valid + unique   :", len(usable))
    print("Gold coverage    :", f"{coverage}/24")
    print("wrote:", out)


if __name__ == "__main__":
    main()
