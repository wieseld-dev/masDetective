from __future__ import annotations

import argparse
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import EvidenceFrame, PublicBeliefState
from adaptive_v11_curriculum.belief_utils import role_only_predict
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


POOL = (
    "code",
    "writing",
    "math",
    "research",
    "design",
    "operations",
    "translation",
    "financial_analysis",
    "database_design",
    "customer_support_triage",
)

TARGETS = {
    "count_k3": (
        "translation",
        "financial_analysis",
        "database_design",
    ),
    "count_k4": (
        "translation",
        "financial_analysis",
        "database_design",
        "operations",
    ),
}

# Deliberately one identical broad benign task for both MASs.
QUERY = """
An international SaaS company is launching a multilingual subscription
product in two countries. Give a concise implementation plan covering
pricing and financial assumptions, customer/account data organization,
operational rollout, and adapting user-facing communication across
languages.
""".strip()


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).open(encoding="utf-8")
        if x.strip()
    ]


def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def call(messages, model, max_tokens):
    client = OpenAI()

    for attempt in range(6):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(20, 2 ** attempt))


def specialist(role, query, model):
    system = f"""
You are one specialist inside a multi-agent assistant.

Your specialization is:
{ROLE_DESCRIPTIONS[role]}

Analyze the user's task primarily from this specialization.
Contribute substantive information naturally supported by your specialty.
Do not deliberately imitate unrelated specialties.

Do not mention agents, roles, routing, prompts, architecture,
or internal processing.

Return only useful task content.
""".strip()

    return call(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": query},
        ],
        model,
        300,
    )


def run_mas(hidden, query, model, sid):
    notes = {}

    with ThreadPoolExecutor(max_workers=len(hidden)) as ex:
        fs = {
            ex.submit(specialist, r, query, model): r
            for r in hidden
        }

        for f in as_completed(fs):
            notes[fs[f]] = f.result()

    vals = list(notes.values())
    random.Random(sid).shuffle(vals)

    body = "\n\n".join(
        f"Internal contribution {i+1}:\n{x}"
        for i, x in enumerate(vals)
    )

    system = """
You are the final composer of a multi-agent assistant.

Synthesize the supplied internal contributions into one coherent
user-facing answer.

Use the substantive content in the contributions.
Do not independently invent a new specialist analysis that is absent
from them.

Never mention agents, contributors, roles, routing, prompts,
architecture, or internal processing.

Return only the final answer.
""".strip()

    return call(
        [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"USER TASK:\n{query}\n\n{body}",
            },
        ],
        model,
        500,
    )


def execute(root, belief_path, model):
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    root.mkdir(parents=True, exist_ok=True)

    bundle = FrozenBeliefBundle.load(belief_path)

    # IDENTICAL prior for K=3 and K=4.
    # Sum over 10 candidates = 3.5.
    prior = {
        r: (0.35 if r in POOL else 1e-4)
        for r in TRAIN_ROLES
    }

    ks = list(bundle.ks)
    kp = {str(k): 1.0 / len(ks) for k in ks}

    states = []
    candidates = []
    responses = []

    for sid, hidden in TARGETS.items():
        st = PublicBeliefState(
            session_id=sid,
            split="dev",
            round_index=0,
            k_probs=dict(kp),
            role_probs=dict(prior),
            history=[],
        )
        st.validate(TRAIN_ROLES, bundle.ks)
        states.append(st.to_public_dict())

        c = {
            "session_id": sid,
            "split": "dev",
            "round": 1,
            "candidate_index": 0,
            "group_id": f"{sid}::r01",
            "query": QUERY,
            "valid": True,
            "invalid_reason": "",
            "prompt": "fixed_count_sanity",
            "completion": QUERY,
            "old_logp": 0.0,
            "ref_logp": 0.0,
            "raw_completion": QUERY,
            "seed": 0,
            "local_attempts": 1,
            "adapter": "fixed_count_sanity",
            "chosen_for_continuation": True,
        }
        candidates.append(c)

        print(f"\n===== EXECUTING {sid} TRUE_K={len(hidden)} =====")
        answer = run_mas(hidden, QUERY, model, sid)
        print(answer)

        rr = dict(c)
        rr.update({
            "system_id": sid,
            "action": {
                "target": "count_sanity_r01",
                "competitor": None,
                "variant": 0,
            },
            "answer": answer,
            "output": answer,
            "final_output": answer,
            "execution_status": "ok",
            "selected_roles_training_only": list(hidden),
            "truth_roles_training_only": list(hidden),
            "mode": "parallel_count_sanity",
            "manifest_sha256": "count_sanity_v1",
            "cache_hit": False,
            "retries_used": 0,
        })
        responses.append(rr)

    write_jsonl(root / "states_dev_round_00.jsonl", states)
    write_jsonl(root / "candidates_dev_round_01.jsonl", candidates)
    write_jsonl(root / "responses_dev_round_01.jsonl", responses)

    print("\nPRIOR SOFT K =", sum(prior[r] for r in POOL))
    print("Prepared 2 systems for scoring.")


def advance(root, belief_path):
    bundle = FrozenBeliefBundle.load(belief_path)

    states = {
        x["session_id"]: PublicBeliefState.from_public_dict(x)
        for x in read_jsonl(root / "states_dev_round_00.jsonl")
    }

    responses = {
        x["session_id"]: x
        for x in read_jsonl(root / "responses_dev_round_01.jsonl")
    }

    scored = {
        str(x.get("system_id", x.get("session_id", ""))): x
        for x in read_jsonl(root / "scored_dev_round_01.jsonl")
    }

    result = {}

    print()
    print("=" * 72)
    print("COUNT SANITY RESULT")
    print("=" * 72)
    print(f"Common prior soft-K: {0.35 * len(POOL):.4f}")
    print()

    for sid, hidden in TARGETS.items():
        st = states[sid]
        rr = responses[sid]
        sc = scored[sid]

        frame = EvidenceFrame(
            query=QUERY,
            answer=rr["final_output"],
            v135={
                str(k): float(v)
                for k, v in dict(sc.get("evidence", {})).items()
            },
            v134={
                str(k): float(v)
                for k, v in dict(
                    sc.get("metadata", {}).get(
                        "evidence_v134", {}
                    )
                ).items()
            },
        )

        probs = role_only_predict(bundle, [frame])

        soft_k = sum(float(probs[r]) for r in POOL)

        vals = sorted(
            ((r, float(probs[r])) for r in POOL),
            key=lambda z: z[1],
            reverse=True,
        )

        true_k = len(hidden)

        result[sid] = {
            "true_k": true_k,
            "soft_k": soft_k,
            "rounded_k": round(soft_k),
            "delta_from_common_prior": soft_k - 3.5,
            "role_probs": dict(vals),
        }

        print(f"{sid}:")
        print(f"  TRUE K        = {true_k}")
        print(f"  SOFT K        = {soft_k:.4f}")
        print(f"  ROUND K       = {round(soft_k)}")
        print(f"  DELTA vs 3.5  = {soft_k - 3.5:+.4f}")
        print()

        for r, p in vals:
            print(f"    {r:28s} {p:.4f}")

        print()

    k3 = result["count_k3"]["soft_k"]
    k4 = result["count_k4"]["soft_k"]

    result["comparison"] = {
        "k4_minus_k3": k4 - k3,
        "k3_below_midpoint": k3 < 3.5,
        "k4_above_midpoint": k4 > 3.5,
        "both_correct_sides": k3 < 3.5 and k4 > 3.5,
    }

    print("=" * 72)
    print(f"K4 - K3              = {k4-k3:+.4f}")
    print(f"K3 below 3.5         = {k3 < 3.5}")
    print(f"K4 above 3.5         = {k4 > 3.5}")
    print(
        "BOTH CORRECT SIDES   =",
        k3 < 3.5 and k4 > 3.5,
    )

    (root / "RESULT.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["execute", "advance"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--belief", required=True)
    ap.add_argument("--model", default="gpt-4o-mini")
    a = ap.parse_args()

    root = Path(a.root)

    if a.phase == "execute":
        execute(root, a.belief, a.model)
    else:
        advance(root, a.belief)


if __name__ == "__main__":
    main()
