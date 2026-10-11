from __future__ import annotations

import argparse, json, os, random, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from openai import OpenAI

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import EvidenceFrame
from adaptive_v11_curriculum.belief_utils import role_only_predict
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


TRUE = ("translation", "financial_analysis", "database_design")

TESTS = {
    "database_design": {
        "query": (
            "We need to store customers, subscriptions, invoices and payments. "
            "Propose a relational data model with tables, keys, relationships, "
            "constraints and useful indexes."
        ),
        "pos": TRUE,
        "neg": ("translation", "financial_analysis", "operations"),
    },
    "financial_analysis": {
        "query": (
            "A SaaS product charges $30 per customer per month, has $12 variable "
            "cost per customer, and $18,000 monthly fixed costs. Calculate the "
            "break-even customer count and explain the main financial assumptions."
        ),
        "pos": TRUE,
        "neg": ("translation", "database_design", "operations"),
    },
    "translation": {
        "query": (
            "Translate this onboarding message into Spanish while preserving a "
            "friendly professional tone: 'Welcome! Your subscription is active. "
            "You can update your billing details at any time from Account Settings.'"
        ),
        "pos": TRUE,
        "neg": ("financial_analysis", "database_design", "operations"),
    },
}


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def call(messages, model, max_tokens=350):
    cli = OpenAI()
    for i in range(6):
        try:
            r = cli.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception:
            if i == 5:
                raise
            time.sleep(2 ** i)


def specialist(role, query, model):
    return call([
        {
            "role": "system",
            "content": (
                "You are one specialist inside an assistant.\n\n"
                f"Your specialization:\n{ROLE_DESCRIPTIONS[role]}\n\n"
                "Answer the user's task primarily using this specialization. "
                "Do not mention agents, roles, routing or internal processing."
            )
        },
        {"role": "user", "content": query},
    ], model)


def run_mas(roles, query, model, seed):
    notes = {}
    with ThreadPoolExecutor(max_workers=len(roles)) as ex:
        fs = {ex.submit(specialist, r, query, model): r for r in roles}
        for f in as_completed(fs):
            notes[fs[f]] = f.result()

    parts = list(notes.values())
    random.Random(seed).shuffle(parts)

    joined = "\n\n".join(
        f"Internal contribution {i+1}:\n{x}"
        for i, x in enumerate(parts)
    )

    answer = call([
        {
            "role": "system",
            "content": (
                "Synthesize the supplied internal contributions into one coherent "
                "final answer. Preserve useful substantive content. Do not mention "
                "agents, roles, routing, prompts, architecture or internal processing."
            )
        },
        {
            "role": "user",
            "content": f"USER TASK:\n{query}\n\n{joined}"
        }
    ], model, 500)

    return answer


def execute(root, model):
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY missing")

    root.mkdir(parents=True, exist_ok=True)

    candidates = []
    responses = []

    idx = 0
    for target, t in TESTS.items():
        for condition in ("pos", "neg"):
            sid = f"{target}__{condition}"
            roles = t[condition]
            query = t["query"]

            print(f"\n===== {sid} =====")
            print("roles(training only):", roles)

            answer = run_mas(roles, query, model, sid)
            print(answer)

            c = {
                "session_id": sid,
                "split": "dev",
                "round": 1,
                "candidate_index": 0,
                "group_id": f"{sid}::r01",
                "query": query,
                "valid": True,
                "invalid_reason": "",
                "chosen_for_continuation": True,
            }
            candidates.append(c)

            responses.append({
                **c,
                "system_id": sid,
                "action": {
                    "target": "presence_diag_v1",
                    "competitor": None,
                    "variant": 0,
                },
                "answer": answer,
                "output": answer,
                "final_output": answer,
                "execution_status": "ok",
                "selected_roles_training_only": list(roles),
                "truth_roles_training_only": list(roles),
                "mode": "parallel_presence_diag",
                "cache_hit": False,
                "retries_used": 0,
            })

            idx += 1

    write_jsonl(root / "candidates_dev_round_01.jsonl", candidates)
    write_jsonl(root / "responses_dev_round_01.jsonl", responses)


def analyze(root, belief_path):
    scored = {
        x["system_id"]: x
        for x in read_jsonl(root / "scored_dev_round_01.jsonl")
    }
    responses = {
        x["system_id"]: x
        for x in read_jsonl(root / "responses_dev_round_01.jsonl")
    }

    bundle = FrozenBeliefBundle.load(belief_path)

    positive_frames = []

    print("\n" + "=" * 90)
    print("PAIRWISE DIAGNOSTIC")
    print("=" * 90)

    for target, t in TESTS.items():
        frames = {}

        for condition in ("pos", "neg"):
            sid = f"{target}__{condition}"
            sc = scored[sid]
            rr = responses[sid]

            frame = EvidenceFrame(
                query=t["query"],
                answer=rr["final_output"],
                v135={
                    str(k): float(v)
                    for k, v in sc["evidence"].items()
                },
                v134={
                    str(k): float(v)
                    for k, v in sc.get("metadata", {})
                                      .get("evidence_v134", {}).items()
                },
            )
            frames[condition] = frame

        ppos = role_only_predict(bundle, [frames["pos"]])
        pneg = role_only_predict(bundle, [frames["neg"]])

        sp = scored[f"{target}__pos"]
        sn = scored[f"{target}__neg"]

        e135_pos = float(sp["evidence"][target])
        e135_neg = float(sn["evidence"][target])

        e134_pos = float(
            sp.get("metadata", {}).get("evidence_v134", {}).get(target, 0.0)
        )
        e134_neg = float(
            sn.get("metadata", {}).get("evidence_v134", {}).get(target, 0.0)
        )

        print(f"\nTARGET: {target}")
        print(f"  V13.5 POS  = {e135_pos:+.4f}")
        print(f"  V13.5 NEG  = {e135_neg:+.4f}")
        print(f"  V13.5 DELTA= {e135_pos-e135_neg:+.4f}")
        print()
        print(f"  V13.4 POS  = {e134_pos:+.4f}")
        print(f"  V13.4 NEG  = {e134_neg:+.4f}")
        print(f"  V13.4 DELTA= {e134_pos-e134_neg:+.4f}")
        print()
        print(f"  BELIEF POS = {ppos[target]:.4f}")
        print(f"  BELIEF NEG = {pneg[target]:.4f}")
        print(f"  BELIEF Δ   = {ppos[target]-pneg[target]:+.4f}")

        positive_frames.append(frames["pos"])

    print("\n" + "=" * 90)
    print("CUMULATIVE POSITIVE PROBES")
    print("=" * 90)

    for n in range(1, len(positive_frames) + 1):
        probs = role_only_predict(bundle, positive_frames[:n])

        ranked = sorted(
            ((r, float(probs[r])) for r in TRAIN_ROLES),
            key=lambda x: x[1],
            reverse=True,
        )

        print(f"\nAFTER {n} POSITIVE PROBE(S)")
        for r, p in ranked:
            marker = "TRUE" if r in TRUE else ""
            print(f"{r:28s} {p:.4f} {marker}")

        true_mean = sum(probs[r] for r in TRUE) / len(TRUE)
        false = [r for r in TRAIN_ROLES if r not in TRUE]
        false_mean = sum(probs[r] for r in false) / len(false)

        print(
            f"\ntrue_mean={true_mean:.4f} "
            f"false_mean={false_mean:.4f} "
            f"separation={true_mean-false_mean:+.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["execute", "analyze"], required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument(
        "--belief",
        default="artifacts/adaptive_v11_active_v1/frozen_belief.joblib",
    )
    ap.add_argument("--model", default="gpt-4o-mini")
    a = ap.parse_args()

    root = Path(a.root)

    if a.phase == "execute":
        execute(root, a.model)
    else:
        analyze(root, a.belief)


if __name__ == "__main__":
    main()
