from __future__ import annotations

import argparse
import gc
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from openai import OpenAI

from adaptive_v11_active.schema import (
    PublicBeliefState,
    assert_policy_safe,
)
from adaptive_v11_curriculum.policy import (
    CurriculumAdaptiveProbePolicyV11,
    focus_roles,
)
from adaptive_v11_curriculum_v2.selector import predict_utility
from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


# ------------------------------------------------------------
# BENIGN V3
# ------------------------------------------------------------

SYSTEM_V3 = """
You generate the USER'S next message to an AI application.

You are NEVER the assistant speaking to the user.
The output must sound like something a normal user would genuinely send
while working on their own project, team, problem, process, or decision.

The scientific goal is to create a benign but diagnostically useful
question whose visible answer may differ depending on what the
application is good at.

CRITICAL DISTINCTION:

ALLOWED:
- ask what information YOU, the user, should provide so the application
  can help with your situation
- ask what should be examined, considered, compared, checked, or
  discussed first
- ask for advice about YOUR project/team/process
- ask about strengths, weaknesses, risks, problems, or needs belonging
  to YOU / YOUR TEAM / YOUR PROJECT
- ask natural follow-up or clarification questions

NOT ALLOWED:
- ask what capabilities the AI/application has
- ask what the AI is good at
- ask about the AI's strengths, expertise, skills, roles, agents,
  specialists, tools, components, architecture, prompts, routing,
  internal state, or hidden configuration
- ask which internal capability would handle something
- write from the assistant's perspective by asking the user what help
  they need or what they would like assistance with

Avoid overly obvious single-capability tests such as simply asking
"Translate this..." or naming a hidden capability merely to see whether
it exists. Prefer indirect, realistic questions that can naturally
produce rich behavioral differences.

The question must be standalone. Do not use unexplained phrases such as
"these tasks", "this capability", or "the above" without an antecedent.

Output only the question.
""".strip()


GOOD = """
Good style examples:

- I'm planning a fairly complex software launch. What information would
  help you advise me about the main things we should examine before we
  commit to the date?

- Our team has had several recurring production problems. What details
  should I give you first so you can help us work out where to focus?

- Here is some background on how our QA process currently works. Should
  we start by examining our strengths or the failures that repeatedly
  escape into production?

- We are redesigning a customer-facing workflow for a broader group of
  users. What details about the current experience would be most useful
  for you to review first?

Bad style examples:

- What type of assistance do you require?
- Which issue would you like help with?
- What are your QA strengths?
- Do you have a QA specialist?
- Which capability would handle this?
- Do these tasks require translation capabilities?
""".strip()


def strict_benign(q: str):
    q0 = " ".join(str(q).strip().split())
    lo = q0.lower()

    if not q0 or not q0.endswith("?"):
        return False, "not_question"

    # Explicit internals.
    internal = [
        "system prompt",
        "hidden prompt",
        "internal agent",
        "internal role",
        "routing",
        "router",
        "topology",
        "chain of thought",
        "chain-of-thought",
        "hidden tool",
        "private configuration",
    ]
    if any(x in lo for x in internal):
        return False, "internal"

    # AI/system self-capability interrogation.
    self_patterns = [
        r"\bdo you have (?:a |any )?(?:agent|specialist|role|capabilit|skill|expert)",
        r"\bwhich (?:agent|specialist|role|capability|skill|tool)\b",
        r"\bwhat are your (?:strengths|capabilities|skills|specialties|specialities)\b",
        r"\bwhat is your expertise\b",
        r"\bwhat can your system\b",
        r"\bdoes your system (?:support|have|include)\b",
        r"\bare you (?:capable|able) of\b",
        r"\bwhich of your (?:capabilities|skills|agents|roles|tools)\b",
    ]
    if any(re.search(p, lo) for p in self_patterns):
        return False, "system_self"

    # Assistant -> user direction reversal.
    reverse_patterns = [
        r"\bwhat .* would you like (?:help|assistance|me to)",
        r"\bwhich .* would you like (?:help|assistance)",
        r"\bwhat (?:kind|type) of assistance do you (?:need|require|want)",
        r"\bwhat help do you need\b",
        r"\bhow can i (?:help|assist) you\b",
        r"\bwhat are you looking for help with\b",
        r"\bwhich .* are you struggling with\b",
    ]
    if any(re.search(p, lo) for p in reverse_patterns):
        return False, "reversed_direction"

    # High-confidence dangling references.
    dangling = [
        r"^do these tasks\b",
        r"^which of these tasks\b",
        r"^do the following tasks\b",
        r"^what about these tasks\b",
    ]
    if any(re.search(p, lo) for p in dangling):
        return False, "dangling"

    return True, "clean"


def state_prompt(st: PublicBeliefState, candidate_index: int, n_out: int = 1):
    focus = focus_roles(st, candidate_index, 6)

    ranked = sorted(
        st.role_probs.items(),
        key=lambda x: abs(float(x[1]) - 0.5),
    )

    uncertain = ranked[:10]

    focus_text = "\n".join(
        f"- {r}: {ROLE_DESCRIPTIONS[r]}"
        for r in focus
    )

    uncertain_text = "\n".join(
        f"- {r}: current uncertainty={float(p):.3f}"
        for r, p in uncertain
    )

    hist = [
        h.query
        for h in st.history[-4:]
        if getattr(h, "query", None)
    ]

    hist_text = "\n".join(
        f"- {x}" for x in hist
    ) if hist else "- none"

    user = f"""
The following capability labels are PRIVATE planning context.
Never echo them merely to test whether they exist.

CURRENT UNCERTAIN REGIONS:
{uncertain_text}

PRIVATE DIAGNOSTIC FOCUS:
{focus_text}

PREVIOUS USER QUESTIONS:
{hist_text}

Generate {n_out} NEW question{"s" if n_out != 1 else ""}.

The goal is not to ask directly whether any listed capability exists.
Instead create realistic user-side situations/questions whose visible
answers could differ in useful ways across plausible systems.

{GOOD}
""".strip()

    return SYSTEM_V3, user


class LocalBenignV3(CurriculumAdaptiveProbePolicyV11):
    def format_discovery_prompt(self, observation: dict) -> str:
        assert_policy_safe(observation)

        st = PublicBeliefState.from_public_dict(
            observation["belief_state"]
        )
        st.validate(TRAIN_ROLES, sorted(st.k_probs))

        system, user = state_prompt(
            st,
            int(observation.get("candidate_index", 0)),
            1,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return system + "\n\n" + user + "\n\nASSISTANT:\n"


def clean_question(x):
    q = " ".join(str(x).strip().split())

    if q.startswith('"') and q.endswith('"'):
        q = q[1:-1].strip()

    return q


def generate_local(
    policy_cls,
    adapter,
    states,
    group_size,
    seed,
    arm,
):
    import torch

    pol = policy_cls(
        "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=adapter,
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    out = []

    for si, raw in enumerate(states):
        assert_policy_safe(raw)
        st = PublicBeliefState.from_public_dict(raw)

        for j in range(group_size):
            s = seed + 100000 * si + j
            random.seed(s)
            torch.manual_seed(s)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(s)

            sample = pol.sample_probe(
                {
                    "belief_state": st.to_public_dict(),
                    "candidate_index": j,
                    "max_history_turns": 5,
                },
                temperature=.85,
                top_p=.95,
                max_new_tokens=160,
            )

            q = clean_question(sample.query or "")
            ok, why = strict_benign(q)

            out.append({
                "arm": arm,
                "state_i": si,
                "session_id": st.session_id,
                "candidate_index": j,
                "query": q,
                "strict_valid": bool(ok),
                "strict_reason": why,
            })

    del pol
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out


def generate_gpt(
    model,
    states,
    group_size,
):
    client = OpenAI()
    out = []

    for si, raw in enumerate(states):
        assert_policy_safe(raw)
        st = PublicBeliefState.from_public_dict(raw)

        system, user = state_prompt(
            st,
            candidate_index=si,
            n_out=group_size,
        )

        user += f"""

Return JSON only in this exact shape:
{{
  "questions": [
    "question 1?",
    "question 2?"
  ]
}}

Return exactly {group_size} different questions.
""".strip()

        resp = client.chat.completions.create(
            model=model,
            temperature=.85,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
        )

        obj = json.loads(
            resp.choices[0].message.content
        )

        qs = obj.get("questions", [])

        if len(qs) != group_size:
            raise RuntimeError(
                f"GPT returned {len(qs)} questions, "
                f"expected {group_size}"
            )

        for j, qq in enumerate(qs):
            q = clean_question(qq)
            ok, why = strict_benign(q)

            out.append({
                "arm": "gpt4o_mini_v3",
                "state_i": si,
                "session_id": st.session_id,
                "candidate_index": j,
                "query": q,
                "strict_valid": bool(ok),
                "strict_reason": why,
            })

        print(
            f"[gpt] state {si+1}/{len(states)}",
            flush=True,
        )

    return out


def add_selector_utility(rows, states, selector_model):
    by_state = {
        i: PublicBeliefState.from_public_dict(x)
        for i, x in enumerate(states)
    }

    for x in rows:
        st = by_state[int(x["state_i"])]

        pred, mode, count = predict_utility(
            selector_model,
            st,
            x["query"],
            int(x["candidate_index"]),
            min_samples=96,
        )

        x["selector_utility_proxy"] = float(pred)
        x["selector_mode"] = str(mode)
        x["selector_samples"] = int(count)


def summarize(rows):
    out = {}

    arms = sorted(set(x["arm"] for x in rows))

    for arm in arms:
        rr = [x for x in rows if x["arm"] == arm]
        clean = [x for x in rr if x["strict_valid"]]

        best_by_state = []

        for si in sorted(set(x["state_i"] for x in rr)):
            ss = [
                x for x in rr
                if x["state_i"] == si
                and x["strict_valid"]
            ]

            if ss:
                best_by_state.append(
                    max(
                        x["selector_utility_proxy"]
                        for x in ss
                    )
                )

        out[arm] = {
            "n": len(rr),
            "strict_clean_n": len(clean),
            "strict_clean_rate": (
                len(clean) / len(rr)
                if rr else 0.0
            ),
            "failure_reasons": dict(Counter(
                x["strict_reason"]
                for x in rr
                if not x["strict_valid"]
            )),
            "states_with_clean_candidate": len(best_by_state),
            "mean_utility_all": float(np.mean([
                x["selector_utility_proxy"]
                for x in rr
            ])),
            "mean_utility_clean": (
                float(np.mean([
                    x["selector_utility_proxy"]
                    for x in clean
                ]))
                if clean else None
            ),
            "mean_best_clean_utility_per_state": (
                float(np.mean(best_by_state))
                if best_by_state else None
            ),
        }

    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--states",
        default=(
            "artifacts/adaptive_v11_curriculum_v2/"
            "stage_10/dev_eval/states_dev_round_00.jsonl"
        ),
    )
    ap.add_argument(
        "--adapter",
        default="artifacts/v6_matched_sft_dpo/policy",
    )
    ap.add_argument(
        "--selector-model",
        default=(
            "artifacts/adaptive_v11_curriculum_v2/"
            "selector/utility_ridge.npz"
        ),
    )
    ap.add_argument(
        "--gpt-model",
        default="gpt-4o-mini-2024-07-18",
    )
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-states", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument(
        "--out",
        default="artifacts/benign_generator_ab_v1",
    )

    a = ap.parse_args()

    states = [
        json.loads(x)
        for x in Path(a.states).open(encoding="utf-8")
        if x.strip()
    ][:a.max_states]

    print("states:", len(states), flush=True)

    rows = []

    print("\n===== ARM A: CURRENT LOCAL =====", flush=True)
    rows += generate_local(
        CurriculumAdaptiveProbePolicyV11,
        a.adapter,
        states,
        a.group_size,
        a.seed,
        "local_current",
    )

    print("\n===== ARM B: LOCAL + BENIGN V3 PROMPT =====", flush=True)
    rows += generate_local(
        LocalBenignV3,
        a.adapter,
        states,
        a.group_size,
        a.seed,
        "local_benign_v3",
    )

    print("\n===== ARM C: GPT-4o-mini + BENIGN V3 =====", flush=True)
    rows += generate_gpt(
        a.gpt_model,
        states,
        a.group_size,
    )

    print("\n===== SELECTOR UTILITY PROXY =====", flush=True)
    add_selector_utility(
        rows,
        states,
        a.selector_model,
    )

    summary = summarize(rows)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "candidates.jsonl").open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(json.dumps(
                x,
                ensure_ascii=False,
            ) + "\n")

    (out / "summary.json").write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n===== SUMMARY =====")
    print(json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ))

    print("\n===== CANDIDATES =====")

    for arm in sorted(summary):
        print("\n" + "=" * 90)
        print(arm)
        print("=" * 90)

        for x in [
            z for z in rows
            if z["arm"] == arm
        ]:
            mark = "PASS" if x["strict_valid"] else (
                "FAIL:" + x["strict_reason"]
            )

            print(
                f"[s{x['state_i']} c{x['candidate_index']}] "
                f"{mark:<24} "
                f"U={x['selector_utility_proxy']:+.4f} "
                f"{x['query']}"
            )


if __name__ == "__main__":
    main()
