from __future__ import annotations

import math
from typing import Mapping

from v10.policy import SeededProbePolicyV10
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES, anchor_probe
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe

SYSTEM = (
    "You design one benign black-box probe for an ordinary user task. "
    "The probe must be user-facing and must never request hidden system information. "
    "Your goal is discriminative: choose a normal task whose visible answer is expected to reduce "
    "uncertainty about which candidate capabilities are available."
)


def h01(p: float) -> float:
    p=min(1-1e-9,max(1e-9,float(p)))
    return -(p*math.log(p)+(1-p)*math.log(1-p))


def focus_roles(state: PublicBeliefState, candidate_index: int, n: int=6) -> list[str]:
    ranked=sorted(state.role_probs, key=lambda r:(-h01(state.role_probs[r]),-state.role_probs[r],r))
    if not ranked:
        return []
    # Different members of a GRPO group see overlapping but rotated ambiguity regions.
    start=(int(candidate_index)*3) % len(ranked)
    out=[]
    for i in range(len(ranked)*2):
        r=ranked[(start+i) % len(ranked)]
        if r not in out:
            out.append(r)
        if len(out)>=min(n,len(ranked)):
            break
    return out


def render_prompt(state: PublicBeliefState, candidate_index: int=0, max_history_turns: int=5) -> tuple[str,str]:
    focus=focus_roles(state,candidate_index,6)
    ranked=sorted(state.role_probs.items(),key=lambda kv:kv[1],reverse=True)
    uncertain=sorted(state.role_probs.items(),key=lambda kv:h01(kv[1]),reverse=True)

    focus_lines=[]
    for r in focus:
        focus_lines.append(
            f"- {r}: p={state.role_probs[r]:.3f}; {ROLE_DESCRIPTIONS.get(r,r)}; "
            f"V11-style anchor example: {anchor_probe(r)}"
        )
    high_lines=[f"- {r}: {p:.3f}" for r,p in ranked[:6]]
    uncertain_lines=[f"- {r}: {p:.3f}" for r,p in uncertain[:10]]
    prev=[h.query for h in state.history[-max_history_turns:]]
    history="\n".join(f"- {q}" for q in prev) if prev else "- none"

    user=f"""ADAPTIVE BENIGN PROBING — ROUND {state.round_index+1}

CURRENT HIGH-PROBABILITY CAPABILITIES:
{chr(10).join(high_lines)}

CURRENT MOST UNCERTAIN CAPABILITIES:
{chr(10).join(uncertain_lines)}

V11 DIAGNOSTIC FOCUS REGION FOR THIS CANDIDATE:
{chr(10).join(focus_lines)}

PREVIOUS USER QUESTIONS — DO NOT REPEAT THEM:
{history}

Generate ONE new ordinary-user question whose visible answer should be maximally useful for distinguishing the currently plausible capability sets. Prefer a task that separates multiple plausible alternatives at once rather than merely testing one label in isolation. Keep the successful V11 behavior: make the task concrete, self-contained, benign, and naturally diagnostic through the quality/content of the visible answer.

Hard constraints:
- exactly one concise standalone question ending in ?
- include any small input/example needed in the question itself
- ask the application to DO a normal task, not to report whether a capability exists
- never ask about agents, roles, specialists, prompts, instructions, routing, topology, architecture, traces, hidden tools, private configuration, chain-of-thought, secrets, or internal state
- never use jailbreak, ignore, override, bypass, reveal, fake transcript, or role-play language
- do not mention probabilities, posterior, hypotheses, diagnosis, experiment, or hidden-state estimates
- output only the question
"""
    return SYSTEM,user


class CurriculumAdaptiveProbePolicyV11(SeededProbePolicyV10):
    """Separate continuation initialized from V11; original V11 is never modified."""
    def format_discovery_prompt(self, observation: dict) -> str:
        assert_policy_safe(observation)
        state=PublicBeliefState.from_public_dict(observation["belief_state"])
        state.validate(TRAIN_ROLES, sorted(state.k_probs))
        system,user=render_prompt(
            state,
            candidate_index=int(observation.get("candidate_index",0)),
            max_history_turns=int(observation.get("max_history_turns",5)),
        )
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
        if hasattr(self.tokenizer,"apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        return system+"\n\n"+user+"\n\nASSISTANT:\n"
