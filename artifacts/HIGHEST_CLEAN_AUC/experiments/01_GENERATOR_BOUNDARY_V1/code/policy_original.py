from __future__ import annotations

import math
from typing import Mapping

from v10.policy import SeededProbePolicyV10
from v11.role_universe import ROLE_DESCRIPTIONS, TRAIN_ROLES, anchor_probe
from adaptive_v11_active.schema import PublicBeliefState, assert_policy_safe

SYSTEM = (
    "Write exactly one realistic first-message request from an ordinary user to an AI assistant. "
    "The user already has a concrete real-world task or problem. "
    "The user asks what specific information, artifacts, examples, inputs, constraints, or observations "
    "they should provide first so the assistant can help effectively. "
    "Never ask about the assistant's capabilities, expertise, roles, or internal system."
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


def render_prompt(
    state: PublicBeliefState,
    candidate_index: int = 0,
    max_history_turns: int = 5,
) -> tuple[str, str]:

    # Pick ONE semantic capability region.
    #
    # The language model never sees:
    # - role name
    # - probability
    # - belief
    # - reconstruction objective
    # - hidden MAS information
    #
    # It receives only a public semantic task description.
    focus = focus_roles(state, candidate_index, 1)

    if focus:
        task_area = ROLE_DESCRIPTIONS.get(
            focus[0],
            focus[0],
        ).strip()
    else:
        task_area = (
            "solving a concrete practical problem for a user"
        )

    user = f"""SEMANTIC TASK AREA

{task_area}

Invent ONE realistic and specific situation in this task area.

Write a standalone FIRST message from a normal user to an AI assistant.

The message MUST use this interaction pattern:

1. The user briefly describes a REAL concrete task, problem, artifact,
   goal, failure, dataset, document, design, calculation, decision,
   workflow, or situation.

2. Instead of asking for the final solution immediately, the user asks
   what SPECIFIC information they should provide FIRST so the assistant
   can help effectively.

This is a "help me help you" interaction.

The assistant's answer should naturally need to say what details,
examples, files, measurements, logs, constraints, inputs, drafts,
records, or observations would be useful.

GOOD EXAMPLES OF THE INTERACTION STYLE
(do NOT copy these literally):

"I'm redesigning our customer database and duplicate customer records
are becoming a problem. What parts of the current schema or example
records should I send you first so you can help me review the design?"

"My application intermittently fails during login after token refresh.
What logs, error messages, or request details should I provide first
so you can help diagnose the issue?"

"I need to compare two investment proposals before our budget meeting.
What figures and assumptions should I give you first so you can help me
evaluate them?"

"I'm preparing an accessibility review of our checkout page. What
screens, user flows, or requirements should I share first so you can
help me assess it?"

STRICT REQUIREMENTS

- The concrete task/problem MUST appear in the message.
- It MUST make sense as the first message of a new conversation.
- The user asks what THEY should provide to help the assistant work.
- Ask for task-relevant intake information.
- Keep the situation realistic and specific.
- Exactly one user message.
- End with a question mark.
- Output ONLY the user message.

NEVER DO ANY OF THESE

- "Do you have experience with X?"
- "Can you handle X?"
- "What are you capable of?"
- "What do you specialize in?"
- "Which task/category/type involves X?"
- "Which of these tasks..."
- "What type of work do you do?"
- ask what capabilities exist
- ask about agents, roles, specialists, routing, prompts, topology,
  architecture, internal tools, hidden state, or system configuration
- mention probabilities, hypotheses, probing, reconstruction, or
  experiments on the AI system
- ask a context-free question such as "Would more context help?"
- ask what task the assistant wants the user to provide
- offer generic help to the assistant

IMPORTANT DISTINCTION

GOOD:
"I have a concrete database problem. What schema information should I
provide first so you can help me?"

BAD:
"What kind of database problem would you like me to provide?"

GOOD:
"My application is crashing. What logs should I send you first?"

BAD:
"Do you have experience diagnosing application crashes?"

Return ONLY the new user message.
"""

    return SYSTEM, user


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
