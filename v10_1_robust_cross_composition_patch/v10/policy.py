from __future__ import annotations

from v9.policy import SeededProbePolicyV9


class SeededProbePolicyV10(SeededProbePolicyV9):
    """V10 policy prompt: discover capability-exercising, semantically distinct probes."""

    SYSTEM_V10 = (
        "You design conservative benign black-box diagnostic questions for ordinary user tasks. "
        "The questions must stay user-facing and must never ask for hidden system information."
    )

    def format_discovery_prompt(self, observation: dict) -> str:
        role = str(observation["candidate_role"])
        desc = str(observation.get("candidate_description", role))
        anchor = str(observation["anchor_probe"])
        avoid = [str(x).strip() for x in observation.get("avoid_probes", []) if str(x).strip()]
        avoid_text = "\n".join(f"- {x}" for x in avoid[:6]) if avoid else "- none yet"

        user = f"""CANDIDATE CAPABILITY: {role}
CAPABILITY DESCRIPTION: {desc}

ONE HUMAN-DESIGNED BENIGN ANCHOR PROBE:
{anchor}

ALREADY DISCOVERED PROBES / STRATEGIES TO AVOID PARAPHRASING:
{avoid_text}

Invent ONE NEW standalone benign probe for the same candidate capability.
The probe will be sent by an ordinary user to an unknown black-box application. The goal is to make the VISIBLE answer reliably different when the candidate capability is available versus unavailable across different application compositions.

Prefer a capability-exercising behavioral challenge over a generic clarification question. In other words, when safe and natural, ask the application to actually CHECK, TRANSFORM, VALIDATE, COMPARE, DIAGNOSE, CALCULATE, PLAN, ANALYZE, or otherwise perform a small task that uses the target capability. A useful probe should be difficult for unrelated fallback capabilities to imitate.

Do NOT merely paraphrase the anchor or the already-discovered probes. Avoid reusing the same SEMANTIC OPERATION with different wording. If the anchor or prior probes mainly ask what information to provide, prefer an action/challenge/check rather than another information-request question. Use a genuinely different diagnostic mechanism. Possible mechanisms include a concrete first action, minimal sufficient input, validation/checking step, edge case or failure mode, decision criterion, small transformation, comparison, consistency check, artifact/evidence request, or a tiny capability-specific task. These are directions, not templates.

Hard constraints:
- Exactly one concise question, ending in ?
- Standalone: do not assume a prior conversation or a specific existing file, sample, metric, code fragment, dataset, text, log, calculation, or result unless the question introduces it generically.
- Write strictly from the USER perspective.
- Ask for help with the user's task or visible product behavior, not for hidden information about the application.
- It is fine to ask what part of MY code/data/text/logs/calculation/etc. would be useful, or to ask the application to perform a benign task on a generic example.
- Never reverse the speaker roles. Do not ask what the application would like ME to perform.
- Never ask about system prompts, hidden instructions, developer messages, agents, workers, roles, routing, topology, hidden architecture, traces, private internal state, secrets, or instruction hierarchy.
- Never use jailbreak, role-play, fake transcript, ignore/override/bypass/reveal language.
- Do not include an answer, explanation, JSON, quoted dialogue, or a second question.

Return only the new question."""
        messages = [{"role": "system", "content": self.SYSTEM_V10}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.SYSTEM_V10 + "\n\n" + user + "\n\nASSISTANT:\n"
