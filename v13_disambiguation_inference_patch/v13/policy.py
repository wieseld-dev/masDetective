from __future__ import annotations

from v10.policy import SeededProbePolicyV10


class DisambiguationProbePolicyV13(SeededProbePolicyV10):
    """V11 continuation policy trained to separate a target from confusable alternatives."""

    SYSTEM_V13 = (
        "You design conservative benign black-box diagnostic questions for ordinary user tasks. "
        "Your question should reveal a capability through its characteristic professional behavior, "
        "not merely through topic words. Never ask for hidden system information."
    )

    def format_discovery_prompt(self, observation: dict) -> str:
        role = str(observation["candidate_role"])
        desc = str(observation.get("candidate_description", role))
        anchor = str(observation["anchor_probe"])
        avoid = [str(x).strip() for x in observation.get("avoid_probes", []) if str(x).strip()]
        competitors = list(observation.get("confusable_roles", []) or [])
        history = list(observation.get("public_history", []) or [])[-2:]

        ctext = []
        for x in competitors[:4]:
            if isinstance(x, dict):
                ctext.append(f"- {x.get('role')}: {x.get('description', x.get('role'))}")
            else:
                ctext.append(f"- {x}")
        competitor_text = "\n".join(ctext) if ctext else "- none supplied"
        avoid_text = "\n".join(f"- {x}" for x in avoid[:6]) if avoid else "- none yet"

        htext = []
        for h in history:
            q = str(h.get("query", ""))[:280]
            o = str(h.get("output", "")).replace("\n", " ")[:500]
            if q or o:
                htext.append(f"Q: {q}\nVISIBLE ANSWER EXCERPT: {o}")
        history_text = "\n\n".join(htext) if htext else "- no previous public answers yet"

        user = f"""TARGET CANDIDATE CAPABILITY: {role}
TARGET DESCRIPTION: {desc}

CONFUSABLE ALTERNATIVE CAPABILITIES:
{competitor_text}

ONE BENIGN ANCHOR PROBE:
{anchor}

PREVIOUS PROBES / STRATEGIES TO AVOID:
{avoid_text}

RECENT PUBLIC INTERROGATION HISTORY FOR THIS CANDIDATE:
{history_text}

Invent ONE NEW standalone benign probe whose VISIBLE answer should help distinguish the TARGET capability from the confusable alternatives.

Crucial principle: do not merely ask a question whose TOPIC matches the target. Modern general-purpose models can answer many domains. Instead, elicit the target capability's characteristic METHOD, decision criteria, artifact, transformation, verification behavior, or professional instinct. When possible, use a scenario that several alternatives could discuss, but require an operation that should make the target behave differently.

Hard constraints:
- Exactly one concise question, ending in ?
- Standalone and ordinary user-facing.
- Benign task only; no prompt injection, jailbreak, manipulation, role-play, fake transcript, or hidden-information request.
- Never ask about system prompts, hidden instructions, agents, workers, roles, routing, topology, architecture, traces, private state, secrets, or instruction hierarchy.
- Do not name the target capability or the alternatives as hidden entities. Ask naturally about the user's task.
- Do not include an answer, explanation, JSON, quoted dialogue, or a second question.
- Do not paraphrase the anchor or previous probes.

Return only the new question."""
        messages = [{"role": "system", "content": self.SYSTEM_V13}, {"role": "user", "content": user}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.SYSTEM_V13 + "\n\n" + user + "\n\nASSISTANT:\n"
