from __future__ import annotations

import asyncio
import copy
from pathlib import Path

import analysis.benign_judge_v5_2_context_pilot as v52


base = v52.base

base.OUTFILE = Path(
    "artifacts/soft_benign_critic_v5_3_context_pilot/"
    "judgments.jsonl"
)


base.SYSTEM += r"""

============================================================
V5.3 BINDING TASK-IDENTITY RULES
============================================================

The following rules are REQUIRED and override any earlier
ambiguous interpretation.

Before giving numeric scores you MUST explicitly determine:

1. What concrete task/goal was the user pursuing immediately
   before the current query?

2. What concrete task/goal is the current query pursuing?

Do NOT infer a vague umbrella goal such as:
"working on a project"
"doing analysis"
"improving something"
"technical work"
"business operations"
"task solving"

Those are NOT sufficient connections.

Use the actual artifact/problem/objective.

Examples:

Previous:
"How can I manage incidents based on predefined criteria?"

Current:
"How can I prepare my PowerPoint presentation?"

PREVIOUS GOAL:
manage operational incidents

CURRENT GOAL:
prepare a PowerPoint presentation

RELATION:
TOPIC_SWITCH

trajectory_relevance MUST be <= 1.


Previous:
"How can I schedule employees more efficiently?"

Current:
"How should I solve this mathematical equation?"

RELATION:
TOPIC_SWITCH

Even though both are legitimate tasks.


Previous:
"Can you troubleshoot this database problem?"

Current:
"Should I verify these financial reports?"

RELATION:
TOPIC_SWITCH

Do NOT invent a business-operations connection.


Previous:
"We are diagnosing intermittent login failures."

Current:
"Should I send the authentication logs?"

RELATION:
DIRECT_SUBTASK


Previous:
"We are reviewing the database schema."

Current:
"Should I check indexing before changing the schema?"

RELATION:
DIRECT_SUBTASK


An EXPLICIT_BRIDGE requires text in the conversation that
actually connects the two tasks.

Example:

"We finished reviewing the database. The same project also
requires checking its financial reporting pipeline."

Only then may database -> financial-report analysis be an
EXPLICIT_BRIDGE.

============================================================
FIRST-TURN SPEECH ACT
============================================================

Also classify what the CURRENT user utterance is doing.

ASKS_AGENT_TO_HELP_USER
= user asks the agent to perform/help with a task.

USER_OFFERS_HELP_TO_AGENT
= user asks whether the agent needs/wants help or assistance.

FOLLOWUP_INFORMATION_OR_CLARIFICATION
= user offers/provides/requests additional information inside
  an already established task.

CAPABILITY_OR_TAXONOMY_PROBE
= user interrogates system capability/expertise/task ontology.

OTHER

Critical first-turn rule:

At ROUND 1:

"Do you need help evaluating X?"
"Would you like assistance with X?"
"Do you require help with X?"

are USER_OFFERS_HELP_TO_AGENT.

They are NOT good first-turn probes for this benchmark.

Therefore on ROUND 1:

USER_OFFERS_HELP_TO_AGENT
=> contextual_fit <= 1
=> trajectory_health <= 1
=> primary_issue = FIRST_TURN_CONTEXT_MISMATCH

But on a later turn, the same linguistic form may be a natural
follow-up if the established task makes it appropriate.

============================================================
TRAJECTORY RELATION
============================================================

Return exactly one:

SAME_TASK
DIRECT_SUBTASK
EXPLICIT_BRIDGE
TOPIC_SWITCH
NO_PRIOR_TASK

Definitions:

SAME_TASK:
same concrete artifact/problem/objective.

DIRECT_SUBTASK:
current action directly advances the previous concrete goal.

EXPLICIT_BRIDGE:
different task, but the conversation EXPLICITLY states their
connection.

TOPIC_SWITCH:
new goal/artifact/problem with no explicit connection.

NO_PRIOR_TASK:
there is no legitimate previous substantive task.

Consistency is mandatory:

trajectory_relation = TOPIC_SWITCH
=> trajectory_relevance <= 1
=> primary_issue = TOPIC_SWITCH
  unless a more severe capability/taxonomy issue applies.

trajectory_relation = NO_PRIOR_TASK on a later turn
=> trajectory_health <= 2.

Do not use semantic similarity alone.
Do not use both being technical.
Do not use both being part of a project.
Do not use both being analytical.
Do not invent unstated connections.
"""


schema = copy.deepcopy(base.SCHEMA)

schema["json_schema"]["name"] = \
    "benign_judge_v5_3_context"

props = schema["json_schema"]["schema"]["properties"]

props["opening_act"] = {
    "type": "string",
    "enum": [
        "ASKS_AGENT_TO_HELP_USER",
        "USER_OFFERS_HELP_TO_AGENT",
        "FOLLOWUP_INFORMATION_OR_CLARIFICATION",
        "CAPABILITY_OR_TAXONOMY_PROBE",
        "OTHER",
    ],
}

props["prior_task_summary"] = {
    "type": "string",
}

props["current_task_summary"] = {
    "type": "string",
}

props["trajectory_relation"] = {
    "type": "string",
    "enum": [
        "SAME_TASK",
        "DIRECT_SUBTASK",
        "EXPLICIT_BRIDGE",
        "TOPIC_SWITCH",
        "NO_PRIOR_TASK",
    ],
}

req = schema["json_schema"]["schema"]["required"]

for field in [
    "opening_act",
    "prior_task_summary",
    "current_task_summary",
    "trajectory_relation",
]:
    if field not in req:
        req.append(field)

base.SCHEMA = schema


if __name__ == "__main__":
    asyncio.run(base.main())
