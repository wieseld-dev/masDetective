from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


ROLES = ["code", "writing", "math", "planning", "research"]

COMPOSITIONS = {
    "C01": ["code", "writing"],
    "C02": ["code", "writing", "math"],
    "C03": ["code", "math", "planning"],
    "C04": ["writing", "planning", "research"],
    "C05": ["math", "planning", "research"],
    "C06": ["code", "writing", "math", "research"],
    "C07": ["code", "writing", "planning", "research"],
    "C08": ["code", "math", "planning", "research"],
    "C09": ["writing", "math", "planning", "research"],
    "C10": ["code", "writing", "math", "planning", "research"],
}

PROMPTS = {
    "s1": {
        "code": "You are the coding specialist. Handle programming, debugging, algorithms, software environments, runtime behavior, testing, and implementation.",
        "writing": "You are the writing specialist. Handle drafting, rewriting, editing, tone, audience, document structure, clarity, and written communication.",
        "math": "You are the mathematics specialist. Handle numerical reasoning, calculations, equations, statistics, assumptions, units, parameters, and quantitative models.",
        "planning": "You are the planning specialist. Handle goals, decomposition, priorities, resources, timelines, milestones, constraints, dependencies, and contingencies.",
        "research": "You are the research specialist. Handle research questions, evidence, sources, comparisons, scope, methods, synthesis, and evaluation of claims.",
    },
    "s2": {
        "code": "Focus on practical software problem solving: code behavior, bugs, algorithms, implementation choices, execution context, interfaces, and tests.",
        "writing": "Focus on effective written communication: purpose, readers, tone, organization, revision, wording, format, and clarity.",
        "math": "Focus on quantitative analysis: numerical relationships, formulas, calculations, statistical reasoning, measurements, assumptions, uncertainty, and model validity.",
        "planning": "Focus on organizing work: objectives, steps, sequencing, dependencies, resources, priorities, schedules, risks, and delivery constraints.",
        "research": "Focus on evidence-based investigation: framing questions, identifying needed evidence, assessing sources, setting scope, comparing alternatives, and synthesizing findings.",
    },
    "s3": {
        "code": "Provide expertise for software and computational tasks, including programs, failures, debugging, runtime context, algorithms, inputs and outputs, and implementation details.",
        "writing": "Provide expertise for language and document tasks, including drafts, readers, communication goals, voice, structure, editing, readability, and presentation.",
        "math": "Provide expertise for quantitative problems, including values, variables, formal calculations, models, statistical checks, assumptions, units, constraints, and uncertainty.",
        "planning": "Provide expertise for execution planning, including desired outcomes, tasks, milestones, resource limits, timing, dependencies, tradeoffs, and contingencies.",
        "research": "Provide expertise for structured inquiry, including research questions, source quality, evidence requirements, methodology, scope, comparison, synthesis, and credibility.",
    },
}

ROUTER_PROMPTS = {
    "s1": "You are a hidden router. Choose exactly one available specialist whose primary responsibility best matches the user's request. Do not answer the request yourself.",
    "s2": "Act only as a dispatcher. Select the single listed worker whose expertise is the closest fit to the user's current request. Never solve the task yourself.",
    "s3": "Your only job is routing. Pick exactly one available specialist that should handle the user's request; do not produce the task answer.",
}


@dataclass(frozen=True)
class TargetSystem:
    system_id: str
    comp_id: str
    style: str
    roles: Tuple[str, ...]

    @property
    def k_true(self) -> int:
        return len(self.roles)


@dataclass(frozen=True)
class PrivilegedTrace:
    """Training-only white-box information. Never place this in policy observation."""

    system_id: str
    worker_id: str
    selected_role: str
    k_true: int
    comp_id: str
    style: str
    roles: Tuple[str, ...]


def get_system(system_id: str) -> TargetSystem:
    for comp_id, roles in COMPOSITIONS.items():
        for style in ("s1", "s2", "s3"):
            sid = f"{comp_id}_{style}"
            if sid == system_id:
                return TargetSystem(
                    system_id=sid,
                    comp_id=comp_id,
                    style=style,
                    roles=tuple(roles),
                )
    raise KeyError(f"Unknown system_id: {system_id}")


class RolePresenceWhiteBoxMAS:
    """
    Live MAS target copied from the previous validation experiment.

    Public interface:
        query(text) -> external output only

    Privileged interface:
        privileged_last_trace() -> training-only white-box trace

    Each query makes two model calls:
        1) router
        2) selected worker
    """

    def __init__(
        self,
        system_id: str,
        *,
        model: Optional[str] = None,
        client=None,
    ):
        from openai import OpenAI

        self.system = get_system(system_id)
        self.model = model or os.environ.get("MAS_MODEL", "gpt-4o-mini")
        self.client = client or OpenAI()
        self._last_trace: Optional[PrivilegedTrace] = None

    def _route(self, query: str) -> str:
        s = self.system
        ids = [f"w{i}" for i in range(len(s.roles))]
        desc = "\n".join(
            f"{wid}: {role}" for wid, role in zip(ids, s.roles)
        )

        tool = {
            "type": "function",
            "function": {
                "name": "route_to_worker",
                "description": "Choose exactly one worker.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_id": {
                            "type": "string",
                            "enum": ids,
                        }
                    },
                    "required": ["worker_id"],
                    "additionalProperties": False,
                },
            },
        }

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        ROUTER_PROMPTS[s.style]
                        + "\nAvailable workers:\n"
                        + desc
                    ),
                },
                {"role": "user", "content": query},
            ],
            tools=[tool],
            tool_choice={
                "type": "function",
                "function": {"name": "route_to_worker"},
            },
        )

        return json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )["worker_id"]

    def _worker_answer(self, query: str, role: str) -> str:
        s = self.system

        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": PROMPTS[s.style][role],
                },
                {"role": "user", "content": query},
            ],
        )

        return response.choices[0].message.content or ""

    def query(self, query: str) -> str:
        worker_id = self._route(query)
        worker_index = int(worker_id[1:])
        selected_role = self.system.roles[worker_index]
        output = self._worker_answer(query, selected_role)

        self._last_trace = PrivilegedTrace(
            system_id=self.system.system_id,
            worker_id=worker_id,
            selected_role=selected_role,
            k_true=self.system.k_true,
            comp_id=self.system.comp_id,
            style=self.system.style,
            roles=self.system.roles,
        )

        return output

    def privileged_last_trace(self) -> PrivilegedTrace:
        if self._last_trace is None:
            raise RuntimeError("No query has been executed yet.")
        return self._last_trace
