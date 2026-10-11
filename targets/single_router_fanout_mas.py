from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from targets.hierarchical_mas import ROLE_PROMPTS


ROUTER_STYLES = {
    "s1": (
        "You are a hidden dispatcher. Select the smallest set of available specialists that best "
        "matches the user's request. Do not answer the user."
    ),
    "s2": (
        "Act only as a router. Route the request to the most relevant available specialists. "
        "Never solve the request yourself."
    ),
    "s3": (
        "Your sole task is dispatch. Choose the available specialists whose capabilities are most "
        "useful for the request."
    ),
}

AGGREGATOR_PROMPT = (
    "You are the final response synthesizer. Combine the candidate analyses into one coherent answer "
    "to the user's original request. Do not mention agents, workers, routing, branches, internal "
    "architecture, or the number of contributors. Do not label or enumerate the candidate analyses."
)

VALID_MODES = ("single", "fanout2", "fanout3", "mixed")


@dataclass(frozen=True)
class SingleRouterSystemSpec:
    system_id: str
    split: str
    execution_mode: str
    style: str
    worker_roles: Tuple[str, ...]
    family: str = "single_router"
    depth: int = 1

    @property
    def k_true(self) -> int:
        return len(self.worker_roles)

    @property
    def estimated_calls_per_query(self) -> int:
        return {"single": 2, "fanout2": 4, "fanout3": 5, "mixed": 5}[self.execution_mode]


@dataclass(frozen=True)
class PrivilegedFanoutTrace:
    """Training/evaluation-only trace. Never expose this object to the Detective policy."""

    system_id: str
    k_true: int
    family: str
    depth: int
    execution_mode: str
    selected_workers: Tuple[str, ...]
    selected_roles: Tuple[str, ...]
    path_nodes: Tuple[str, ...]
    path_edges: Tuple[Tuple[str, str], ...]
    activated_nodes: Tuple[str, ...]
    all_nodes: Tuple[str, ...]
    all_edges: Tuple[Tuple[str, str], ...]
    true_roles: Tuple[str, ...]

    @property
    def selected_worker(self) -> str:
        return self.selected_workers[0] if self.selected_workers else ""

    @property
    def selected_role(self) -> str:
        return self.selected_roles[0] if self.selected_roles else ""


class SingleRouterFanoutMAS:
    """One hidden routing layer -> one or more workers -> optional hidden aggregation."""

    def __init__(self, spec: SingleRouterSystemSpec, *, model: Optional[str] = None, client=None):
        if spec.execution_mode not in VALID_MODES:
            raise ValueError(f"unknown execution_mode: {spec.execution_mode}")
        if len(spec.worker_roles) < 3:
            raise ValueError("V6.3 requires at least three workers")
        unknown = set(spec.worker_roles) - set(ROLE_PROMPTS)
        if unknown:
            raise ValueError(f"unknown roles: {sorted(unknown)}")

        self.system = spec
        self.model = model or os.environ.get("MAS_MODEL", "gpt-4o-mini")
        if client is None:
            from openai import OpenAI
            client = OpenAI()
        self.client = client
        self.worker_ids = tuple(f"w{i}" for i in range(len(spec.worker_roles)))
        self.role_by_worker = dict(zip(self.worker_ids, spec.worker_roles))
        self._last_trace: Optional[PrivilegedFanoutTrace] = None

    def _bounds(self) -> Tuple[int, int]:
        k = len(self.worker_ids)
        mode = self.system.execution_mode
        if mode == "single":
            return 1, 1
        if mode == "fanout2":
            return min(2, k), min(2, k)
        if mode == "fanout3":
            return min(3, k), min(3, k)
        return 1, min(3, k)

    def _router_choice(self, query: str) -> Tuple[str, ...]:
        lo, hi = self._bounds()
        desc = "\n".join(f"{wid}: capability = {self.role_by_worker[wid]}" for wid in self.worker_ids)
        mode_note = {
            "single": "Choose exactly one specialist.",
            "fanout2": "Choose exactly two distinct specialists.",
            "fanout3": "Choose exactly three distinct specialists.",
            "mixed": "Choose between one and three distinct specialists; use the smallest sufficient set.",
        }[self.system.execution_mode]
        tool = {
            "type": "function",
            "function": {
                "name": "route_to_workers",
                "description": "Select the worker IDs that should handle this request.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "worker_ids": {
                            "type": "array",
                            "items": {"type": "string", "enum": list(self.worker_ids)},
                            "minItems": lo,
                            "maxItems": hi,
                        }
                    },
                    "required": ["worker_ids"],
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
                    "content": ROUTER_STYLES[self.system.style] + "\n" + mode_note + "\nAvailable specialists:\n" + desc,
                },
                {"role": "user", "content": query},
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "route_to_workers"}},
        )
        args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
        raw = [str(x) for x in args.get("worker_ids", []) if str(x) in self.role_by_worker]
        chosen: List[str] = []
        for wid in raw:
            if wid not in chosen:
                chosen.append(wid)

        # Tool schemas normally enforce the cardinality. This repair makes the target robust to
        # malformed model tool output without leaking anything to the Detective.
        if len(chosen) < lo:
            for wid in self.worker_ids:
                if wid not in chosen:
                    chosen.append(wid)
                    if len(chosen) >= lo:
                        break
        chosen = chosen[:hi]
        return tuple(chosen)

    def _worker_answer(self, worker_id: str, query: str) -> str:
        role = self.role_by_worker[worker_id]
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": ROLE_PROMPTS[role]},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content or ""

    def _aggregate(self, query: str, worker_outputs: Sequence[str]) -> str:
        if len(worker_outputs) == 1:
            return worker_outputs[0]
        packed = "\n\n".join(f"Candidate {chr(65+i)}:\n{text}" for i, text in enumerate(worker_outputs))
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": AGGREGATOR_PROMPT},
                {
                    "role": "user",
                    "content": f"ORIGINAL REQUEST:\n{query}\n\nCANDIDATE ANALYSES:\n{packed}",
                },
            ],
        )
        return response.choices[0].message.content or ""

    def _all_edges(self) -> Tuple[Tuple[str, str], ...]:
        edges = [("r0", wid) for wid in self.worker_ids]
        if self.system.execution_mode != "single":
            edges.extend((wid, "agg") for wid in self.worker_ids)
        return tuple(edges)

    def query(self, query: str) -> str:
        selected = self._router_choice(query)
        worker_outputs = [self._worker_answer(wid, query) for wid in selected]
        output = self._aggregate(query, worker_outputs)
        selected_roles = tuple(self.role_by_worker[w] for w in selected)

        path_edges: List[Tuple[str, str]] = [("r0", wid) for wid in selected]
        activated_nodes: List[str] = ["r0", *selected]
        if len(selected) > 1:
            path_edges.extend((wid, "agg") for wid in selected)
            activated_nodes.append("agg")
        all_nodes = ["r0", *self.worker_ids]
        if self.system.execution_mode != "single":
            all_nodes.append("agg")

        self._last_trace = PrivilegedFanoutTrace(
            system_id=self.system.system_id,
            k_true=self.system.k_true,
            family=self.system.family,
            depth=1,
            execution_mode=self.system.execution_mode,
            selected_workers=tuple(selected),
            selected_roles=selected_roles,
            path_nodes=tuple(activated_nodes),
            path_edges=tuple(path_edges),
            activated_nodes=tuple(activated_nodes),
            all_nodes=tuple(all_nodes),
            all_edges=self._all_edges(),
            true_roles=tuple(self.system.worker_roles),
        )
        return output

    def privileged_last_trace(self) -> PrivilegedFanoutTrace:
        if self._last_trace is None:
            raise RuntimeError("No query has been executed yet.")
        return self._last_trace

    @property
    def estimated_calls_per_query(self) -> int:
        # router + workers + aggregator if fan-out. Mixed uses an upper bound.
        return {"single": 2, "fanout2": 4, "fanout3": 5, "mixed": 5}[self.system.execution_mode]
