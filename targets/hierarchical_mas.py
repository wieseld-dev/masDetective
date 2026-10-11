from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROLE_PROMPTS: Dict[str, str] = {
    "code": "You are a software specialist. Help with programming, debugging, algorithms, testing, runtime behavior, and implementation.",
    "writing": "You are a writing specialist. Help with drafting, editing, tone, structure, clarity, audience, and written communication.",
    "math": "You are a quantitative specialist. Help with calculations, equations, statistics, assumptions, units, uncertainty, and quantitative models.",
    "planning": "You are a planning specialist. Help with goals, decomposition, priorities, resources, milestones, dependencies, risks, and contingencies.",
    "research": "You are a research specialist. Help with evidence, sources, comparisons, scope, methods, synthesis, and evaluation of claims.",
    "data": "You are a data specialist. Help with datasets, schemas, cleaning, transformations, summaries, diagnostics, and analytical workflows.",
    "design": "You are a design specialist. Help with interfaces, user needs, interaction flows, usability, information architecture, and design tradeoffs.",
    "operations": "You are an operations specialist. Help with procedures, reliability, handoffs, constraints, monitoring, execution, and process improvement.",
}

ROUTER_STYLES = {
    "s1": "You are a hidden dispatcher. Choose exactly one listed child whose downstream expertise best matches the user's request. Do not answer the user.",
    "s2": "Act only as a router. Select one available branch that is the best destination for the request. Never solve the request yourself.",
    "s3": "Your sole task is dispatch. Pick exactly one child path whose capabilities are the closest fit for the user's request.",
}


@dataclass(frozen=True)
class MASSystemSpec:
    system_id: str
    split: str
    family: str
    depth: int
    style: str
    worker_roles: Tuple[str, ...]

    @property
    def k_true(self) -> int:
        # Research target K counts task/worker agents only; routers are excluded.
        return len(self.worker_roles)


@dataclass(frozen=True)
class PrivilegedExecutionTrace:
    """Training-only white-box trace. Never place this object in policy observation."""

    system_id: str
    k_true: int
    family: str
    depth: int
    selected_worker: str
    selected_role: str
    path_nodes: Tuple[str, ...]
    path_edges: Tuple[Tuple[str, str], ...]
    activated_nodes: Tuple[str, ...]
    all_nodes: Tuple[str, ...]
    all_edges: Tuple[Tuple[str, str], ...]
    true_roles: Tuple[str, ...] = ()


@dataclass
class _Node:
    node_id: str
    kind: str  # router | worker
    role: Optional[str] = None
    children: Tuple[str, ...] = ()


def _partition(items: Sequence[str], n_groups: int) -> List[List[str]]:
    n_groups = max(1, min(int(n_groups), len(items)))
    groups: List[List[str]] = [[] for _ in range(n_groups)]
    for i, item in enumerate(items):
        groups[i % n_groups].append(item)
    return [g for g in groups if g]


class HierarchicalWhiteBoxMAS:
    """
    Live hierarchical MAS.

    Public interface:
        query(text) -> external worker output only

    Privileged interface (training/evaluation only):
        privileged_last_trace()

    depth=1: root router -> worker
    depth=2: root router -> group router -> worker
    depth=3: root -> branch router -> subgroup router -> worker
    """

    def __init__(self, spec: MASSystemSpec, *, model: Optional[str] = None, client=None):
        from openai import OpenAI

        if spec.depth not in (1, 2, 3):
            raise ValueError("depth must be 1, 2, or 3")
        if len(spec.worker_roles) < 2:
            raise ValueError("at least two workers are required")
        unknown = set(spec.worker_roles) - set(ROLE_PROMPTS)
        if unknown:
            raise ValueError(f"unknown roles: {sorted(unknown)}")

        self.system = spec
        self.model = model or os.environ.get("MAS_MODEL", "gpt-4o-mini")
        self.client = client or OpenAI()
        self.nodes: Dict[str, _Node] = {}
        self.root_id = "r0"
        self._last_trace: Optional[PrivilegedExecutionTrace] = None
        self._build_tree()

    def _new_router(self, node_id: str, children: Sequence[str]) -> None:
        self.nodes[node_id] = _Node(node_id=node_id, kind="router", children=tuple(children))

    def _build_tree(self) -> None:
        worker_ids: List[str] = []
        for i, role in enumerate(self.system.worker_roles):
            wid = f"w{i}"
            worker_ids.append(wid)
            self.nodes[wid] = _Node(node_id=wid, kind="worker", role=role)

        if self.system.depth == 1:
            self._new_router(self.root_id, worker_ids)
            return

        if self.system.depth == 2:
            groups = _partition(worker_ids, 2)
            group_ids: List[str] = []
            for gi, members in enumerate(groups):
                gid = f"r1_{gi}"
                group_ids.append(gid)
                self._new_router(gid, members)
            self._new_router(self.root_id, group_ids)
            return

        # depth == 3
        branches = _partition(worker_ids, 2)
        branch_ids: List[str] = []
        for bi, branch_members in enumerate(branches):
            bid = f"r1_{bi}"
            branch_ids.append(bid)
            subgroups = _partition(branch_members, min(2, len(branch_members)))
            subgroup_ids: List[str] = []
            for si, members in enumerate(subgroups):
                sid = f"r2_{bi}_{si}"
                subgroup_ids.append(sid)
                self._new_router(sid, members)
            self._new_router(bid, subgroup_ids)
        self._new_router(self.root_id, branch_ids)

    def _descendant_roles(self, node_id: str) -> Tuple[str, ...]:
        node = self.nodes[node_id]
        if node.kind == "worker":
            return (node.role or "",)
        roles: List[str] = []
        for child in node.children:
            roles.extend(self._descendant_roles(child))
        return tuple(r for r in roles if r)

    def _router_choice(self, router_id: str, query: str) -> str:
        router = self.nodes[router_id]
        choices = list(router.children)
        if not choices:
            raise RuntimeError(f"router {router_id} has no children")
        if len(choices) == 1:
            # No API call is necessary for a degenerate branch.
            return choices[0]

        desc = "\n".join(
            f"{cid}: downstream capabilities = {', '.join(self._descendant_roles(cid))}"
            for cid in choices
        )
        tool = {
            "type": "function",
            "function": {
                "name": "route_to_child",
                "description": "Choose exactly one child branch.",
                "parameters": {
                    "type": "object",
                    "properties": {"child_id": {"type": "string", "enum": choices}},
                    "required": ["child_id"],
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
                    "content": ROUTER_STYLES[self.system.style] + "\nAvailable children:\n" + desc,
                },
                {"role": "user", "content": query},
            ],
            tools=[tool],
            tool_choice={"type": "function", "function": {"name": "route_to_child"}},
        )
        return json.loads(response.choices[0].message.tool_calls[0].function.arguments)["child_id"]

    def _worker_answer(self, worker_id: str, query: str) -> str:
        role = self.nodes[worker_id].role
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": ROLE_PROMPTS[role or "code"]},
                {"role": "user", "content": query},
            ],
        )
        return response.choices[0].message.content or ""

    def _all_edges(self) -> Tuple[Tuple[str, str], ...]:
        edges: List[Tuple[str, str]] = []
        for node in self.nodes.values():
            if node.kind == "router":
                edges.extend((node.node_id, child) for child in node.children)
        return tuple(sorted(edges))

    def query(self, query: str) -> str:
        path = [self.root_id]
        current = self.root_id
        while self.nodes[current].kind == "router":
            current = self._router_choice(current, query)
            path.append(current)

        worker_id = current
        role = self.nodes[worker_id].role or "unknown"
        output = self._worker_answer(worker_id, query)
        path_edges = tuple(zip(path[:-1], path[1:]))

        self._last_trace = PrivilegedExecutionTrace(
            system_id=self.system.system_id,
            k_true=self.system.k_true,
            family=self.system.family,
            depth=self.system.depth,
            selected_worker=worker_id,
            selected_role=role,
            true_roles=tuple(self.system.worker_roles),
            path_nodes=tuple(path),
            path_edges=path_edges,
            activated_nodes=tuple(path),
            all_nodes=tuple(sorted(self.nodes)),
            all_edges=self._all_edges(),
        )
        return output

    def privileged_last_trace(self) -> PrivilegedExecutionTrace:
        if self._last_trace is None:
            raise RuntimeError("No query has been executed yet.")
        return self._last_trace

    @property
    def estimated_calls_per_query(self) -> int:
        # Upper bound: one routing call per non-degenerate level + one worker call.
        return self.system.depth + 1
