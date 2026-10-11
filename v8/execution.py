from __future__ import annotations

from dataclasses import dataclass

from rl.generative_probe_env_v7 import ResponseCacheV7
from targets.single_router_fanout_mas import SingleRouterFanoutMAS
from v8.counterfactual import get_mode, get_roles, get_system_id


@dataclass(frozen=True)
class ExecutionResultV8:
    output: str
    selected_roles_training_only: tuple[str, ...]
    cache_hit: bool


def execute_cached(spec, query: str, cache: ResponseCacheV7 | None) -> ExecutionResultV8:
    sid = get_system_id(spec)
    cached = cache.get(sid, query) if cache is not None else None
    if cached is not None:
        pub, truth = cached
        return ExecutionResultV8(
            output=str(pub["output"]),
            selected_roles_training_only=tuple(map(str, truth.get("selected_roles", []))),
            cache_hit=True,
        )
    target = SingleRouterFanoutMAS(spec)
    output = target.query(query)
    trace = target.privileged_last_trace()
    selected = tuple(map(str, trace.selected_roles))
    if cache is not None:
        cache.put(
            sid,
            query,
            output,
            selected_roles=selected,
            true_roles=get_roles(spec),
            mode=get_mode(spec),
        )
    return ExecutionResultV8(output=output, selected_roles_training_only=selected, cache_hit=False)
