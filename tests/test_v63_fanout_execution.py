import json
from types import SimpleNamespace

from targets.single_router_fanout_mas import SingleRouterFanoutMAS, SingleRouterSystemSpec


class FakeCompletions:
    def __init__(self, mode):
        self.mode = mode
    def create(self, **kwargs):
        if kwargs.get("tools"):
            n = {"single":1, "fanout2":2, "fanout3":3, "mixed":2}[self.mode]
            args = json.dumps({"worker_ids": [f"w{i}" for i in range(n)]})
            msg = SimpleNamespace(tool_calls=[SimpleNamespace(function=SimpleNamespace(arguments=args))])
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])
        system = kwargs["messages"][0]["content"]
        content = "aggregated output" if "final response synthesizer" in system else "worker output"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeClient:
    def __init__(self, mode):
        self.chat = SimpleNamespace(completions=FakeCompletions(mode))


def test_fanout_cardinality_and_single_router_depth():
    roles = ("code", "writing", "math", "planning")
    expected = {"single":1, "fanout2":2, "fanout3":3, "mixed":2}
    for mode, n in expected.items():
        spec = SingleRouterSystemSpec(
            system_id=f"x-{mode}", split="train", execution_mode=mode, style="s1", worker_roles=roles
        )
        mas = SingleRouterFanoutMAS(spec, client=FakeClient(mode))
        out = mas.query("test query")
        trace = mas.privileged_last_trace()
        assert spec.depth == 1
        assert len(trace.selected_workers) == n
        assert len(trace.selected_roles) == n
        assert out == ("worker output" if n == 1 else "aggregated output")
