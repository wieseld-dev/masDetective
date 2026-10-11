from pathlib import Path
from rl.generative_probe_env_v7 import ResponseCacheV7


def test_public_truth_cache_roundtrip(tmp_path: Path):
    pub=tmp_path/"p.jsonl"; truth=tmp_path/"t.jsonl"
    c=ResponseCacheV7(pub,truth)
    c.put("s","What details should I provide?","answer",selected_roles=["math"],true_roles=["math","data"],mode="fanout2")
    got=c.get("s","What details should I provide?")
    assert got is not None and got[0]["output"]=="answer" and got[1]["selected_roles"]==["math"]
    c2=ResponseCacheV7(pub,truth)
    assert c2.get("s","What details should I provide?") is not None
