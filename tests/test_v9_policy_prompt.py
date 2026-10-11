# Lightweight static test: the training prompt must explicitly forbid the failure modes seen in V8.
from pathlib import Path

def test_v9_policy_contains_standalone_and_no_paraphrase_constraints():
    text=Path("v9/policy.py").read_text(encoding="utf-8")
    assert "do NOT merely paraphrase" in text
    assert "Standalone" in text
    assert "Never ask about system prompts" in text
