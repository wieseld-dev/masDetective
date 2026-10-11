from __future__ import annotations

import inspect
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence


def discover_dual_scorer_cli(script: str | Path = "adaptive_score_v135.py") -> tuple[str, str, str]:
    """Discover existing scorer input/output flags without assuming undocumented names."""
    p = Path(script)
    if not p.exists():
        raise RuntimeError(f"missing existing dual scorer script: {p}")
    proc = subprocess.run([sys.executable, str(p), "--help"], capture_output=True, text=True)
    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    input_candidates = ("--input", "--responses", "--response-input", "--input-jsonl")
    output_candidates = ("--output", "--replay-bank", "--output-jsonl")
    inp = next((f for f in input_candidates if f in help_text), None)
    out = next((f for f in output_candidates if f in help_text), None)
    if inp is None or out is None:
        raise RuntimeError(
            "Could not discover input/output flags for adaptive_score_v135.py. "
            "This is intentionally a hard preflight failure rather than silently overwriting its defaults.\n" + help_text[:5000]
        )
    return inp, out, help_text


def extract_final_output(result) -> str:
    if isinstance(result, str):
        return result
    for name in ("output", "final_output", "answer", "response"):
        if hasattr(result, name):
            value = getattr(result, name)
            if value is not None:
                return str(value)
    raise RuntimeError(f"cannot extract final output from {type(result)} fields={dir(result)[:80]}")


def policy_save_capability(policy) -> str:
    if hasattr(policy, "save_adapter"):
        return "save_adapter"
    model = getattr(policy, "model", None)
    if model is not None and hasattr(model, "save_pretrained"):
        return "model.save_pretrained"
    raise RuntimeError("policy exposes no known adapter save interface")
