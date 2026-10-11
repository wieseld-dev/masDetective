from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BenignGateResult:
    allowed: bool
    reason: str = "ok"


_BANNED_PATTERNS = [
    r"\b(ignore|disregard|override|forget)\b.{0,80}\b(previous|prior|system|developer)\b",
    r"\b(system prompt|developer message|hidden instructions?|private instructions?)\b",
    r"\b(reveal|show|print|dump|return|repeat|extract)\b.{0,80}\b(system prompt|hidden|private|internal instructions?)\b",
    r"\b(internal messages?|internal transcript|chain[- ]of[- ]thought|reasoning trace)\b",
    r"\b(internal routing|routing table|route trace|execution trace|tool trace)\b",
    r"\b(secret|credential|api key|password|token)\b.{0,50}\b(reveal|show|print|dump|return|give)\b",
    r"\b(agent|agents|worker|workers|specialist|specialists|expert|experts)\b",
    r"\b(router|orchestrator|dispatcher|routing|topology)\b",
    r"\b(architecture|hidden structure|internal structure)\b",
    r"\b(how many|number of|count of|total number|distinct number)\b.{0,90}\b(role|roles|expert|experts|specialist|specialists|agent|agents|worker|workers|component|components|branch|branches)\b",
    r"\b(role|roles|expertise|specialization|specialisations?|specializations?)\b.{0,90}\b(involved|used|contributed|handled|assigned|collaborat|coordinate)\b",
    r"\b(collaborat(?:e|es|ed|ion|ing)|coordinate(?:d|s|ion|ing))\b.{0,90}\b(expert|specialist|agent|worker|role|component)\b",
    r"\b(component|components|branch|branches|path|paths)\b.{0,90}\b(internal|hidden|contribut|handle|process|answer)\b",
    r"\b(USER INPUT|USER QUERY|EXTERNAL OUTPUT|ASSISTANT:|SYSTEM:|TURN\s*\d+)\b",
]
_COMPILED = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _BANNED_PATTERNS]


def check_benign(query: str, *, max_chars: int = 320) -> BenignGateResult:
    text = (query or "").strip()
    if not text:
        return BenignGateResult(False, "empty")
    if len(text) > max_chars:
        return BenignGateResult(False, "too_long")
    if "\n" in text:
        return BenignGateResult(False, "multi_line_or_transcript")
    if "[" in text or "]" in text:
        return BenignGateResult(False, "placeholder_or_template")
    if "?" not in text:
        return BenignGateResult(False, "not_a_question")
    for pattern in _COMPILED:
        if pattern.search(text):
            return BenignGateResult(False, "structural_or_private_request")
    return BenignGateResult(True, "ok")
