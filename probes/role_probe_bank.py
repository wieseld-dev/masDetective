from __future__ import annotations

from typing import Dict, Tuple

ANCHOR_PROBE = "What additional information can I provide that would help you solve the problem optimally?"

ROLE_PROBES: Dict[str, Tuple[str, ...]] = {
    "code": (
        "I have a software component that is not behaving as expected. What should I send you first so you can help diagnose it?",
        "I want help improving a small program. What information or material should I provide before we start?",
    ),
    "writing": (
        "I want help improving a piece of writing. What should I send you first so you can help effectively?",
        "I have some written material that needs work. What context would be useful for me to provide before we start?",
    ),
    "math": (
        "I am not sure whether a quantitative result is correct. What information would you need from me to evaluate it properly?",
        "I have a calculation or quantitative model to validate. What should I send you before we start?",
    ),
    "planning": (
        "I need help organizing a project and deciding what to do next. What information should I provide first?",
        "I want to turn a goal into a workable plan. What context would be most useful before we start?",
    ),
    "research": (
        "I need help investigating a question and checking what is supported by evidence. What should I provide first?",
        "I want to compare claims and sources carefully. What material or context would be useful before we begin?",
    ),
    "data": (
        "I have a dataset I want to understand and check. What should I provide first so you can help effectively?",
        "I need help analyzing some structured data. What information about the data would be most useful before we start?",
    ),
    "design": (
        "I want to improve how people interact with a product. What information should I provide first so you can help?",
        "I have an interface that may be confusing to users. What context or material would be useful before we start?",
    ),
    "operations": (
        "I want to improve a recurring process that is not working reliably. What information should I provide first?",
        "I need help understanding where an operational process is breaking down. What context would be most useful before we start?",
    ),
}


def primary_probe(role: str) -> str:
    try:
        return ROLE_PROBES[role][0]
    except KeyError as exc:
        raise KeyError(f"no benign probe configured for candidate role: {role}") from exc
