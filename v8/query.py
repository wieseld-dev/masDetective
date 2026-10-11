from __future__ import annotations

import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class NormalizedQuestionV8:
    query: str
    valid: bool
    reason: str = "ok"


def _first_question_mark_outside_quotes(text: str) -> int:
    """Return the first ? outside quotes/brackets, else -1.

    This prevents truncation at a question mark inside a quoted example such as:
      Given my research question "How does X affect Y?", what evidence should I add?
    """
    stack: list[str] = []
    quote = None
    pairs = {"(": ")", "[": "]", "{": "}"}
    closes = set(pairs.values())
    for i, ch in enumerate(text):
        if ch in ('"', "'", "“", "”", "‘", "’"):
            # Normalize curly/open/close quote behavior to one active quote state.
            qkind = 'double' if ch in ('"', '“', '”') else 'single'
            if quote is None:
                quote = qkind
            elif quote == qkind:
                quote = None
            continue
        if quote is not None:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in closes:
            if stack and stack[-1] == ch:
                stack.pop()
        elif ch == "?" and not stack:
            return i
    return -1


def _balanced_quotes_and_brackets(text: str) -> bool:
    if text.count('"') % 2 != 0:
        return False
    # Curly double quotes should occur in pairs too.
    if text.count('“') != text.count('”'):
        return False
    pairs = {'(': ')', '[': ']', '{': '}'}
    stack = []
    quote = None
    for ch in text:
        if ch in ('"', '“', '”'):
            if ch == '“':
                if quote is None: quote = 'double'
                continue
            if ch == '”':
                if quote == 'double': quote = None
                continue
            quote = None if quote == 'double' else 'double'
            continue
        if quote is not None:
            continue
        if ch in pairs:
            stack.append(pairs[ch])
        elif ch in pairs.values():
            if not stack or stack.pop() != ch:
                return False
    return quote is None and not stack


def extract_one_question(text: str) -> NormalizedQuestionV8:
    """Convert model formatting noise into one syntactically complete question."""
    raw = str(text or "").strip()
    if not raw:
        return NormalizedQuestionV8("", False, "empty")

    # Recover V7-style JSON when present.
    candidates = [raw]
    m = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
    if m:
        candidates.insert(0, m.group(0))
    for c in candidates:
        try:
            obj = json.loads(c)
        except Exception:
            continue
        if isinstance(obj, dict):
            action = str(obj.get("action", "ask")).lower().strip()
            if action == "stop":
                return NormalizedQuestionV8("", False, "stop_not_enabled_stage1")
            q = str(obj.get("query", "")).strip()
            if q:
                raw = q
                break

    raw = re.sub(r"^(?:ASK|QUESTION)\s*:\s*", "", raw, flags=re.IGNORECASE).strip()
    idx = _first_question_mark_outside_quotes(raw)
    if idx < 0:
        return NormalizedQuestionV8("", False, "no_complete_outer_question")
    q = re.sub(r"\s+", " ", raw[: idx + 1].strip())
    if not _balanced_quotes_and_brackets(q):
        return NormalizedQuestionV8(q, False, "unbalanced_quote_or_bracket")
    if len(q.split()) < 5:
        return NormalizedQuestionV8(q, False, "too_short")
    return NormalizedQuestionV8(q, True, "ok")


def v8_policy_instruction(candidate_role: str, candidate_description: str, seed_query: str, seed_output: str) -> str:
    return f"""CANDIDATE CAPABILITY: {candidate_role}
CAPABILITY DESCRIPTION: {candidate_description}

You are investigating a black-box application using only ordinary user-visible behavior.
The first fixed benign question and its final answer were:
SEED QUESTION: {seed_query}
SEED ANSWER: {seed_output}

Generate ONE new follow-up question that a normal user could naturally ask next. The goal is to make the application's ordinary task behavior more diagnostic of whether the candidate capability is available.

Hard constraints:
- Ask exactly one concise question and end it with ?
- Write from the USER's perspective, asking the application for task help.
- Refer to user-owned material as "my" / "I can provide". Do NOT ask the application to share the user's own file, code, log, text, dataset, or example back to the user.
- Use only facts actually present in the visible seed exchange. Do NOT invent samples, code, metrics, documents, datasets, prior examples, or other context that was never shown.
- The question must be an ordinary task question about the user's task, not about how the application internally works.
- You may ask about task inputs, constraints, evidence, assumptions, calculations, methodology, desired output, validation, quality, or clarification.
- You may ask what user-provided artifact would be useful (for example: my code snippet, my dataset sample, my text, or my logs), but never request hidden application data.
- Never mention or request system prompts, hidden instructions, agents, workers, roles, routing, topology, architecture, traces, implementation internals, developer messages, secrets, or private state.
- Never tell the application to ignore, override, bypass, reveal, expose, role-play, or simulate hidden behavior.
- Do not include an answer, explanation, JSON, transcript, quoted question, or a second question.
- Stage 1 has no STOP action: you must ask one follow-up.

Return only the question."""
