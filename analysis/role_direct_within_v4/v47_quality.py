#!/usr/bin/env python3
"""Conservative, deterministic textual QC for benign MAS5 candidate questions.

This is a syntax/structure/known-failure guard, NOT a measure of actual
PRESENT/ABSENT discriminability. Borderline rows require human review.
"""
from __future__ import annotations
import re
from collections import Counter

SLOT_TYPES = ('intake', 'intake', 'ambiguity', 'ambiguity', 'validation', 'validation', 'direct', 'direct')

# Bad examples actually observed in the user's October 10 manual review.
BAD_TERMS = (
    (r'(?i)\btarget\s+(?:capabilit(?:y|ies)|role|agent|specialist)\s*(?:\*\*)?\s*[:\-—]', 'explicit_target_label'),
    (r'(?i)\b(?:internal|hidden)\s+(?:agent|roles?|instructions?|prompts?)\b', 'internal_architecture_request'),
    (r'(?i)\b(?:routing trace|system prompt|installed roles?|agent topology|list your agents)\b', 'internal_architecture_request'),
    (r'(?i)\bignore\s+(?:all\s+)?previous\s+instructions\b', 'prompt_injection'),
    (r'(?i)\b(?:i\s*am|i\'m)\s+working\s+on\s+this\s+situation\s*:', 'generic_template'),
    (r'(?i)^\s*(?:\*\*)?(?:user\'s scenario|user scenario)\s*:', 'third_person_user_wrapper'),
    (r'(?i)\bwhat\s+do\s+you\s+need\s+from\s+me\s+next\s*\?\s*\*\*?\s*to\s+assist', 'mixed_task_intent'),
    (r'(?i)\bplease\s+share\s+the\s+error\s+messages\b', 'reversed_direction'),
    (r'(?i)\bvalidate\s+that\s+the\s+primary\s+key\s+on\s+the\s+orders\s+table\s+matches\s+the\s+id\s+column\s+in\s+the\s+products\s+table\b', 'incorrect_database_constraint'),
    (r'(?i)(?:\[(?:topic|specific|insert|task)|\{(?:topic|specific|insert|task)|<[^>]{1,35}>)', 'unfilled_placeholder'),
    (r'(?i)\b(?:target_role|probe_family|strategy_index|selected_roles_training_only)\b', 'metadata_leakage'),
)

# Subjective weaknesses: inspect rather than silently deleting promising probes.
REVIEW_TERMS = (
    (r'(?i)\b(?:what\s+details\s+should\s+i\s+ensure\s+before\s+proceeding|latest\s+algorithmic\s+techniques)\b', 'generic_scope'),
    (r'(?i)^\s*(?:\*\*)?scenario\s*:', 'scenario_wrapper_requires_review'),
    (r'(?i)\bwhat\s+information\s+should\s+i\s+gather\b', 'underspecified_elicitation'),
    (r'(?i)\bplease\s+(?:provide|generate|implement|research|create|design|write)\b', 'may_request_execution_instead_of_elicitation'),
    (r'(?i)\bwhat\s+(?:specific\s+)?(?:insights|details|information)\s+would\s+you\s+(?:kindly\s+)?share\b', 'ambiguous_information_direction'),
    (r'(?i)\b(?:the\s+user|they\s+seek|a\s+researcher\s+is|a\s+journalist\s+is)\b', 'third_person_perspective'),
    (r'(?i)\b(?:our\s+(?:current|existing)\s+(?:code|stack|project|application)|my\s+code)\b', 'missing_artifact_possible'),
    (r'(?i)\b(?:20%\s+over\s+three\s+months|improvements?\s+on\s+average\s+page\s+load)\b', 'unsupported_performance_claim'),
    (r'(?i)\b(?:five\s+pieces\s+of\s+information|rank\s+them\s+by\s+(?:expected\s+impact|importance))\b', 'repeated_rank_five_template'),
)

# Signs that the question genuinely asks what the *user* should supply.
HMHY_PATTERNS = (
    r'(?i)\bwhat\s+(?:specific\s+)?(?:information|data|details|artifacts|examples|constraints|inputs|context|evidence|tests|checks|measurements|requirements|questions|documents|files)\s+(?:would\s+you\s+need|should\s+i\s+(?:provide|prepare|send|share|gather|collect)|do\s+you\s+need)\b',
    r'(?i)\bwhat\s+(?:should|could|can)\s+i\s+(?:provide|prepare|send|share|gather|collect)\b',
    r'(?i)\bwhat\s+do\s+you\s+need\s+from\s+me\b',
    r'(?i)\bwhat\s+(?:specific\s+)?(?:information|details|artifacts|evidence|constraints|examples|tests|context|material|inputs)\s+should\s+i\s+(?:provide|share|send|prepare|gather)\b',
    r'(?i)\bwhat\s+should\s+i\s+(?:send|prepare|share)\s+(?:for\s+you|so\s+(?:that\s+)?you)\b',
    r'(?i)\bwhat\s+(?:specific\s+)?(?:information|details|artifacts|examples|constraints|context|data|evidence)\s+(?:should|could|can)\s+i\s+(?:give|supply)\b',
    r'(?i)\bwhat(?:\s+\w+){0,9}\s+should\s+i\s+(?:provide|prepare|send|share|gather|collect)\b',
    r'(?i)\bwhat(?:\s+\w+){0,8}\s+(?:would|do)\s+you\s+need\s+from\s+me\b',
    r'(?i)\bwhat(?:\s+\w+){0,9}\s+do\s+you\s+need\s+from\s+my\s+(?:results|materials|documents|code|study)\b',
    r'(?i)\bwhat\s+(?:specific\s+)?(?:details|information|examples|data|evidence|constraints|artifacts)\s+should\s+i\s+(?:provide|prepare)\b',
    r'(?i)\bwhat\s+(?:checks|questions|counterexamples|measurements|context)\s+should\s+i\s+(?:prepare|answer|provide|share)\b',
    r'(?i)\bwhat\s+(?:do\s+you\s+need|would\s+you\s+need)\s+(?:from\s+me|to\s+(?:assess|design|verify|validate|debug|recommend))\b',
)

GENERIC_MARKERS=(
    'project','information','details','help','approach','scenario','context','requirements',
    'specific','important','ensure','necessary','relevant','provide','questions','task',
    'examples','constraints','please','current','before','work','working','would','should',
    'need','system','review','assist','correctly','what','which','more','five','rank',
    'my','i','the','to','and','for','with','you','your','me','our','we','of','a','an',
)


def assess(question: str, role: str, index: int):
    """Return ('KEEP'|'REJECT'|'REVIEW', [reasons]).

    REVIEW rows must be explicitly approved or rejected by the user before repair.
    This function makes NO claim of role activation/discriminability.
    """
    q=' '.join(str(question).split())
    reasons=[]; hard=[]; soft=[]
    if index not in range(8):return 'REJECT',['invalid_slot']
    if not (25 <= len(q.split()) <= 180) and index < 6:hard.append('hmhy_length_outside_25_180')
    if len(q)<45:soft.append('very_short')
    if len(q)>1800:hard.append('too_long')
    if q.count('?')==0 and index<6:hard.append('missing_question')
    if re.search(r'(?i)\b(?:i|my|we|our|me)\b',q) is None and index<6:soft.append('not_first_person')
    if any(role.lower()==word.lower() for word in re.findall(r'\b[a-z]+_[a-z_]+\b',q)):
        hard.append('snake_case_role_leaked')
    for pattern,reason in BAD_TERMS:
        if re.search(pattern,q):hard.append(reason)
    if index<6 and re.search(r'(?i)\bplease\s+provide\s*:\s*(?:1\.|[0-9]+\s+\*\*|\*\*)',q):
        hard.append('asks_assistant_to_supply_items')
    if index<6:
        if not any(re.search(pattern,q) for pattern in HMHY_PATTERNS):
            # An explicit variation may still be valid; not enough to auto-delete.
            soft.append('elicitation_direction_uncertain')
        if re.search(r'(?i)\b(?:can you|please)\s+(?:provide|summarize|design|generate|create|write|implement|conduct|research|compare|calculate|identify|rank)\b',q):
            soft.append('secondary_execution_request')
        if re.search(r'(?i)\b(?:question|targeted question)\s*:',q):soft.append('unnatural_label')
    else:
        if re.search(r'(?i)\b(?:greatest common divisor|sum of all even numbers|average of three numbers)\b',q):
            soft.append('trivial_direct_control')
        if re.search(r'(?i)\b(?:our|my)\s+(?:current|existing)\s+(?:code|application|project|stack)\b',q):
            soft.append('missing_current_artifact')
    for pattern,reason in REVIEW_TERMS:
        if index>=6 and reason in ('may_request_execution_instead_of_elicitation','repeated_rank_five_template','underspecified_elicitation','ambiguous_information_direction'):
            continue
        if re.search(pattern,q):soft.append(reason)
    # This is only a warning, not an automatic rejection: technical terminology
    # can be expressed with multiword phrases not seen by a lexical filter.
    content=[w for w in re.findall(r'[a-z]{3,}',q.lower()) if w not in GENERIC_MARKERS]
    if len(set(content))<5:soft.append('weak_domain_specificity')
    reasons=list(dict.fromkeys(hard+soft))
    if hard:return 'REJECT',reasons
    if soft:return 'REVIEW',reasons
    return 'KEEP',['no_known_structural_defect']


def strictly_accept(question,role,index):
    decision,reasons=assess(question,role,index)
    # A generated candidate cannot receive human review during GPU sampling;
    # reject REVIEW as well. This is deliberately strict.
    return decision=='KEEP',','.join(reasons)
