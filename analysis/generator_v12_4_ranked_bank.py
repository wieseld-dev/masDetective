from __future__ import annotations

import json
import random
import re
from pathlib import Path

import torch

from analysis.benign_generator_ab_v1 import strict_benign
from adaptive_v11_active.io import stable_int
from adaptive_v12_benign.targeted_policy_v12_2 import (
    TargetedRichProbePolicyV12_2,
)
from adaptive_v12_benign.gate_v12_1 import (
    INTERNAL_PATTERNS,
    DIRECTION_REVERSAL_PATTERNS,
    TASK_VERBS,
    _contains_any_pattern,
    _richness_score,
)
from v11.role_universe import (
    ROLE_DESCRIPTIONS,
    anchor_probe,
)


ROLES = [
    "code",
    "writing",
    "math",
    "planning",
    "research",
    "data",
    "design",
    "operations",
]

TARGET_CUES = {
    "code": {
        "code", "function", "script", "debug", "bug",
        "implement", "implementation", "python", "java",
        "algorithm", "software", "test", "api",
    },
    "writing": {
        "draft", "write", "rewrite", "edit", "tone",
        "clarity", "article", "email", "paragraph",
        "structure", "reader",
    },
    "math": {
        "calculate", "calculation", "equation",
        "statistics", "statistical", "probability",
        "uncertainty", "variance", "average",
        "numerical", "units",
    },
    "planning": {
        "plan", "roadmap", "timeline", "milestone",
        "milestones", "priority", "prioritize",
        "dependency", "dependencies", "schedule",
        "resource", "risk",
    },
    "research": {
        "research", "evidence", "source", "sources",
        "literature", "credibility", "peer-reviewed",
        "study", "studies", "synthesize", "claims",
        "compare",
    },
    "data": {
        "dataset", "data", "clean", "cleaning",
        "preprocess", "preprocessing", "pandas",
        "dataframe", "outlier", "missing",
        "transform", "schema", "analysis",
    },
    "design": {
        "interface", "ux", "user experience",
        "usability", "interaction", "navigation",
        "prototype", "layout", "user flow",
        "information architecture",
    },
    "operations": {
        "process", "procedure", "workflow",
        "handoff", "monitoring", "reliability",
        "incident", "bottleneck", "maintenance",
        "operational", "operations",
    },
}

CLARIFICATION_PATTERNS = [
    r"\bwhat should i send\b",
    r"\bwhat information should i provide\b",
    r"\bwhat .* should i provide\b",
    r"\bcould you please clarify\b",
    r"\bplease clarify\b",
    r"\bwhat specific .* do you need\b",
    r"\bprovide me with an initial\b",
    r"\bwhat kind of information\b",
    r"\bwhat details .* need\b",
]


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean(x):
    s = str(x or "").strip()

    s = re.sub(
        r"^\s*(?:question|query|probe|task)\s*:\s*",
        "",
        s,
        flags=re.I,
    ).strip()

    if (
        len(s) >= 2
        and s[0] == s[-1]
        and s[0] in {"'", '"'}
    ):
        s = s[1:-1].strip()

    return s


def recover(sample):
    for field in ("query", "completion", "raw_completion"):
        q = clean(getattr(sample, field, ""))

        if q:
            return q, field

    return "", "empty"


def hard_gate(q):
    q = str(q or "").strip()

    words = re.findall(
        r"\b[\w'-]+\b",
        q,
    )

    if len(words) < 8:
        return False, "fragment"

    if len(words) > 180:
        return False, "too_long"

    ok, reason = strict_benign(q)

    # Imperative tasks are allowed in V12.4.
    if not ok and reason != "not_question":
        return False, f"strict_benign:{reason}"

    lower = q.lower()

    if _contains_any_pattern(
        lower,
        INTERNAL_PATTERNS,
    ):
        return False, "internal_meta"

    if _contains_any_pattern(
        lower,
        DIRECTION_REVERSAL_PATTERNS,
    ):
        return False, "direction_reversal"

    if any(
        x in lower
        for x in (
            "<topic>",
            "<task>",
            "{topic}",
            "{task}",
            "<insert",
        )
    ):
        return False, "template_residue"

    return True, ""


def rank_candidate(role, q):
    lower = q.lower()

    words = re.findall(
        r"\b[\w'-]+\b",
        q,
    )
    n = len(words)

    richness = _richness_score(q)

    tokens = set(
        re.findall(
            r"[a-z]+",
            lower,
        )
    )

    has_task_verb = bool(
        tokens & TASK_VERBS
    )

    target_hits = sum(
        int(cue in lower)
        for cue in TARGET_CUES[role]
    )

    clarification = any(
        re.search(
            p,
            lower,
            flags=re.I,
        )
        for p in CLARIFICATION_PATTERNS
    )

    if 30 <= n <= 80:
        length_score = 3.0
    elif 22 <= n <= 100:
        length_score = 1.5
    elif n < 18:
        length_score = -2.0
    elif n > 120:
        length_score = -2.0
    else:
        length_score = 0.0

    score = (
        2.5 * min(target_hits, 4)
        + 1.5 * min(richness, 4)
        + 1.0 * int(has_task_verb)
        + length_score
        - 5.0 * int(clarification)
    )

    return {
        "score": float(score),
        "words": n,
        "target_hits": target_hits,
        "richness": richness,
        "task_verb": has_task_verb,
        "clarification": clarification,
    }


def main():
    out = Path(
        "artifacts/generator_v12_4"
    )
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    policy = TargetedRichProbePolicyV12_2(
        "Qwen/Qwen2.5-1.5B-Instruct",
        adapter_path=None,
        trainable=False,
        use_4bit=False,
        max_seq_length=4096,
    )

    all_rows = []
    selected = []

    for role in ROLES:
        role_rows = []

        for j in range(4):
            seed = stable_int(
                20260927,
                "v12_4",
                role,
                j,
            )

            seed_all(seed)

            obs = {
                "candidate_role":
                    role,
                "candidate_description":
                    ROLE_DESCRIPTIONS[role],
                "anchor_probe":
                    anchor_probe(role),
                "avoid_probes":
                    [anchor_probe(role)],
            }

            sample = policy.sample_probe(
                obs,
                temperature=0.90,
                top_p=0.96,
                max_new_tokens=224,
            )

            q, source = recover(sample)

            valid, reason = hard_gate(q)

            metrics = (
                rank_candidate(role, q)
                if valid
                else {
                    "score": -999.0,
                    "words": 0,
                    "target_hits": 0,
                    "richness": 0,
                    "task_verb": False,
                    "clarification": False,
                }
            )

            row = {
                "role": role,
                "candidate": j,
                "query": q,
                "hard_benign": valid,
                "reason": reason,
                "source": source,
                **metrics,
            }

            role_rows.append(row)
            all_rows.append(row)

        valid_rows = [
            x
            for x in role_rows
            if x["hard_benign"]
        ]

        valid_rows.sort(
            key=lambda x: (
                x["score"],
                x["target_hits"],
                x["richness"],
            ),
            reverse=True,
        )

        winners = valid_rows[:2]

        for rank, x in enumerate(
            winners,
            1,
        ):
            selected.append({
                **x,
                "selected_rank":
                    rank,
            })

    with (
        out / "all_candidates.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in all_rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (
        out / "selected_bank.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in selected:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n" + "=" * 110)
    print("V12.4 SELECTED BANK")
    print("=" * 110)

    for role in ROLES:
        print(
            "\n###",
            role,
        )

        xs = [
            x
            for x in selected
            if x["role"] == role
        ]

        for x in xs:
            print(
                f"\nrank={x['selected_rank']} "
                f"score={x['score']:.2f} "
                f"words={x['words']} "
                f"target_hits={x['target_hits']} "
                f"richness={x['richness']}"
            )
            print(x["query"])


if __name__ == "__main__":
    main()
