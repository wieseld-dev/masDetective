from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter


SRC = Path("artifacts/v7_repair_pilot_v1/sources.jsonl")
OUT = Path("artifacts/v7_repair_pilot_v3")
OUT.mkdir(parents=True, exist_ok=True)


DOMAIN_TERMS = {
    "accessibility", "accessible", "financial", "finance", "budget",
    "database", "databases", "query", "queries", "data", "dataset",
    "datasets", "statistical", "statistics", "mathematical", "math",
    "interface", "interfaces", "incident", "incidents", "risk", "risks",
    "customer", "support", "survey", "research", "factual", "facts",
    "claims", "documentation", "document", "documents", "planning",
    "project", "systems", "system", "design", "designing",
    "implementation", "market", "analysis", "analyses",
}


def norm(s):
    return " ".join(str(s or "").strip().split())


def substantive_tokens(s):
    return {
        x.lower()
        for x in re.findall(r"[A-Za-z][A-Za-z_-]+", s)
        if x.lower() in DOMAIN_TERMS
    }


def strict_valid(q):
    lo = norm(q).lower()
    reasons = []

    if not q or not q.endswith("?"):
        reasons.append("not_question")

    bad_dangling = [
        "these tasks",
        "these projects",
        "these assignments",
        "these activities",
        "this capability",
        "these capabilities",
    ]
    if any(x in lo for x in bad_dangling):
        reasons.append("dangling")

    system_self_patterns = [
        r"\bdo you have experience\b",
        r"\bare you experienced\b",
        r"\bare you knowledgeable\b",
        r"\bhave you handled\b",
        r"\bwhat .* have you handled\b",
        r"\bwhat are your (?:skills|strengths|capabilities|expertise)\b",
        r"\bwhat is your expertise\b",
        r"\bdoes your system\b",
        r"\bare you capable\b",
    ]
    if any(re.search(p, lo) for p in system_self_patterns):
        reasons.append("system_self")

    reversed_patterns = [
        r"\bwould you like assistance\b",
        r"\bwould you like help\b",
        r"\bdo you need assistance\b",
        r"\bdo you need help\b",
        r"\bwhat help do you need\b",
        r"\bhow can i help you\b",
        r"\bhow can i assist you\b",
    ]
    if any(re.search(p, lo) for p in reversed_patterns):
        reasons.append("reversed_direction")

    if "[" in q or "]" in q:
        reasons.append("placeholder")

    return len(reasons) == 0, reasons


def clean_phrase(x):
    x = norm(x)
    x = re.sub(r"\*\*", "", x)
    x = re.sub(r"\s+\?$", "", x)
    return x.strip(" .?,")


def repair_direction(q):
    patterns = [
        r"^What specific aspect of (.+?) would you like assistance with\??$",
        r"^What specific types? of (.+?) would you like assistance with\??$",
        r"^Which specific kind of (.+?) would you like help with\??$",
        r"^Which specific type of (.+?) would you like assistance with(?: regarding .+)?\??$",
        r"^Which type of (.+?) would you like assistance with\??$",
        r"^What kind of (.+?) do you need assistance with\??$",
        r"^What specific type of (.+?) would you like assistance with\??$",
        r"^Which specific aspect of (.+?) would you like assistance with\??$",
    ]

    for p in patterns:
        m = re.match(p, q, flags=re.I)
        if m:
            topic = clean_phrase(m.group(1))

            if not substantive_tokens(topic):
                return None, "generic_direction_topic"

            return (
                f"I'm working on {topic}. "
                f"What information should I provide so you can help me "
                f"determine what to focus on first?",
                "direction_minimal",
            )

    return None, "unparsed_direction"


def repair_system_self(q):
    patterns = [
        r"^Do you have experience with (.+?)\??$",
        r"^Do you have experience in (.+?)\??$",
        r"^Do you have experience (.+?)\??$",
        r"^Do you have experience working with (.+?)\??$",
    ]

    for p in patterns:
        m = re.match(p, q, flags=re.I)
        if m:
            activity = clean_phrase(m.group(1))

            if not substantive_tokens(activity):
                return None, "generic_system_self_topic"

            return (
                f"I'm working on {activity}. "
                f"What information should I provide so you can help me "
                f"determine what to examine first?",
                "system_self_minimal",
            )

    return None, "unparsed_system_self"


def repair_dangling(q):
    patterns = [
        r"^Do these tasks (.+?)\??$",
        r"^Do these projects (.+?)\??$",
        r"^Do these assignments (.+?)\??$",
        r"^Do these activities (.+?)\??$",
    ]

    for p in patterns:
        m = re.match(p, q, flags=re.I)
        if m:
            predicate = clean_phrase(m.group(1))

            if not substantive_tokens(predicate):
                return None, "generic_dangling_topic"

            # Preserve the original predicate almost verbatim.
            return (
                f"I'm working on a project that may {predicate}. "
                f"What information should I provide so you can help me "
                f"determine what to focus on?",
                "dangling_minimal",
            )

    return None, "unparsed_dangling"


def make_repair(row):
    fam = row["benign_label"]
    q = norm(row["query"])

    if fam == "DIRECTION_BAD":
        return repair_direction(q)

    if fam == "SYSTEM_SELF_BAD":
        return repair_system_self(q)

    if fam == "DANGLING_BAD":
        return repair_dangling(q)

    return None, "unknown_family"


rows = [
    json.loads(x)
    for x in SRC.open(encoding="utf-8")
    if x.strip()
]

out_rows = []

for src in rows:
    repair, method = make_repair(src)

    if repair is None:
        out_rows.append({
            **src,
            "repair": None,
            "method": method,
            "eligible": False,
            "skip_reason": method,
        })
        continue

    ok, reasons = strict_valid(repair)

    src_terms = substantive_tokens(src["query"])
    rep_terms = substantive_tokens(repair)

    preserved = sorted(src_terms & rep_terms)

    # Because this is deliberately a minimal repair, require all
    # recognized source-domain terms to survive.
    topic_preserved = bool(src_terms) and src_terms <= rep_terms

    eligible = ok and topic_preserved

    out_rows.append({
        **src,
        "repair": repair,
        "method": method,
        "strict_valid": ok,
        "strict_reasons": reasons,
        "source_terms": sorted(src_terms),
        "repair_terms": sorted(rep_terms),
        "preserved_terms": preserved,
        "topic_preserved": topic_preserved,
        "eligible": eligible,
        "skip_reason": None if eligible else (
            "validator" if not ok else "topic_not_preserved"
        ),
    })


with (OUT / "repairs.jsonl").open("w", encoding="utf-8") as f:
    for x in out_rows:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")


eligible = [x for x in out_rows if x.get("eligible")]

summary = {
    "sources": len(out_rows),
    "eligible": len(eligible),
    "eligible_rate": len(eligible) / len(out_rows),
    "by_family": {},
    "skip_reasons": dict(Counter(
        x.get("skip_reason")
        for x in out_rows
        if not x.get("eligible")
    )),
}

for fam in sorted(set(x["benign_label"] for x in out_rows)):
    xs = [x for x in out_rows if x["benign_label"] == fam]
    ys = [x for x in xs if x.get("eligible")]
    summary["by_family"][fam] = {
        "n": len(xs),
        "eligible": len(ys),
    }

(OUT / "summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("===== V7 MINIMAL REPAIR V3 =====")
print(json.dumps(summary, indent=2))

for x in out_rows:
    print("\n" + "=" * 100)
    print(
        f"{x['benign_label']} "
        f"adv={x['local_advantage']:+.3f} "
        f"reward={x['reward']:+.3f}"
    )
    print("BAD :", x["query"])

    if x.get("eligible"):
        print("GOOD:", x["repair"])
        print("TERMS:", x["source_terms"])
    else:
        print("SKIP:", x.get("skip_reason"))
