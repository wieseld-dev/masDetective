import json
from pathlib import Path

SRC = Path(
    "artifacts/v7_retention_pilot_v1/"
    "selected_sources.jsonl"
)

OUT = Path(
    "artifacts/v7_substantive_repair_v4"
)
OUT.mkdir(parents=True, exist_ok=True)

rows = [
    json.loads(x)
    for x in SRC.open()
    if x.strip()
]

# pair_id follows the frozen V7 retention pilot order.
REPAIRS = {
    0:
        "I'm reviewing a website for accessibility and want to identify "
        "the issues that would matter most to users. What should I examine "
        "first, and what kinds of problems should I look for?",

    1:
        "I'm reviewing our financial performance and want to understand "
        "which parts deserve closer analysis. What should I examine first "
        "and what indicators would be most informative?",

    2:
        "I have a dataset I need to analyze but I'm not yet sure which "
        "patterns or checks will be most useful. What should I examine "
        "first and why?",

    3:
        "I'm analyzing a dataset for an upcoming decision and want to "
        "avoid focusing on the wrong signals. What should I examine first "
        "and how would you prioritize the analysis?",

    4:
        "I'm trying to improve several database queries that are becoming "
        "slow as the amount of data grows. What should I examine first to "
        "improve their design and performance?",

    5:
        "We're redesigning a database that needs to remain reliable as "
        "data volume and traffic increase. What architectural choices "
        "should I evaluate first to keep it scalable?",

    6:
        "We're redesigning a complex interface with several user workflows "
        "and inconsistent navigation. What should I examine first to "
        "identify the most important usability problems?",

    7:
        "I'm reviewing a database that has become harder to maintain and "
        "query efficiently. What should I examine first to improve its "
        "structure and performance?",

    8:
        "Our support team receives many different customer inquiries and "
        "we need a consistent way to handle them. What factors should we "
        "consider first when deciding how to triage and respond?",

    9:
        "I have a dataset with several variables and noisy measurements, "
        "and I'm deciding whether advanced statistical modeling is "
        "appropriate. What should I examine before choosing a method?",

    10:
        "I need to make a decision using a large dataset with many "
        "variables and observations. What should I examine first before "
        "drawing conclusions from the data?",

    11:
        "I'm reviewing an interface with several user flows and some "
        "inconsistent navigation. What should I examine first to identify "
        "the most important design problems?",
}

assert len(rows) == 12
assert set(REPAIRS) == set(range(12))

out = []

for pair_id, src in enumerate(rows):
    q = REPAIRS[pair_id]

    out.append({
        **src,
        "pair_id": pair_id,
        "v3_repair": src["repair"],
        "repair": q,
        "repair_version": "V4_SUBSTANTIVE",
    })

    print("=" * 100)
    print("PAIR", pair_id, src["benign_label"])
    print("BAD :", src["query"])
    print("V3  :", src["repair"])
    print("V4  :", q)

with (OUT / "sources.jsonl").open(
    "w",
    encoding="utf-8",
) as f:
    for x in out:
        f.write(json.dumps(x, ensure_ascii=False) + "\n")

print("\nWROTE:", OUT / "sources.jsonl")
