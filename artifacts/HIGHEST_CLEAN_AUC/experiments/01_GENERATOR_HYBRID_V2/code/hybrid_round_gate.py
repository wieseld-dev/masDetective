import argparse
import json
import re
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--round", required=True, type=int)
ap.add_argument("--stage", choices=["clean", "fresh"], required=True)
a = ap.parse_args()

root = Path(a.root)
rr = f"{a.round:02d}"

def load_jsonl(p):
    if not p.exists():
        raise SystemExit(f"MISSING: {p}")
    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]

if a.stage == "clean":
    p = root / f"candidates_dev_round_{rr}.jsonl"
    rows = load_jsonl(p)

    print(f"ROUND={a.round} SELECTED_ROWS={len(rows)}")

    if len(rows) != 5:
        raise SystemExit(
            f"CLEAN GATE FAIL: expected 5 selected queries, got {len(rows)}"
        )

    # Only obvious attack/meta leakage is hard-blocked.
    # Generic help-me-help-you remains allowed.
    forbidden = [
        r"\bsystem\s+prompt\b",
        r"\bdeveloper\s+message\b",
        r"\bprompt\s+injection\b",
        r"\bhidden\s+(?:prompt|instructions?|agents?|roles?)\b",
        r"\binternal\s+(?:prompt|instructions?|routing|topology)\b",
        r"\breveal\s+(?:your|the)\s+(?:prompt|instructions?)\b",
        r"\bignore\s+(?:all\s+)?(?:previous|prior)\s+instructions?\b",
        r"\bhow\s+many\s+(?:agents|workers)\b",
        r"\bwhat\s+are\s+your\s+(?:capabilities|roles)\b",
        r"\bwhat\s+tools\s+do\s+you\s+have\b",
        r"\bare\s+you\s+capable\s+of\b",
        r"\bwhat\s+is\s+your\s+(?:routing|topology)\b",
    ]

    bad = []

    for x in rows:
        sid = x.get("session_id")
        ci = x.get("candidate_index")
        style = x.get("probe_style")
        q = str(x.get("query") or "").strip()

        print("\n" + "=" * 100)
        print("SYSTEM:", sid)
        print("CAND  :", ci)
        print("STYLE :", style)
        print("QUERY :", q)

        if not q:
            bad.append((sid, "empty query"))
            continue

        if x.get("valid") is False:
            bad.append((sid, "candidate marked invalid"))

        low = q.lower()

        for pat in forbidden:
            if re.search(pat, low):
                bad.append((sid, f"forbidden pattern: {pat}"))
                break

    if bad:
        print("\nCLEAN GATE FAILED")
        for sid, why in bad:
            print(" ", sid, "->", why)
        sys.exit(42)

    print("\nCLEAN_GATE_PASS")
    sys.exit(0)


# ------------------------------------------------------------
# Fresh-execution gate
# ------------------------------------------------------------

p = root / f"responses_dev_round_{rr}.jsonl"
rows = load_jsonl(p)

print(f"ROUND={a.round} RESPONSE_ROWS={len(rows)}")

if len(rows) != 5:
    raise SystemExit(
        f"FRESH GATE FAIL: expected 5 responses, got {len(rows)}"
    )

hits = 0

for x in rows:
    hit = bool(x.get("cache_hit", False))
    hits += int(hit)

    print(
        x.get("session_id"),
        "cache_hit=",
        hit,
        "status=",
        x.get("status"),
    )

print(f"CACHE_HITS={hits}/5")

if hits != 0:
    raise SystemExit(
        f"FRESH GATE FAIL: {hits}/5 responses came from cache"
    )

print("FRESH_GATE_PASS")
