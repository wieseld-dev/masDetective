import json
import math
from pathlib import Path
from collections import defaultdict
import statistics

S5 = Path(
    "artifacts/v14_R7_q38selectorV41_G8_simple5_v1"
)

DATA = Path(
    "artifacts/q38_selector_v41_data"
)

def read(p):
    if not p.exists():
        return []
    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ]

def entropy_like(rp):
    vals = [float(v) for v in rp.values()]
    return -sum(
        p*math.log(max(p,1e-12))
        +
        (1-p)*math.log(max(1-p,1e-12))
        for p in vals
    )

def describe(name, xs):
    if not xs:
        print(name, "EMPTY")
        return

    xs = list(map(float, xs))

    print(
        f"{name:<34}"
        f" n={len(xs):5d}"
        f" mean={statistics.mean(xs):.4f}"
        f" med={statistics.median(xs):.4f}"
        f" min={min(xs):.4f}"
        f" max={max(xs):.4f}"
    )

# ------------------------------------------------
# Simple5 online states.
# ------------------------------------------------

s5_entropy = []
s5_hlen = []
s5_uncertain = []
s5_query_len = []

for p in sorted(S5.glob("states_dev_round_*.jsonl")):
    for x in read(p):
        rp = {
            str(k): float(v)
            for k,v in x["role_probs"].items()
        }

        s5_entropy.append(
            entropy_like(rp)
        )

        s5_hlen.append(
            len(x.get("history", []) or [])
        )

        s5_uncertain.append(
            sum(
                abs(v - .5) < .15
                for v in rp.values()
            )
        )

for p in sorted(S5.glob("candidate_pool_dev_round_*.jsonl")):
    for x in read(p):
        s5_query_len.append(
            len(
                str(x.get("query",""))
                .split()
            )
        )

# ------------------------------------------------
# Offline dataset.
# Adapt to whichever nested representation exists.
# ------------------------------------------------

off_entropy = []
off_hlen = []
off_uncertain = []
off_query_len = []

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

for p in DATA.glob("*.jsonl"):
    for row in read(p):
        for x in walk(row):

            rp = None

            for k in (
                "role_probs",
                "role_belief",
                "role_probs_before",
                "full_role_belief",
            ):
                if isinstance(x.get(k), dict):
                    rp = x[k]
                    break

            if rp:
                try:
                    rp = {
                        str(k): float(v)
                        for k,v in rp.items()
                    }
                    if len(rp) >= 10:
                        off_entropy.append(
                            entropy_like(rp)
                        )
                        off_uncertain.append(
                            sum(
                                abs(v-.5)<.15
                                for v in rp.values()
                            )
                        )
                except Exception:
                    pass

            hist = x.get("history")
            if isinstance(hist, list):
                off_hlen.append(len(hist))

            q = x.get("query")
            if isinstance(q, str) and q.strip():
                off_query_len.append(
                    len(q.split())
                )

print("===== BELIEF ENTROPY-LIKE =====")
describe("OFFLINE", off_entropy)
describe("SIMPLE5", s5_entropy)

print("\n===== HISTORY LENGTH =====")
describe("OFFLINE", off_hlen)
describe("SIMPLE5", s5_hlen)

print("\n===== N ROLES NEAR 0.5 =====")
describe("OFFLINE", off_uncertain)
describe("SIMPLE5", s5_uncertain)

print("\n===== QUERY LENGTH WORDS =====")
describe("OFFLINE", off_query_len)
describe("SIMPLE5", s5_query_len)
