import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path(
    "artifacts/v14_R7_q38selectorV41_G8_simple5_v1"
)

manifest = json.loads(
    (ROOT/"manifest_privileged.json").read_text()
)

truth = {}

for x in manifest["systems"]:
    sid = str(
        x.get(
            "session_id",
            x.get("system_id")
        )
    )
    truth[sid] = set(
        map(str, x["roles"])
    )

def read(p):
    return [
        json.loads(x)
        for x in p.read_text().splitlines()
        if x.strip()
    ] if p.exists() else []

n = 0
true_n = 0

bins = Counter()
focuses = Counter()

for r in range(1,9):
    diagp = ROOT / f"selector_diag_dev_round_{r:02d}.jsonl"
    statep = ROOT / f"states_dev_round_{r-1:02d}.jsonl"

    if not diagp.exists() or not statep.exists():
        continue

    states = {
        str(x["session_id"]): x
        for x in read(statep)
    }

    for x in read(diagp):
        if not x.get("selector_selected"):
            continue

        sid = str(x["session_id"])

        m = re.search(
            r"TARGET CANDIDATE CAPABILITY:\s*([A-Za-z0-9_]+)",
            str(x.get("prompt",""))
        )

        focus = (
            m.group(1)
            if m
            else "UNKNOWN"
        )

        p = states[sid]["role_probs"].get(
            focus
        )

        is_true = (
            focus in truth[sid]
        )

        n += 1
        true_n += int(is_true)
        focuses[focus] += 1

        if p is not None:
            p = float(p)

            if p < .2:
                b = "<.20"
            elif p < .3:
                b = ".20-.30"
            elif p < .4:
                b = ".30-.40"
            elif p < .6:
                b = ".40-.60"
            else:
                b = ">.60"

            bins[b] += 1

print("selected =", n)
print("true-focus =", true_n)
print(
    "true-focus rate =",
    true_n/max(n,1)
)

print("\nFOCUS COUNTS")
for k,v in focuses.most_common():
    print(f"{k:28} {v}")

print("\nP_BEFORE BINS")
for k,v in bins.items():
    print(k,v)
