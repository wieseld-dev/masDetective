#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path("artifacts/role_direct_routing_pilot_v1")


def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sid(x):
    return str(
        x.get("system_id")
        or x.get("session_id")
    )


def pair_auc(pos, neg):
    if not pos or not neg:
        return None

    wins = []

    for p in pos:
        for n in neg:
            if p > n:
                wins.append(1.0)
            elif p == n:
                wins.append(0.5)
            else:
                wins.append(0.0)

    return mean(wins)


manifest = json.loads(
    (ROOT / "manifest_privileged.json").read_text()
)

truth = {
    str(x["session_id"]): set(map(str, x["roles"]))
    for x in manifest["systems"]
    if str(x["split"]) == "dev"
}

candidates = read_jsonl(
    ROOT / "candidates_dev_round_01.jsonl"
)

responses = read_jsonl(
    ROOT / "responses_dev_round_01.jsonl"
)

scored = read_jsonl(
    ROOT / "scored_dev_round_01.jsonl"
)

assert len(candidates) == 300, len(candidates)
assert len(responses) == 300, len(responses)
assert len(scored) == 300, len(scored)

resp_by = {
    (sid(x), str(x["query"])): x
    for x in responses
}

score_by = {
    (sid(x), str(x["query"])): x
    for x in scored
}

rows = []

for c in candidates:
    system = str(c["session_id"])
    q = str(c["query"])
    role = str(c["target_role"])

    key = (system, q)

    assert key in resp_by, ("missing response", key)
    assert key in score_by, ("missing score", key)

    r = resp_by[key]
    s = score_by[key]

    selected = set(
        map(
            str,
            r.get("selected_roles_training_only", [])
        )
    )

    v135 = float(
        s["evidence"][role]
    )

    v134 = float(
        s["metadata"]["evidence_v134"][role]
    )

    rows.append({
        "system_id": system,
        "target_role": role,
        "query": q,
        "present": role in truth[system],
        "activated": role in selected,
        "v135": v135,
        "v134": v134,
    })


by_role = defaultdict(list)

for x in rows:
    by_role[x["target_role"]].append(x)


report = {}
evaluable = []

for role in sorted(by_role):
    rr = by_role[role]

    pos = [x for x in rr if x["present"]]
    neg = [x for x in rr if not x["present"]]

    if not pos:
        report[role] = {
            "positive_probes": 0,
            "negative_probes": len(neg),
        }
        continue

    # ----------------------------------------------------------
    # Raw probe-level AUC.
    # ----------------------------------------------------------

    probe_auc_135 = pair_auc(
        [x["v135"] for x in pos],
        [x["v135"] for x in neg],
    )

    probe_auc_134 = pair_auc(
        [x["v134"] for x in pos],
        [x["v134"] for x in neg],
    )

    # ----------------------------------------------------------
    # Same exact query: positive MAS vs four negative MAS.
    # This is the cleanest metric here.
    # ----------------------------------------------------------

    by_query = defaultdict(list)

    for x in rr:
        by_query[x["query"]].append(x)

    matched_135 = []
    matched_134 = []

    for q, qr in by_query.items():

        qp = [x for x in qr if x["present"]]
        qn = [x for x in qr if not x["present"]]

        assert len(qp) == 1, (role, q, len(qp))
        assert len(qn) == 4, (role, q, len(qn))

        p = qp[0]

        for n in qn:
            matched_135.append(
                1.0 if p["v135"] > n["v135"]
                else 0.5 if p["v135"] == n["v135"]
                else 0.0
            )

            matched_134.append(
                1.0 if p["v134"] > n["v134"]
                else 0.5 if p["v134"] == n["v134"]
                else 0.0
            )

    # ----------------------------------------------------------
    # Average the 3 probes per system, then rank positive system
    # against the four negative systems.
    # ----------------------------------------------------------

    by_system = defaultdict(list)

    for x in rr:
        by_system[x["system_id"]].append(x)

    pos_mean_135 = []
    neg_mean_135 = []
    pos_mean_134 = []
    neg_mean_134 = []

    for system, sr in by_system.items():

        m135 = mean(x["v135"] for x in sr)
        m134 = mean(x["v134"] for x in sr)

        if sr[0]["present"]:
            pos_mean_135.append(m135)
            pos_mean_134.append(m134)
        else:
            neg_mean_135.append(m135)
            neg_mean_134.append(m134)

    system_auc_135 = pair_auc(
        pos_mean_135,
        neg_mean_135,
    )

    system_auc_134 = pair_auc(
        pos_mean_134,
        neg_mean_134,
    )

    result = {
        "routing_when_present":
            mean(float(x["activated"]) for x in pos),

        "probe_auc_v135":
            probe_auc_135,

        "probe_auc_v134":
            probe_auc_134,

        "query_matched_win_v135":
            mean(matched_135),

        "query_matched_win_v134":
            mean(matched_134),

        "system_mean_auc_v135":
            system_auc_135,

        "system_mean_auc_v134":
            system_auc_134,

        "mean_present_v135":
            mean(x["v135"] for x in pos),

        "mean_absent_v135":
            mean(x["v135"] for x in neg),
    }

    report[role] = result
    evaluable.append(result)


def macro(key):
    vals = [
        x[key]
        for x in evaluable
        if x.get(key) is not None
    ]
    return mean(vals)


summary = {
    "rows": len(rows),
    "evaluable_roles": len(evaluable),

    "routing_target_when_present":
        mean(
            float(x["activated"])
            for x in rows
            if x["present"]
        ),

    "macro_probe_auc_v135":
        macro("probe_auc_v135"),

    "macro_query_matched_win_v135":
        macro("query_matched_win_v135"),

    "macro_system_mean_auc_v135":
        macro("system_mean_auc_v135"),

    "macro_probe_auc_v134":
        macro("probe_auc_v134"),

    "macro_query_matched_win_v134":
        macro("query_matched_win_v134"),

    "macro_system_mean_auc_v134":
        macro("system_mean_auc_v134"),

    "per_role": report,
}

(
    ROOT / "sensor_pilot_report.json"
).write_text(
    json.dumps(summary, indent=2)
)


print("=" * 100)
print("ROLE DIRECT — SENSOR PILOT")
print("=" * 100)

print("rows                    =", summary["rows"])
print("evaluable roles         =", summary["evaluable_roles"])
print("routing                 =", round(summary["routing_target_when_present"], 4))

print()
print("V13.5")
print("  probe AUC              =", round(summary["macro_probe_auc_v135"], 4))
print("  QUERY-MATCHED WIN      =", round(summary["macro_query_matched_win_v135"], 4))
print("  system-mean AUC        =", round(summary["macro_system_mean_auc_v135"], 4))

print()
print("V13.4")
print("  probe AUC              =", round(summary["macro_probe_auc_v134"], 4))
print("  QUERY-MATCHED WIN      =", round(summary["macro_query_matched_win_v134"], 4))
print("  system-mean AUC        =", round(summary["macro_system_mean_auc_v134"], 4))

print()
print("=" * 100)
print("PER ROLE — V13.5")
print("=" * 100)

for role in sorted(report):
    x = report[role]

    if "routing_when_present" not in x:
        print(f"{role:26s} NO POSITIVE SIMPLE5 SYSTEM")
        continue

    print(
        f"{role:26s} "
        f"route={x['routing_when_present']:.3f}  "
        f"probeAUC={x['probe_auc_v135']:.3f}  "
        f"matched={x['query_matched_win_v135']:.3f}  "
        f"meanAUC={x['system_mean_auc_v135']:.3f}"
    )

print()
print("REPORT =", ROOT / "sensor_pilot_report.json")
