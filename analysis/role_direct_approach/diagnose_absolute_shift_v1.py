#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path("artifacts/role_direct_sensor_v1")

report = json.loads(
    (ROOT / "absolute_detection_report.json").read_text()
)

K = "3"

thresholds = report["results"][K]["thresholds"]

by_role = defaultdict(list)

for x in thresholds:
    by_role[x["role"]].append(x["role_threshold"])

print("=" * 100)
print("K=3 ROLE THRESHOLD STABILITY")
print("=" * 100)

for role in sorted(by_role):
    vv = by_role[role]

    print(
        f"{role:26s} "
        f"mean={mean(vv):+.3f} "
        f"std={pstdev(vv):.3f} "
        f"range=[{min(vv):+.3f},{max(vv):+.3f}]"
    )

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)

stds = [
    pstdev(v)
    for v in by_role.values()
]

ranges = [
    max(v) - min(v)
    for v in by_role.values()
]

print("mean threshold std   =", round(mean(stds), 4))
print("mean threshold range =", round(mean(ranges), 4))
print("max threshold std    =", round(max(stds), 4))
print("max threshold range  =", round(max(ranges), 4))
