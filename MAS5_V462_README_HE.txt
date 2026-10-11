MAS5 V4.6.2 — RESUME HOTFIX FROM EXISTING FAILED RUN (not a fresh run)

Observed results: Generator Bootstrap SFT PASS in 47 sec (30 train/5 val,
heldout NLL 2.6838 -> 2.62195); Generator probe bank 158/160, missing
one math and one qa_testing. No MAS/API training or MAS5 test started.

This patch saves the 158 validated probes, samples ONLY the two missing slots
from existing trained Generator adapter (96 attempts/slot max), validates all
160 and preserves original probes, then atomically restores the full training
bank with a permanent backup. If quality/diversity still fails, it stops
BEFORE any MAS API execution. Regular test-phase generation now allows
64 attempts/slot rather than 18; it never relaxes the quality rules.

Resume exactly prior RUN directory: V46_ROOT=.../mas5_night_20261009_224735_1879051
NO need to rerun bootstrap. Afterok DAG: repair -> prepare -> train execute
(up to 960 API-backed MAS executions) -> evidence/gates/Generator DPO,
Sensor/Selector -> MAS5 test (up to 200 MAS executions), conditional.

CAUTION: 960+200 are MAS executions, each may cost multiple model API calls.
No guarantee DPO evidence gate or subsequent evaluation passes; teacher scorer
V4.2 was below TF-IDF baseline. Training MAS configurations exclude identical
MAS5 triple compositions but are not new sealed independent workloads.

Install: unzip -o mas5_v462_resume_hotfix.zip
Submit: V46_APPROVE_API_MAS_EXECUTIONS=1160 \
 V46_ROOT="$PWD/artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051" \
 bash cluster/role_direct_within_v4/v462_resume.sh
Expected repair time: 2-20 min on RTX4090 plus queue. Full chain hours.
