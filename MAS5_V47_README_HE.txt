MAS5 V4.7 — Curate 158 probes and regenerate only rejected/missing slots
===================================================================

SAFETY/VALIDITY
- No MAS/API executions occur during AUDIT, CURATE, GPU REPAIR, or INSPECT.
- Existing successful HMHY bootstrap SFT and its adapter are reused.
- Original JSONL is never overwritten until explicit FINALIZE; a backup is saved.
- The automatic quality filter detects structural flaws and flags borderline
  questions for human inspection. It CANNOT establish target activation or
  present/absent fingerprint discrimination. Only matched execution can do that.
- During GPU regeneration, the trained model writes full first-person HMHY
  questions (6 of 8 per role); there is no fixed appended question suffix.
- DPO only occurs AFTER separate controlled training MAS evidence is collected
  and the gates in the existing V4.6 path pass. No DPO happens in the repair job.
- MAS5 is a previously explored DEV set, not a never-seen held-out benchmark.

Install this zip into /home/wieseld/mas_rl_detective with `unzip -n`.

RUN=/home/wieseld/mas_rl_detective/artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051
export V46_ROOT="$RUN"
PY="$HOME/.conda/envs/masdetective/bin/python"

1. AUDIT (CPU, about 2-10 seconds):
$PY -m analysis.role_direct_within_v4.v47_curate audit --root "$V46_ROOT"

2. Inspect the human review file, including borderline and rejected rows:
column -t -s $'\t' "$RUN/train/v47_quality/review.tsv" | less -SR
Edit its `decision` column using your preferred editor; choose KEEP or REJECT.
Automatically REJECT entries cannot be set to KEEP without clearing the hard
validation fault. REVIEW entries can be retained (KEEP) or regenerated (REJECT).
Only review.tsv can be edited, not the original JSONL.

3. CURATE (CPU, 2-10 seconds):
$PY -m analysis.role_direct_within_v4.v47_curate curate --root "$V46_ROOT"
If you consciously want to discard ALL ambiguous questions instead of reading
and deciding, optional `--reject-review` treats every REVIEW as REJECT. This
may discard valid probes; it is intentionally not the default.

4. Generate missing slots on RTX 4090 (10-90 minutes + queue typical estimate;
   longer is possible; no external API):
bash cluster/role_direct_within_v4/v47_submit_repair.sh

5. Check job status and inspect new probes:
squeue --me
sacct -j JOB_ID -X --format=JobID,JobName%20,State,ExitCode,Elapsed
$PY -m analysis.role_direct_within_v4.v47_curate inspect --root "$V46_ROOT" \
  | tee "$RUN/train/v47_quality/preview.txt"

6. If you have reviewed the proposed bank, finalize (CPU <10 s):
$PY -m analysis.role_direct_within_v4.v47_curate finalize \
  --root "$V46_ROOT" --confirm I_REVIEWED_PROBES

7. OPTIONAL, SEPARATE: only if you deliberately authorize paid MAS runs,
   continue original training DPO/Sensor/Selector and MAS5 evaluation:
V46_APPROVE_API_MAS_EXECUTIONS=1160 \
  bash cluster/role_direct_within_v4/v47_continue.sh
The bound is 960 matched training MAS executions + 200 MAS5 executions;
each MAS execution may incur multiple chargeable API calls. Fees unknown.
Jobs have a 20-hour per-stage limit; total chain may exceed 20 hours.

If GPU fails leaving a partial generated bank, retry with
V47_RETRY=1 bash cluster/role_direct_within_v4/v47_submit_repair.sh
The retry reuses the partial bank, does not delete original questions, and
still has no external API calls.
