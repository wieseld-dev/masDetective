# MAS5 V5.9 — Same five systems, **NEW interactions**, frozen Sensor before/after

Goal: test whether Sensor V5.7 generalizes to **new questions and newly executed answers on the SAME five MAS5 systems**. It does NOT use 200 training interactions for validation, and it does NOT claim new-composition generalization.

## What happens

1. Preflight picks **2 UNEXECUTED HMHY probes per role (40 distinct questions)** out of the 160-slot previously generated bank, preferentially from the 134 human-reviewed approvals. It excludes **all identified training queries** from V5.6, V5.4 paired, V4.4 paired (if available), and V137 train; it checks exact normalized query hashes and near duplicates for the same target role. If fewer than two role-specific holdouts exist, **it FAILS before any paid execution** rather than recycling the old questions.
2. With user approval, wait for V5.7's **Slurm `afterok` completion**; execute 100 fresh MAS calls in R1, 100 in R2 using the existing MAS5 target executor and the same five role compositions. New run folders and output paths; no cached responses are read or reused.
3. Score all 200 new single observations plus 100 pairs of observations using **both** the frozen pre-V5.7 Sensor and the **selected V5.7 trained checkpoint** (sequentially, on RTX 4090); same 1280-token input limit; same role-conditioned template, and the same data for both. The combined `joint` input uses the V5.7 two-observation template. Model weights are **never updated** during evaluation.
4. Report **Within-MAS AUC (20 roles, 3 positive) averaged over five systems**, `Exact Top-3`, F1@3 for R1, R2, two-logit mean (primary), and joint (secondary). Also report input truncation, duplicate old response text audit, provenance.

**Important:** Questions existed earlier in the **unexecuted** generator bank, but they were not presented to the MAS in V5.6 or to the Sensor during V5.7 supervised training (per available training-source audit). The responses are generated **now**, not loaded from V5.6. Do not confuse question holdout with a new system holdout.

## Install

On your laptop:

```bash
scp ~/Downloads/mas5_v59_fresh_sensor_test.zip wieseld@slurm.bgu.ac.il:~/mas_rl_detective/
```

On BGU:

```bash
cd ~/mas_rl_detective
unzip -n mas5_v59_fresh_sensor_test.zip
python tests/test_v59_cpu.py
```

## One command to prepare and submit the **full** experiment

```bash
V59_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v59_submit.sh
```

**Approval warning:** triggers up to 200 **new billable MAS executions**, each potentially making several provider-model requests; approximate planning example 1000 model calls without retries, no guaranteed price cap. Training V5.7 must finish successfully first, so `execute1` is scheduled with `afterok:<V57 training job>`; `execute2`, `score_old`, `score_v57`, and `report` follow the previous step via `afterok`.

If you wish to inspect question quality first **without spending**, instead run:

```bash
bash cluster/role_direct_within_v4/v59_submit.sh
```

Then after inspecting `review_40_fresh_questions.tsv`, submit with:

```bash
RUN=$(ls -dt artifacts/role_direct_within_v4/fresh_test_v59/fresh_* | head -1)
V59_APPROVE_MAS_EXECUTIONS=200 bash cluster/role_direct_within_v4/v59_launch.sh "$RUN"
```

Never submit the same run twice. If a step fails, inspect logs and consult before rerunning billable jobs.

## Status & morning results

```bash
cd ~/mas_rl_detective
RUN=$(ls -dt artifacts/role_direct_within_v4/fresh_test_v59/fresh_* | head -1)
bash cluster/role_direct_within_v4/v59_watch.sh "$RUN"
cat "$RUN/fresh_comparison_summary.tsv" 2>/dev/null || echo 'Waiting for fresh evaluation'
```

**Runtime**: 2x100 MAS calls typically tens of minutes to a few hours plus queue. Two 300-observation Sensor scorers may take roughly 10–50 minutes each, depending on device queue/throughput. Each Slurm stage `--time=20:00:00`. A V5.7 training dependency could add several hours. **No GPU/API execution has been tested in this local build.**

### Safety and scientific caveats

- V5.7 was selected using the old V5.6 training interaction set; this test holds out every query selected for V5.9 against the actual identified training query sources.
- Old and new Sensors receive the **same new observations**. No labels, traces or routing enter their inputs. Ground-truth role sets are loaded only at report time.
- The test is same-composition, new-query generalization; one cannot claim transfer to unseen MAS compositions from these results.
- Some answers may still be byte-identical to prior answers. The report counts exact answer duplicates so the result is not misrepresented.
- No Generator/Selector or new Sensor training is performed in this experiment. The 40 queries are distinct holdout questions, not adaptive.
