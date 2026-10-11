MAS5 V5.8 Dependency-only overnight patch

Contents:
- cluster/role_direct_within_v4/v58_launch.sh: now supports a pending/running R1 Preview by setting `--dependency=afterok:<preview_job_id>` on the first paid execute job. All remaining generate/execute/score jobs are chained via afterok.
- cluster/role_direct_within_v4/v58_overnight.sh: one-command wrapper. With RUN argument, uses that prepared V5.8 run; without RUN argument, prepares a new run and submits the full chain.

No existing V5.6/V5.7 jobs are cancelled. R1 Preview is reused and not regenerated.
The existing RUN/jobs.tsv lock prevents a second paid submission for the same run.
No MAS/API requests are issued by the install or submit commands themselves. Paid execution begins when Slurm reaches each execute job; authorise with V58_APPROVE_MAS_EXECUTIONS=160.

Usage (existing prepared RUN):
  V58_APPROVE_MAS_EXECUTIONS=160 bash cluster/role_direct_within_v4/v58_overnight.sh "$RUN"

Usage (no prepared RUN):
  V58_APPROVE_MAS_EXECUTIONS=160 bash cluster/role_direct_within_v4/v58_overnight.sh

Expected end-to-end time: several hours, potentially overnight, depending on queue, generation, and API. Each Slurm job time limit remains 20 hours.
