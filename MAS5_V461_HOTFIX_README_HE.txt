MAS5 V4.6.1 BOOTSTRAP HOTFIX (NO API CHANGES)

ROOT CAUSE: V4.6 SFT bootstrap reserves one validation example PER ROLE; the sparse V4.3 HMHY bank cannot support 30 train + 5 validation.
FIX: select exactly FIVE validation items from FIVE DISTINCT roles with >=2 qualified examples. Keep all remaining rows in training, with all roles represented. Print counts before training. Preflight checks qualified HMHY count BEFORE any job is scheduled.
No change to DPO reward, sensor, selector, MAS5 runtime, 960+200 cap. Other stages may still fail their gates and cannot be guaranteed.
To install on BGU (no existing running jobs in last diagnostic):
  cd ~/mas_rl_detective
  unzip -o mas5_v461_bootstrap_hotfix.zip
  V46_APPROVE_API_MAS_EXECUTIONS=1160 bash cluster/role_direct_within_v4/v46_submit.sh
Runtime: bootstrap minutes to tens of minutes, dependent on GPU; pipeline potentially many hours and may stop on gates.
