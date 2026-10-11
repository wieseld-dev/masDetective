V4.3 HMHy probing prototype, October 9, 2026
- Uses existing trained V4 Generator LoRA without modifying it.
- Five concrete execution-task strategies plus three anchored elicitation strategies
  (intake, ambiguity, validation), for a maximum of 8 per each of 20 target roles.
- Adapts v12.10 scenario + deterministic wrapper; adapted for 20-role catalog.
- Outputs candidate bank and generation metadata ONLY. No MAS execution,
  activation verification, discrimination scoring, Selector training, or new LoRA training.
- Validate candidates manually and with matched PRESENT/ABSENT before DPO.
- Safe install without overwriting any existing file: unzip -n archive.zip.
- Submit: bash cluster/role_direct_within_v4/v43_submit.sh
- Logs: logs/v43-hmhy-<jobid>.out and .err
- Result: <printed RUN>/v43_hmhy_candidate_probes_round_01.jsonl
