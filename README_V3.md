# Adaptive Replay V3

This replaces the slow `run_value_pilot`.

What it does:
- no API calls;
- trains on already-recorded MAS outputs;
- Q2/Q3 genuinely depend on the current black-box belief state;
- policy state contains belief + used-probe mask + remaining budget;
- ground truth is used only to produce a scalar training reward;
- compares Random, strong Fixed, Uncertainty, and REINFORCE.

Important: this is still a constrained pilot because actions are limited to the 10
probes that already have recorded outputs. It validates the adaptive learning loop;
it is not the final generated-query research system.
