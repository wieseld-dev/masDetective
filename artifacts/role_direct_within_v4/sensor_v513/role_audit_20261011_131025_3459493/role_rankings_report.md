# V5.13 — Role ranking error attribution (retrospective)

- V58 installed roles: **code, planning, research**
- V58 source: `/home/wieseld/mas_rl_detective/artifacts/role_direct_within_v4/adaptive_v58/mas5_20261010_210843_2874280`
- V512 reference: `/home/wieseld/mas_rl_detective/artifacts/role_direct_within_v4/sensor_v512/calibration_20261011_125719_3414025`
- **Frozen 20 role-specific TF-IDF verifiers; no MAS calls and no weight updates.**
- V58 composition was seen during training (V56); only exact interaction overlap was audited.
- V137 DEV has previously been inspected; this is diagnostic, not independent model selection.
- Model inputs are only user question/visible answer per arm. Installed roles are used only when evaluating.

## V58 rounds: ranking of 20 roles

### Round 1

- Raw mean: AUC **0.784**, Top-3 `research,incident_management,operations`, correct **1/3**; misses `code,planning`.
- Negative-centered mean: AUC **0.392**, Top-3 `database_design,operations,risk_assessment`, correct **0/3**.

| Role | Present? | Raw logit mean | Raw rank | Null median (VAL) | Centered rank |
|---|---:|---:|---:|---:|---:|
| `research` | YES | -0.1195 | 1 | -0.0766 | 14 |
| `incident_management` | no | -0.1247 | 2 | -0.4025 | 5 |
| `operations` | no | -0.1456 | 3 | -0.5517 | 2 |
| `planning` | YES | -0.1457 | 4 | -0.3121 | 7 |
| `requirements_analysis` | no | -0.1597 | 5 | -0.4707 | 4 |
| `translation` | no | -0.1775 | 6 | -0.2249 | 11 |
| `writing` | no | -0.1850 | 7 | -0.1513 | 13 |
| `risk_assessment` | no | -0.1851 | 8 | -0.5541 | 3 |
| `customer_support_triage` | no | -0.2031 | 9 | -0.3725 | 6 |
| `documentation` | no | -0.2489 | 10 | -0.4121 | 8 |
| `qa_testing` | no | -0.2523 | 11 | -0.1159 | 17 |
| `code` | YES | -0.2886 | 12 | -0.1731 | 16 |
| `accessibility_review` | no | -0.2904 | 13 | -0.1031 | 18 |
| `data` | no | -0.3171 | 14 | -0.2569 | 15 |
| `database_design` | no | -0.4331 | 15 | -0.9442 | 1 |
| `experiment_design` | no | -0.5458 | 16 | -0.6392 | 10 |
| `design` | no | -0.5961 | 17 | -0.5628 | 12 |
| `fact_checking` | no | -0.6315 | 18 | -0.7343 | 9 |
| `math` | no | -0.9406 | 19 | -0.0928 | 20 |
| `financial_analysis` | no | -1.0552 | 20 | -0.3591 | 19 |

### Round 2

- Raw mean: AUC **0.706**, Top-3 `research,incident_management,risk_assessment`, correct **1/3**; misses `code,planning`.
- Negative-centered mean: AUC **0.412**, Top-3 `database_design,risk_assessment,operations`, correct **0/3**.

| Role | Present? | Raw logit mean | Raw rank | Null median (VAL) | Centered rank |
|---|---:|---:|---:|---:|---:|
| `research` | YES | -0.0730 | 1 | -0.0766 | 12 |
| `incident_management` | no | -0.1057 | 2 | -0.4025 | 4 |
| `risk_assessment` | no | -0.1304 | 3 | -0.5541 | 2 |
| `operations` | no | -0.1462 | 4 | -0.5517 | 3 |
| `translation` | no | -0.1485 | 5 | -0.2249 | 10 |
| `planning` | YES | -0.1597 | 6 | -0.3121 | 8 |
| `requirements_analysis` | no | -0.1762 | 7 | -0.4707 | 5 |
| `writing` | no | -0.2063 | 8 | -0.1513 | 15 |
| `documentation` | no | -0.2105 | 9 | -0.4121 | 6 |
| `customer_support_triage` | no | -0.2118 | 10 | -0.3725 | 7 |
| `qa_testing` | no | -0.2462 | 11 | -0.1159 | 17 |
| `data` | no | -0.2623 | 12 | -0.2569 | 13 |
| `accessibility_review` | no | -0.2650 | 13 | -0.1031 | 18 |
| `code` | YES | -0.2849 | 14 | -0.1731 | 16 |
| `database_design` | no | -0.3512 | 15 | -0.9442 | 1 |
| `experiment_design` | no | -0.5348 | 16 | -0.6392 | 9 |
| `design` | no | -0.6117 | 17 | -0.5628 | 14 |
| `fact_checking` | no | -0.7097 | 18 | -0.7343 | 11 |
| `financial_analysis` | no | -0.9932 | 19 | -0.3591 | 19 |
| `math` | no | -1.0323 | 20 | -0.0928 | 20 |

### Round 3

- Raw mean: AUC **0.706**, Top-3 `research,incident_management,risk_assessment`, correct **1/3**; misses `code,planning`.
- Negative-centered mean: AUC **0.412**, Top-3 `database_design,risk_assessment,operations`, correct **0/3**.

| Role | Present? | Raw logit mean | Raw rank | Null median (VAL) | Centered rank |
|---|---:|---:|---:|---:|---:|
| `research` | YES | -0.0641 | 1 | -0.0766 | 11 |
| `incident_management` | no | -0.1031 | 2 | -0.4025 | 4 |
| `risk_assessment` | no | -0.1253 | 3 | -0.5541 | 2 |
| `operations` | no | -0.1371 | 4 | -0.5517 | 3 |
| `translation` | no | -0.1616 | 5 | -0.2249 | 10 |
| `planning` | YES | -0.1729 | 6 | -0.3121 | 8 |
| `requirements_analysis` | no | -0.1971 | 7 | -0.4707 | 5 |
| `writing` | no | -0.1975 | 8 | -0.1513 | 15 |
| `customer_support_triage` | no | -0.2032 | 9 | -0.3725 | 7 |
| `qa_testing` | no | -0.2163 | 10 | -0.1159 | 16 |
| `documentation` | no | -0.2293 | 11 | -0.4121 | 6 |
| `data` | no | -0.2548 | 12 | -0.2569 | 13 |
| `accessibility_review` | no | -0.2681 | 13 | -0.1031 | 18 |
| `code` | YES | -0.2758 | 14 | -0.1731 | 17 |
| `database_design` | no | -0.3414 | 15 | -0.9442 | 1 |
| `experiment_design` | no | -0.5044 | 16 | -0.6392 | 9 |
| `design` | no | -0.6028 | 17 | -0.5628 | 14 |
| `fact_checking` | no | -0.7246 | 18 | -0.7343 | 12 |
| `financial_analysis` | no | -0.9269 | 19 | -0.3591 | 19 |
| `math` | no | -1.0172 | 20 | -0.0928 | 20 |

### Round 4

- Raw mean: AUC **0.706**, Top-3 `research,incident_management,risk_assessment`, correct **1/3**; misses `code,planning`.
- Negative-centered mean: AUC **0.392**, Top-3 `database_design,risk_assessment,operations`, correct **0/3**.

| Role | Present? | Raw logit mean | Raw rank | Null median (VAL) | Centered rank |
|---|---:|---:|---:|---:|---:|
| `research` | YES | -0.0458 | 1 | -0.0766 | 11 |
| `incident_management` | no | -0.1040 | 2 | -0.4025 | 4 |
| `risk_assessment` | no | -0.1216 | 3 | -0.5541 | 2 |
| `operations` | no | -0.1289 | 4 | -0.5517 | 3 |
| `translation` | no | -0.1744 | 5 | -0.2249 | 10 |
| `planning` | YES | -0.1853 | 6 | -0.3121 | 9 |
| `writing` | no | -0.1939 | 7 | -0.1513 | 15 |
| `customer_support_triage` | no | -0.2020 | 8 | -0.3725 | 7 |
| `requirements_analysis` | no | -0.2104 | 9 | -0.4707 | 5 |
| `qa_testing` | no | -0.2201 | 10 | -0.1159 | 16 |
| `documentation` | no | -0.2383 | 11 | -0.4121 | 6 |
| `data` | no | -0.2594 | 12 | -0.2569 | 12 |
| `accessibility_review` | no | -0.2675 | 13 | -0.1031 | 18 |
| `code` | YES | -0.2846 | 14 | -0.1731 | 17 |
| `database_design` | no | -0.3234 | 15 | -0.9442 | 1 |
| `experiment_design` | no | -0.4884 | 16 | -0.6392 | 8 |
| `design` | no | -0.5868 | 17 | -0.5628 | 13 |
| `fact_checking` | no | -0.7726 | 18 | -0.7343 | 14 |
| `financial_analysis` | no | -0.9039 | 19 | -0.3591 | 19 |
| `math` | no | -1.0004 | 20 | -0.0928 | 20 |

### Round 5

- Raw mean: AUC **0.686**, Top-3 `research,incident_management,risk_assessment`, correct **1/3**; misses `code,planning`.
- Negative-centered mean: AUC **0.373**, Top-3 `database_design,risk_assessment,operations`, correct **0/3**.

| Role | Present? | Raw logit mean | Raw rank | Null median (VAL) | Centered rank |
|---|---:|---:|---:|---:|---:|
| `research` | YES | -0.0632 | 1 | -0.0766 | 12 |
| `incident_management` | no | -0.1086 | 2 | -0.4025 | 4 |
| `risk_assessment` | no | -0.1334 | 3 | -0.5541 | 2 |
| `operations` | no | -0.1399 | 4 | -0.5517 | 3 |
| `writing` | no | -0.1686 | 5 | -0.1513 | 13 |
| `translation` | no | -0.1796 | 6 | -0.2249 | 10 |
| `planning` | YES | -0.1923 | 7 | -0.3121 | 9 |
| `requirements_analysis` | no | -0.2004 | 8 | -0.4707 | 5 |
| `customer_support_triage` | no | -0.2125 | 9 | -0.3725 | 7 |
| `qa_testing` | no | -0.2170 | 10 | -0.1159 | 16 |
| `data` | no | -0.2401 | 11 | -0.2569 | 11 |
| `documentation` | no | -0.2491 | 12 | -0.4121 | 6 |
| `accessibility_review` | no | -0.2643 | 13 | -0.1031 | 18 |
| `code` | YES | -0.2825 | 14 | -0.1731 | 17 |
| `database_design` | no | -0.3360 | 15 | -0.9442 | 1 |
| `experiment_design` | no | -0.5017 | 16 | -0.6392 | 8 |
| `design` | no | -0.5924 | 17 | -0.5628 | 14 |
| `fact_checking` | no | -0.7684 | 18 | -0.7343 | 15 |
| `financial_analysis` | no | -0.8961 | 19 | -0.3591 | 19 |
| `math` | no | -1.0198 | 20 | -0.0928 | 20 |

## V137 DEV (24 systems)

- Number of systems: 24; role-specific logits and both ranks in `v137_role_predictions.tsv`.
- `v137_system_errors.tsv` isolates systems where absent roles enter Top-3.
- V58 changes in a role's raw logit **do not measure per-role binary AUC**, because one MAS has a fixed role-presence label.
- Frozen-model historical replay **cannot prove an adaptive improvement** or new-composition generalization.
- Read `v58_probe_excerpts.tsv` to compare probes and visible response excerpts for potential shortcuts/truncation.
