# Curriculum V2 Final Audit

This report uses existing artifacts only; no MAS/LLM execution.

## Generator / Selector

| Stage | Groups | G | Positive pool | Oracle reward | Random reward | Chosen reward | Greedy reward | Chosen regret | Greedy regret | Greedy exact-best | Random exact-best | Greedy top-2 | Spearman |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 96 | 6.00 | 0.948 | 1.310 | -0.123 | 0.107 | 0.014 | 1.203 | 1.296 | 0.146 | 0.167 | 0.323 | 0.080 |
| 2 | 168 | 6.00 | 0.923 | 1.219 | -0.077 | 0.109 | 0.125 | 1.110 | 1.094 | 0.179 | 0.167 | 0.375 | 0.139 |
| 3 | 168 | 6.00 | 0.923 | 1.442 | 0.196 | 0.402 | 0.433 | 1.041 | 1.009 | 0.238 | 0.167 | 0.435 | 0.106 |
| 4 | 256 | 6.00 | 0.973 | 1.416 | 0.310 | 0.382 | 0.459 | 1.033 | 0.956 | 0.180 | 0.167 | 0.375 | 0.068 |
| 5 | 256 | 8.00 | 0.969 | 1.307 | 0.184 | 0.241 | 0.205 | 1.066 | 1.102 | 0.121 | 0.125 | 0.238 | -0.010 |
| 6 | 360 | 8.00 | 0.961 | 1.209 | 0.180 | 0.218 | 0.242 | 0.991 | 0.967 | 0.139 | 0.125 | 0.272 | 0.032 |
| 7 | 360 | 8.00 | 0.950 | 1.255 | 0.129 | 0.138 | 0.135 | 1.117 | 1.121 | 0.114 | 0.125 | 0.289 | -0.029 |
| 8 | 480 | 8.00 | 0.944 | 1.204 | 0.142 | 0.202 | 0.210 | 1.002 | 0.994 | 0.144 | 0.125 | 0.258 | 0.044 |
| 9 | 616 | 8.00 | 0.971 | 1.386 | 0.247 | 0.328 | 0.353 | 1.058 | 1.033 | 0.138 | 0.125 | 0.282 | 0.075 |
| 10 | 768 | 8.00 | 0.977 | 1.297 | 0.303 | 0.388 | 0.400 | 0.909 | 0.896 | 0.172 | 0.125 | 0.286 | 0.059 |

## Generator validity / leakage sanity

- Stage 1: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.31, forbidden-preexecution-fields=[]
- Stage 2: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.27, forbidden-preexecution-fields=[]
- Stage 3: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.24, forbidden-preexecution-fields=[]
- Stage 4: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.22, forbidden-preexecution-fields=[]
- Stage 5: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.26, forbidden-preexecution-fields=[]
- Stage 6: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.25, forbidden-preexecution-fields=[]
- Stage 7: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.22, forbidden-preexecution-fields=[]
- Stage 8: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.18, forbidden-preexecution-fields=[]
- Stage 9: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.09, forbidden-preexecution-fields=[]
- Stage 10: candidate-valid=1.000, duplicate-query-rate=0.000, mean-local-attempts=1.06, forbidden-preexecution-fields=[]

## Paired DEV effects

| Stage | Metric | Rfirst→Rlast | N | Raw delta | Improvement | 95% CI | Fraction improved | sign-test p |
|---:|---|---|---:|---:|---:|---|---:|---:|
| 1 | role_log_loss | 1→2 | 12 | 0.01446 | -0.01446 | [-0.03098, -0.00070] | 0.250 | 0.1460 |
| 1 | role_brier | 1→2 | 12 | 0.00579 | -0.00579 | [-0.01252, -0.00034] | 0.250 | 0.1460 |
| 1 | role_auc | 1→2 | 12 | -0.04575 | -0.04575 | [-0.09150, 0.00163] | 0.250 | 0.2266 |
| 1 | true_false_margin | 1→2 | 12 | -0.01421 | -0.01421 | [-0.02569, -0.00440] | 0.167 | 0.0386 |
| 1 | oracle_k_role_recall | 1→2 | 12 | -0.02778 | -0.02778 | [-0.13889, 0.08333] | 0.167 | 1.0000 |
| 2 | role_log_loss | 1→3 | 12 | -0.01115 | 0.01115 | [-0.00428, 0.02725] | 0.583 | 0.7744 |
| 2 | role_brier | 1→3 | 12 | -0.00373 | 0.00373 | [-0.00195, 0.00959] | 0.667 | 0.3877 |
| 2 | role_auc | 1→3 | 12 | 0.06046 | 0.06046 | [-0.04902, 0.17484] | 0.500 | 1.0000 |
| 2 | true_false_margin | 1→3 | 12 | 0.02037 | 0.02037 | [-0.00009, 0.04316] | 0.583 | 0.7744 |
| 2 | oracle_k_role_recall | 1→3 | 12 | 0.08333 | 0.08333 | [-0.05556, 0.22222] | 0.417 | 0.4531 |
| 3 | role_log_loss | 1→3 | 12 | -0.01400 | 0.01400 | [0.00563, 0.02181] | 0.917 | 0.0063 |
| 3 | role_brier | 1→3 | 12 | -0.00463 | 0.00463 | [0.00150, 0.00748] | 0.917 | 0.0063 |
| 3 | role_auc | 1→3 | 12 | 0.10879 | 0.10879 | [0.04322, 0.17157] | 0.833 | 0.0117 |
| 3 | true_false_margin | 1→3 | 12 | 0.01466 | 0.01466 | [0.00459, 0.02494] | 0.750 | 0.1460 |
| 3 | oracle_k_role_recall | 1→3 | 12 | 0.20139 | 0.20139 | [0.11111, 0.29167] | 0.667 | 0.0078 |
| 4 | role_log_loss | 1→4 | 14 | -0.01617 | 0.01617 | [0.00428, 0.02868] | 0.714 | 0.1796 |
| 4 | role_brier | 1→4 | 14 | -0.00552 | 0.00552 | [0.00136, 0.00987] | 0.643 | 0.4240 |
| 4 | role_auc | 1→4 | 14 | 0.08789 | 0.08789 | [-0.01315, 0.19031] | 0.571 | 0.7905 |
| 4 | true_false_margin | 1→4 | 14 | 0.02368 | 0.02368 | [0.00683, 0.04092] | 0.714 | 0.1796 |
| 4 | oracle_k_role_recall | 1→4 | 14 | 0.07143 | 0.07143 | [-0.10714, 0.24420] | 0.357 | 1.0000 |
| 5 | role_log_loss | 1→4 | 14 | -0.00812 | 0.00812 | [0.00032, 0.01666] | 0.643 | 0.4240 |
| 5 | role_brier | 1→4 | 14 | -0.00287 | 0.00287 | [-0.00011, 0.00612] | 0.643 | 0.4240 |
| 5 | role_auc | 1→4 | 14 | 0.05565 | 0.05565 | [0.01213, 0.09591] | 0.714 | 0.0923 |
| 5 | true_false_margin | 1→4 | 14 | 0.01138 | 0.01138 | [0.00206, 0.02167] | 0.643 | 0.4240 |
| 5 | oracle_k_role_recall | 1→4 | 14 | 0.05357 | 0.05357 | [-0.03214, 0.14286] | 0.357 | 0.4531 |
| 6 | role_log_loss | 1→5 | 14 | -0.00885 | 0.00885 | [-0.00268, 0.02144] | 0.571 | 0.7905 |
| 6 | role_brier | 1→5 | 14 | -0.00296 | 0.00296 | [-0.00076, 0.00705] | 0.643 | 0.4240 |
| 6 | role_auc | 1→5 | 14 | 0.07266 | 0.07266 | [-0.02088, 0.17095] | 0.500 | 0.5488 |
| 6 | true_false_margin | 1→5 | 14 | 0.01211 | 0.01211 | [0.00215, 0.02406] | 0.786 | 0.0574 |
| 6 | oracle_k_role_recall | 1→5 | 14 | 0.06071 | 0.06071 | [-0.05000, 0.18214] | 0.286 | 1.0000 |
| 7 | role_log_loss | 1→5 | 16 | -0.02449 | 0.02449 | [0.01325, 0.03628] | 0.875 | 0.0042 |
| 7 | role_brier | 1→5 | 16 | -0.00833 | 0.00833 | [0.00393, 0.01289] | 0.812 | 0.0213 |
| 7 | role_auc | 1→5 | 16 | 0.12943 | 0.12943 | [0.07902, 0.18628] | 0.875 | 0.0010 |
| 7 | true_false_margin | 1→5 | 16 | 0.01832 | 0.01832 | [0.00826, 0.02946] | 0.812 | 0.0213 |
| 7 | oracle_k_role_recall | 1→5 | 16 | 0.14375 | 0.14375 | [0.07917, 0.20833] | 0.625 | 0.0020 |
| 8 | role_log_loss | 1→6 | 16 | -0.01696 | 0.01696 | [0.00334, 0.03149] | 0.750 | 0.0768 |
| 8 | role_brier | 1→6 | 16 | -0.00667 | 0.00667 | [0.00123, 0.01253] | 0.750 | 0.0768 |
| 8 | role_auc | 1→6 | 16 | 0.09839 | 0.09839 | [0.03265, 0.16423] | 0.750 | 0.0768 |
| 8 | true_false_margin | 1→6 | 16 | 0.01943 | 0.01943 | [0.00465, 0.03572] | 0.625 | 0.4545 |
| 8 | oracle_k_role_recall | 1→6 | 16 | 0.10417 | 0.10417 | [-0.00833, 0.21042] | 0.562 | 0.1460 |
| 9 | role_log_loss | 1→7 | 18 | -0.03298 | 0.03298 | [0.01947, 0.04780] | 0.889 | 0.0013 |
| 9 | role_brier | 1→7 | 18 | -0.01377 | 0.01377 | [0.00831, 0.01983] | 0.889 | 0.0013 |
| 9 | role_auc | 1→7 | 18 | 0.11415 | 0.11415 | [0.05727, 0.16727] | 0.833 | 0.0075 |
| 9 | true_false_margin | 1→7 | 18 | 0.03208 | 0.03208 | [0.02113, 0.04334] | 0.889 | 0.0013 |
| 9 | oracle_k_role_recall | 1→7 | 18 | 0.08565 | 0.08565 | [0.01686, 0.15377] | 0.556 | 0.0923 |
| 10 | role_log_loss | 1→8 | 18 | -0.02269 | 0.02269 | [0.00677, 0.03903] | 0.667 | 0.2379 |
| 10 | role_brier | 1→8 | 18 | -0.00883 | 0.00883 | [0.00234, 0.01566] | 0.722 | 0.0963 |
| 10 | role_auc | 1→8 | 18 | 0.13825 | 0.13825 | [0.09108, 0.18928] | 0.889 | 0.0003 |
| 10 | true_false_margin | 1→8 | 18 | 0.02846 | 0.02846 | [0.01681, 0.04108] | 0.889 | 0.0013 |
| 10 | oracle_k_role_recall | 1→8 | 18 | 0.16071 | 0.16071 | [0.10351, 0.22124] | 0.722 | 0.0002 |

## Stage 10 stopping / saturation

- role_log_loss: best round=7 value=0.63331; final R8=0.63535
- role_brier: best round=7 value=0.22183; final R8=0.22268
- role_auc: best round=7 value=0.76098; final R8=0.75469
- true_false_margin: best round=7 value=0.05254; final R8=0.05121
- oracle_k_role_recall: best round=7 value=0.64187; final R8=0.64054

## Stage 10 automatic checks

- Generator positive-pool fraction: 0.977
- Oracle best reward: 1.297
- Random expected reward: 0.303
- Greedy selector reward: 0.400
- Greedy selector uplift over random: 0.097
- Remaining oracle gap after greedy selector: 0.896
- Greedy exact-best rate: 0.172 vs random baseline 0.125
- Greedy top-2 rate: 0.286
- Mean within-pool predicted-utility/reward Spearman: 0.059

## Reward breakdown fields

See `reward_breakdown.csv`. It separates oracle-best, actual chosen, greedy-selector and expected-random values for every numeric field available in `reward_breakdown_training_only`.

## Files

- `stage_summary.csv`
- `selector_groups.csv`
- `dev_trajectory.csv`
- `paired_effects.csv`
- `reward_breakdown.csv`
- `summary.json`
