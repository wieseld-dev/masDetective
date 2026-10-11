# MAS5 V4.7.2 — Review of newly regenerated probes

This is a **textual quality review**, not an empirical assessment of role activation or PRESENT/ABSENT discrimination.

- New candidates reviewed: **99**
- Accepted unchanged: **41**
- Reject/regenerate: **58**
- Previously accepted from earlier review: **61**
- Provisional total: **102/160** retained, **58** remaining

## Per-role decisions

| Role | Keep new | Reject new |
|---|---:|---:|
| `code` | 2 | 3 |
| `writing` | 3 | 1 |
| `math` | 1 | 4 |
| `planning` | 2 | 5 |
| `research` | 2 | 2 |
| `data` | 2 | 5 |
| `design` | 0 | 3 |
| `operations` | 2 | 3 |
| `qa_testing` | 2 | 3 |
| `translation` | 5 | 2 |
| `financial_analysis` | 2 | 2 |
| `database_design` | 3 | 2 |
| `accessibility_review` | 1 | 3 |
| `fact_checking` | 2 | 3 |
| `requirements_analysis` | 4 | 3 |
| `incident_management` | 1 | 2 |
| `experiment_design` | 2 | 1 |
| `risk_assessment` | 2 | 3 |
| `documentation` | 2 | 4 |
| `customer_support_triage` | 1 | 4 |

## Structural findings

1. Some HMHY-labeled questions still request the model to **execute tasks without input** or ask the model to provide artifacts rather than the user.
2. Metadata such as `Question Family`, `Target Capability Inquiry`, or `Help Me Help You Request` should not be visible in the submitted user query.
3. Some direct control probes refer to **missing source inputs** (datasets, snippets, financial statements, incident reports) or require access the system does not have.
4. No evidence in these files demonstrates that a retained question discriminates TARGET_PRESENT from TARGET_ABSENT. Only a controlled matched test can do that.
5. **Original 61 questions are not in this upload**, so near-duplicate checks against them and validation of the full 160-row bank remain necessary.

## Important handling

Keep source query text and metadata unchanged for KEEP. Do not automatically treat rejected questions as successful model-generated probes after manual rewriting.
Retain old 160-row bank and training responses, if any, as immutable artifacts. Use a new isolated output directory for any regenerated bank.
Train and test must be on separate system compositions. MAS5 has been used for development and is not a fresh sealed external test.

## Review provenance

SHA256 of uploaded file: `5784d3b76e4d9275c606d8175fb80572a5ef8b3901d5fd57bb51eae44cf98b4c`

Each of the 99 per-query decisions and a brief reason is in the accompanying TSV.
