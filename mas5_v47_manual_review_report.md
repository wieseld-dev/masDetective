# MAS5 V4.7 — Manual Review of 160 Candidate Probes

Scope: semantic/formulation curation only; no MAS executions or measured discrimination.

Total: 160; KEEP 61; REJECT 99.

Original-to-final transitions:
- KEEP → KEEP: 31
- KEEP → REJECT: 28
- REJECT → REJECT: 43
- REVIEW → KEEP: 30
- REVIEW → REJECT: 28

## Role breakdown

| Role | KEEP | REJECT | Regenerate slots |
|---|---:|---:|---|
| `code` | 3 | 5 | 1, 2, 4, 6, 7 |
| `writing` | 4 | 4 | 0, 2, 4, 5 |
| `math` | 3 | 5 | 0, 1, 2, 4, 6 |
| `planning` | 1 | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `research` | 4 | 4 | 1, 2, 4, 7 |
| `data` | 1 | 7 | 1, 2, 3, 4, 5, 6, 7 |
| `design` | 5 | 3 | 0, 1, 2 |
| `operations` | 3 | 5 | 0, 1, 2, 3, 6 |
| `qa_testing` | 3 | 5 | 0, 1, 2, 3, 4 |
| `translation` | 1 | 7 | 0, 2, 3, 4, 5, 6, 7 |
| `financial_analysis` | 4 | 4 | 2, 4, 6, 7 |
| `database_design` | 3 | 5 | 0, 1, 2, 4, 7 |
| `accessibility_review` | 4 | 4 | 3, 4, 6, 7 |
| `fact_checking` | 3 | 5 | 0, 1, 2, 4, 6 |
| `requirements_analysis` | 1 | 7 | 0, 1, 2, 4, 5, 6, 7 |
| `incident_management` | 5 | 3 | 2, 4, 6 |
| `experiment_design` | 5 | 3 | 1, 3, 4 |
| `risk_assessment` | 3 | 5 | 0, 2, 3, 6, 7 |
| `documentation` | 2 | 6 | 1, 2, 3, 4, 5, 6 |
| `customer_support_triage` | 3 | 5 | 2, 3, 4, 6, 7 |

## Human-review policy

KEEP means structurally credible role-relevant benign prompt or useful direct control, not empirically effective fingerprinting.

REJECT reasons included mislabeled target-role text, instructions asking the assistant to provide missing input, nonexistent current artifacts/data, boilerplate HMHY, implausible constraints, and uninformative trivial direct controls.

## Important

The curated TSV retains all original query text, role, slot and family fields byte-for-byte at the data-field level, changing only `decision` and adding human rationales to `reasons`.
No source MAS question bank should be overwritten if training-candidate or response artifacts for that run already exist. Regeneration should occur inside an isolated V4.7 quality folder.
