# MAS5 — Final 42 question review (V4.8.0)

This is a **textual and methodological review**, not experimental PRESENT/ABSENT evidence. Keeping a question does not prove target activation or discrimination.

## Summary

- Reviewed: **42** (originally approved bank: **118**).
- **KEEP: 16**. **REJECT: 26**.
- Combined candidate quality-approved bank: **134/160**, if the 118 earlier approvals are intact and unmodified.
- The two manually authored `customer_support_triage` HMHY probes (slots 3, 4) both pass this review.
- Do **not** substitute a 134/160 text-reviewed bank for the 160-slot MAS5 evaluation without explicitly changing the fixed-size experiment design.

## Decision by role

| Role | KEEP | REJECT |
|---|---:|---:|
| `accessibility_review` | 1 | 2 |
| `code` | 1 | 1 |
| `customer_support_triage` | 2 | 2 |
| `data` | 2 | 2 |
| `database_design` | 0 | 1 |
| `design` | 0 | 1 |
| `documentation` | 1 | 2 |
| `fact_checking` | 2 | 0 |
| `financial_analysis` | 0 | 2 |
| `incident_management` | 1 | 0 |
| `math` | 0 | 2 |
| `operations` | 1 | 1 |
| `planning` | 1 | 1 |
| `qa_testing` | 2 | 1 |
| `requirements_analysis` | 0 | 3 |
| `research` | 0 | 1 |
| `risk_assessment` | 0 | 3 |
| `translation` | 1 | 1 |
| `writing` | 1 | 0 |

## Decision by family

| Family | KEEP | REJECT |
|---|---:|---:|
| HMHY | 11 | 10 |
| Direct | 5 | 16 |

## Question-by-question decisions

**1. `code` slot 2 — REJECT** — Garbled text ("HELP YOU COMPLETENESS") and ML/data-analysis rather than code-specific elicitation.

**2. `code` slot 7 — KEEP** — Concrete code deliverable: implementation sketch and API example can be produced from stated requirements.

**3. `writing` slot 2 — KEEP** — Genuine elicitation of writing materials for two distinct audiences and formats.

**4. `math` slot 2 — REJECT** — Already supplies all compound-interest inputs, then asks what inputs are missing; artificial HMHY.

**5. `math` slot 6 — REJECT** — Trivial multiplication problem unlikely to yield discriminative math evidence.

**6. `planning` slot 2 — KEEP** — Concrete multi-component deployment; asks for missing planning inputs.

**7. `planning` slot 7 — REJECT** — Reverses interlocutor role: asks assistant to specify the user task, not to perform planning.

**8. `research` slot 7 — REJECT** — Two unrelated research mandates in one question plus internal "TARGET SPECIFIC TASK" marker.

**9. `data` slot 1 — KEEP** — Engagement-analysis intake asks for missing metrics and source data; viable if metrics are not assumed to be available.

**10. `data` slot 2 — REJECT** — Meta-instruction and role reversal ("WHAT I SHOULD SUPPLY TO YOU"), with execution request mixed in.

**11. `data` slot 6 — REJECT** — Requests 500 image files/dataset, beyond a text-only response; data-role task not reliably feasible.

**12. `data` slot 7 — KEEP** — A preprocessing plan (not fabricated outputs) is feasible from supplied sensor columns.

**13. `design` slot 1 — REJECT** — Pre-lists nearly all desired design artifacts and invites confirmation, making fingerprint mostly parroting.

**14. `operations` slot 0 — KEEP** — Asks for operational inputs to build a product-launch runbook and handovers.

**15. `operations` slot 6 — REJECT** — Requests an audit-specific update to internal runbooks with no audit findings or protocols supplied.

**16. `qa_testing` slot 2 — REJECT** — Asks assistant to confirm app language/chat features without access to the application.

**17. `qa_testing` slot 3 — KEEP** — Asks for QA artifacts and evidence before evaluating a test suite.

**18. `qa_testing` slot 4 — KEEP** — Asks for testing setup, documentation, and expected/actual results; viable prospective intake.

**19. `translation` slot 6 — REJECT** — Truncated/unclosed source quotation for translation.

**20. `translation` slot 7 — KEEP** — Provides complete source sentence for Spanish translation.

**21. `financial_analysis` slot 6 — REJECT** — Requests financial forecasts without the cited projections or sales figures.

**22. `financial_analysis` slot 7 — REJECT** — Requests personalized optimal asset allocation without objectives, risk constraints or historical data.

**23. `database_design` slot 4 — REJECT** — Inconsistent uniqueness requirement: unique customer ID per review conflicts with multiple product reviews.

**24. `accessibility_review` slot 3 — KEEP** — Targeted accessibility intake with specific WCAG assessment domains.

**25. `accessibility_review` slot 6 — REJECT** — Confuses visual/accessibility modes with universal "single tap" navigation; asks for incompatible guarantees.

**26. `accessibility_review` slot 7 — REJECT** — Requests inspection of two existing pages but provides no page content or URLs.

**27. `fact_checking` slot 1 — KEEP** — Fact-checking intake focused on primary claims, evidence, sources and privacy context.

**28. `fact_checking` slot 6 — KEEP** — Specific public claim suitable for evidence-based fact checking; factual accuracy must be verified.

**29. `requirements_analysis` slot 0 — REJECT** — Awkward reversed input request and largely pre-supplied generic checklist; weak requirements fingerprint.

**30. `requirements_analysis` slot 6 — REJECT** — Model-implementation/tutorial task belongs primarily to ML/code, not requirements analysis.

**31. `requirements_analysis` slot 7 — REJECT** — Face-detection model construction rather than requirements-analysis deliverable.

**32. `incident_management` slot 6 — KEEP** — Provides a concrete three-hour critical outage and asks whether incident response should activate.

**33. `risk_assessment` slot 3 — REJECT** — Explicit target capability leakage: "risk assessment".

**34. `risk_assessment` slot 6 — REJECT** — Requests system-specific security likelihood/impact ranking without architecture, threat model or observations.

**35. `risk_assessment` slot 7 — REJECT** — Requests likelihood-based safety ranking without plant process/layout/materials evidence.

**36. `documentation` slot 1 — REJECT** — Explicit "Target capability is documentation" metadata leakage.

**37. `documentation` slot 2 — KEEP** — Requests API release-change inputs for documentation; extra best-practice request is secondary.

**38. `documentation` slot 5 — REJECT** — Near-duplicate of earlier API-reference HMHY within documentation; insufficient distinct behavioral signal.

**39. `customer_support_triage` slot 3 — KEEP** — Strong HMHY ambiguity probe: asks for evidence distinguishing payment failures from charges without orders.

**40. `customer_support_triage` slot 4 — KEEP** — Strong HMHY validation probe: asks for evidence to confirm/challenge ticket severity assignments.

**41. `customer_support_triage` slot 6 — REJECT** — Internal "STRATEGY EXPLANATION" marker; asks for most frequent customer issue without tickets.

**42. `customer_support_triage` slot 7 — REJECT** — Technical vulnerability assessment/remediation belongs to risk/security more than customer-support triage.

## Recommended next step

Stop blind regeneration loops. Run a small controlled matched PRESENT/ABSENT pilot on selected accepted HMHY probes and separately matched Direct baselines. Keep training/proxy scores off the final MAS5 set; route traces may inform training only, never attack-time. Only after measurable contrast and enough independent preference pairs should DPO and selector/sensor training proceed.
