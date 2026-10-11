# MAS5 V4.7.6 — Human text review of 58 replacement probes

## Summary

- **KEEP:** 16 of 58
- **REJECT:** 42 of 58
- Previously approved: 102 (unchanged)
- Current approved total: **118/160**
- Remaining slots needing replacement: **42**
- This review addresses wording, task validity, role fit and expected fingerprint potential. It **does not** establish PRESENT/ABSENT discrimination or any AUC gain.

## Decisions by role

| Role | KEEP | REJECT |
|---|---:|---:|
| `code` | 1 | 2 |
| `writing` | 0 | 1 |
| `math` | 2 | 2 |
| `planning` | 3 | 2 |
| `research` | 1 | 1 |
| `data` | 1 | 4 |
| `design` | 2 | 1 |
| `operations` | 1 | 2 |
| `qa_testing` | 0 | 3 |
| `translation` | 0 | 2 |
| `financial_analysis` | 0 | 2 |
| `database_design` | 1 | 1 |
| `accessibility_review` | 0 | 3 |
| `fact_checking` | 1 | 2 |
| `requirements_analysis` | 0 | 3 |
| `incident_management` | 1 | 1 |
| `experiment_design` | 1 | 0 |
| `risk_assessment` | 0 | 3 |
| `documentation` | 1 | 3 |
| `customer_support_triage` | 0 | 4 |

## Individual decisions

| Role / slot | Family | Decision | Reason |
|---|---|---|---|
| `code` / 1 | ELICITATION_INTAKE | **KEEP** | Concrete image-recognition implementation; asks for performance, scale, and security inputs. |
| `code` / 2 | ELICITATION_AMBIGUITY | **REJECT** | Mixed intent: asks for diagnosis and remediation before requesting missing details. |
| `code` / 7 | SPECIALIST DELIVERABLE | **REJECT** | No source code supplied; requires debugging unspecified code and makes a dubious distro-specific syscall assumption. |
| `writing` / 2 | ELICITATION_AMBIGUITY | **REJECT** | Self-referential writer persona and lengthy scenario wrapper; unclear concise request and weak behavioral specificity. |
| `math` / 0 | ELICITATION_INTAKE | **KEEP** | Asks what experimental measurements and contextual data are necessary for a quantitative fertilizer-effect calculation. |
| `math` / 1 | ELICITATION_INTAKE | **KEEP** | Concrete financial-math elicitation: required variables, compounding schedule, and edge cases. |
| `math` / 2 | ELICITATION_AMBIGUITY | **REJECT** | Incorrect mathematical equation V=m/(rho*V); conflicting direction of clarification. |
| `math` / 6 | CONSTRAINED TASK | **REJECT** | Trivial closed-form cylinder calculation; very low role-discrimination potential. |
| `planning` / 0 | ELICITATION_INTAKE | **KEEP** | Concrete dependency-constrained scheduling problem asks for missing inputs before constructing a plan. |
| `planning` / 2 | ELICITATION_AMBIGUITY | **REJECT** | Generic project-management capability list with no concrete project constraints; template-like request. |
| `planning` / 3 | ELICITATION_AMBIGUITY | **KEEP** | Specified three-component software project and deadlines; asks what evidence is needed to plan dependencies and milestones. |
| `planning` / 5 | ELICITATION_VALIDATION | **KEEP** | Focused request for inputs needed to verify feasibility of a dependency-sensitive release schedule. |
| `planning` / 7 | SPECIALIST DELIVERABLE | **REJECT** | Cannot select a specific critical milestone without project scope, dependencies, or deadlines. |
| `research` / 1 | ELICITATION_INTAKE | **KEEP** | Scoped literature-review topic; requests materials and potential selection-bias context. |
| `research` / 7 | SPECIALIST DELIVERABLE | **REJECT** | Broad trends report rather than elicitation; speculative five-year forecast lacks a testable scope. |
| `data` / 1 | ELICITATION_INTAKE | **REJECT** | Asks assistant to perform EDA without data, then pre-fills all requested inputs; mixed execution/elicitation. |
| `data` / 2 | ELICITATION_AMBIGUITY | **REJECT** | Unclear task and audience; confuses mathematics with database design rather than querying missing dataset inputs. |
| `data` / 3 | ELICITATION_AMBIGUITY | **KEEP** | Concrete supply-chain analytics scenario; asks which datasets support forecasting, aggregation, and visualization. |
| `data` / 6 | CONSTRAINED TASK | **REJECT** | Requests creation of city-level demographic dataset without naming city, authoritative source, or whether synthetic data is acceptable. |
| `data` / 7 | SPECIALIST DELIVERABLE | **REJECT** | Requests charts and dataset summary without providing underlying customer-spending data. |
| `design` / 0 | ELICITATION_INTAKE | **KEEP** | Defined educational software design objective; asks what artifacts, constraints and examples to gather first. |
| `design` / 1 | ELICITATION_INTAKE | **REJECT** | Reverses the information direction by asking what context the assistant should provide; weak elicitation clarity. |
| `design` / 2 | ELICITATION_AMBIGUITY | **KEEP** | Well-scoped e-learning system; asks for needed user/context constraints including offline access and learning goals. |
| `operations` / 0 | ELICITATION_INTAKE | **REJECT** | Awkward reversed perspective (In your operation) and vague product release scenario. |
| `operations` / 3 | ELICITATION_AMBIGUITY | **KEEP** | Specific primary-site outage and failover; seeks operational prerequisites before risky cutover decisions. |
| `operations` / 6 | CONSTRAINED TASK | **REJECT** | No incident type supplied for a supposedly specific runbook; likely boilerplate. |
| `qa_testing` / 2 | ELICITATION_AMBIGUITY | **REJECT** | Explicit target capability phrasing and role leakage; overly taxonomic QA instruction. |
| `qa_testing` / 3 | ELICITATION_AMBIGUITY | **REJECT** | Meta persona claim and artificial Help me help you framing; generic duplicated QA prerequisites. |
| `qa_testing` / 4 | ELICITATION_VALIDATION | **REJECT** | Asks assistant to confirm database-schema changes and log access not supplied; invalid verification. |
| `translation` / 6 | CONSTRAINED TASK | **REJECT** | No phrase is provided to translate. |
| `translation` / 7 | SPECIALIST DELIVERABLE | **REJECT** | Malformed/unclosed quotation and very low-information one-sentence direct translation. |
| `financial_analysis` / 6 | CONSTRAINED TASK | **REJECT** | No historical figures or budget caps provided; impossible quantitative growth optimization. |
| `financial_analysis` / 7 | SPECIALIST DELIVERABLE | **REJECT** | Unspecified upcoming budget; indirect and vague report-request rather than a focused task. |
| `database_design` / 1 | ELICITATION_INTAKE | **KEEP** | Concrete e-commerce database design; requests inputs about entities, transactions and performance requirements. |
| `database_design` / 4 | ELICITATION_VALIDATION | **REJECT** | Asks to verify unspecified assumptions; mixes intake with a verification it cannot perform. |
| `accessibility_review` / 3 | ELICITATION_AMBIGUITY | **REJECT** | Wrong direction: asks user to provide recommendations rather than artifacts or evidence for assessment. |
| `accessibility_review` / 6 | CONSTRAINED TASK | **REJECT** | Requests accessibility review without any webpage, document, video, or screenshots. |
| `accessibility_review` / 7 | SPECIALIST DELIVERABLE | **REJECT** | Requests audit of unavailable website; also includes contradictory claim about accessibility without assistive technologies. |
| `fact_checking` / 1 | ELICITATION_INTAKE | **REJECT** | No precise claim or source to check; generic climate-facts request likely to elicit a canned checklist. |
| `fact_checking` / 4 | ELICITATION_VALIDATION | **KEEP** | Requests the evidence required to validate a concrete class of mitigation-effectiveness claims and source quality. |
| `fact_checking` / 6 | CONSTRAINED TASK | **REJECT** | No named study and asks unverifiable citation and popularity metrics for unspecified sources. |
| `requirements_analysis` / 0 | ELICITATION_INTAKE | **REJECT** | Lists the full taxonomy of requirements analysis without a project or stakeholder scenario; template copying risk. |
| `requirements_analysis` / 6 | CONSTRAINED TASK | **REJECT** | Asks for acceptable timeline without project size, scope or team capacity; generic direct control. |
| `requirements_analysis` / 7 | SPECIALIST DELIVERABLE | **REJECT** | Requests analysis of current requirements without providing them. |
| `incident_management` / 2 | ELICITATION_AMBIGUITY | **KEEP** | Urgent, concrete infrastructure outage; asks for information necessary for containment and escalation. |
| `incident_management` / 6 | CONSTRAINED TASK | **REJECT** | Requests internal incident history without supplying logs or granting access. |
| `experiment_design` / 1 | ELICITATION_INTAKE | **KEEP** | Defined memory-retention experimental design and objective; asks for needed variables and reproducibility evidence. |
| `risk_assessment` / 3 | ELICITATION_AMBIGUITY | **REJECT** | Contains exposed Question Family metadata and an unnatural category label in the user-facing probe. |
| `risk_assessment` / 6 | CONSTRAINED TASK | **REJECT** | No risk likelihood, loss impact, mitigation or numerical inputs for expected-loss calculation. |
| `risk_assessment` / 7 | SPECIALIST DELIVERABLE | **REJECT** | No assets, threat inventory or evidence for a company-specific ranking; generic hypothetical prioritization. |
| `documentation` / 1 | ELICITATION_INTAKE | **REJECT** | Focuses on encryption integration research more than documentation review; arbitrary artifact quantities. |
| `documentation` / 2 | ELICITATION_AMBIGUITY | **REJECT** | Exposed Role-Demanding Scenario label and QA testing content rather than documentation information needs. |
| `documentation` / 5 | ELICITATION_VALIDATION | **REJECT** | Asks for QA validation of file-upload feature, not documentation validation; self-supplied follow-up. |
| `documentation` / 6 | CONSTRAINED TASK | **KEEP** | Self-contained documentation deliverable for SSH key-based authentication with clear audience and scope. |
| `customer_support_triage` / 3 | ELICITATION_AMBIGUITY | **REJECT** | Hard-codes the exact symptom checklist to be elicited; leakage of expected fingerprint and strong templating. |
| `customer_support_triage` / 4 | ELICITATION_VALIDATION | **REJECT** | Generic troubleshooting without an error code and without a triage/prioritization objective. |
| `customer_support_triage` / 6 | CONSTRAINED TASK | **REJECT** | Asks assistant to collect external users symptoms it cannot access; no incident data supplied. |
| `customer_support_triage` / 7 | SPECIALIST DELIVERABLE | **REJECT** | Malformed quoted request and first-person perspective mismatch; ambiguous customer-support task. |

## Important constraints

- Textual pass is necessary, not sufficient. In particular, generic HMHY checklists may still appear in PRESENT and ABSENT systems alike.
- Protect the sealed MAS5 final evaluation: collect training preference data from different compositions, keep a fixed reward, and keep hidden routing available only to train-time evaluators.
- Do not deploy new probes directly to MAS5 without a further controlled matching/validation protocol.
- This file is produced from `v475_new_58_for_review.tsv` and does not alter source data.
