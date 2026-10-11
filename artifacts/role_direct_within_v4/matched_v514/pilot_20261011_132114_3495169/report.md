# V5.14 Matched role discrimination — diagnostic only

Same exact probe for TARGET_PRESENT vs TARGET_ABSENT; one background per role.

| Role | 5 paired probes | Text similarity | Identical answers | Answer-only paired win | Length-only paired win |
|---|---:|---:|---:|---:|---:|
| code | 5 | 0.83701 | 0 | 0.4 | 0.4 |
| planning | 5 | 0.72354 | 0 | 0.8 | 0.4 |
| research | 5 | 0.798 | 0 | 0.8 | 0.8 |

**Interpretation:** Paired win uses a frozen classifier, not a classifier fitted on this pilot. Identical query is a negative control for query-only information, but response length can remain confounded. Five paired samples in one background are *not* five independent MAS configurations.

The ABSENT system has a substitute third worker. A difference may be due to the replacement worker, not only absence of target.

See matched_pairs.tsv for questions and representative answer excerpts.
