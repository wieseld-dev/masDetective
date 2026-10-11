# V11.2 strong evaluation v1

This package does not modify the running V11.2 training pipeline.

Evaluation layers:
1. **Matched generator comparison**: frozen original V11 vs final V11.2, 3 new probes per role, same seed schedule.
2. **Locked known-role hard-confuser benchmark**: 24 fresh synthetic base systems (6 per topology mode), composition-disjoint from V11 training, 108 target incidences, hard-confuser MINUS replacements.
3. **Unseen-role hard regression**: the six V11 unseen role types, using hard-confuser replacements. This is development evidence, not a pristine final paper test.
4. **V13.5 downstream test**: on the locked 20-role benchmark only, compare whether V11.2 probes create larger target evidence margins, lower absent-target false positives/top-3 errors, and better target ranks.
5. **Paired role bootstrap**: 95% CIs over role-level deltas.

Primary generator diagnostics include strict-specific rate, robust specificity, Q25, activation, positive pair rate, best-of-3, and worst-confuser specificity. The benchmark manifest is written and SHA256-locked before outputs are evaluated.
