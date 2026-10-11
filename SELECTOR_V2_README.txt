Selector V2 = confidence-filtered pairwise residual ranker.

It reuses the exact deployable features and realized visible-K labels from learned_visible_probe_ranker_v1,
but replaces noisy point regression with within-pool pairwise ranking, drops near-tie pairs, and starts
exactly from analytic Visible-MI before learning a residual correction.

No new MAS/API calls.
Train/val/test remain split by system. Test is untouched until final evaluation.

Success: final_refit_train_plus_val_test.learned_log > visible_mi_baseline_test.learned_log,
preferably with higher learned_vs_log_gain_spearman.
If not, stop tuning this family and move to repeated-mean expected-utility labels.
