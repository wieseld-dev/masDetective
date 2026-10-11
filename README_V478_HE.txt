MAS5 V4.7.8 - fix for existing V4.7.7 resume conflict.
118 manually approved questions are immutable. Generated questions conflicting with
same-role accepted entries are quarantined for regeneration. All existing checkpoint
files are retained and a reconciliation JSON audit is produced.
No MAS/API calls. No model retraining.

cd ~/mas_rl_detective
unzip -o mas5_v478_resume_conflict_fix.zip
export V46_ROOT="$PWD/artifacts/role_direct_within_v4/overnight_v46/mas5_night_20261009_224735_1879051"
bash cluster/role_direct_within_v4/v478_submit.sh

Watch:
squeue --me
ls -t logs/v478-mas5-v478-regen-*.out | head -1 | xargs -r tail -n 40
cat "$V46_ROOT/train/v47_quality/v472_reviewed_second_pass/v476_manual_review/v477_regeneration/v478_similarity_reconciliation.json"
