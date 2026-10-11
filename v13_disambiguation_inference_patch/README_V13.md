# V13 — Hard-Negative Behavioral Disambiguation + Sequential Inference

V13 does **not** throw away V11. It continues the trained V11 LoRA and changes the curriculum from "probe the topic" to "separate the target capability from semantically confusable alternatives".

## What changes

1. Hard neighbors are computed automatically from the **public role descriptions**.
2. Training counterfactuals replace the target with a nearest semantic neighbor while K and FANOUT mode stay fixed.
3. The V13 prompt asks for a behavioral fingerprint: method, artifact, decision criterion, validation action, etc., rather than topic words.
4. A learned inference head is fit from TRAIN-only counterfactual outputs. It models the **output-minus-query behavioral residual** and uses per-role cross-fitted calibration.
5. At test time the frozen V13 policy gets only public candidate descriptions, previous public probes/answers and current public candidate list. It never gets true roles, K, routing, traces or counterfactual branch identity.
6. The controller keeps interrogating uncertain candidates and passes the most confusable alternatives to the probe generator.

## Scientific status

This is a **development** continuation and development end-to-end evaluation. V12 has already been inspected, so its compositions are explicitly excluded from V13 training. V13 evaluation uses fresh compositions disjoint from V11, the inspected V12 grid and V13 training. If V13 works, freeze policy + inference head + thresholds + stopping rule, then create a new untouched final set.

## Run

Install:

```bash
cd ~/mas_rl_detective
tar -xf ~/v13_disambiguation_inference_patch.tar
bash v13_disambiguation_inference_patch/apply_v13.sh
```

Plan only (no MAS API calls):

```bash
conda activate masdetective
PYTHONPATH="$PWD" python training/train_disambiguation_v13.py
```

Train:

```bash
sbatch cluster/v13_disambiguation_train_gpu.sbatch
```

After training finishes, plan evaluation:

```bash
PYTHONPATH="$PWD" python analysis/adaptive_fanout_count_v13.py
```

Then evaluate:

```bash
sbatch cluster/v13_fanout_eval_gpu.sbatch
```

Monitor:

```bash
squeue --me
tail -f "$(ls -t logs/v13-disamb-*.out | head -1)"
# later
tail -f "$(ls -t logs/v13-count-*.out | head -1)"
```

Primary outputs:

- `artifacts/v13_policy/`
- `artifacts/v13_summary.json`
- `artifacts/v13_probe_disambiguation.jsonl`
- `artifacts/v13_fanout_count/summary.json`
- `artifacts/v13_fanout_count/detector_oof.json`

The old `artifacts/v11_policy/` is never modified.
