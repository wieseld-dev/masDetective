#!/usr/bin/env python3

import hashlib
import inspect
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch.utils.data import Dataset

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


ROOT = Path("artifacts/v137_presence_v1")
TRAIN_PATH = ROOT / "dataset_v1/train_v137.jsonl"
DEV_PATH = ROOT / "dataset_v1/dev_v137.jsonl"
OUT = ROOT / "model_v1"


def readj(path):
    with Path(path).open() as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def sha(path):
    return hashlib.sha256(
        Path(path).read_bytes()
    ).hexdigest()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


def binary_metrics(y, p, threshold=0.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)

    out = {
        "balanced_accuracy":
            float(balanced_accuracy_score(y, pred)),
        "precision":
            float(
                precision_score(
                    y, pred, zero_division=0
                )
            ),
        "recall":
            float(
                recall_score(
                    y, pred, zero_division=0
                )
            ),
        "f1":
            float(
                f1_score(
                    y, pred, zero_division=0
                )
            ),
    }

    if len(np.unique(y)) == 2:
        out["roc_auc"] = float(
            roc_auc_score(y, p)
        )
        out["pr_auc"] = float(
            average_precision_score(y, p)
        )

    return out


def poisson_binomial(probs):
    dist = np.array([1.0], dtype=np.float64)

    for p in probs:
        nxt = np.zeros(
            len(dist) + 1,
            dtype=np.float64,
        )

        nxt[:-1] += dist * (1.0 - p)
        nxt[1:] += dist * p

        dist = nxt

    return dist


def resolve_base_model():
    hits = []

    for p in Path("artifacts").rglob(
        "adapter_config.json"
    ):
        try:
            obj = json.loads(p.read_text())
        except Exception:
            continue

        base = str(
            obj.get(
                "base_model_name_or_path",
                "",
            )
        )

        task = str(
            obj.get(
                "task_type",
                "",
            )
        ).upper()

        if (
            task == "SEQ_CLS"
            and "Qwen2.5-3B" in base
        ):
            hits.append(
                (str(p), base)
            )

    # Prefer previous V13.6b lineage.
    preferred = [
        base
        for path, base in hits
        if (
            "v13_6b" in path.lower()
            or "v136b" in path.lower()
            or "v13.6b" in path.lower()
        )
    ]

    preferred = sorted(set(preferred))

    if len(preferred) == 1:
        return preferred[0], "previous V13.6b config"

    unique = sorted({
        base
        for _, base in hits
    })

    if len(unique) == 1:
        return unique[0], "existing SEQ_CLS config"

    old = Path(
        "analysis/train_v13_6b_activation_judge.py"
    )

    if old.exists():
        found = re.findall(
            r"""["']([^"']*Qwen2\.5-3B[^"']*)["']""",
            old.read_text(errors="ignore"),
        )

        found = sorted(set(
            x
            for x in found
            if "/" in x
            and len(x) < 180
        ))

        if len(found) == 1:
            return found[0], "V13.6b source"

    return "Qwen/Qwen2.5-3B", "fallback"


class PresenceDataset(Dataset):
    def __init__(
        self,
        rows,
        tokenizer,
        max_length,
    ):
        self.rows = rows

        enc = tokenizer(
            [
                x["model_input"]
                for x in rows
            ],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        self.ids = enc["input_ids"]
        self.mask = enc["attention_mask"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.rows[i]

        return {
            "input_ids": self.ids[i],
            "attention_mask": self.mask[i],
            "labels": float(x["label"]),
            "sample_weight":
                float(
                    x.get(
                        "sample_weight",
                        1.0,
                    )
                ),
        }


class PresenceCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        base = [
            {
                "input_ids":
                    x["input_ids"],
                "attention_mask":
                    x["attention_mask"],
            }
            for x in features
        ]

        batch = self.tokenizer.pad(
            base,
            padding=True,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(
            [
                x["labels"]
                for x in features
            ],
            dtype=torch.float32,
        )

        batch["sample_weight"] = torch.tensor(
            [
                x["sample_weight"]
                for x in features
            ],
            dtype=torch.float32,
        )

        return batch


class WeightedBCETrainer(Trainer):
    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        weights = inputs.pop(
            "sample_weight"
        ).float()

        labels = inputs.pop(
            "labels"
        ).float()

        outputs = model(**inputs)

        logits = (
            outputs.logits
            .reshape(-1)
            .float()
        )

        labels = labels.reshape(-1)
        weights = weights.reshape(-1)

        loss_each = (
            F.binary_cross_entropy_with_logits(
                logits,
                labels,
                reduction="none",
            )
        )

        loss = (
            (loss_each * weights).sum()
            / weights.sum().clamp_min(1e-8)
        )

        if return_outputs:
            return loss, outputs

        return loss


def main():
    seed = 1377

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU REQUIRED"
        )

    torch.cuda.manual_seed_all(seed)

    train_rows = readj(TRAIN_PATH)
    dev_rows = readj(DEV_PATH)

    if len(train_rows) != 910:
        raise RuntimeError(
            f"Expected 910 TRAIN rows, "
            f"got {len(train_rows)}"
        )

    if len(dev_rows) != 760:
        raise RuntimeError(
            f"Expected 760 DEV rows, "
            f"got {len(dev_rows)}"
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_name, model_source = (
        resolve_base_model()
    )

    gpu = torch.cuda.get_device_properties(0)

    print("=" * 100)
    print("V13.7 TARGETED INSTALLED-PRESENCE SENSOR")
    print("=" * 100)

    print("base model  :", model_name)
    print("model source:", model_source)
    print("GPU         :", gpu.name)
    print(
        "GPU VRAM    :",
        round(
            gpu.total_memory / 1024**3,
            2,
        ),
        "GiB",
    )

    print("TRAIN rows  :", len(train_rows))
    print("DEV rows    :", len(dev_rows))
    print("TRAIN SHA   :", sha(TRAIN_PATH))
    print("DEV SHA     :", sha(DEV_PATH))

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    # ----------------------------------------------------------
    # Pick sequence length from the actual V13.7 dataset.
    # ----------------------------------------------------------

    lengths = []

    for row in train_rows + dev_rows:
        ids = tokenizer(
            row["model_input"],
            truncation=False,
            add_special_tokens=True,
        )["input_ids"]

        lengths.append(len(ids))

    lengths = np.asarray(lengths)

    p95 = int(
        np.percentile(lengths, 95)
    )
    p99 = int(
        np.percentile(lengths, 99)
    )
    mx = int(lengths.max())

    max_length = min(
        2048,
        max(
            768,
            int(
                math.ceil(p99 / 128)
                * 128
            ),
        ),
    )

    truncated = int(
        (lengths > max_length).sum()
    )

    truncation_rate = (
        truncated / len(lengths)
    )

    print()
    print("TOKEN p95    :", p95)
    print("TOKEN p99    :", p99)
    print("TOKEN max    :", mx)
    print("MAX LENGTH   :", max_length)
    print(
        "TRUNCATED    :",
        truncated,
        f"({truncation_rate:.2%})",
    )

    if truncation_rate > 0.02:
        raise RuntimeError(
            "More than 2% of examples would "
            "be truncated."
        )

    train_ds = PresenceDataset(
        train_rows,
        tokenizer,
        max_length,
    )

    dev_ds = PresenceDataset(
        dev_rows,
        tokenizer,
        max_length,
    )

    use_bf16 = (
        torch.cuda.is_bf16_supported()
    )

    dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            model_name,
            num_labels=1,
            quantization_config=quant_config,
            device_map={"": 0},
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    )

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        modules_to_save=[
            "score",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "TOTAL PARAMETERS    :",
        f"{total_params:,}",
    )
    print(
        "TRAINABLE PARAMETERS:",
        f"{trainable_params:,}",
    )
    print(
        "TRAINABLE %         :",
        f"{100 * trainable_params / total_params:.4f}%",
    )

    def compute_metrics(eval_pred):
        raw = eval_pred.predictions
        y = eval_pred.label_ids

        if isinstance(raw, tuple):
            raw = raw[0]

        logits = (
            np.asarray(raw)
            .reshape(-1)
        )

        y = (
            np.asarray(y)
            .reshape(-1)
            .astype(int)
        )

        probs = sigmoid(logits)

        return binary_metrics(
            y,
            probs,
            threshold=0.5,
        )

    training_kwargs = dict(
        output_dir=str(
            OUT / "trainer"
        ),
        num_train_epochs=5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=29,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        greater_is_better=True,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=seed,
        data_seed=seed,
    )

    params = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if "eval_strategy" in params:
        training_kwargs[
            "eval_strategy"
        ] = "epoch"
    else:
        training_kwargs[
            "evaluation_strategy"
        ] = "epoch"

    training_args = TrainingArguments(
        **training_kwargs
    )

    trainer = WeightedBCETrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=PresenceCollator(
            tokenizer
        ),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=2
            )
        ],
    )

    print()
    print("===== TRAIN START =====")

    trainer.train()

    print()
    print("===== DEV PREDICTION =====")

    prediction = trainer.predict(
        dev_ds
    )

    raw = prediction.predictions

    if isinstance(raw, tuple):
        raw = raw[0]

    logits = (
        np.asarray(raw)
        .reshape(-1)
    )

    probs = sigmoid(logits)

    labels = np.asarray(
        [
            int(x["label"])
            for x in dev_rows
        ],
        dtype=int,
    )

    overall = binary_metrics(
        labels,
        probs,
        threshold=0.5,
    )

    # ----------------------------------------------------------
    # Per-role diagnostics.
    # ----------------------------------------------------------

    per_role = {}

    all_roles = sorted({
        x["target_role"]
        for x in dev_rows
    })

    for role in all_roles:
        idx = [
            i
            for i, x in enumerate(dev_rows)
            if x["target_role"] == role
        ]

        yy = labels[idx]
        pp = probs[idx]

        m = binary_metrics(
            yy,
            pp,
            threshold=0.5,
        )

        m["n"] = len(idx)
        m["positive_n"] = int(
            yy.sum()
        )

        m["mean_p_positive"] = float(
            pp[yy == 1].mean()
        )

        m["mean_p_negative"] = float(
            pp[yy == 0].mean()
        )

        per_role[role] = m

    # ----------------------------------------------------------
    # End-to-end raw K decoding on DEV.
    #
    # Important:
    # These are weighted-BCE sigmoid probabilities, not yet
    # calibrated. Exact-K here is therefore DIAGNOSTIC.
    # ----------------------------------------------------------

    by_system = defaultdict(list)

    for i, row in enumerate(dev_rows):
        by_system[
            row["system_id"]
        ].append(
            (
                row,
                float(probs[i]),
            )
        )

    system_rows = []

    correct_all = 0
    correct_k3 = 0
    correct_k4 = 0

    n_k3 = 0
    n_k4 = 0

    k_abs_errors = []

    exact_role_set_topk = 0

    for sid, items in sorted(
        by_system.items()
    ):
        if len(items) != 20:
            raise RuntimeError(
                f"{sid}: expected 20 probes, "
                f"got {len(items)}"
            )

        target_roles = {
            row["target_role"]
            for row, _ in items
        }

        if len(target_roles) != 20:
            raise RuntimeError(
                f"{sid}: target-role coverage != 20"
            )

        ks = {
            int(row["K_eval_only"])
            for row, _ in items
        }

        if len(ks) != 1:
            raise RuntimeError(
                f"{sid}: inconsistent K"
            )

        true_k = next(iter(ks))

        role_probs = {
            row["target_role"]: p
            for row, p in items
        }

        true_roles = {
            row["target_role"]
            for row, _ in items
            if int(row["label"]) == 1
        }

        ordered = sorted(role_probs)

        posterior = poisson_binomial(
            [
                role_probs[r]
                for r in ordered
            ]
        )

        # Experimental universe is K=3..8.
        pred_k = max(
            range(3, 9),
            key=lambda k:
                posterior[k],
        )

        correct = int(
            pred_k == true_k
        )

        correct_all += correct
        k_abs_errors.append(
            abs(pred_k - true_k)
        )

        if true_k == 3:
            n_k3 += 1
            correct_k3 += correct

        elif true_k == 4:
            n_k4 += 1
            correct_k4 += correct

        ranked = sorted(
            ordered,
            key=lambda r:
                role_probs[r],
            reverse=True,
        )

        pred_roles = set(
            ranked[:pred_k]
        )

        exact_role_set_topk += int(
            pred_roles == true_roles
        )

        system_rows.append({
            "system_id": sid,
            "true_K": true_k,
            "pred_K_raw": pred_k,
            "posterior_K_raw":
                [
                    float(z)
                    for z in posterior
                ],
            "role_probabilities_raw":
                role_probs,
            "true_installed_roles":
                sorted(true_roles),
            "pred_roles_top_raw_K":
                sorted(pred_roles),
        })

    n_systems = len(system_rows)

    k_diag = {
        "note":
            "RAW diagnostic only; probabilities "
            "not calibrated yet.",
        "systems": n_systems,
        "exact_K_raw":
            correct_all / n_systems,
        "K3_n": n_k3,
        "K3_exact_raw":
            correct_k3 / n_k3,
        "K4_n": n_k4,
        "K4_exact_raw":
            correct_k4 / n_k4,
        "K_MAE_raw":
            float(
                np.mean(k_abs_errors)
            ),
        "exact_role_set_top_raw_K":
            exact_role_set_topk
            / n_systems,
    }

    # ----------------------------------------------------------
    # Save best adapter.
    # ----------------------------------------------------------

    adapter_dir = (
        OUT / "best_adapter"
    )

    trainer.save_model(
        str(adapter_dir)
    )

    tokenizer.save_pretrained(
        str(adapter_dir)
    )

    # Verify that the SEQ_CLS score head was saved.
    from safetensors import safe_open

    score_keys = []

    for sf in adapter_dir.glob(
        "*.safetensors"
    ):
        with safe_open(
            str(sf),
            framework="pt",
            device="cpu",
        ) as f:
            score_keys.extend(
                [
                    k
                    for k in f.keys()
                    if "score" in k
                ]
            )

    if not score_keys:
        raise RuntimeError(
            "CLASSIFICATION SCORE HEAD "
            "NOT FOUND IN SAVED ADAPTER"
        )

    # ----------------------------------------------------------
    # Save row predictions.
    # ----------------------------------------------------------

    with (
        OUT / "dev_predictions.jsonl"
    ).open("w") as f:
        for row, prob, logit in zip(
            dev_rows,
            probs,
            logits,
        ):
            f.write(
                json.dumps({
                    "example_id":
                        row["example_id"],
                    "system_id":
                        row["system_id"],
                    "target_role":
                        row["target_role"],
                    "true_label":
                        int(row["label"]),
                    "K_eval_only":
                        int(
                            row["K_eval_only"]
                        ),
                    "prob_installed_raw":
                        float(prob),
                    "logit":
                        float(logit),
                })
                + "\n"
            )

    with (
        OUT / "dev_system_K_raw.jsonl"
    ).open("w") as f:
        for row in system_rows:
            f.write(
                json.dumps(row)
                + "\n"
            )

    results = {
        "version":
            "v137_presence_v1",
        "base_model":
            model_name,
        "base_model_source":
            model_source,
        "gpu":
            gpu.name,
        "gpu_vram_gib":
            float(
                gpu.total_memory
                / 1024**3
            ),
        "parameters_total":
            int(total_params),
        "parameters_trainable":
            int(trainable_params),
        "trainable_pct":
            float(
                100
                * trainable_params
                / total_params
            ),
        "token_length_p95":
            p95,
        "token_length_p99":
            p99,
        "token_length_max":
            mx,
        "max_length":
            max_length,
        "truncated_examples":
            truncated,
        "overall_presence":
            overall,
        "per_role":
            per_role,
        "raw_K_diagnostic":
            k_diag,
        "saved_score_head_keys":
            score_keys,
        "internal_test_used":
            False,
    }

    (
        OUT / "DEV_METRICS.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("=" * 100)
    print("V13.7 DEV RESULTS")
    print("=" * 100)

    print(
        "ROC-AUC :",
        round(
            overall["roc_auc"],
            4,
        ),
    )

    print(
        "PR-AUC  :",
        round(
            overall["pr_auc"],
            4,
        ),
    )

    print(
        "BalAcc@0.5:",
        round(
            overall[
                "balanced_accuracy"
            ],
            4,
        ),
    )

    print()
    print("RAW K DIAGNOSTIC")

    print(
        "Exact K:",
        f"{k_diag['exact_K_raw']:.2%}",
    )

    print(
        "K=3 exact:",
        f"{k_diag['K3_exact_raw']:.2%}",
        f"(n={n_k3})",
    )

    print(
        "K=4 exact:",
        f"{k_diag['K4_exact_raw']:.2%}",
        f"(n={n_k4})",
    )

    print(
        "K MAE:",
        round(
            k_diag["K_MAE_raw"],
            4,
        ),
    )

    print(
        "Exact role set:",
        f"{k_diag['exact_role_set_top_raw_K']:.2%}",
    )

    print()
    print(
        "SCORE HEAD SAVE CHECK: PASS"
    )
    print(
        "INTERNAL TEST USED: NO"
    )
    print(
        "TRAINING + DEV EVALUATION: PASS"
    )


if __name__ == "__main__":
    main()
