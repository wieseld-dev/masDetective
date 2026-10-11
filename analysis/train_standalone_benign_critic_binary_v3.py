from __future__ import annotations

import json
import math
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import random
import shutil
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import (
    train_test_split,
)

from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from scipy.optimize import minimize_scalar


ROOT = Path(
    "artifacts/"
    "standalone_benign_critic_binary_v3"
)

ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

DATA = Path(
    "artifacts/"
    "standalone_benign_teacher_v3/"
    "trainable_clean_nostrict.jsonl"
)

STRICT = Path(
    "artifacts/"
    "standalone_benign_teacher_v1/"
    "strict_gold.jsonl"
)

MODEL_NAME = os.environ.get(
    "BENIGN_BASE_MODEL",
    "Qwen/Qwen2.5-7B-Instruct",
)

LABELS = [
    "NON_BENIGN",
    "BENIGN",
]

L2I = {
    x: i
    for i, x in enumerate(LABELS)
}

I2L = {
    i: x
    for x, i in L2I.items()
}

BENIGN_IDX = L2I["BENIGN"]

SEED = 20260927

MAX_LEN = 320
TRAIN_BATCH = 1
EVAL_BATCH = 4
GRAD_ACCUM = 16

EPOCHS = 6
LR = 1.0e-4
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.04

MAX_GRAD_NORM = 1.0


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path):
    with Path(path).open(
        encoding="utf-8"
    ) as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


class QDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x = self.rows[i]

        return {
            "text":
                "USER QUERY:\n"
                + str(x["query"]).strip(),
            "label":
                int(L2I["BENIGN" if x["label"] == "BENIGN" else "NON_BENIGN"]),
        }


def make_collator(tok):
    def collate(batch):
        enc = tok(
            [x["text"] for x in batch],
            padding=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )

        enc["labels"] = torch.tensor(
            [
                x["label"]
                for x in batch
            ],
            dtype=torch.long,
        )

        return enc

    return collate


@torch.no_grad()
def eval_loader(
    model,
    loader,
):
    model.eval()

    logits_all = []
    y_all = []

    for batch in loader:
        labels = batch.pop(
            "labels"
        ).cuda()

        batch = {
            k: v.cuda()
            for k, v in batch.items()
        }

        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16,
        ):
            out = model(**batch)

        logits_all.append(
            out.logits.float()
            .cpu().numpy()
        )

        y_all.append(
            labels.cpu().numpy()
        )

    logits = np.concatenate(
        logits_all,
        axis=0,
    )

    y = np.concatenate(
        y_all,
        axis=0,
    )

    p = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()

    pred = p.argmax(axis=1)

    metrics = {
        "accuracy":
            float(
                accuracy_score(
                    y,
                    pred,
                )
            ),
        "macro_f1":
            float(
                f1_score(
                    y,
                    pred,
                    average="macro",
                    zero_division=0,
                )
            ),
        "binary_benign_auc":
            float(
                roc_auc_score(
                    (y == BENIGN_IDX)
                    .astype(int),
                    p[:, BENIGN_IDX],
                )
            ),
        "binary_benign_f1_at_05":
            float(
                f1_score(
                    (y == BENIGN_IDX)
                    .astype(int),
                    (
                        p[:, BENIGN_IDX]
                        >= 0.5
                    ).astype(int),
                    zero_division=0,
                )
            ),
        "nonbenign_recall_at_05":
            float(
                np.mean(
                    (
                        p[
                            y != BENIGN_IDX,
                            BENIGN_IDX,
                        ]
                        < 0.5
                    )
                )
            ),
        "confusion_matrix":
            confusion_matrix(
                y,
                pred,
                labels=list(
                    range(
                        len(LABELS)
                    )
                ),
            ).tolist(),
    }

    return metrics, logits, y


@torch.no_grad()
def strict_eval(
    model,
    tok,
    strict_rows,
):
    ds = QDataset(strict_rows)

    dl = DataLoader(
        ds,
        batch_size=EVAL_BATCH,
        shuffle=False,
        collate_fn=
            make_collator(tok),
    )

    metrics, logits, y = eval_loader(
        model,
        dl,
    )

    p = torch.softmax(
        torch.tensor(logits),
        dim=-1,
    ).numpy()

    pred = p.argmax(axis=1)

    binary_truth = (
        y == BENIGN_IDX
    ).astype(int)

    binary_pred = (
        p[:, BENIGN_IDX]
        >= 0.5
    ).astype(int)

    metrics[
        "strict_binary_accuracy"
    ] = float(
        accuracy_score(
            binary_truth,
            binary_pred,
        )
    )

    details = []

    for row, yy, pp, probs in zip(
        strict_rows,
        y,
        pred,
        p,
    ):
        details.append({
            "query": row["query"],
            "expected":
                I2L[int(yy)],
            "predicted":
                I2L[int(pp)],
            "p_benign":
                float(
                    probs[BENIGN_IDX]
                ),
            "binary_correct":
                bool(
                    (yy == BENIGN_IDX)
                    ==
                    (
                        probs[
                            BENIGN_IDX
                        ]
                        >= 0.5
                    )
                ),
        })

    return metrics, details


def fit_temperature(
    logits,
    y,
):
    logits_t = torch.tensor(
        logits,
        dtype=torch.float32,
    )

    y_t = torch.tensor(
        y,
        dtype=torch.long,
    )

    def objective(log_t):
        T = math.exp(
            float(log_t)
        )

        return float(
            F.cross_entropy(
                logits_t / T,
                y_t,
            )
        )

    r = minimize_scalar(
        objective,
        bounds=(-3.0, 3.0),
        method="bounded",
    )

    return float(
        math.exp(r.x)
    )


def best_binary_threshold(
    logits,
    y,
    temperature,
):
    p = torch.softmax(
        torch.tensor(
            logits,
            dtype=torch.float32,
        )
        / temperature,
        dim=-1,
    ).numpy()[:, BENIGN_IDX]

    truth = (
        y == BENIGN_IDX
    ).astype(int)

    best = None

    for t in np.linspace(
        0.05,
        0.95,
        181,
    ):
        pred = (
            p >= t
        ).astype(int)

        f1 = f1_score(
            truth,
            pred,
            zero_division=0,
        )

        acc = accuracy_score(
            truth,
            pred,
        )

        score = (
            0.7 * f1
            + 0.3 * acc
        )

        if (
            best is None
            or score > best[0]
        ):
            best = (
                score,
                float(t),
                float(f1),
                float(acc),
            )

    return {
        "threshold":
            best[1],
        "binary_f1":
            best[2],
        "binary_accuracy":
            best[3],
    }


def main():
    seed_all(SEED)

    if not DATA.exists():
        raise SystemExit(
            f"missing {DATA}"
        )

    rows = read_jsonl(DATA)

    VALID_SOURCE_LABELS = {
        "BENIGN",
        "CAPABILITY",
        "TAXONOMY",
        "INTERNAL_META",
        "NONSTANDALONE",
        "REVERSED_DIRECTION",
    }

    rows = [
        x for x in rows
        if x.get("label")
        in VALID_SOURCE_LABELS
    ]

    # exact dedup
    seen = set()
    clean = []

    for x in rows:
        q = " ".join(
            str(
                x["query"]
            ).split()
        )

        k = q.casefold()

        if k in seen:
            continue

        seen.add(k)

        clean.append({
            **x,
            "query": q,
        })

    rows = clean

    counts = Counter(
        "BENIGN"
        if x["label"] == "BENIGN"
        else "NON_BENIGN"
        for x in rows
    )

    print(
        "===== DATA ====="
    )

    print(
        "rows:",
        len(rows),
    )

    print(
        "labels:",
        dict(counts),
    )

    if len(counts) < 2:
        raise RuntimeError(
            "teacher dataset has "
            "insufficient class diversity"
        )

    labels = [
        L2I["BENIGN" if x["label"] == "BENIGN" else "NON_BENIGN"]
        for x in rows
    ]

    train_rows, temp_rows = \
        train_test_split(
            rows,
            test_size=0.20,
            random_state=SEED,
            stratify=labels,
        )

    temp_labels = [
        L2I["BENIGN" if x["label"] == "BENIGN" else "NON_BENIGN"]
        for x in temp_rows
    ]

    dev_rows, test_rows = \
        train_test_split(
            temp_rows,
            test_size=0.50,
            random_state=SEED + 1,
            stratify=temp_labels,
        )

    strict_rows = read_jsonl(
        STRICT
    )

    print(
        f"train={len(train_rows)} "
        f"dev={len(dev_rows)} "
        f"test={len(test_rows)} "
        f"strict={len(strict_rows)}"
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=
            torch.bfloat16,
    )

    model = \
        AutoModelForSequenceClassification\
        .from_pretrained(
            MODEL_NAME,
            num_labels=len(LABELS),
            quantization_config=quant,
            device_map={"": 0},
            torch_dtype=torch.bfloat16,
        )

    model.config.pad_token_id = \
        tok.pad_token_id

    model.config.use_cache = False

    model = \
        prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )

    lora = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        modules_to_save=[
            "score",
        ],
    )

    model = get_peft_model(
        model,
        lora,
    )

    model.print_trainable_parameters()

    collate = make_collator(tok)

    # Moderate class balancing:
    # inverse sqrt frequency, not full inverse.
    train_counts = Counter(
        L2I["BENIGN" if x["label"] == "BENIGN" else "NON_BENIGN"]
        for x in train_rows
    )

    sample_weights = [
        1.0 / math.sqrt(
            train_counts[
                L2I["BENIGN" if x["label"] == "BENIGN" else "NON_BENIGN"]
            ]
        )
        for x in train_rows
    ]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(train_rows),
        replacement=True,
    )

    train_dl = DataLoader(
        QDataset(train_rows),
        batch_size=TRAIN_BATCH,
        sampler=sampler,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    dev_dl = DataLoader(
        QDataset(dev_rows),
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    test_dl = DataLoader(
        QDataset(test_rows),
        batch_size=EVAL_BATCH,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate,
    )

    params = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        params,
        lr=LR,
        weight_decay=
            WEIGHT_DECAY,
    )

    opt_steps_per_epoch = math.ceil(
        len(train_dl)
        / GRAD_ACCUM
    )

    total_steps = (
        opt_steps_per_epoch
        * EPOCHS
    )

    sched = \
        get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=max(
                1,
                int(
                    total_steps
                    * WARMUP_FRAC
                ),
            ),
            num_training_steps=
                total_steps,
        )

    best_score = -1e9
    best_epoch = -1

    best_dev_logits = None
    best_dev_y = None

    epoch_results = []

    started = time.time()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        opt.zero_grad(
            set_to_none=True
        )

        running = 0.0
        micro = 0
        opt_step = 0

        for bi, batch in enumerate(
            train_dl,
            1,
        ):
            labels_t = batch.pop(
                "labels"
            ).cuda()

            batch = {
                k: v.cuda(
                    non_blocking=True
                )
                for k, v
                in batch.items()
            }

            with torch.autocast(
                "cuda",
                dtype=torch.bfloat16,
            ):
                logits = model(
                    **batch
                ).logits

                loss = F.cross_entropy(
                    logits.float(),
                    labels_t,
                    label_smoothing=0.03,
                )

                loss = (
                    loss
                    / GRAD_ACCUM
                )

            loss.backward()

            running += (
                float(loss.detach())
                * GRAD_ACCUM
            )

            micro += 1

            if (
                bi % GRAD_ACCUM == 0
                or bi == len(train_dl)
            ):
                torch.nn.utils\
                    .clip_grad_norm_(
                        params,
                        MAX_GRAD_NORM,
                    )

                opt.step()
                sched.step()

                opt.zero_grad(
                    set_to_none=True
                )

                opt_step += 1

                if opt_step % 100 == 0:
                    elapsed = (
                        time.time()
                        - started
                    ) / 3600.0

                    print(
                        f"[train] "
                        f"epoch={epoch}/{EPOCHS} "
                        f"step={opt_step}/"
                        f"{opt_steps_per_epoch} "
                        f"loss="
                        f"{running/max(1,micro):.4f} "
                        f"lr="
                        f"{sched.get_last_lr()[0]:.2e} "
                        f"elapsed_h="
                        f"{elapsed:.2f}",
                        flush=True,
                    )

        dev_metrics, dev_logits, dev_y = \
            eval_loader(
                model,
                dev_dl,
            )

        strict_metrics, strict_details = \
            strict_eval(
                model,
                tok,
                strict_rows,
            )

        selection_score = (
            0.60
            * dev_metrics[
                "binary_benign_auc"
            ]
            + 0.25
            * dev_metrics[
                "macro_f1"
            ]
            + 0.15
            * dev_metrics[
                "nonbenign_recall_at_05"
            ]
        )

        rec = {
            "epoch": epoch,
            "train_mean_loss":
                running
                / max(1, micro),
            "dev": dev_metrics,
            "strict":
                strict_metrics,
            "selection_score":
                float(
                    selection_score
                ),
            "elapsed_hours":
                (
                    time.time()
                    - started
                ) / 3600.0,
        }

        epoch_results.append(rec)

        print()
        print(
            "===== EPOCH RESULT ====="
        )

        print(
            json.dumps(
                rec,
                indent=2,
            ),
            flush=True,
        )

        if selection_score > best_score:
            best_score = \
                float(selection_score)

            best_epoch = epoch

            best_dev_logits = \
                dev_logits.copy()

            best_dev_y = \
                dev_y.copy()

            best = ROOT / "best_adapter"

            if best.exists():
                shutil.rmtree(
                    best
                )

            model.save_pretrained(
                best
            )

            tok.save_pretrained(
                best
            )

            (
                ROOT /
                "best_strict_details.json"
            ).write_text(
                json.dumps(
                    strict_details,
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            print(
                f"[best] epoch={epoch} "
                f"score="
                f"{best_score:.6f}",
                flush=True,
            )

    if best_dev_logits is None:
        raise RuntimeError(
            "no best model"
        )

    temperature = fit_temperature(
        best_dev_logits,
        best_dev_y,
    )

    threshold_info = \
        best_binary_threshold(
            best_dev_logits,
            best_dev_y,
            temperature,
        )

    calibration = {
        "temperature":
            temperature,
        "benign_index":
            BENIGN_IDX,
        "benign_label":
            "BENIGN",
        "labels":
            LABELS,
        **threshold_info,
    }

    (
        ROOT / "calibration.json"
    ).write_text(
        json.dumps(
            calibration,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Final test is evaluated with current model
    # for general monitoring. Model selection was
    # based only on dev + strict set.
    test_metrics, _, _ = \
        eval_loader(
            model,
            test_dl,
        )

    summary = {
        "base_model":
            MODEL_NAME,
        "train_examples":
            len(train_rows),
        "dev_examples":
            len(dev_rows),
        "test_examples":
            len(test_rows),
        "strict_examples":
            len(strict_rows),
        "label_counts":
            dict(counts),
        "epochs":
            EPOCHS,
        "best_epoch":
            best_epoch,
        "best_selection_score":
            best_score,
        "calibration":
            calibration,
        "test_last_epoch":
            test_metrics,
        "epoch_results":
            epoch_results,
        "runtime_hours":
            (
                time.time()
                - started
            ) / 3600.0,
    }

    (
        ROOT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "===== FINAL SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
