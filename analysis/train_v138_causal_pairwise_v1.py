#!/usr/bin/env python3

import json
import math
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from torch.nn.utils import clip_grad_norm_
from torch.utils.data import (
    DataLoader,
    Dataset,
    WeightedRandomSampler,
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


ROOT = Path(
    "artifacts/v138_pairwise_v2"
)

PAIR_PATH = (
    ROOT / "train_pairs_v2.jsonl"
)

DEV_PATH = (
    ROOT / "dev_v138.jsonl"
)

OUT = (
    ROOT / "model_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

BASE_MODEL = (
    "Qwen/Qwen2.5-3B-Instruct"
)

SEED = 1388
EPOCHS = 5
LR = 5e-5
GRAD_ACCUM = 8
LAMBDA_RANK = 1.0


def readj(path):
    with Path(path).open() as f:
        return [
            json.loads(x)
            for x in f
            if x.strip()
        ]


def sigmoid(x):
    x = np.asarray(
        x,
        dtype=float,
    )

    return (
        1.0
        / (
            1.0
            + np.exp(-x)
        )
    )


def poisson_binomial(probs):
    d = np.array(
        [1.0],
        dtype=float,
    )

    for p in probs:
        n = np.zeros(
            len(d) + 1,
            dtype=float,
        )

        n[:-1] += (
            d * (1.0 - p)
        )

        n[1:] += (
            d * p
        )

        d = n

    return d


def prompt_for(
    role,
    desc,
    response,
):
    # Response first so left-side truncation preserves
    # the target information and the end of the response.
    return f"""You are analyzing the visible behavior of a black-box multi-agent application.

VISIBLE FINAL RESPONSE:
{response}

TARGET CAPABILITY:
{role}

PUBLIC DESCRIPTION OF TARGET CAPABILITY:
{desc}

Decide which explanation is more consistent with the visible response:
A = the target capability is installed in the application
B = the target capability is absent from the application

Answer with A or B only.

Answer:"""


class PairDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU REQUIRED"
        )

    torch.cuda.manual_seed_all(
        SEED
    )

    pairs = readj(
        PAIR_PATH
    )

    dev = readj(
        DEV_PATH
    )

    if len(pairs) != 266:
        raise RuntimeError(
            f"Expected 266 pairs, got {len(pairs)}"
        )

    if len(dev) != 760:
        raise RuntimeError(
            f"Expected 760 DEV rows, got {len(dev)}"
        )

    gpu = (
        torch.cuda
        .get_device_properties(0)
    )

    print("=" * 100)
    print("V13.8 NATIVE-LM PAIRWISE PRESENCE SENSOR")
    print("=" * 100)

    print(
        "GPU        :",
        gpu.name,
    )

    print(
        "VRAM GiB   :",
        round(
            gpu.total_memory
            / 1024**3,
            2,
        ),
    )

    print(
        "BASE MODEL :",
        BASE_MODEL,
    )

    print(
        "TRAIN PAIRS:",
        len(pairs),
    )

    print(
        "DEV ROWS   :",
        len(dev),
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            BASE_MODEL,
            trust_remote_code=True,
        )
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    tokenizer.padding_side = (
        "left"
    )

    tokenizer.truncation_side = (
        "left"
    )


    # ------------------------------------------------------
    # Find two single-token choices.
    # No random classification head.
    # ------------------------------------------------------

    candidates = [
        (" A", " B"),
        ("A", "B"),
        (" 1", " 0"),
        ("1", "0"),
        (" YES", " NO"),
        ("YES", "NO"),
    ]

    choice = None

    for a, b in candidates:
        ai = tokenizer(
            a,
            add_special_tokens=False,
        )["input_ids"]

        bi = tokenizer(
            b,
            add_special_tokens=False,
        )["input_ids"]

        if (
            len(ai) == 1
            and len(bi) == 1
            and ai[0] != bi[0]
        ):
            choice = (
                a,
                b,
                ai[0],
                bi[0],
            )
            break

    if choice is None:
        raise RuntimeError(
            "Could not find single-token A/B choices"
        )

    (
        installed_text,
        absent_text,
        installed_id,
        absent_id,
    ) = choice

    print()
    print(
        "INSTALLED choice:",
        repr(installed_text),
        installed_id,
    )

    print(
        "ABSENT choice   :",
        repr(absent_text),
        absent_id,
    )


    # ------------------------------------------------------
    # Determine sequence length from actual prompts.
    # ------------------------------------------------------

    length_texts = []

    for x in pairs:
        length_texts.append(
            prompt_for(
                x["target_role"],
                x[
                    "public_role_description"
                ],
                x[
                    "positive_response"
                ],
            )
        )

        length_texts.append(
            prompt_for(
                x["target_role"],
                x[
                    "public_role_description"
                ],
                x[
                    "negative_response"
                ],
            )
        )

    for x in dev:
        length_texts.append(
            prompt_for(
                x["target_role"],
                x[
                    "public_role_description"
                ],
                x["final_output"],
            )
        )

    lengths = []

    for text in length_texts:
        lengths.append(
            len(
                tokenizer(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )["input_ids"]
            )
        )

    arr = np.asarray(
        lengths
    )

    p95 = int(
        np.percentile(
            arr,
            95,
        )
    )

    p99 = int(
        np.percentile(
            arr,
            99,
        )
    )

    mx = int(
        arr.max()
    )

    max_length = min(
        1536,
        max(
            768,
            int(
                math.ceil(
                    p99 / 128
                )
                * 128
            ),
        ),
    )

    trunc_n = int(
        (
            arr
            > max_length
        ).sum()
    )

    print()
    print(
        "TOKEN p95 :",
        p95,
    )

    print(
        "TOKEN p99 :",
        p99,
    )

    print(
        "TOKEN max :",
        mx,
    )

    print(
        "MAX LENGTH:",
        max_length,
    )

    print(
        "TRUNCATED :",
        trunc_n,
        "/",
        len(arr),
    )


    # ------------------------------------------------------
    # QLoRA. Much smaller adapter than V13.7.
    # ------------------------------------------------------

    use_bf16 = (
        torch.cuda
        .is_bf16_supported()
    )

    dtype = (
        torch.bfloat16
        if use_bf16
        else torch.float16
    )

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    model = (
        AutoModelForCausalLM
        .from_pretrained(
            BASE_MODEL,
            quantization_config=qconf,
            device_map={"": 0},
            trust_remote_code=True,
            dtype=dtype,
        )
    )

    model.config.use_cache = False

    model = (
        prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
        )
    )

    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",

        # Attention only.
        # Deliberately much lower capacity than V13.7.
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora,
    )

    total = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "TRAINABLE:",
        f"{trainable:,}",
    )

    print(
        "PARAM %  :",
        f"{100 * trainable / total:.4f}%",
    )


    # ------------------------------------------------------
    # Equal expected contribution from each role.
    # ------------------------------------------------------

    role_count = Counter(
        x["target_role"]
        for x in pairs
    )

    sample_weights = [
        1.0
        / role_count[
            x["target_role"]
        ]
        for x in pairs
    ]

    sampler = (
        WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(pairs),
            replacement=True,
        )
    )

    loader = DataLoader(
        PairDataset(pairs),
        batch_size=1,
        sampler=sampler,
        num_workers=0,
    )


    optimizer = (
        torch.optim.AdamW(
            [
                p
                for p in model.parameters()
                if p.requires_grad
            ],
            lr=LR,
            weight_decay=0.01,
        )
    )

    optimizer_steps_per_epoch = (
        math.ceil(
            len(loader)
            / GRAD_ACCUM
        )
    )

    total_optimizer_steps = (
        optimizer_steps_per_epoch
        * EPOCHS
    )

    warmup_steps = max(
        1,
        int(
            0.10
            * total_optimizer_steps
        ),
    )

    scheduler = (
        get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=
                warmup_steps,
            num_training_steps=
                total_optimizer_steps,
        )
    )

    print()
    print(
        "optimizer steps:",
        total_optimizer_steps,
    )

    print(
        "warmup steps   :",
        warmup_steps,
    )

    print(
        "LR             :",
        LR,
    )

    print(
        "rank lambda    :",
        LAMBDA_RANK,
    )


    def score_prompts(
        texts,
        grad,
    ):
        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        enc = {
            k: v.to("cuda")
            for k, v
            in enc.items()
        }

        context = (
            torch.enable_grad()
            if grad
            else torch.no_grad()
        )

        with context:
            out = model(
                **enc,
                use_cache=False,
            )

            last = (
                out.logits[
                    :,
                    -1,
                    :
                ]
                .float()
            )

            score = (
                last[
                    :,
                    installed_id
                ]
                -
                last[
                    :,
                    absent_id
                ]
            )

        return score


    def evaluate():
        model.eval()

        scores = []

        batch_size = 2

        for start in range(
            0,
            len(dev),
            batch_size,
        ):
            rr = dev[
                start:
                start + batch_size
            ]

            texts = [
                prompt_for(
                    x["target_role"],
                    x[
                        "public_role_description"
                    ],
                    x[
                        "final_output"
                    ],
                )
                for x in rr
            ]

            s = (
                score_prompts(
                    texts,
                    grad=False,
                )
                .detach()
                .cpu()
                .numpy()
            )

            scores.extend(
                map(
                    float,
                    s,
                )
            )

        scores = np.asarray(
            scores
        )

        probs = sigmoid(
            scores
        )

        y = np.asarray(
            [
                int(x["label"])
                for x in dev
            ],
            dtype=int,
        )

        global_auc = float(
            roc_auc_score(
                y,
                probs,
            )
        )

        pr_auc = float(
            average_precision_score(
                y,
                probs,
            )
        )

        bal = float(
            balanced_accuracy_score(
                y,
                (
                    probs >= 0.5
                ).astype(int),
            )
        )

        per_role = {}

        for role in sorted({
            x["target_role"]
            for x in dev
        }):
            idx = [
                i
                for i, x
                in enumerate(dev)
                if (
                    x["target_role"]
                    == role
                )
            ]

            yy = y[idx]
            pp = probs[idx]

            per_role[role] = float(
                roc_auc_score(
                    yy,
                    pp,
                )
            )

        macro_role_auc = float(
            np.mean(
                list(
                    per_role.values()
                )
            )
        )


        # ----------------------------------------------
        # Raw end-to-end K.
        # ----------------------------------------------

        by_system = defaultdict(
            list
        )

        for i, x in enumerate(
            dev
        ):
            by_system[
                x["system_id"]
            ].append(
                (
                    x,
                    float(
                        probs[i]
                    ),
                )
            )

        exact = 0
        exact3 = 0
        exact4 = 0

        n3 = 0
        n4 = 0

        mae = []

        k_dist = Counter()

        system_rows = []

        for sid, rr in sorted(
            by_system.items()
        ):
            if len(rr) != 20:
                raise RuntimeError(
                    f"{sid}: "
                    f"{len(rr)} roles"
                )

            true_k = {
                int(
                    x[
                        "K_eval_only"
                    ]
                )
                for x, _
                in rr
            }

            if len(true_k) != 1:
                raise RuntimeError(
                    f"{sid}: inconsistent K"
                )

            true_k = next(
                iter(true_k)
            )

            role_p = {
                x["target_role"]: p
                for x, p
                in rr
            }

            posterior = (
                poisson_binomial(
                    [
                        role_p[r]
                        for r
                        in sorted(
                            role_p
                        )
                    ]
                )
            )

            pred_k = max(
                range(3, 9),
                key=lambda k:
                    posterior[k],
            )

            ok = int(
                pred_k == true_k
            )

            exact += ok

            mae.append(
                abs(
                    pred_k
                    - true_k
                )
            )

            k_dist[
                pred_k
            ] += 1

            if true_k == 3:
                n3 += 1
                exact3 += ok

            if true_k == 4:
                n4 += 1
                exact4 += ok

            system_rows.append({
                "system_id": sid,
                "true_K": true_k,
                "pred_K_raw":
                    pred_k,
                "posterior_K_raw":
                    [
                        float(z)
                        for z
                        in posterior
                    ],
            })

        return {
            "scores": scores,
            "probs": probs,

            "global_auc":
                global_auc,

            "pr_auc":
                pr_auc,

            "balanced_accuracy":
                bal,

            "macro_role_auc":
                macro_role_auc,

            "per_role_auc":
                per_role,

            "exact_K":
                exact
                / len(by_system),

            "K3_exact":
                exact3 / n3,

            "K4_exact":
                exact4 / n4,

            "K_MAE":
                float(
                    np.mean(
                        mae
                    )
                ),

            "pred_K_distribution":
                {
                    str(k): int(v)
                    for k, v
                    in sorted(
                        k_dist.items()
                    )
                },

            "system_rows":
                system_rows,
        }


    # ------------------------------------------------------
    # Epoch-zero diagnostic.
    # ------------------------------------------------------

    print()
    print(
        "===== ZERO-SHOT DEV ====="
    )

    z = evaluate()

    print(
        "zero global AUC:",
        round(
            z["global_auc"],
            4,
        ),
    )

    print(
        "zero macro role AUC:",
        round(
            z[
                "macro_role_auc"
            ],
            4,
        ),
    )

    print(
        "zero exact K:",
        f"{z['exact_K']:.2%}",
    )


    # ------------------------------------------------------
    # Training.
    # ------------------------------------------------------

    best_macro = -1.0
    best_epoch = None
    best_metrics = None

    optimizer.zero_grad(
        set_to_none=True
    )

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        model.train()

        epoch_loss = 0.0
        epoch_bce = 0.0
        epoch_rank = 0.0
        rank_correct = 0
        pair_seen = 0

        for step, pair in enumerate(
            loader,
            start=1,
        ):
            role = pair[
                "target_role"
            ][0]

            desc = pair[
                "public_role_description"
            ][0]

            pos_response = pair[
                "positive_response"
            ][0]

            neg_response = pair[
                "negative_response"
            ][0]

            pos_prompt = (
                prompt_for(
                    role,
                    desc,
                    pos_response,
                )
            )

            neg_prompt = (
                prompt_for(
                    role,
                    desc,
                    neg_response,
                )
            )

            scores = score_prompts(
                [
                    pos_prompt,
                    neg_prompt,
                ],
                grad=True,
            )

            pos_score = scores[0]
            neg_score = scores[1]

            # Balanced binary likelihood.
            bce = 0.5 * (
                F.softplus(
                    -pos_score
                )
                +
                F.softplus(
                    neg_score
                )
            )

            # Bradley-Terry / logistic ranking loss.
            rank = F.softplus(
                -(
                    pos_score
                    - neg_score
                )
            )

            loss = (
                bce
                +
                LAMBDA_RANK
                * rank
            )

            (
                loss
                / GRAD_ACCUM
            ).backward()

            epoch_loss += float(
                loss.detach()
            )

            epoch_bce += float(
                bce.detach()
            )

            epoch_rank += float(
                rank.detach()
            )

            rank_correct += int(
                (
                    pos_score
                    > neg_score
                ).item()
            )

            pair_seen += 1

            if (
                step
                % GRAD_ACCUM
                == 0
                or step
                == len(loader)
            ):
                clip_grad_norm_(
                    [
                        p
                        for p
                        in model.parameters()
                        if p.requires_grad
                    ],
                    1.0,
                )

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

        print()
        print(
            f"===== EPOCH {epoch} TRAIN ====="
        )

        print(
            "loss:",
            round(
                epoch_loss
                / pair_seen,
                4,
            ),
        )

        print(
            "bce:",
            round(
                epoch_bce
                / pair_seen,
                4,
            ),
        )

        print(
            "rank:",
            round(
                epoch_rank
                / pair_seen,
                4,
            ),
        )

        print(
            "pair accuracy:",
            f"{rank_correct/pair_seen:.2%}",
        )

        ev = evaluate()

        print(
            f"===== EPOCH {epoch} DEV ====="
        )

        print(
            "global AUC:",
            round(
                ev["global_auc"],
                4,
            ),
        )

        print(
            "PR-AUC:",
            round(
                ev["pr_auc"],
                4,
            ),
        )

        print(
            "macro role AUC:",
            round(
                ev[
                    "macro_role_auc"
                ],
                4,
            ),
        )

        print(
            "BalAcc@0.5:",
            round(
                ev[
                    "balanced_accuracy"
                ],
                4,
            ),
        )

        print(
            "raw exact K:",
            f"{ev['exact_K']:.2%}",
        )

        print(
            "K3:",
            f"{ev['K3_exact']:.2%}",
            "K4:",
            f"{ev['K4_exact']:.2%}",
        )

        print(
            "K dist:",
            ev[
                "pred_K_distribution"
            ],
        )

        if (
            ev["macro_role_auc"]
            > best_macro
        ):
            best_macro = (
                ev[
                    "macro_role_auc"
                ]
            )

            best_epoch = epoch

            # JSON-safe copy.
            best_metrics = {
                k: v
                for k, v
                in ev.items()
                if k not in {
                    "scores",
                    "probs",
                    "system_rows",
                }
            }

            best_dir = (
                OUT
                / "best_adapter"
            )

            if best_dir.exists():
                shutil.rmtree(
                    best_dir
                )

            model.save_pretrained(
                best_dir
            )

            tokenizer.save_pretrained(
                best_dir
            )

            # Save predictions corresponding to THIS best model.
            with (
                OUT
                / "dev_predictions.jsonl"
            ).open("w") as f:
                for row, score, prob in zip(
                    dev,
                    ev["scores"],
                    ev["probs"],
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
                                int(
                                    row["label"]
                                ),

                            "K_eval_only":
                                int(
                                    row[
                                        "K_eval_only"
                                    ]
                                ),

                            "choice_logit_difference":
                                float(
                                    score
                                ),

                            "prob_installed_raw":
                                float(
                                    prob
                                ),
                        })
                        + "\n"
                    )

            with (
                OUT
                / "dev_system_K_raw.jsonl"
            ).open("w") as f:
                for r in (
                    ev[
                        "system_rows"
                    ]
                ):
                    f.write(
                        json.dumps(r)
                        + "\n"
                    )

            print(
                "NEW BEST ADAPTER SAVED"
            )


    result = {
        "version":
            "v138_native_lm_pairwise_v1",

        "base_model":
            BASE_MODEL,

        "best_epoch":
            best_epoch,

        "best_macro_role_auc":
            best_macro,

        "best_dev":
            best_metrics,

        "choice_tokens": {
            "installed":
                installed_text,

            "absent":
                absent_text,

            "installed_id":
                installed_id,

            "absent_id":
                absent_id,
        },

        "training": {
            "pairs":
                len(pairs),

            "epochs":
                EPOCHS,

            "lr":
                LR,

            "gradient_accumulation":
                GRAD_ACCUM,

            "lambda_rank":
                LAMBDA_RANK,

            "lora_rank":
                8,

            "lora_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],
        },

        "model_input_includes_raw_query":
            False,

        "internal_test_used":
            False,
    }

    (
        OUT
        / "DEV_METRICS.json"
    ).write_text(
        json.dumps(
            result,
            indent=2,
            sort_keys=True,
        )
    )

    print()
    print("=" * 100)
    print("V13.8 FINAL")
    print("=" * 100)

    print(
        "BEST EPOCH:",
        best_epoch,
    )

    print(
        "BEST MACRO ROLE AUC:",
        round(
            best_macro,
            4,
        ),
    )

    print(
        "GLOBAL AUC:",
        round(
            best_metrics[
                "global_auc"
            ],
            4,
        ),
    )

    print(
        "RAW EXACT K:",
        f"{best_metrics['exact_K']:.2%}",
    )

    print(
        "K3:",
        f"{best_metrics['K3_exact']:.2%}",
    )

    print(
        "K4:",
        f"{best_metrics['K4_exact']:.2%}",
    )

    print(
        "RAW QUERY USED: NO"
    )

    print(
        "RANDOM CLASSIFICATION HEAD USED: NO"
    )

    print(
        "INTERNAL TEST USED: NO"
    )

    print(
        "V13.8 TRAINING: DONE"
    )


if __name__ == "__main__":
    main()
