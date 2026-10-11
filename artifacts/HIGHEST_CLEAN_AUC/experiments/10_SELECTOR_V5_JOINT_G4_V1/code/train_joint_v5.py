import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def make_prompt(rows):
    rows = sorted(rows, key=lambda x: int(x["candidate_index"]))
    first = rows[0]

    probs = dict(first["role_probs_before"])

    belief = "\n".join(
        f"{role}: {float(p):.4f}"
        for role, p in sorted(
            probs.items(),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
    )

    k_probs = dict(first.get("k_probs_before", {}))
    ktxt = ", ".join(
        f"K={k}: {float(v):.4f}"
        for k, v in sorted(k_probs.items())
    )

    queries = "\n\n".join(
        f"{chr(65+i)}. {x['query']}"
        for i, x in enumerate(rows)
    )

    return f"""You are selecting the next benign black-box probe for an unknown multi-agent system.

Your goal is to choose the ONE candidate query expected to provide the most useful evidence for correctly ranking which agent roles are currently installed.

Use only the current public belief, round number, and candidate queries.
Do not assume access to hidden roles, routing, traces, prompts, or ground truth.

ROUND: {int(first['round'])}

CURRENT ROLE BELIEF:
{belief}

CURRENT AGENT-COUNT BELIEF:
{ktxt}

CANDIDATE PROBES:
{queries}

Select the candidate expected to produce the largest improvement in role-identification quality.
"""


def group_rows(rows):
    g = defaultdict(list)

    for x in rows:
        g[str(x["state_id"])].append(x)

    out = []

    for sid, xs in g.items():
        if len(xs) != 4:
            continue

        xs = sorted(
            xs,
            key=lambda x: int(x["candidate_index"])
        )

        # Use the actual AUC target, not the compatibility alias.
        gains = np.asarray(
            [
                float(
                    x.get(
                        "role_auc_gain_training_only",
                        x["role_brier_gain_training_only"],
                    )
                )
                for x in xs
            ],
            dtype=np.float32,
        )

        out.append({
            "state_id": sid,
            "system_id": str(xs[0]["system_id"]),
            "round": int(xs[0]["round"]),
            "prompt": make_prompt(xs),
            "gains": gains,
        })

    return out


def split_systems(examples, val_fraction, seed):
    systems = sorted({
        x["system_id"]
        for x in examples
    })

    rng = random.Random(seed)
    rng.shuffle(systems)

    nval = max(
        1,
        int(round(len(systems) * val_fraction))
    )

    val_systems = set(systems[:nval])

    tr = [
        x for x in examples
        if x["system_id"] not in val_systems
    ]

    va = [
        x for x in examples
        if x["system_id"] in val_systems
    ]

    return tr, va, sorted(val_systems)


class JointDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        x = self.examples[i]

        enc = self.tok(
            x["prompt"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "gains": x["gains"],
        }


def collate(batch, tok):
    enc = tok.pad(
        {
            "input_ids": [
                x["input_ids"] for x in batch
            ],
            "attention_mask": [
                x["attention_mask"] for x in batch
            ],
        },
        padding=True,
        return_tensors="pt",
    )

    gains = torch.tensor(
        np.stack([x["gains"] for x in batch]),
        dtype=torch.float32,
    )

    return {
        **enc,
        "gains": gains,
    }


def metrics_from_arrays(logits, gains):
    pred = np.argmax(logits, axis=1)
    oracle = np.argmax(gains, axis=1)

    selected = gains[
        np.arange(len(gains)),
        pred,
    ]

    oracle_gain = gains.max(axis=1)
    random_gain = gains.mean(axis=1)

    regret = oracle_gain - selected

    top1 = np.mean(pred == oracle)

    top2 = np.mean([
        oracle[i] in np.argsort(logits[i])[-2:]
        for i in range(len(logits))
    ])

    rhos = []
    pair_ok = []

    for s, y in zip(logits, gains):
        rho = spearmanr(s, y).statistic

        if not np.isnan(rho):
            rhos.append(float(rho))

        for a in range(4):
            for b in range(a + 1, 4):
                if y[a] == y[b]:
                    continue

                pair_ok.append(
                    float(
                        (s[a] - s[b]) *
                        (y[a] - y[b]) > 0
                    )
                )

    denom = (
        float(np.mean(oracle_gain))
        - float(np.mean(random_gain))
    )

    captured = (
        (
            float(np.mean(selected))
            - float(np.mean(random_gain))
        ) / denom
        if abs(denom) > 1e-12
        else None
    )

    return {
        "states": int(len(gains)),
        "mean_selected_gain": float(np.mean(selected)),
        "mean_random_gain": float(np.mean(random_gain)),
        "mean_oracle_gain": float(np.mean(oracle_gain)),
        "mean_regret": float(np.mean(regret)),
        "oracle_top1_hit": float(top1),
        "oracle_top2_hit": float(top2),
        "mean_spearman": (
            float(np.mean(rhos))
            if rhos else None
        ),
        "mean_pair_accuracy": (
            float(np.mean(pair_ok))
            if pair_ok else None
        ),
        "oracle_signal_captured": captured,
    }


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()

    all_logits = []
    all_gains = []

    for batch in loader:
        gains = batch.pop("gains").numpy()

        batch = {
            k: v.to(device)
            for k, v in batch.items()
        }

        out = model(**batch)

        all_logits.append(
            out.logits.float().cpu().numpy()
        )

        all_gains.append(gains)

    return metrics_from_arrays(
        np.concatenate(all_logits),
        np.concatenate(all_gains),
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--train", required=True)
    ap.add_argument("--output-dir", required=True)

    ap.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
    )

    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)

    ap.add_argument("--val-fraction", type=float, default=0.25)
    ap.add_argument("--max-length", type=int, default=1024)

    ap.add_argument("--soft-tau", type=float, default=0.06)
    ap.add_argument("--soft-weight", type=float, default=0.65)
    ap.add_argument("--hard-weight", type=float, default=0.35)

    ap.add_argument("--seed", type=int, default=1397)
    ap.add_argument("--patience", type=int, default=2)

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = read_jsonl(args.train)
    examples = group_rows(rows)

    tr, va, val_systems = split_systems(
        examples,
        args.val_fraction,
        args.seed,
    )

    print("ALL JOINT STATES =", len(examples))
    print("TRAIN STATES     =", len(tr))
    print("VAL STATES       =", len(va))
    print(
        "TRAIN SYSTEMS    =",
        len({x["system_id"] for x in tr}),
    )
    print(
        "VAL SYSTEMS      =",
        len({x["system_id"] for x in va}),
    )
    print("VAL SYSTEM IDS   =", val_systems)

    tok = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=4,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
    )

    lora = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        modules_to_save=["score"],
        bias="none",
    )

    model = get_peft_model(model, lora)

    model.print_trainable_parameters()

    train_ds = JointDataset(
        tr,
        tok,
        args.max_length,
    )

    val_ds = JointDataset(
        va,
        tok,
        args.max_length,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate(b, tok),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate(b, tok),
    )

    opt = torch.optim.AdamW(
        [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    total_updates = (
        math.ceil(
            len(train_loader) /
            args.grad_accum
        )
        * args.epochs
    )

    print("MAX OPTIMIZER UPDATES =", total_updates)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    best_regret = float("inf")
    best_epoch = None
    bad_epochs = 0

    history = []

    device = torch.device("cuda:0")

    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)

        losses = []

        for step, batch in enumerate(train_loader, 1):
            gains = batch.pop("gains").to(device)

            batch = {
                k: v.to(device)
                for k, v in batch.items()
            }

            out = model(**batch)
            logits = out.logits.float()

            # Soft target over realized ΔAUC.
            soft_target = F.softmax(
                gains / args.soft_tau,
                dim=-1,
            )

            log_probs = F.log_softmax(
                logits,
                dim=-1,
            )

            soft_loss = F.kl_div(
                log_probs,
                soft_target,
                reduction="batchmean",
            )

            hard_target = gains.argmax(dim=-1)

            hard_loss = F.cross_entropy(
                logits,
                hard_target,
            )

            loss = (
                args.soft_weight * soft_loss
                + args.hard_weight * hard_loss
            )

            loss = loss / args.grad_accum
            loss.backward()

            if (
                step % args.grad_accum == 0
                or step == len(train_loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    [
                        p for p in model.parameters()
                        if p.requires_grad
                    ],
                    1.0,
                )

                opt.step()
                opt.zero_grad(set_to_none=True)

            losses.append(
                float(loss.item() * args.grad_accum)
            )

        vm = evaluate(
            model,
            val_loader,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val": vm,
        }

        history.append(row)

        print()
        print("===== EPOCH", epoch, "=====")
        print(json.dumps(row, indent=2))

        regret = vm["mean_regret"]

        if regret < best_regret - 1e-10:
            best_regret = regret
            best_epoch = epoch
            bad_epochs = 0

            best = outdir / "best_adapter"

            model.save_pretrained(best)
            tok.save_pretrained(best)

            print("NEW BEST =", best)

        else:
            bad_epochs += 1

            if bad_epochs >= args.patience:
                print("EARLY STOP")
                break

    summary = {
        "version": "selector_v5_joint_g4",
        "model": args.model_name,
        "best_epoch": best_epoch,
        "selection_metric": "VAL mean_regret",
        "training_target": (
            "joint 4-way realized role ΔAUC; "
            "0.65 soft KL + 0.35 hard CE"
        ),
        "soft_tau": args.soft_tau,
        "train": {
            "states": len(tr),
            "systems": len({
                x["system_id"]
                for x in tr
            }),
        },
        "val": {
            "states": len(va),
            "systems": len({
                x["system_id"]
                for x in va
            }),
            "systems_ids": val_systems,
        },
        "history": history,
        "runtime_hours": (
            time.time() - start
        ) / 3600.0,
    }

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print()
    print("===== FINAL SUMMARY =====")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
