from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
    get_cosine_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)


MODEL = "Qwen/Qwen3.8-27B"

DATA = Path(
    os.environ.get(
        "DATA_ROOT",
        "artifacts/q38_selector_v41_data",
    )
)

OUT = Path(
    os.environ.get(
        "OUT_DIR",
        "artifacts/q38_selector_v41",
    )
)

MAX_LEN = int(
    os.environ.get(
        "MAX_LEN",
        "1024",
    )
)

EPOCHS = int(
    os.environ.get(
        "EPOCHS",
        "3",
    )
)

MAX_MICRO = int(
    os.environ.get(
        "MAX_MICRO",
        "18000",
    )
)

PERMS_PER_STATE = int(
    os.environ.get(
        "PERMS_PER_STATE",
        "4",
    )
)

EVAL_PERMS = int(
    os.environ.get(
        "EVAL_PERMS",
        "3",
    )
)

GRAD_ACC = int(
    os.environ.get(
        "GRAD_ACC",
        "8",
    )
)

LR = float(
    os.environ.get(
        "LR",
        "6e-6",
    )
)

TARGET_TEMP = float(
    os.environ.get(
        "TARGET_TEMP",
        "2.0",
    )
)

PAIR_WEIGHT = float(
    os.environ.get(
        "PAIR_WEIGHT",
        "0.50",
    )
)

PATIENCE = int(
    os.environ.get(
        "PATIENCE",
        "2",
    )
)

LETTERS = list(
    "ABCDEFGH"
)

SYSTEM = (
    "You select the most diagnostically useful benign probe "
    "for black-box inference of an application's installed roles. "
    "Compare the candidate probes against the current public belief "
    "and previous visible interactions. "
    "Do not answer the probes. "
    "Select the probe expected to best separate plausible competing "
    "role configurations and improve the role ranking."
)


def read_jsonl(path):
    return [
        json.loads(x)
        for x in path.read_text().splitlines()
        if x.strip()
    ]


def dump(path, obj):
    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )


def stable_seed(*parts):
    s = "::".join(
        map(str, parts)
    )

    h = hashlib.sha256(
        s.encode()
    ).hexdigest()

    return int(
        h[:16],
        16,
    )


print("=" * 100)
print(
    "QWEN3.8-27B SELECTOR V4.1"
)
print(
    "RESTRICTED LISTWISE + PAIRWISE"
)
print("=" * 100)

print("MODEL =", MODEL)
print("MAX_LEN =", MAX_LEN)
print("EPOCHS =", EPOCHS)
print("MAX_MICRO =", MAX_MICRO)
print(
    "PERMS_PER_STATE =",
    PERMS_PER_STATE,
)
print(
    "EVAL_PERMS =",
    EVAL_PERMS,
)
print("GRAD_ACC =", GRAD_ACC)
print("LR =", LR)
print(
    "TARGET_TEMP =",
    TARGET_TEMP,
)
print(
    "PAIR_WEIGHT =",
    PAIR_WEIGHT,
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

train_groups = read_jsonl(
    DATA / "train_groups.jsonl"
)

val_groups = read_jsonl(
    DATA / "val_groups.jsonl"
)

challenge_groups = read_jsonl(
    DATA / "challenge_groups.jsonl"
)

print(
    f"TRAIN_STATES={len(train_groups)} "
    f"VAL_STATES={len(val_groups)} "
    f"CHALLENGE_STATES={len(challenge_groups)}"
)


###############################################################################
# TOKENIZER / CHAT TEMPLATE
###############################################################################

processor = AutoProcessor.from_pretrained(
    MODEL
)

tokenizer = getattr(
    processor,
    "tokenizer",
    processor,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = (
        tokenizer.eos_token
    )


def apply_chat(user):
    messages = [
        {
            "role": "system",
            "content": SYSTEM,
        },
        {
            "role": "user",
            "content": user,
        },
    ]

    # Qwen reasoning models may otherwise want to begin with
    # reasoning/thinking tokens. We need direct ranking logits.
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


###############################################################################
# LABEL TOKENS
###############################################################################

label_ids = {}

for letter in LETTERS:

    found = None

    for text in (
        letter,
        " " + letter,
    ):
        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) == 1:
            found = int(ids[0])
            break

    if found is None:
        raise RuntimeError(
            f"No one-token encoding for {letter}"
        )

    label_ids[letter] = found

print(
    "LABEL TOKEN IDS =",
    label_ids,
)


###############################################################################
# PROMPT RENDERING
###############################################################################

def role_block(g):
    rp = {
        str(k): float(v)
        for k, v
        in g[
            "role_probs_before"
        ].items()
    }

    ordered = sorted(
        rp,
        key=lambda r: (
            abs(rp[r] - 0.5),
            -rp[r],
            r,
        ),
    )

    return " | ".join(
        f"{r}={rp[r]:.3f}"
        for r in ordered
    )


def render_user(
    g,
    order,
    history_chars,
    query_chars,
):
    history = " ".join(
        str(
            g.get(
                "history_text",
                "- none",
            )
        ).split()
    )

    if len(history) > history_chars:
        history = (
            history[:history_chars]
            + "..."
        )

    candidates = []

    for display_i, original_i in enumerate(
        order
    ):
        q = " ".join(
            str(
                g["queries"][
                    original_i
                ]
            ).split()
        )

        if len(q) > query_chars:
            q = (
                q[:query_chars]
                + "..."
            )

        candidates.append(
            f"{LETTERS[display_i]}. {q}"
        )

    return (
        "CURRENT ROLE BELIEF\n"
        + role_block(g)
        + "\n\n"
        + "RECENT PUBLIC HISTORY\n"
        + history
        + "\n\n"
        + "CANDIDATES\n"
        + "\n".join(candidates)
        + "\n\n"
        + "Rank these probes comparatively. "
          "Prefer the probe most likely to resolve current ambiguity, "
          "distinguish competing role configurations, or elicit "
          "distinctive behavior visible in the final response. "
          "Do not prefer a probe merely because its topic resembles "
          "a likely role."
    )


def make_prompt(g, order):
    # NEVER silently truncate away belief/candidates.
    # Reduce optional history first.
    settings = [
        (1200, 320),
        (900, 300),
        (650, 280),
        (400, 260),
        (200, 240),
        (0, 220),
    ]

    for hchars, qchars in settings:

        user = render_user(
            g,
            order,
            hchars,
            qchars,
        )

        text = apply_chat(
            user
        )

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) <= MAX_LEN:
            return (
                text,
                ids,
                hchars,
                qchars,
            )

    raise RuntimeError(
        f"Prompt cannot fit MAX_LEN={MAX_LEN}: "
        f"state={g['state_id']}"
    )


###############################################################################
# TRAIN ITEMS
###############################################################################

class SelectorDataset(Dataset):

    def __init__(self, groups):

        self.items = []

        lengths = []
        compressed = 0

        t0 = time.time()

        for gi, g in enumerate(groups):

            n = len(
                g["queries"]
            )

            for p in range(
                PERMS_PER_STATE
            ):

                order = list(
                    range(n)
                )

                rng = random.Random(
                    stable_seed(
                        g["state_id"],
                        "train",
                        p,
                    )
                )

                rng.shuffle(
                    order
                )

                (
                    text,
                    ids,
                    hchars,
                    qchars,
                ) = make_prompt(
                    g,
                    order,
                )

                lengths.append(
                    len(ids)
                )

                if hchars < 1200:
                    compressed += 1

                utilities = [
                    float(
                        g["pair_gains"][i]
                    )
                    for i in order
                ]

                auc_gains = [
                    float(
                        g["auc_gains"][i]
                    )
                    for i in order
                ]

                spread = (
                    max(utilities)
                    - min(utilities)
                )

                # Ambiguous states still matter, but should not dominate
                # a confident preference state.
                state_weight = (
                    0.50
                    + 0.50
                    * min(
                        1.0,
                        spread / 4.0,
                    )
                )

                self.items.append({
                    "input_ids":
                        ids,

                    "utilities":
                        utilities,

                    "auc_gains":
                        auc_gains,

                    "state_weight":
                        state_weight,
                })

        print(
            "TRAIN EXAMPLES =",
            len(self.items),
        )

        print(
            "TOKEN LENGTH "
            f"mean={np.mean(lengths):.1f} "
            f"p95={np.percentile(lengths,95):.1f} "
            f"max={max(lengths)}"
        )

        print(
            "HISTORY_COMPRESSED =",
            compressed,
        )

        print(
            "TOKENIZATION_SECONDS =",
            round(
                time.time() - t0,
                1,
            ),
        )

    def __len__(self):
        return len(
            self.items
        )

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    if len(batch) != 1:
        raise RuntimeError(
            "Expected microbatch=1"
        )

    x = batch[0]

    ids = torch.tensor(
        [x["input_ids"]],
        dtype=torch.long,
    )

    mask = torch.ones_like(
        ids
    )

    return {
        "input_ids":
            ids,

        "attention_mask":
            mask,

        "utilities":
            torch.tensor(
                x["utilities"],
                dtype=torch.float32,
            ),

        "state_weight":
            torch.tensor(
                float(
                    x["state_weight"]
                ),
                dtype=torch.float32,
            ),
    }


train_ds = SelectorDataset(
    train_groups
)

loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=True,
    collate_fn=collate,
)


###############################################################################
# MODEL
###############################################################################

print()
print(
    "Loading Qwen3.8-27B BF16..."
)

load_t0 = time.time()

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    device_map={"": 0},
    low_cpu_mem_usage=True,
)

print(
    "LOAD_SECONDS =",
    round(
        time.time()
        - load_t0,
        1,
    ),
)

if hasattr(
    model.config,
    "use_cache",
):
    model.config.use_cache = False

model.gradient_checkpointing_enable()

if hasattr(
    model,
    "enable_input_require_grads",
):
    model.enable_input_require_grads()


wanted = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

available = {
    name.rsplit(".", 1)[-1]
    for name, _
    in model.named_modules()
}

targets = [
    x
    for x in wanted
    if x in available
]

print(
    "LoRA targets =",
    targets,
)

if not targets:
    raise RuntimeError(
        "No LoRA targets"
    )

lora = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=targets,
)

model = get_peft_model(
    model,
    lora,
)

model.print_trainable_parameters()


###############################################################################
# LOSS
###############################################################################

def restricted_scores(
    logits,
    n_candidates,
):
    # logits: vocab scores for NEXT generated token
    ids = torch.tensor(
        [
            label_ids[
                LETTERS[i]
            ]
            for i
            in range(
                n_candidates
            )
        ],
        device=logits.device,
        dtype=torch.long,
    )

    return logits.index_select(
        0,
        ids,
    ).float()


def selector_loss(
    scores,
    utilities,
    state_weight,
):
    utilities = utilities.to(
        scores.device
    )

    # ---------------------------------------------------------
    # SOFT LISTWISE TARGET
    #
    # A one-shot realized winner is noisy.
    # Preserve full within-state ordering and near-ties.
    # ---------------------------------------------------------

    centered_u = (
        utilities
        - utilities.mean()
    )

    centered_u = torch.clamp(
        centered_u,
        -12.0,
        12.0,
    )

    target = torch.softmax(
        centered_u
        / TARGET_TEMP,
        dim=0,
    )

    logp = torch.log_softmax(
        scores,
        dim=0,
    )

    list_loss = -(
        target
        * logp
    ).sum()

    # ---------------------------------------------------------
    # PAIRWISE RANKING AUXILIARY
    #
    # Only enforce ordering when realized utility differs by
    # at least one corrected true-vs-false role pair.
    # ---------------------------------------------------------

    pair_losses = []
    pair_weights = []

    n = len(scores)

    for i in range(n):
        for j in range(
            i + 1,
            n,
        ):

            du = (
                utilities[i]
                - utilities[j]
            )

            if abs(
                float(
                    du.detach().cpu()
                )
            ) < 1.0:
                continue

            sign = torch.sign(
                du
            )

            margin = (
                scores[i]
                - scores[j]
            )

            pl = F.softplus(
                -sign
                * margin
            )

            w = min(
                2.0,
                abs(
                    float(
                        du.detach().cpu()
                    )
                )
                / 3.0,
            )

            pair_losses.append(
                pl
            )

            pair_weights.append(
                w
            )

    if pair_losses:

        pw = torch.tensor(
            pair_weights,
            device=scores.device,
            dtype=scores.dtype,
        )

        pv = torch.stack(
            pair_losses
        )

        pair_loss = (
            pv
            * pw
        ).sum() / pw.sum()

    else:
        pair_loss = (
            scores.sum()
            * 0.0
        )

    # Tiny scale regularizer, only to stop arbitrary exploding
    # A-H logits. It does not determine ranking.
    centered_s = (
        scores
        - scores.mean()
    )

    reg = (
        1e-4
        * centered_s
        .pow(2)
        .mean()
    )

    sw = state_weight.to(
        scores.device
    )

    total = (
        sw
        * (
            list_loss
            + PAIR_WEIGHT
            * pair_loss
        )
        + reg
    )

    return (
        total,
        list_loss.detach(),
        pair_loss.detach(),
    )


###############################################################################
# EVALUATION — PERMUTATION ENSEMBLE
###############################################################################

def rankdata(a):
    a = np.asarray(
        a,
        dtype=float,
    )

    order = np.argsort(
        a
    )

    ranks = np.empty(
        len(a),
        dtype=float,
    )

    i = 0

    while i < len(a):

        j = i + 1

        while (
            j < len(a)
            and a[order[j]]
            == a[order[i]]
        ):
            j += 1

        r = (
            i + j - 1
        ) / 2.0

        for k in range(
            i,
            j,
        ):
            ranks[
                order[k]
            ] = r

        i = j

    return ranks


def spearman(
    x,
    y,
):
    rx = rankdata(x)
    ry = rankdata(y)

    if (
        np.std(rx) < 1e-12
        or np.std(ry) < 1e-12
    ):
        return None

    return float(
        np.corrcoef(
            rx,
            ry,
        )[0, 1]
    )


@torch.inference_mode()
def score_group(
    g,
    tag,
):
    model.eval()

    n = len(
        g["queries"]
    )

    aggregate = np.zeros(
        n,
        dtype=np.float64,
    )

    for p in range(
        EVAL_PERMS
    ):
        order = list(
            range(n)
        )

        rng = random.Random(
            stable_seed(
                g["state_id"],
                tag,
                p,
            )
        )

        rng.shuffle(
            order
        )

        _, ids, _, _ = make_prompt(
            g,
            order,
        )

        input_ids = torch.tensor(
            [ids],
            dtype=torch.long,
            device="cuda",
        )

        attention = torch.ones_like(
            input_ids
        )

        out = model(
            input_ids=input_ids,
            attention_mask=attention,
        )

        raw = restricted_scores(
            out.logits[
                0,
                -1,
                :
            ],
            n,
        )

        raw = (
            raw
            - raw.mean()
        )

        sd = raw.std(
            unbiased=False
        )

        if float(sd) > 1e-6:
            raw = raw / sd

        raw = (
            raw.detach()
            .cpu()
            .numpy()
        )

        # Map display positions back to original candidate.
        for display_i, original_i in enumerate(
            order
        ):
            aggregate[
                original_i
            ] += raw[
                display_i
            ]

    return (
        aggregate
        / EVAL_PERMS
    )


@torch.inference_mode()
def evaluate(
    groups,
    name,
):
    selected = []
    random_gain = []
    oracle_gain = []

    top1 = []
    top2 = []
    rhos = []

    pair_correct = 0
    pair_total = 0

    for gi, g in enumerate(
        groups,
        1,
    ):

        scores = score_group(
            g,
            name,
        )

        gains = np.asarray(
            g["auc_gains"],
            dtype=float,
        )

        pick = int(
            np.argmax(
                scores
            )
        )

        selected.append(
            float(
                gains[pick]
            )
        )

        random_gain.append(
            float(
                np.mean(gains)
            )
        )

        best = float(
            np.max(gains)
        )

        oracle_gain.append(
            best
        )

        best_set = {
            i
            for i, z
            in enumerate(gains)
            if abs(
                z - best
            ) < 1e-12
        }

        rank = list(
            np.argsort(
                scores
            )[::-1]
        )

        top1.append(
            float(
                pick in best_set
            )
        )

        top2.append(
            float(
                any(
                    x in best_set
                    for x
                    in rank[:2]
                )
            )
        )

        rho = spearman(
            scores,
            gains,
        )

        if rho is not None:
            rhos.append(
                rho
            )

        n = len(gains)

        for i in range(n):
            for j in range(
                i + 1,
                n,
            ):

                d = (
                    gains[i]
                    - gains[j]
                )

                if abs(d) < 1e-12:
                    continue

                pair_correct += int(
                    d
                    * (
                        scores[i]
                        - scores[j]
                    )
                    > 0
                )

                pair_total += 1

        if (
            gi % 100 == 0
            or gi == len(groups)
        ):
            print(
                f"[{name}] "
                f"{gi}/{len(groups)}",
                flush=True,
            )

    ms = float(
        np.mean(selected)
    )

    mr = float(
        np.mean(random_gain)
    )

    mo = float(
        np.mean(oracle_gain)
    )

    result = {
        "states":
            len(groups),

        "mean_selected_gain":
            ms,

        "mean_random_gain":
            mr,

        "mean_oracle_gain":
            mo,

        "mean_regret":
            mo - ms,

        "oracle_top1_hit":
            float(
                np.mean(top1)
            ),

        "oracle_top2_hit":
            float(
                np.mean(top2)
            ),

        "mean_spearman":
            (
                float(
                    np.mean(rhos)
                )
                if rhos
                else None
            ),

        "mean_pair_accuracy":
            (
                pair_correct
                / max(
                    1,
                    pair_total,
                )
            ),

        "oracle_signal_captured":
            (
                (ms - mr)
                / (mo - mr)
                if (
                    mo
                    > mr
                    + 1e-12
                )
                else None
            ),
    }

    print()
    print(
        name,
        json.dumps(
            result,
            indent=2,
        ),
        flush=True,
    )

    return result


###############################################################################
# ZERO-SHOT — WITH THINKING DISABLED + PERMUTATION ENSEMBLE
###############################################################################

history = []

print()
print("=" * 100)
print(
    "EPOCH 0 — CLEAN ZERO-SHOT"
)
print("=" * 100)

base_val = evaluate(
    val_groups,
    "VAL_BASE",
)

history.append({
    "epoch":
        0,

    "train_loss":
        None,

    "val":
        base_val,
})


###############################################################################
# BEST STATE
###############################################################################

def clone_adapter():
    state = (
        get_peft_model_state_dict(
            model
        )
    )

    return {
        k:
            v.detach()
            .cpu()
            .clone()
        for k, v
        in state.items()
    }


best_regret = float(
    base_val[
        "mean_regret"
    ]
)

best_epoch = 0

best_state = clone_adapter()

print(
    f"[BEST] epoch=0 "
    f"regret={best_regret:.6f}",
    flush=True,
)


###############################################################################
# OPTIMIZER
###############################################################################

trainable = [
    p
    for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable,
    lr=LR,
    weight_decay=0.01,
)

possible_micro = (
    len(loader)
    * EPOCHS
)

micro_budget = min(
    possible_micro,
    MAX_MICRO,
)

total_steps = max(
    1,
    math.ceil(
        micro_budget
        / GRAD_ACC
    ),
)

warmup = max(
    10,
    int(
        0.03
        * total_steps
    ),
)

scheduler = (
    get_cosine_schedule_with_warmup(
        optimizer,
        warmup,
        total_steps,
    )
)

print()
print(
    "TRAIN_EXAMPLES_PER_EPOCH =",
    len(loader),
)

print(
    "MICRO_BUDGET =",
    micro_budget,
)

print(
    "OPTIMIZER_STEPS =",
    total_steps,
)

print(
    "WARMUP_STEPS =",
    warmup,
)


###############################################################################
# TRAIN LOOP
###############################################################################

torch.cuda.reset_peak_memory_stats()

optimizer.zero_grad(
    set_to_none=True
)

global_micro = 0
opt_steps = 0
non_improve = 0
train_t0 = time.time()

stop_all = False


for epoch in range(
    1,
    EPOCHS + 1,
):

    model.train()

    total_loss_sum = 0.0
    list_loss_sum = 0.0
    pair_loss_sum = 0.0

    epoch_n = 0
    accum = 0

    for batch in loader:

        batch = {
            k:
                (
                    v.to("cuda")
                    if torch.is_tensor(v)
                    else v
                )
            for k, v
            in batch.items()
        }

        out = model(
            input_ids=
                batch[
                    "input_ids"
                ],

            attention_mask=
                batch[
                    "attention_mask"
                ],
        )

        n_candidates = int(
            len(
                batch[
                    "utilities"
                ]
            )
        )

        scores = restricted_scores(
            out.logits[
                0,
                -1,
                :
            ],
            n_candidates,
        )

        (
            loss,
            list_loss,
            pair_loss,
        ) = selector_loss(
            scores,
            batch[
                "utilities"
            ],
            batch[
                "state_weight"
            ],
        )

        (
            loss
            / GRAD_ACC
        ).backward()

        total_loss_sum += float(
            loss.detach().cpu()
        )

        list_loss_sum += float(
            list_loss.cpu()
        )

        pair_loss_sum += float(
            pair_loss.cpu()
        )

        epoch_n += 1
        global_micro += 1
        accum += 1

        if accum >= GRAD_ACC:

            torch.nn.utils.clip_grad_norm_(
                trainable,
                1.0,
            )

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            opt_steps += 1
            accum = 0

        if (
            global_micro % 50
            == 0
        ):

            hours = (
                time.time()
                - train_t0
            ) / 3600

            print(
                f"epoch={epoch} "
                f"micro={global_micro}/"
                f"{micro_budget} "
                f"opt={opt_steps}/"
                f"{total_steps} "
                f"loss="
                f"{total_loss_sum/epoch_n:.4f} "
                f"list="
                f"{list_loss_sum/epoch_n:.4f} "
                f"pair="
                f"{pair_loss_sum/epoch_n:.4f} "
                f"hours={hours:.2f}",
                flush=True,
            )

        if (
            global_micro
            >= micro_budget
        ):
            stop_all = True
            break

    if accum > 0:

        torch.nn.utils.clip_grad_norm_(
            trainable,
            1.0,
        )

        optimizer.step()
        scheduler.step()

        optimizer.zero_grad(
            set_to_none=True
        )

        opt_steps += 1

    val = evaluate(
        val_groups,
        f"VAL_E{epoch}",
    )

    mean_loss = (
        total_loss_sum
        / max(
            1,
            epoch_n,
        )
    )

    record = {
        "epoch":
            epoch,

        "train_loss":
            mean_loss,

        "list_loss":
            (
                list_loss_sum
                / max(
                    1,
                    epoch_n,
                )
            ),

        "pair_loss":
            (
                pair_loss_sum
                / max(
                    1,
                    epoch_n,
                )
            ),

        "micro_seen":
            global_micro,

        "optimizer_steps":
            opt_steps,

        "val":
            val,

        "elapsed_hours":
            (
                time.time()
                - train_t0
            ) / 3600,
    }

    history.append(
        record
    )

    dump(
        OUT
        / "training_history.json",
        history,
    )

    ckpt = (
        OUT
        / "checkpoints"
        / f"epoch_{epoch:02d}"
    )

    ckpt.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        ckpt
    )

    regret = float(
        val[
            "mean_regret"
        ]
    )

    if (
        regret
        < best_regret
        - 1e-6
    ):

        best_regret = regret
        best_epoch = epoch
        best_state = clone_adapter()
        non_improve = 0

        print(
            f"[BEST] epoch={epoch} "
            f"regret={regret:.6f}",
            flush=True,
        )

    else:

        non_improve += 1

        print(
            "[NO IMPROVEMENT] "
            f"{non_improve}/"
            f"{PATIENCE}",
            flush=True,
        )

    if stop_all:
        print(
            "MICRO BUDGET REACHED",
            flush=True,
        )
        break

    if (
        non_improve
        >= PATIENCE
    ):
        print(
            "EARLY STOP",
            flush=True,
        )
        break


###############################################################################
# RESTORE BEST
###############################################################################

set_peft_model_state_dict(
    model,
    best_state,
)

best_dir = (
    OUT
    / "best_adapter"
)

best_dir.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    best_dir
)

tokenizer.save_pretrained(
    best_dir
)


###############################################################################
# FINAL EVALUATION
###############################################################################

print()
print("=" * 100)
print(
    f"FINAL BEST EPOCH = {best_epoch}"
)
print("=" * 100)

final_val = evaluate(
    val_groups,
    "VAL_FINAL",
)

challenge = evaluate(
    challenge_groups,
    "CHALLENGE_FINAL",
)

runtime = (
    time.time()
    - train_t0
) / 3600

summary = {
    "version":
        "q38_selector_v41",

    "model":
        MODEL,

    "best_epoch":
        best_epoch,

    "objective":
        (
            "restricted A-H soft listwise CE "
            "+ weighted pairwise logistic"
        ),

    "thinking_disabled":
        True,

    "candidate_order_training":
        "random permutations",

    "evaluation":
        (
            f"{EVAL_PERMS}-permutation "
            "score ensemble"
        ),

    "duplicate_denoising":
        "exact same-query utility averaged",

    "train_states":
        len(train_groups),

    "train_examples_per_epoch":
        len(train_ds),

    "global_micro":
        global_micro,

    "optimizer_steps":
        opt_steps,

    "epoch0_zero_shot":
        base_val,

    "val":
        final_val,

    "challenge":
        challenge,

    "runtime_hours":
        runtime,

    "peak_gpu_gib":
        (
            torch.cuda
            .max_memory_allocated()
            / 1024**3
        ),
}

dump(
    OUT
    / "summary.json",
    summary,
)

dump(
    OUT
    / "training_history.json",
    history,
)

print()
print(
    json.dumps(
        summary,
        indent=2,
    )
)

print()
print(
    "Q38 SELECTOR V4.1 COMPLETE"
)
