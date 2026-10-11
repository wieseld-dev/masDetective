from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from peft import (
    PeftConfig,
    PeftModel,
)

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


LETTERS = list("ABCDEFGH")

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
    path = Path(path)
    return [
        json.loads(x)
        for x in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp = path.with_suffix(
        path.suffix + ".tmp"
    )

    with tmp.open(
        "w",
        encoding="utf-8",
    ) as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )

    tmp.replace(path)


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


def history_text(state):
    history = list(
        state.get("history", []) or []
    )

    if not history:
        return "- no previous public interaction"

    blocks = []

    for h in history[-2:]:
        q = " ".join(
            str(
                h.get("query", "")
            ).split()
        )

        a = " ".join(
            str(
                h.get(
                    "answer",
                    h.get("output", ""),
                )
            ).split()
        )

        if len(a) > 320:
            a = a[:320] + "..."

        blocks.append(
            f"Q: {q}\n"
            f"VISIBLE ANSWER: {a}"
        )

    return "\n\n".join(blocks)


def role_block(state):
    rp = {
        str(k): float(v)
        for k, v
        in dict(
            state["role_probs"]
        ).items()
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
    state,
    candidates,
    order,
    history_chars,
    query_chars,
):
    hist = " ".join(
        history_text(
            state
        ).split()
    )

    if len(hist) > history_chars:
        hist = (
            hist[:history_chars]
            + "..."
        )

    lines = []

    for display_i, original_i in enumerate(
        order
    ):
        q = " ".join(
            str(
                candidates[
                    original_i
                ]["query"]
            ).split()
        )

        if len(q) > query_chars:
            q = (
                q[:query_chars]
                + "..."
            )

        lines.append(
            f"{LETTERS[display_i]}. {q}"
        )

    return (
        "CURRENT ROLE BELIEF\n"
        + role_block(state)
        + "\n\n"
        + "RECENT PUBLIC HISTORY\n"
        + hist
        + "\n\n"
        + "CANDIDATES\n"
        + "\n".join(lines)
        + "\n\n"
        + "Rank these probes comparatively. "
          "Prefer the probe most likely to resolve current ambiguity, "
          "distinguish competing role configurations, or elicit "
          "distinctive behavior visible in the final response. "
          "Do not prefer a probe merely because its topic resembles "
          "a likely role."
    )


def apply_chat(
    tokenizer,
    user,
):
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


def make_prompt(
    tokenizer,
    state,
    candidates,
    order,
    max_len,
):
    # Same explicit compression strategy as V4.1 training.
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
            state,
            candidates,
            order,
            hchars,
            qchars,
        )

        text = apply_chat(
            tokenizer,
            user,
        )

        ids = tokenizer.encode(
            text,
            add_special_tokens=False,
        )

        if len(ids) <= max_len:
            return ids

    raise RuntimeError(
        "Unable to fit selector prompt "
        f"inside max_len={max_len}"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--root",
        required=True,
    )

    ap.add_argument(
        "--round",
        required=True,
        type=int,
    )

    ap.add_argument(
        "--split",
        default="dev",
    )

    ap.add_argument(
        "--adapter",
        default=(
            "artifacts/"
            "q38_selector_v41/"
            "best_adapter"
        ),
    )

    ap.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    ap.add_argument(
        "--eval-perms",
        type=int,
        default=3,
    )

    args = ap.parse_args()

    root = Path(
        args.root
    )

    adapter = Path(
        args.adapter
    )

    cand_path = (
        root
        / f"candidates_{args.split}_round_{args.round:02d}.jsonl"
    )

    pool_path = (
        root
        / f"candidate_pool_{args.split}_round_{args.round:02d}.jsonl"
    )

    state_path = (
        root
        / f"states_{args.split}_round_{args.round-1:02d}.jsonl"
    )

    diag_path = (
        root
        / f"selector_diag_{args.split}_round_{args.round:02d}.jsonl"
    )

    if not adapter.exists():
        raise RuntimeError(
            f"Missing selector adapter: {adapter}"
        )

    candidates = read_jsonl(
        cand_path
    )

    states = {
        str(x["session_id"]): x
        for x in read_jsonl(
            state_path
        )
    }

    # Preserve all generated candidates before
    # replacing candidates_* with one chosen probe/system.
    shutil.copyfile(
        cand_path,
        pool_path,
    )

    grouped = defaultdict(list)

    for c in candidates:
        grouped[
            str(c["session_id"])
        ].append(c)

    cfg = PeftConfig.from_pretrained(
        str(adapter)
    )

    base_name = (
        cfg.base_model_name_or_path
        or "Qwen/Qwen3.8-27B"
    )

    print(
        "SELECTOR ADAPTER =",
        adapter,
        flush=True,
    )

    print(
        "SELECTOR BASE    =",
        base_name,
        flush=True,
    )

    processor = (
        AutoProcessor
        .from_pretrained(
            base_name
        )
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
                found = int(
                    ids[0]
                )
                break

        if found is None:
            raise RuntimeError(
                f"No one-token label for {letter}"
            )

        label_ids[
            letter
        ] = found

    print(
        "LABEL IDS =",
        label_ids,
        flush=True,
    )

    print(
        "Loading Qwen3.8-27B...",
        flush=True,
    )

    base = (
        AutoModelForMultimodalLM
        .from_pretrained(
            base_name,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
    )

    model = (
        PeftModel
        .from_pretrained(
            base,
            str(adapter),
            is_trainable=False,
        )
    )

    model.eval()

    chosen_rows = []
    diag_rows = []

    for sid in sorted(
        states
    ):
        if sid not in grouped:
            raise RuntimeError(
                f"No candidates for {sid}"
            )

        state = states[sid]

        viable = [
            dict(x)
            for x in sorted(
                grouped[sid],
                key=lambda z:
                    int(
                        z["candidate_index"]
                    ),
            )
            if x.get(
                "valid",
                True,
            )
        ]

        n = len(
            viable
        )

        if n < 2:
            raise RuntimeError(
                f"{sid}: only {n} valid candidates"
            )

        if n > len(
            LETTERS
        ):
            raise RuntimeError(
                f"{sid}: {n} candidates > "
                f"{len(LETTERS)} labels"
            )

        aggregate = np.zeros(
            n,
            dtype=np.float64,
        )

        per_perm = []

        for p in range(
            args.eval_perms
        ):

            order = list(
                range(n)
            )

            rng = random.Random(
                stable_seed(
                    sid,
                    args.round,
                    "ONLINE_FIXED",
                    p,
                )
            )

            rng.shuffle(
                order
            )

            ids = make_prompt(
                tokenizer,
                state,
                viable,
                order,
                args.max_length,
            )

            input_ids = torch.tensor(
                [ids],
                dtype=torch.long,
                device="cuda:0",
            )

            attention_mask = (
                torch.ones_like(
                    input_ids
                )
            )

            with torch.inference_mode():
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            logits = (
                out.logits[
                    0,
                    -1,
                    :
                ]
                .float()
            )

            display_scores = torch.stack([
                logits[
                    label_ids[
                        LETTERS[j]
                    ]
                ]
                for j in range(n)
            ])

            display_scores = (
                display_scores
                - display_scores.mean()
            )

            sd = display_scores.std(
                unbiased=False
            )

            if float(
                sd.detach().cpu()
            ) > 1e-6:
                display_scores = (
                    display_scores
                    / sd
                )

            ds = (
                display_scores
                .detach()
                .cpu()
                .numpy()
            )

            original_scores = np.zeros(
                n,
                dtype=np.float64,
            )

            for display_i, original_i in enumerate(
                order
            ):
                original_scores[
                    original_i
                ] = float(
                    ds[
                        display_i
                    ]
                )

            aggregate += (
                original_scores
            )

            per_perm.append(
                original_scores.tolist()
            )

        aggregate /= float(
            args.eval_perms
        )

        best_local = int(
            np.argmax(
                aggregate
            )
        )

        chosen = dict(
            viable[
                best_local
            ]
        )

        chosen[
            "selector_score"
        ] = float(
            aggregate[
                best_local
            ]
        )

        chosen[
            "selector_model"
        ] = (
            "q38_selector_v41"
        )

        chosen_rows.append(
            chosen
        )

        ranking = sorted(
            range(n),
            key=lambda i:
                -float(
                    aggregate[i]
                ),
        )

        print()
        print(
            "=" * 100
        )

        print(
            f"ROUND={args.round} "
            f"SID={sid} "
            f"valid={n}"
        )

        for rank, i in enumerate(
            ranking,
            1,
        ):
            selected = (
                i == best_local
            )

            c = viable[i]

            print(
                f"{rank:02d} "
                f"score={aggregate[i]:+.5f} "
                f"cand={int(c['candidate_index']):02d} "
                f"{'* SELECTED' if selected else ''}"
            )

            print(
                "   ",
                c["query"],
            )

            d = dict(c)

            d[
                "selector_score"
            ] = float(
                aggregate[i]
            )

            d[
                "selector_rank"
            ] = rank

            d[
                "selector_selected"
            ] = selected

            d[
                "selector_model"
            ] = (
                "q38_selector_v41"
            )

            d[
                "selector_permutation_scores"
            ] = [
                float(z[i])
                for z in per_perm
            ]

            diag_rows.append(
                d
            )

    if len(
        chosen_rows
    ) != len(
        states
    ):
        raise RuntimeError(
            f"selected {len(chosen_rows)} "
            f"for {len(states)} systems"
        )

    write_jsonl(
        diag_path,
        diag_rows,
    )

    write_jsonl(
        cand_path,
        chosen_rows,
    )

    print()
    print(
        f"[done] pool={len(candidates)} "
        f"selected={len(chosen_rows)}"
    )

    print(
        "POOL     =",
        pool_path,
    )

    print(
        "SELECTED =",
        cand_path,
    )

    print(
        "DIAG     =",
        diag_path,
    )


if __name__ == "__main__":
    main()
