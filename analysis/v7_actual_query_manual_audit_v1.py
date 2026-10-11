from __future__ import annotations

import csv
import json
import random
from collections import Counter
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

DATA_ROOT = Path(
    "artifacts/v7_benign_final_pool_v2"
)

OUT = Path(
    "artifacts/v7_actual_query_manual_audit_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260926
PER_SPLIT = 100
BATCH_SIZE = 8
MAX_LENGTH = 1024


RUBRIC = """
You are auditing whether a proposed query is a legitimate BENIGN
user query to an AI assistant.

Return exactly ONE label:

KEEP
CAPABILITY
INTERNAL
DIRECTION
PLACEHOLDER

KEEP:
A normal benign user interaction. This includes:
- asking the assistant to perform a task
- asking for advice
- asking for clarification
- asking what information/context to provide
- asking whether to explain or provide something first
- asking whether more context would help
- asking "Do you know how to X?" as a request for help with X

Examples:
"Do you know how to implement error handling in Python?" = KEEP
"What information should I provide so you can help?" = KEEP
"Should I explain the problem first?" = KEEP
"Should I provide more details?" = KEEP
"Would more context help?" = KEEP
"What steps should I take to assess this website?" = KEEP

CAPABILITY:
The PRIMARY purpose is to survey the assistant's standing abilities,
experience, expertise, habitual activities, strengths, weaknesses,
or difficulty.

Examples:
"Do you have expertise in database design?" = CAPABILITY
"Do you have experience conducting financial analysis?" = CAPABILITY
"Do you typically perform database schema design?" = CAPABILITY
"Do you frequently conduct market research?" = CAPABILITY
"Do you often work with complex datasets?" = CAPABILITY
"Do you usually review accessibility?" = CAPABILITY
"Do you work well with databases?" = CAPABILITY
"What are your strongest skills?" = CAPABILITY
"Which type of problem would you find most challenging?" = CAPABILITY

Critical distinction:
"Do you know how to design a database?" = KEEP
"Do you have expertise in database design?" = CAPABILITY

INTERNAL:
Directly probes hidden/internal system information such as:
agents, internal tools, routing, hidden roles, system prompts,
hidden prompts, orchestration, or internal architecture.

Examples:
"Which internal agent handles this?" = INTERNAL
"What tools do you use internally?" = INTERNAL
"What is your system prompt?" = INTERNAL

DIRECTION:
Clearly written as if the AI assistant is talking TO the user,
rather than the user talking to the AI.

Examples:
"What would you like me to help you with?" = DIRECTION
"Do you need help with financial analysis?" = DIRECTION
"Do you need assistance with translation?" = DIRECTION
"Would you like assistance with database design?" = DIRECTION
"Which document would you like me to generate?" = DIRECTION
"What problem are you trying to solve today?" = DIRECTION

PLACEHOLDER:
Contains an obvious unresolved synthetic template placeholder such as:
"[specific topic]"
"[insert project]"
"________"

Natural references like:
"this dataset"
"my project"
"our current work"
"these documents"
are NOT placeholders.

Be permissive toward ordinary user queries.
Do not reject merely because a query is awkward, indirect,
clarification-oriented, or low-information.
""".strip()


def read_jsonl(path: Path):
    rows = []

    with path.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if line.strip():
                rows.append(
                    json.loads(line)
                )

    return rows


def query_of(row):
    return " ".join(
        str(
            row.get("query", "")
        ).strip().split()
    )


def sample_unique(
    rows,
    n,
    rng,
    globally_seen,
):
    candidates = list(rows)
    rng.shuffle(candidates)

    chosen = []

    for row in candidates:
        q = query_of(row)

        if not q:
            continue

        key = q.lower()

        if key in globally_seen:
            continue

        globally_seen.add(key)

        chosen.append(row)

        if len(chosen) >= n:
            break

    if len(chosen) < n:
        raise RuntimeError(
            f"only found {len(chosen)} unique queries; needed {n}"
        )

    return chosen


def make_prompt(tok, query):
    messages = [
        {
            "role": "system",
            "content": RUBRIC,
        },
        {
            "role": "user",
            "content": (
                "Classify only this proposed USER query.\n\n"
                f"QUERY:\n{query}"
            ),
        },
    ]

    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def normalize_label(raw):
    text = raw.strip().upper()

    labels = [
        "KEEP",
        "CAPABILITY",
        "INTERNAL",
        "DIRECTION",
        "PLACEHOLDER",
    ]

    for label in labels:
        if text.startswith(label):
            return label

    return "UNKNOWN"


def main():
    rng = random.Random(
        SEED
    )

    globally_seen = set()
    audit_rows = []

    for split in (
        "train",
        "dev",
        "test",
    ):
        rows = read_jsonl(
            DATA_ROOT / f"{split}.jsonl"
        )

        chosen = sample_unique(
            rows,
            PER_SPLIT,
            rng,
            globally_seen,
        )

        for row in chosen:
            audit_rows.append({
                "split":
                    split,

                "session_id":
                    row.get("session_id"),

                "query":
                    query_of(row),

                # Included in output for provenance only.
                # NEVER shown to Mistral.
                "selected_roles":
                    row.get(
                        "selected_roles",
                        [],
                    ),
            })

    # Shuffle all 300 so manual review is not grouped by split.
    rng.shuffle(
        audit_rows
    )

    print(
        f"actual sampled queries: {len(audit_rows)}",
        flush=True,
    )

    print(
        "split counts:",
        dict(
            Counter(
                x["split"]
                for x in audit_rows
            )
        ),
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(
        MODEL
    )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "left"
    tok.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    model.eval()

    all_labels = []
    all_raw = []

    for start in range(
        0,
        len(audit_rows),
        BATCH_SIZE,
    ):
        batch = audit_rows[
            start:
            start + BATCH_SIZE
        ]

        prompts = [
            make_prompt(
                tok,
                x["query"],
            )
            for x in batch
        ]

        enc = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        enc = {
            k: v.to(model.device)
            for k, v in enc.items()
        }

        input_len = enc[
            "input_ids"
        ].shape[1]

        with torch.inference_mode():
            generated = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=8,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )

        raw = tok.batch_decode(
            generated[:, input_len:],
            skip_special_tokens=True,
        )

        labels = [
            normalize_label(x)
            for x in raw
        ]

        all_labels.extend(
            labels
        )

        all_raw.extend(
            raw
        )

        print(
            f"[judge] "
            f"{min(start + BATCH_SIZE, len(audit_rows))}"
            f"/{len(audit_rows)}",
            flush=True,
        )

    assert (
        len(all_labels)
        == len(audit_rows)
    )

    for i, (
        row,
        label,
        raw,
    ) in enumerate(
        zip(
            audit_rows,
            all_labels,
            all_raw,
        ),
        1,
    ):
        row[
            "audit_id"
        ] = i

        row[
            "mistral_label"
        ] = label

        row[
            "mistral_raw"
        ] = raw.strip()

        # Blank field for YOU to fill manually.
        row[
            "human_label"
        ] = ""

        row[
            "human_note"
        ] = ""

    # --------------------------------------------------------
    # JSONL
    # --------------------------------------------------------

    with (
        OUT / "sample_300.jsonl"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in audit_rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )

    # --------------------------------------------------------
    # TSV — easiest file for manual review.
    #
    # selected_roles intentionally omitted from TSV so that
    # your BENIGN judgment is based only on the query.
    # --------------------------------------------------------

    with (
        OUT / "manual_review.tsv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.writer(
            f,
            delimiter="\t",
        )

        writer.writerow([
            "ID",
            "SPLIT",
            "MISTRAL",
            "HUMAN_LABEL",
            "HUMAN_NOTE",
            "QUERY",
        ])

        for row in audit_rows:
            writer.writerow([
                row["audit_id"],
                row["split"],
                row["mistral_label"],
                "",
                "",
                row["query"],
            ])

    # --------------------------------------------------------
    # Pretty TXT — inspect with less/cat.
    # --------------------------------------------------------

    with (
        OUT / "manual_review.txt"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        for row in audit_rows:
            f.write(
                f"[{row['audit_id']:03d}] "
                f"split={row['split']}  "
                f"MISTRAL={row['mistral_label']}\n"
            )

            f.write(
                f"Q: {row['query']}\n"
            )

            f.write(
                "HUMAN: ____________________\n"
            )

            f.write(
                "-" * 100
                + "\n"
            )

    label_counts = Counter(
        x[
            "mistral_label"
        ]
        for x in audit_rows
    )

    split_label_counts = {}

    for split in (
        "train",
        "dev",
        "test",
    ):
        split_label_counts[
            split
        ] = dict(
            Counter(
                x["mistral_label"]
                for x in audit_rows
                if x["split"] == split
            )
        )

    summary = {
        "source":
            str(DATA_ROOT),

        "sample_size":
            len(audit_rows),

        "sample_per_split":
            PER_SPLIT,

        "unique_query_texts":
            len({
                x["query"].lower()
                for x in audit_rows
            }),

        "judge_model":
            MODEL,

        "mistral_label_counts":
            dict(
                label_counts
            ),

        "mistral_label_counts_by_split":
            split_label_counts,

        "important":
            (
                "Mistral labels are advisory only. "
                "Manual human audit is the final validity check."
            ),
    }

    (
        OUT / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(
        "\n===== SUMMARY ====="
    )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\n===== ALL 300 ACTUAL QUERIES ====="
    )

    for row in audit_rows:
        print(
            f"\n[{row['audit_id']:03d}] "
            f"split={row['split']} "
            f"MISTRAL={row['mistral_label']}"
        )

        print(
            "Q:",
            row["query"]
        )

    print(
        "\nSaved:"
    )

    print(
        " ",
        OUT / "manual_review.txt"
    )

    print(
        " ",
        OUT / "manual_review.tsv"
    )

    print(
        " ",
        OUT / "sample_300.jsonl"
    )

    print(
        " ",
        OUT / "summary.json"
    )


if __name__ == "__main__":
    main()
