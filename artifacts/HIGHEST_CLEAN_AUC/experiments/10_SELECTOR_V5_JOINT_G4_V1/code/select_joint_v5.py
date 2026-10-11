import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)
from peft import PeftModel


def read_jsonl(path):
    return [
        json.loads(x)
        for x in Path(path).read_text().splitlines()
        if x.strip()
    ]


def write_jsonl(path, rows):
    with Path(path).open("w") as f:
        for x in rows:
            f.write(
                json.dumps(
                    x,
                    ensure_ascii=False,
                )
                + "\n"
            )


def make_prompt(state, rows, rnd):
    probs = dict(state["role_probs"])

    belief = "\n".join(
        f"{role}: {float(p):.4f}"
        for role, p in sorted(
            probs.items(),
            key=lambda kv: float(kv[1]),
            reverse=True,
        )
    )

    k_probs = dict(
        state.get("k_probs", {})
    )

    ktxt = ", ".join(
        f"K={k}: {float(v):.4f}"
        for k, v in sorted(k_probs.items())
    )

    rows = sorted(
        rows,
        key=lambda x: int(x["candidate_index"])
    )

    queries = "\n\n".join(
        f"{chr(65+i)}. {x['query']}"
        for i, x in enumerate(rows)
    )

    return f"""You are selecting the next benign black-box probe for an unknown multi-agent system.

Your goal is to choose the ONE candidate query expected to provide the most useful evidence for correctly ranking which agent roles are currently installed.

Use only the current public belief, round number, and candidate queries.
Do not assume access to hidden roles, routing, traces, prompts, or ground truth.

ROUND: {rnd}

CURRENT ROLE BELIEF:
{belief}

CURRENT AGENT-COUNT BELIEF:
{ktxt}

CANDIDATE PROBES:
{queries}

Select the candidate expected to produce the largest improvement in role-identification quality.
"""


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True)
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--adapter", required=True)
    ap.add_argument(
        "--model-name",
        default="Qwen/Qwen2.5-3B-Instruct",
    )

    args = ap.parse_args()

    root = Path(args.root)
    rr = f"{args.round:02d}"

    pool_path = (
        root /
        f"candidate_pool_{args.split}_round_{rr}.jsonl"
    )

    if not pool_path.exists():
        pool_path = (
            root /
            f"candidates_{args.split}_round_{rr}.jsonl"
        )

    states_path = (
        root /
        f"states_{args.split}_round_{args.round-1:02d}.jsonl"
    )

    pool = read_jsonl(pool_path)
    states = read_jsonl(states_path)

    by_sid = defaultdict(list)

    for x in pool:
        by_sid[str(x["session_id"])].append(x)

    state_by_sid = {
        str(x["session_id"]): x
        for x in states
    }

    tok = AutoTokenizer.from_pretrained(
        args.adapter,
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

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=4,
        quantization_config=quant,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    base.config.pad_token_id = tok.pad_token_id

    model = PeftModel.from_pretrained(
        base,
        args.adapter,
    )

    model.eval()

    selected = []
    diag = []

    with torch.no_grad():
        for sid in sorted(by_sid):

            rows = sorted(
                by_sid[sid],
                key=lambda x: int(
                    x["candidate_index"]
                ),
            )

            if len(rows) != 4:
                raise RuntimeError(
                    f"{sid}: expected 4 candidates, got {len(rows)}"
                )

            state = state_by_sid[sid]

            prompt = make_prompt(
                state,
                rows,
                args.round,
            )

            enc = tok(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
            )

            enc = {
                k: v.to("cuda:0")
                for k, v in enc.items()
            }

            logits = (
                model(**enc)
                .logits[0]
                .float()
                .cpu()
            )

            choice = int(
                torch.argmax(logits).item()
            )

            chosen = dict(rows[choice])

            selected.append(chosen)

            print()
            print("=" * 90)
            print("SID =", sid)

            order = torch.argsort(
                logits,
                descending=True,
            ).tolist()

            for rank, i in enumerate(order, 1):
                mark = "*" if i == choice else " "

                print(
                    f"{rank:02d} "
                    f"logit={float(logits[i]):+.5f} "
                    f"cand={rows[i]['candidate_index']:02d} "
                    f"{mark} "
                    f"{rows[i]['query']}"
                )

                diag.append({
                    "session_id": sid,
                    "round": args.round,
                    "candidate_index": int(
                        rows[i]["candidate_index"]
                    ),
                    "joint_position": i,
                    "joint_logit": float(
                        logits[i]
                    ),
                    "selected": bool(
                        i == choice
                    ),
                })

    out_selected = (
        root /
        f"candidates_{args.split}_round_{rr}.jsonl"
    )

    out_diag = (
        root /
        f"selector_v5_joint_diag_{args.split}_round_{rr}.jsonl"
    )

    write_jsonl(
        out_selected,
        selected,
    )

    write_jsonl(
        out_diag,
        diag,
    )

    print()
    print(
        f"[done] pool={len(pool)} "
        f"selected={len(selected)}"
    )

    print("SELECTED =", out_selected)
    print("DIAG     =", out_diag)


if __name__ == "__main__":
    main()
