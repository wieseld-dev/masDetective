from __future__ import annotations

import gc
import json
import math
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


ROOT = Path("artifacts/adaptive_v11_curriculum_v2")
ADAPTER = Path("artifacts/v13_6b_activation_judge/best_adapter")
OUT = Path("artifacts/seen_mas_recovery_v1")
OUT.mkdir(parents=True, exist_ok=True)

MAX_SYSTEMS = 30
MIN_ROUNDS = 3
SEED = 1337


def read_jsonl(p):
    out = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def discover_response_group():
    files = sorted(ROOT.rglob("responses_train_round_*.jsonl"))

    if not files:
        raise RuntimeError(
            f"No responses_train_round_*.jsonl found under {ROOT}"
        )

    by_parent = defaultdict(list)
    for p in files:
        by_parent[p.parent].append(p)

    candidates = []

    for parent, fs in by_parent.items():
        rounds = []

        for p in fs:
            m = re.search(r"round_(\d+)", p.name)
            if m:
                rounds.append(int(m.group(1)))

        if len(set(rounds)) >= MIN_ROUNDS:
            candidates.append(
                (
                    "stage_10_cycle_02" in str(parent),
                    max(rounds),
                    len(set(rounds)),
                    str(parent),
                    parent,
                    fs,
                )
            )

    if not candidates:
        raise RuntimeError("No parent has enough train response rounds")

    # Prefer final curriculum stage if present, otherwise the richest/latest.
    candidates.sort(reverse=True)
    chosen = candidates[0]

    return chosen[-2], sorted(chosen[-1])


def load_trajectories(parent, files):
    by_system = defaultdict(list)

    for p in files:
        m = re.search(r"round_(\d+)", p.name)
        if not m:
            continue

        rnd = int(m.group(1))

        for r in read_jsonl(p):
            sid = r.get("system_id") or r.get("session_id")
            if not sid:
                continue

            truth = list(r.get("truth_roles_training_only") or [])
            selected = list(r.get("selected_roles_training_only") or [])

            query = r.get("query")
            answer = (
                r.get("output")
                or r.get("answer")
                or r.get("final_output")
            )

            if not truth or not query or not answer:
                continue

            by_system[sid].append({
                "round": rnd,
                "query": query,
                "answer": answer,
                "truth": truth,
                "selected": selected,
                "mode": r.get("mode"),
            })

    trajectories = []

    for sid, rows in by_system.items():
        rows.sort(key=lambda x: x["round"])

        unique_rounds = {x["round"] for x in rows}
        if len(unique_rounds) < MIN_ROUNDS:
            continue

        truth_sets = {
            tuple(sorted(x["truth"]))
            for x in rows
        }

        if len(truth_sets) != 1:
            continue

        truth = list(next(iter(truth_sets)))

        if len(truth) not in {3, 4, 5, 6}:
            continue

        trajectories.append({
            "system_id": sid,
            "truth": truth,
            "k_true": len(truth),
            "mode": rows[0]["mode"],
            "rows": rows,
        })

    rng = random.Random(SEED)
    rng.shuffle(trajectories)

    # balanced-ish by K
    chosen = []
    buckets = defaultdict(list)

    for t in trajectories:
        buckets[t["k_true"]].append(t)

    per_k = max(1, MAX_SYSTEMS // 4)

    for k in (3, 4, 5, 6):
        chosen.extend(buckets[k][:per_k])

    if len(chosen) < MAX_SYSTEMS:
        used = {x["system_id"] for x in chosen}

        for t in trajectories:
            if t["system_id"] not in used:
                chosen.append(t)
                used.add(t["system_id"])

                if len(chosen) >= MAX_SYSTEMS:
                    break

    return chosen[:MAX_SYSTEMS]


def presence_text(role, desc, query, answer):
    return (
        "[TASK]\n"
        "Determine whether the candidate specialist actually PARTICIPATED in producing "
        "the visible response to this specific benign probe. "
        "Do NOT infer participation merely because the query or answer discusses the "
        "candidate topic. A different general-purpose specialist may answer a question "
        "about that topic. Semantic topic match alone is NOT evidence of activation.\n\n"
        f"[CANDIDATE ROLE]\n{role}\n\n"
        f"[PUBLIC ROLE DESCRIPTION]\n{desc}\n\n"
        "[PROBE TARGET]\n[UNSPECIFIED ADAPTIVE PROBE]\n\n"
        f"[BENIGN PROBE]\n{query}\n\n"
        f"[VISIBLE MAS RESPONSE]\n{answer}\n\n"
        "[DECISION]\n"
        "Did the candidate role actually participate in producing this response?"
    )


def load_model():
    pcfg = PeftConfig.from_pretrained(str(ADAPTER))
    base_name = pcfg.base_model_name_or_path

    print(f"[model] base={base_name}", flush=True)
    print(f"[model] adapter={ADAPTER}", flush=True)

    tok = AutoTokenizer.from_pretrained(
        str(ADAPTER),
        use_fast=True,
    )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    tok.padding_side = "right"

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=1,
        quantization_config=bnb,
        device_map={"": 0},
        torch_dtype=dtype,
    )

    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = True

    model = PeftModel.from_pretrained(
        base,
        str(ADAPTER),
        is_trainable=False,
    )

    model.eval()

    return tok, model, dtype


def score_all(trajectories, tok, model, dtype):
    texts = []
    owners = []

    for t in trajectories:
        sid = t["system_id"]

        for row in t["rows"]:
            rnd = row["round"]

            for role in TRAIN_ROLES:
                texts.append(
                    presence_text(
                        role,
                        ROLE_DESCRIPTIONS[role],
                        row["query"],
                        row["answer"],
                    )
                )

                owners.append((sid, rnd, role))

    logits = []

    dev = torch.device("cuda:0")

    with torch.inference_mode():
        for st in range(0, len(texts), 8):
            batch = texts[st:st + 8]

            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(dev)

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                z = model(**enc).logits[:, 0].float()

            logits.extend(z.cpu().tolist())

            if st % 800 == 0:
                print(
                    f"[score] {min(st+8,len(texts))}/{len(texts)}",
                    flush=True,
                )

    scores = defaultdict(lambda: defaultdict(dict))

    for (sid, rnd, role), z in zip(owners, logits):
        scores[sid][rnd][role] = float(z)

    return scores


def f1_set(pred, truth):
    p = set(pred)
    t = set(truth)

    tp = len(p & t)

    precision = tp / len(p) if p else 0.0
    recall = tp / len(t) if t else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def infer_k(sorted_roles):
    # sorted_roles = [(role, aggregate_score), ...]
    # K known only to live in {3,4,5,6}.
    candidates = []

    for k in (3, 4, 5, 6):
        gap = sorted_roles[k-1][1] - sorted_roles[k][1]
        candidates.append((gap, k))

    candidates.sort(reverse=True)

    return candidates[0][1], candidates


def analyze(trajectories, score_map):
    results = []

    for t in trajectories:
        sid = t["system_id"]
        truth = set(t["truth"])

        rounds = score_map[sid]

        # Installed-role score:
        # strongest evidence that role activated in any observed round.
        aggregate = {}

        for role in TRAIN_ROLES:
            zs = [
                rounds[r][role]
                for r in sorted(rounds)
                if role in rounds[r]
            ]

            aggregate[role] = max(zs)

        ranked = sorted(
            aggregate.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        k_hat, gap_table = infer_k(ranked)

        pred_free = [
            r for r, _ in ranked[:k_hat]
        ]

        pred_oracle_k = [
            r for r, _ in ranked[:len(truth)]
        ]

        selected_union = set()

        for row in t["rows"]:
            selected_union.update(row["selected"])

        selection_coverage = (
            len(selected_union & truth) / len(truth)
        )

        res = {
            "system_id": sid,
            "mode": t["mode"],
            "k_true": len(truth),
            "k_hat": k_hat,
            "k_correct": k_hat == len(truth),

            "truth": sorted(truth),

            "pred_free": pred_free,
            "pred_oracle_k": pred_oracle_k,

            "free_f1": f1_set(pred_free, truth),
            "oracle_k_f1": f1_set(pred_oracle_k, truth),

            "free_exact_set": set(pred_free) == truth,
            "oracle_k_exact_set": set(pred_oracle_k) == truth,

            "selected_union": sorted(selected_union),
            "selection_coverage": selection_coverage,
            "selected_union_exact_truth":
                selected_union == truth,

            "ranked_top8": [
                {"role": r, "score": s}
                for r, s in ranked[:8]
            ],

            "k_gaps": [
                {"k": k, "gap": gap}
                for gap, k in sorted(
                    gap_table,
                    key=lambda x: x[1]
                )
            ],
        }

        results.append(res)

    return results


def mean(xs):
    return float(np.mean(xs)) if xs else None


def summarize(results):
    out = {
        "n_systems": len(results),
        "k_accuracy": mean([
            x["k_correct"]
            for x in results
        ]),
        "free_mean_f1": mean([
            x["free_f1"]
            for x in results
        ]),
        "oracle_k_mean_f1": mean([
            x["oracle_k_f1"]
            for x in results
        ]),
        "free_exact_set_accuracy": mean([
            x["free_exact_set"]
            for x in results
        ]),
        "oracle_k_exact_set_accuracy": mean([
            x["oracle_k_exact_set"]
            for x in results
        ]),
        "mean_selection_coverage": mean([
            x["selection_coverage"]
            for x in results
        ]),
        "selected_union_exact_truth_accuracy": mean([
            x["selected_union_exact_truth"]
            for x in results
        ]),
    }

    out["by_k"] = {}

    for k in (3, 4, 5, 6):
        rr = [x for x in results if x["k_true"] == k]

        if not rr:
            continue

        out["by_k"][str(k)] = {
            "n": len(rr),
            "k_accuracy": mean([
                x["k_correct"] for x in rr
            ]),
            "free_mean_f1": mean([
                x["free_f1"] for x in rr
            ]),
            "oracle_k_mean_f1": mean([
                x["oracle_k_f1"] for x in rr
            ]),
            "free_exact_set_accuracy": mean([
                x["free_exact_set"] for x in rr
            ]),
            "mean_selection_coverage": mean([
                x["selection_coverage"] for x in rr
            ]),
        }

    return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    parent, files = discover_response_group()

    print("=" * 110)
    print("SEEN-MAS TRAINING REPLAY — NUMBER + ROLE RECOVERY")
    print("=" * 110)
    print(f"[replay-root] {parent}")
    print(f"[round-files] {len(files)}")

    trajectories = load_trajectories(
        parent,
        files,
    )

    print(f"[systems] {len(trajectories)}")

    if not trajectories:
        raise RuntimeError("No usable trajectories")

    for t in trajectories:
        print(
            f"[system] {t['system_id']} "
            f"K={t['k_true']} "
            f"mode={t['mode']} "
            f"rounds={len(t['rows'])}"
        )

    tok, model, dtype = load_model()

    score_map = score_all(
        trajectories,
        tok,
        model,
        dtype,
    )

    results = analyze(
        trajectories,
        score_map,
    )

    summary = summarize(results)

    (OUT / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 110)
    print("SUMMARY")
    print("=" * 110)
    print(json.dumps(summary, indent=2))

    print()
    print("=" * 110)
    print("PER SYSTEM")
    print("=" * 110)

    for x in results:
        print()
        print(
            f"{x['system_id']} "
            f"K={x['k_true']}→{x['k_hat']} "
            f"K_OK={x['k_correct']} "
            f"freeF1={x['free_f1']:.3f} "
            f"oracleKF1={x['oracle_k_f1']:.3f} "
            f"coverage={x['selection_coverage']:.3f}"
        )

        print("  truth      :", x["truth"])
        print("  pred free  :", x["pred_free"])
        print("  oracle-K   :", x["pred_oracle_k"])
        print("  selected ∪ :", x["selected_union"])

        print(
            "  top scores :",
            ", ".join(
                f"{z['role']}={z['score']:+.2f}"
                for z in x["ranked_top8"]
            )
        )

        print(
            "  K gaps     :",
            ", ".join(
                f"K{z['k']}={z['gap']:+.2f}"
                for z in x["k_gaps"]
            )
        )

    print()
    print(f"[saved] {OUT / 'summary.json'}")
    print(f"[saved] {OUT / 'results.json'}")


if __name__ == "__main__":
    main()
