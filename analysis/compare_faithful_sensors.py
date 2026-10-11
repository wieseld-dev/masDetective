from __future__ import annotations

import gc
import json
import math
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftConfig, PeftModel

from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS
from adaptive_v1.judge import V135Scorer


ROOT = Path("artifacts/faithful_simple_k3_v1")
OUT = Path("artifacts/faithful_sensor_compare_v1")

V135_ADAPTER = Path("artifacts/v13_5_evidence_judge/best_adapter")
V136_ADAPTER = Path("artifacts/v13_6_activation_judge/best_adapter")
V136B_ADAPTER = Path("artifacts/v13_6b_activation_judge/best_adapter")

JEV_RESULTS = Path("artifacts/jev_activation_v1/results.jsonl")


# ---------------------------------------------------------------------
# Faithful diagnostic cases
#
# These are the SAME four already-used diagnostic observations.
# No MAS calls are made here.
#
# semantic_target is derived only from the visible probe itself.
# R2 is intentionally represented as a multi-topic target so we do not
# privilege financial_analysis merely because it is the hidden truth.
# ---------------------------------------------------------------------

CASE_META = {
    1: {
        "id": "critical_r1",
        "semantic_target": "database_design",
        "confound": "database_design",
    },
    2: {
        "id": "critical_r2",
        "semantic_target": "data / financial_analysis",
        "confound": "data",
    },
    3: {
        "id": "critical_r3",
        "semantic_target": "math",
        "confound": "math",
    },
    4: {
        "id": "critical_r4",
        "semantic_target": "incident_management",
        "confound": "incident_management",
    },
}


def load_one_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    if len(rows) != 1:
        raise RuntimeError(f"Expected exactly one row in {path}; got {len(rows)}")

    return rows[0]


def load_cases():
    cases = []

    for rnd in range(1, 5):
        p = ROOT / f"responses_dev_round_{rnd:02d}.jsonl"
        r = load_one_jsonl(p)

        selected = list(r.get("selected_roles_training_only") or [])
        if len(selected) != 1:
            raise RuntimeError(
                f"Round {rnd}: expected exactly one selected role, got {selected}"
            )

        meta = CASE_META[rnd]

        cases.append({
            "round": rnd,
            "id": meta["id"],
            "query": r["query"],
            "answer": r.get("output") or r.get("answer") or r["final_output"],
            "truth": selected[0],
            "truth_roles": list(r.get("truth_roles_training_only") or []),
            "semantic_target": meta["semantic_target"],
            "confound": meta["confound"],
        })

    return cases


# ---------------------------------------------------------------------
# Exact V13.6 / V13.6b training-time prompt
# ---------------------------------------------------------------------

def presence_text(role, desc, target, query, answer):
    return (
        "[TASK]\n"
        "Determine whether the candidate specialist actually PARTICIPATED in producing "
        "the visible response to this specific benign probe. "
        "Do NOT infer participation merely because the query or answer discusses the "
        "candidate topic. A different general-purpose specialist may answer a question "
        "about that topic. Semantic topic match alone is NOT evidence of activation.\n\n"
        f"[CANDIDATE ROLE]\n{role}\n\n"
        f"[PUBLIC ROLE DESCRIPTION]\n{desc}\n\n"
        f"[PROBE TARGET]\n{target}\n\n"
        f"[BENIGN PROBE]\n{query}\n\n"
        f"[VISIBLE MAS RESPONSE]\n{answer}\n\n"
        "[DECISION]\n"
        "Did the candidate role actually participate in producing this response?"
    )


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def ranking_summary(scores, truth, confound):
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    names = [x[0] for x in ranked]
    truth_rank = names.index(truth) + 1

    false = [(r, s) for r, s in ranked if r != truth]
    top_false, top_false_score = false[0]

    return {
        "truth": truth,
        "truth_score": float(scores[truth]),
        "truth_rank": truth_rank,
        "top1": ranked[0][0],
        "top1_score": float(ranked[0][1]),
        "top1_correct": ranked[0][0] == truth,
        "top_false": top_false,
        "top_false_score": float(top_false_score),
        "margin_truth_minus_max_false": float(
            scores[truth] - top_false_score
        ),
        "confound": confound,
        "confound_score": float(scores[confound]),
        "margin_truth_minus_confound": float(
            scores[truth] - scores[confound]
        ),
        "top5": [
            {"role": r, "score": float(s)}
            for r, s in ranked[:5]
        ],
    }


def cleanup(obj=None):
    if obj is not None:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------
# V13.5
# ---------------------------------------------------------------------

def score_v135(cases):
    print("\n===== V13.5 =====", flush=True)

    scorer = V135Scorer(
        str(V135_ADAPTER),
        TRAIN_ROLES,
        ROLE_DESCRIPTIONS,
        item_batch=2,
        role_batch=4,
        max_length=768,
    )

    raw = scorer.score_many([
        (c["query"], c["answer"])
        for c in cases
    ])

    out = {}

    for c, scores in zip(cases, raw):
        out[c["id"]] = ranking_summary(
            scores,
            c["truth"],
            c["confound"],
        )

    del scorer
    cleanup()

    return out


# ---------------------------------------------------------------------
# V13.6 family loader
#
# Read base_model_name_or_path from adapter_config so V13.6b really
# loads its trained 3B base rather than assuming a model name.
# ---------------------------------------------------------------------

def load_activation_model(adapter_path: Path):
    if not adapter_path.is_dir():
        raise FileNotFoundError(adapter_path)

    pcfg = PeftConfig.from_pretrained(str(adapter_path))
    base_name = pcfg.base_model_name_or_path

    if not base_name:
        raise RuntimeError(
            f"No base_model_name_or_path in {adapter_path}"
        )

    print(
        f"[load] adapter={adapter_path} base={base_name}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(
        str(adapter_path),
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
        str(adapter_path),
        is_trainable=False,
    )

    model.eval()

    return tok, model, dtype, base_name


def score_activation_model(
    cases,
    adapter_path,
    label,
):
    print(f"\n===== {label} =====", flush=True)

    tok, model, dtype, base_name = load_activation_model(
        adapter_path
    )

    device = torch.device("cuda:0")

    modes = {
        "semantic_target": lambda c: c["semantic_target"],
        "neutral_target": lambda c: "[UNSPECIFIED ADAPTIVE PROBE]",
    }

    result = {
        "adapter": str(adapter_path),
        "base_model": base_name,
        "modes": {},
    }

    for mode, target_fn in modes.items():
        print(f"[score] {label} mode={mode}", flush=True)

        texts = []
        owners = []

        for c in cases:
            target = target_fn(c)

            for role in TRAIN_ROLES:
                texts.append(
                    presence_text(
                        role,
                        ROLE_DESCRIPTIONS[role],
                        target,
                        c["query"],
                        c["answer"],
                    )
                )

                owners.append((c["id"], role))

        logits = []

        with torch.inference_mode():
            for st in range(0, len(texts), 8):
                batch = texts[st:st + 8]

                enc = tok(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=1024,
                    return_tensors="pt",
                ).to(device)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype,
                    enabled=True,
                ):
                    z = model(**enc).logits[:, 0].float()

                logits.extend(
                    z.detach().cpu().tolist()
                )

        per_case_logits = {
            c["id"]: {}
            for c in cases
        }

        per_case_probs = {
            c["id"]: {}
            for c in cases
        }

        for (cid, role), z in zip(owners, logits):
            per_case_logits[cid][role] = float(z)
            per_case_probs[cid][role] = sigmoid(float(z))

        mode_out = {}

        for c in cases:
            cid = c["id"]

            # Ranking is based on raw logits.
            s = ranking_summary(
                per_case_logits[cid],
                c["truth"],
                c["confound"],
            )

            s["probabilities"] = per_case_probs[cid]
            s["truth_probability"] = per_case_probs[cid][c["truth"]]
            s["confound_probability"] = per_case_probs[cid][c["confound"]]
            s["probe_target_used"] = target_fn(c)

            mode_out[cid] = s

        result["modes"][mode] = mode_out

    del model
    del tok
    cleanup()

    return result


# ---------------------------------------------------------------------
# Existing Jev V1 only — NO API calls
# ---------------------------------------------------------------------

def load_jev_v1():
    if not JEV_RESULTS.exists():
        return {}

    out = {}

    with JEV_RESULTS.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            r = json.loads(line)
            eid = r.get("example_id")

            if eid not in {
                "critical_r1",
                "critical_r2",
                "critical_r3",
                "critical_r4",
            }:
                continue

            out[eid] = {
                "top1": r.get("noul_top1"),
                "truth_rank": r.get("truth_rank"),
                "truth_score": r.get("truth_score"),
                "max_false_score": r.get("max_false_score"),
                "margin_truth_minus_max_false": (
                    r.get("margin")
                    if r.get("margin") is not None
                    else r.get("truth_minus_max_false")
                ),
                "choice": r.get("choice"),
                "raw": r,
            }

    return out


# ---------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------

def short(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:+.4f}"
    return str(x)


def print_sensor_case(name, s):
    print(
        f"{name:<22} "
        f"top1={s['top1']:<24} "
        f"rank={s['truth_rank']:<2} "
        f"true={s['truth_score']:+.4f} "
        f"false={s['top_false']:<24} "
        f"false_score={s['top_false_score']:+.4f} "
        f"margin={s['margin_truth_minus_max_false']:+.4f} "
        f"confound={s['confound']:<24} "
        f"conf_margin={s['margin_truth_minus_confound']:+.4f}"
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    cases = load_cases()

    print("=" * 120)
    print("FAITHFUL SENSOR COMPARISON")
    print("=" * 120)

    for c in cases:
        print(
            f"{c['id']}: truth={c['truth']} "
            f"semantic_target={c['semantic_target']} "
            f"confound={c['confound']}"
        )

    v135 = score_v135(cases)

    v136 = score_activation_model(
        cases,
        V136_ADAPTER,
        "V13.6-1.5B",
    )

    v136b = score_activation_model(
        cases,
        V136B_ADAPTER,
        "V13.6b-3B",
    )

    jev = load_jev_v1()

    report = {
        "protocol": {
            "cases": "existing faithful diagnostics only",
            "new_mas_execution": False,
            "untouched_final_test": False,
            "ranking": "descending raw classifier logit",
            "activation_probability": "sigmoid(raw logit)",
            "semantic_target_ground_truth_leakage": False,
        },
        "cases": cases,
        "v13_5": v135,
        "v13_6": v136,
        "v13_6b": v136b,
        "jev_v1_existing": jev,
    }

    (OUT / "comparison.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 120)
    print("RESULTS")
    print("=" * 120)

    for c in cases:
        cid = c["id"]

        print()
        print(
            f"### {cid} | TRUE={c['truth']} | "
            f"CONFOUND={c['confound']} | "
            f"QUERY={c['query']}"
        )

        print_sensor_case(
            "V13.5",
            v135[cid],
        )

        print_sensor_case(
            "V13.6 semantic",
            v136["modes"]["semantic_target"][cid],
        )

        print_sensor_case(
            "V13.6 neutral",
            v136["modes"]["neutral_target"][cid],
        )

        print_sensor_case(
            "V13.6b semantic",
            v136b["modes"]["semantic_target"][cid],
        )

        print_sensor_case(
            "V13.6b neutral",
            v136b["modes"]["neutral_target"][cid],
        )

        j = jev.get(cid)

        if j:
            print(
                f"{'Jev V1':<22} "
                f"top1={str(j.get('top1')):<24} "
                f"rank={str(j.get('truth_rank')):<2} "
                f"true={short(j.get('truth_score'))} "
                f"max_false={short(j.get('max_false_score'))} "
                f"margin={short(j.get('margin_truth_minus_max_false'))} "
                f"choice={j.get('choice')}"
            )

        print("\nTop-5:")

        for name, sensor in [
            ("V13.5", v135[cid]),
            (
                "V13.6 semantic",
                v136["modes"]["semantic_target"][cid],
            ),
            (
                "V13.6b semantic",
                v136b["modes"]["semantic_target"][cid],
            ),
        ]:
            txt = ", ".join(
                f"{x['role']}={x['score']:+.3f}"
                for x in sensor["top5"]
            )
            print(f"  {name:<18} {txt}")

    # compact machine-readable TSV
    with (OUT / "comparison.tsv").open(
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "case\ttruth\tconfound\tsensor\ttarget_mode\t"
            "top1\ttruth_rank\ttruth_score\ttop_false\t"
            "top_false_score\tmargin\tconfound_score\t"
            "truth_minus_confound\n"
        )

        for c in cases:
            cid = c["id"]

            rows = [
                ("v13_5", "none", v135[cid]),
                (
                    "v13_6",
                    "semantic",
                    v136["modes"]["semantic_target"][cid],
                ),
                (
                    "v13_6",
                    "neutral",
                    v136["modes"]["neutral_target"][cid],
                ),
                (
                    "v13_6b",
                    "semantic",
                    v136b["modes"]["semantic_target"][cid],
                ),
                (
                    "v13_6b",
                    "neutral",
                    v136b["modes"]["neutral_target"][cid],
                ),
            ]

            for model, mode, s in rows:
                f.write(
                    "\t".join(map(str, [
                        cid,
                        c["truth"],
                        c["confound"],
                        model,
                        mode,
                        s["top1"],
                        s["truth_rank"],
                        s["truth_score"],
                        s["top_false"],
                        s["top_false_score"],
                        s["margin_truth_minus_max_false"],
                        s["confound_score"],
                        s["margin_truth_minus_confound"],
                    ]))
                    + "\n"
                )

    print()
    print(f"[saved] {OUT / 'comparison.json'}")
    print(f"[saved] {OUT / 'comparison.tsv'}")


if __name__ == "__main__":
    main()
