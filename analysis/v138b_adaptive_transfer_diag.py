#!/usr/bin/env python3
import argparse, json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import analysis.observation_model_v1 as om
import analysis.train_v13_6b_activation_judge as tr


def load_raw(root):
    raw = {}
    for p in sorted(root.rglob("responses_*_round_*.jsonl")):
        parent = str(p.parent)
        with p.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                x = json.loads(line)

                sid = str(
                    x.get("system_id")
                    or x.get("session_id")
                    or ""
                )
                rnd = x.get("round")
                ci = x.get("candidate_index")
                q = str(x.get("query") or "")

                if not sid or rnd is None or ci is None:
                    continue

                key = (parent, sid, int(rnd), int(ci), q)

                raw[key] = {
                    "answer": str(
                        x.get("final_output")
                        or x.get("answer")
                        or x.get("output")
                        or ""
                    ),
                    "action": x.get("action"),
                    "execution_status": x.get("execution_status"),
                }
    return raw


def load_model(adapter):
    from peft import PeftModel
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

    adapter = Path(adapter)
    cfg = json.loads(
        (adapter / "adapter_config.json").read_text()
    )
    base_name = cfg["base_model_name_or_path"]

    dtype = (
        torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16
    )

    try:
        tok = AutoTokenizer.from_pretrained(
            adapter, use_fast=True
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained(
            base_name, use_fast=True
        )

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        base_name,
        num_labels=1,
        quantization_config=qconf,
        device_map={"": 0},
        dtype=dtype,
    )
    base.config.pad_token_id = tok.pad_token_id
    base.config.use_cache = False

    model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=False,
    )
    model.eval()

    return tok, model, dtype


@torch.no_grad()
def score_one(model, tok, text, dtype, max_length):
    x = tok(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    x = {k: v.to("cuda:0") for k, v in x.items()}

    with torch.autocast(
        device_type="cuda",
        dtype=dtype,
        enabled=True,
    ):
        z = model(**x).logits[:, 0].float()

    return float(z.item())


def target_for(mode, row, role):
    if mode == "self":
        return role

    if mode == "neutral":
        return "adaptive_probe"

    if mode == "recorded":
        a = row["_raw"].get("action")
        if isinstance(a, dict) and a.get("target"):
            return str(a["target"])
        return f"adaptive_r{int(row['round']):02d}"

    raise ValueError(mode)


def summarize(records, roles):
    y = np.asarray(
        [r["label"] for r in records], dtype=int
    )
    s = np.asarray(
        [r["score"] for r in records], dtype=float
    )

    micro = float(roc_auc_score(y, s))

    per_role = []
    for role in roles:
        rr = [r for r in records if r["role"] == role]
        yy = np.asarray([r["label"] for r in rr])
        ss = np.asarray([r["score"] for r in rr])

        if len(np.unique(yy)) == 2:
            per_role.append(
                float(roc_auc_score(yy, ss))
            )

    groups = defaultdict(list)
    for r in records:
        groups[r["row_id"]].append(r)

    row_auc = []
    topk = []
    margin = []

    for rr in groups.values():
        yy = np.asarray(
            [r["label"] for r in rr], dtype=int
        )
        ss = np.asarray(
            [r["score"] for r in rr], dtype=float
        )

        row_auc.append(
            float(roc_auc_score(yy, ss))
        )

        k = int(yy.sum())
        order = np.argsort(-ss)[:k]
        topk.append(float(yy[order].sum() / k))

        margin.append(
            float(
                ss[yy == 1].mean()
                - ss[yy == 0].mean()
            )
        )

    return {
        "n_rows": len(groups),
        "n_endpoints": len(records),
        "micro_auc": micro,
        "macro_role_auc": (
            float(np.mean(per_role))
            if per_role else None
        ),
        "roles_with_auc": len(per_role),
        "mean_within_row_auc": float(np.mean(row_auc)),
        "mean_topk_selected_recall": float(np.mean(topk)),
        "mean_selected_minus_absent_margin":
            float(np.mean(margin)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        default="artifacts/adaptive_v11_curriculum_v2",
    )
    ap.add_argument(
        "--obs-dir",
        default="artifacts/observation_model_v1",
    )
    ap.add_argument(
        "--adapter",
        default=(
            "artifacts/"
            "v13_8b_balanced_repair_2048_run1/"
            "best_adapter"
        ),
    )
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1388)
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument(
        "--out",
        default="artifacts/v138b_adaptive_transfer_diag",
    )
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    root = Path(args.root)
    obs = Path(args.obs_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/4] loading examples + raw outputs", flush=True)

    examples, roles, _, _ = om.collect_examples(root)
    roles = list(roles)
    rd = dict(tr.ROLE_DESCRIPTIONS)

    split = json.loads(
        (obs / "split_systems.json").read_text()
    )
    val_systems = set(map(str, split["val"]))

    raw = load_raw(root)

    by_sid = defaultdict(list)

    for x in examples:
        sid = str(x["system_id"])

        if sid not in val_systems:
            continue

        key = (
            str(x["parent"]),
            sid,
            int(x["round"]),
            int(x["candidate_index"]),
            str(x["query"]),
        )

        rr = raw.get(key)
        if rr is None or not rr["answer"]:
            continue

        if not x.get("selected_roles"):
            continue

        z = dict(x)
        z["_raw"] = rr
        by_sid[sid].append(z)

    rng = random.Random(args.seed)
    sids = sorted(by_sid)
    rng.shuffle(sids)

    chosen = []
    for sid in sids[:args.rows]:
        xs = by_sid[sid]
        chosen.append(xs[rng.randrange(len(xs))])

    assert len(chosen) == args.rows, (
        len(chosen), args.rows
    )

    print(
        f"VAL systems selected: {len(chosen)} / "
        f"{len(val_systems)}",
        flush=True,
    )

    print("[2/4] loading frozen V13.8b", flush=True)
    tok, model, dtype = load_model(args.adapter)
    print(
        "GPU:", torch.cuda.get_device_name(0),
        flush=True
    )

    results = {}

    for mode in ("recorded", "neutral", "self"):
        print(f"\n[3/4] mode={mode}", flush=True)

        records = []

        for i, x in enumerate(chosen, 1):
            selected = set(map(str, x["selected_roles"]))
            answer = x["_raw"]["answer"]
            query = str(x["query"])

            row_id = (
                f"{x['system_id']}::"
                f"r{int(x['round']):02d}::"
                f"c{int(x['candidate_index']):02d}"
            )

            for role in roles:
                target = target_for(mode, x, role)

                text = tr.presence_text(
                    role,
                    rd[role],
                    target,
                    query,
                    answer,
                )

                score = score_one(
                    model,
                    tok,
                    text,
                    dtype,
                    args.max_length,
                )

                records.append({
                    "row_id": row_id,
                    "role": role,
                    "label": int(role in selected),
                    "score": score,
                })

            print(
                f"  {mode}: {i}/{len(chosen)}",
                flush=True,
            )

        results[mode] = summarize(records, roles)

    (out / "summary.json").write_text(
        json.dumps(
            results,
            indent=2,
            sort_keys=True,
        )
    )

    print("\n[4/4] COMPARISON")
    print(
        f"{'MODE':<12}"
        f"{'MICRO':>9}"
        f"{'MACRO':>9}"
        f"{'ROW_AUC':>10}"
        f"{'TOPK':>9}"
        f"{'MARGIN':>10}"
    )

    for mode in ("recorded", "neutral", "self"):
        m = results[mode]

        macro = (
            "NA"
            if m["macro_role_auc"] is None
            else f"{m['macro_role_auc']:.4f}"
        )

        print(
            f"{mode:<12}"
            f"{m['micro_auc']:>9.4f}"
            f"{macro:>9}"
            f"{m['mean_within_row_auc']:>10.4f}"
            f"{m['mean_topk_selected_recall']:>9.4f}"
            f"{m['mean_selected_minus_absent_margin']:>+10.4f}"
        )

    print("\nwrote:", out / "summary.json")


if __name__ == "__main__":
    main()
