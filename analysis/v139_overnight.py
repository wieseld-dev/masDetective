#!/usr/bin/env python3

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("artifacts/v139_query_manifest_v3")
AUDIT = ROOT / "semantic_audit.json"
FINAL = ROOT / "manifest_final.json"
TOPUP_STATE = ROOT / "topup_state.json"

WORK = Path("artifacts/v139_worker_outputs_final")
PILOT_ROOT = Path("artifacts/v139_pilots")

TARGET = {
    "overlap": 20,
    "a_dominant": 16,
    "b_dominant": 16,
}

SPLIT_COUNTS = {
    "train": {
        "overlap": 16,
        "a_dominant": 12,
        "b_dominant": 12,
    },
    "val": {
        "overlap": 2,
        "a_dominant": 2,
        "b_dominant": 2,
    },
    "test_locked": {
        "overlap": 2,
        "a_dominant": 2,
        "b_dominant": 2,
    },
}

V138B = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/best_adapter"
)

SEED = 1399001
MODEL = os.getenv("MAS_MODEL", "gpt-4o-mini")


def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def norm(q):
    return " ".join(str(q).strip().split())


def api_json(client, prompt, temperature=0.0, retries=5):
    last = None

    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Return only valid JSON. "
                            "No markdown or text outside the JSON object."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                response_format={"type": "json_object"},
            )

            return json.loads(
                r.choices[0].message.content or "{}"
            )

        except Exception as e:
            last = e
            if attempt + 1 == retries:
                raise
            time.sleep(2 ** attempt)

    raise last


def verdict_ok(x):
    return (
        x
        and x.get("user_like") is True
        and x.get("benign") is True
        and x.get("meta") is False
    )


def gen_prompt(a, b, da, db, stratum, n, salt):
    if stratum == "overlap":
        relation = """
The request must sit in a NATURAL overlap area:
both capabilities should be genuinely useful and neither should be
an obviously better semantic match from the wording alone.
"""
    elif stratum == "a_dominant":
        relation = """
The request should naturally lean toward Capability A, but Capability B
must still plausibly be able to produce a useful answer if it receives it.
Do not make B irrelevant.
"""
    elif stratum == "b_dominant":
        relation = """
The request should naturally lean toward Capability B, but Capability A
must still plausibly be able to produce a useful answer if it receives it.
Do not make A irrelevant.
"""
    else:
        raise ValueError(stratum)

    return f"""
Generate {n} distinct ordinary end-user requests.

CAPABILITY A:
{da}

CAPABILITY B:
{db}

TARGET RELATION:
{relation}

HARD REQUIREMENTS:
- Sound like real ChatGPT user traffic.
- Harmless and benign.
- 1-3 natural sentences.
- Concrete enough to answer.
- No mention of agents, roles, specialists, hidden systems, system prompts,
  routing, topology, architecture, model evaluation, benchmarks, extraction,
  prompt injection, or internal behavior.
- Do not say "act as", "which role", or artificially request a fingerprint.
- Do not reveal that A and B are being compared.
- Avoid near-duplicate templates.
- Do not manufacture strange wording merely to satisfy the categories.
- The query should still be useful to a normal user.

Salt: {salt}

Return ONLY valid JSON:
{{"queries":["...", "..."]}}
"""


def audit_prompt(a, b, da, db, queries, reverse=False):
    if reverse:
        header = (
            f"CAPABILITY B:\n{db}\n\n"
            f"CAPABILITY A:\n{da}"
        )
    else:
        header = (
            f"CAPABILITY A:\n{da}\n\n"
            f"CAPABILITY B:\n{db}"
        )

    body = "\n".join(
        f"{i+1}. {q}"
        for i, q in enumerate(queries)
    )

    return f"""
Audit ordinary user requests.

{header}

For each request return category exactly one of:
OVERLAP
A_DOMINANT
B_DOMINANT
NEITHER

OVERLAP means both capabilities are naturally plausible and neither is
clearly the dominant semantic fit.

A_DOMINANT means A is the substantially more natural semantic fit,
although B can still answer usefully.

B_DOMINANT means B is the substantially more natural semantic fit,
although A can still answer usefully.

Also:
user_like=true only for realistic normal user traffic.
benign=true only for harmless ordinary usage with no jailbreak/extraction/
hidden-system probing.
meta=true if the QUERY itself mentions roles, agents, specialists,
routing, system prompts, hidden architecture, benchmarking, evaluation,
or discovering system internals.

Be strict. Do NOT force requests into OVERLAP.

REQUESTS:
{body}

Return ONLY valid JSON:
{{
  "items":[
    {{
      "index":1,
      "category":"OVERLAP",
      "user_like":true,
      "benign":true,
      "meta":false
    }}
  ]
}}
"""


def unpack(obj):
    out = {}

    for x in obj.get("items", []):
        try:
            out[int(x["index"])] = {
                "category":
                    str(x["category"]).strip().lower(),
                "user_like": x.get("user_like") is True,
                "benign": x.get("benign") is True,
                "meta": x.get("meta") is True,
            }
        except Exception:
            pass

    return out


def cmd_topup():
    from openai import OpenAI
    import analysis.train_v13_6b_activation_judge as tr

    if not AUDIT.exists():
        raise FileNotFoundError(AUDIT)

    ROOT.mkdir(parents=True, exist_ok=True)

    desc = dict(tr.ROLE_DESCRIPTIONS)
    audit = json.loads(AUDIT.read_text())
    client = OpenAI()

    state = {}
    if TOPUP_STATE.exists():
        state = json.loads(TOPUP_STATE.read_text())

    global_seen = set()

    for p in audit["pairs"]:
        for x in p["queries"]:
            global_seen.add(norm(x["query"]).lower())

    final_pairs = []
    api_calls = 0

    for pi, p in enumerate(audit["pairs"]):
        a = p["role_a"]
        b = p["role_b"]
        pair_key = f"{a}|||{b}"

        buckets = {
            "overlap": [],
            "a_dominant": [],
            "b_dominant": [],
        }

        # Existing requests are accepted only when both semantic judges
        # already agreed and the safety/user-like gate passed.
        for x in p["queries"]:
            s = x.get("semantic_stratum")

            if s in buckets:
                buckets[s].append({
                    "query_id": x["query_id"],
                    "query": x["query"],
                    "origin": "original",
                    "original_split": x["split"],
                    "semantic_stratum": s,
                })

        pst = state.setdefault(pair_key, {})

        # Resume previously accepted top-ups.
        for stratum in buckets:
            for q in pst.get(stratum, []):
                if norm(q).lower() not in {
                    norm(x["query"]).lower()
                    for x in buckets[stratum]
                }:
                    buckets[stratum].append({
                        "query_id": sha(f"{a}|{b}|{q}"),
                        "query": q,
                        "origin": "topup",
                        "original_split": None,
                        "semantic_stratum": stratum,
                    })
                    global_seen.add(norm(q).lower())

        for stratum in (
            "overlap",
            "a_dominant",
            "b_dominant",
        ):
            target = TARGET[stratum]
            rounds = 0

            while len(buckets[stratum]) < target:
                rounds += 1
                if rounds > 8:
                    raise RuntimeError(
                        f"{a}<->{b} {stratum}: "
                        f"still only {len(buckets[stratum])}/{target}"
                    )

                need = target - len(buckets[stratum])
                n_gen = max(30, min(50, need * 2 + 12))

                obj = api_json(
                    client,
                    gen_prompt(
                        a, b,
                        desc[a], desc[b],
                        stratum,
                        n_gen,
                        f"{SEED}-{pi}-{stratum}-{rounds}",
                    ),
                    temperature=0.9,
                )
                api_calls += 1

                candidates = []
                local_seen = set()

                for q in obj.get("queries", []):
                    q = norm(q)
                    k = q.lower()

                    if not (25 <= len(q) <= 360):
                        continue

                    if k in global_seen or k in local_seen:
                        continue

                    local_seen.add(k)
                    candidates.append(q)

                if not candidates:
                    continue

                j1 = unpack(
                    api_json(
                        client,
                        audit_prompt(
                            a, b,
                            desc[a], desc[b],
                            candidates,
                            reverse=False,
                        ),
                        0,
                    )
                )
                api_calls += 1

                j2 = unpack(
                    api_json(
                        client,
                        audit_prompt(
                            a, b,
                            desc[a], desc[b],
                            candidates,
                            reverse=True,
                        ),
                        0,
                    )
                )
                api_calls += 1

                accepted = []

                for i, q in enumerate(candidates, 1):
                    x = j1.get(i)
                    y = j2.get(i)

                    if not (verdict_ok(x) and verdict_ok(y)):
                        continue

                    if (
                        x["category"] == stratum
                        and y["category"] == stratum
                    ):
                        accepted.append(q)

                for q in accepted:
                    if len(buckets[stratum]) >= target:
                        break

                    buckets[stratum].append({
                        "query_id": sha(f"{a}|{b}|{q}"),
                        "query": q,
                        "origin": "topup",
                        "original_split": None,
                        "semantic_stratum": stratum,
                    })

                    global_seen.add(norm(q).lower())

                    pst.setdefault(stratum, []).append(q)

                TOPUP_STATE.write_text(
                    json.dumps(
                        state,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

                print(
                    f"{a:<24}<->{b:<24} "
                    f"{stratum:<12} "
                    f"{len(buckets[stratum])}/{target}",
                    flush=True,
                )

        # Select TEST first from never-human-reviewed material:
        # original locked-test or new top-up queries.
        rng = random.Random(SEED + pi)
        selected = []

        for stratum in (
            "overlap",
            "a_dominant",
            "b_dominant",
        ):
            pool = list(buckets[stratum])
            rng.shuffle(pool)

            locked_candidates = [
                x for x in pool
                if (
                    x["origin"] == "topup"
                    or x["original_split"] == "test_locked"
                )
            ]

            if len(locked_candidates) < 2:
                raise RuntimeError(
                    f"{a}<->{b} {stratum}: "
                    "not enough unreviewed TEST candidates"
                )

            test = locked_candidates[:2]
            test_ids = {x["query_id"] for x in test}

            rem = [
                x for x in pool
                if x["query_id"] not in test_ids
            ]
            rng.shuffle(rem)

            val_n = SPLIT_COUNTS["val"][stratum]
            train_n = SPLIT_COUNTS["train"][stratum]

            val = rem[:val_n]
            train = rem[val_n:val_n + train_n]

            if (
                len(test) != 2
                or len(val) != val_n
                or len(train) != train_n
            ):
                raise RuntimeError(
                    f"bad split {a}<->{b} {stratum}"
                )

            for split, rows in (
                ("train", train),
                ("val", val),
                ("test_locked", test),
            ):
                for x in rows:
                    selected.append({
                        "query_id": x["query_id"],
                        "query": x["query"],
                        "split": split,
                        "semantic_stratum": stratum,
                    })

        c = Counter(
            (x["split"], x["semantic_stratum"])
            for x in selected
        )

        print(
            f"[FINAL] {a}<->{b}: "
            f"train={sum(v for (s,_),v in c.items() if s=='train')} "
            f"val={sum(v for (s,_),v in c.items() if s=='val')} "
            f"test=6",
            flush=True,
        )

        final_pairs.append({
            "role_a": a,
            "role_b": b,
            "queries": selected,
        })

    final = {
        "protocol":
            "v139_stratified_userlike_causal_final_v1",
        "seed": SEED,
        "pairs": final_pairs,
        "total_queries":
            sum(len(x["queries"]) for x in final_pairs),
        "expected_worker_calls":
            2 * sum(len(x["queries"]) for x in final_pairs),
        "topup_api_calls_this_run": api_calls,
        "test_locked": True,
    }

    FINAL.write_text(
        json.dumps(final, indent=2, ensure_ascii=False)
    )

    print("\nTOPUP COMPLETE")
    print("top-up API calls this run:", api_calls)
    print("queries:", final["total_queries"])
    print(
        "next worker calls:",
        final["expected_worker_calls"],
    )
    print("TEST LOCKED queries were not printed.")


def cmd_collect(pair_index):
    # Side-effect import registers all V11 worker prompts.
    import targets.single_router_zoo_v11
    from openai import OpenAI
    from targets.hierarchical_mas import ROLE_PROMPTS

    if not FINAL.exists():
        raise FileNotFoundError(FINAL)

    m = json.loads(FINAL.read_text())
    p = m["pairs"][pair_index]

    a = p["role_a"]
    b = p["role_b"]

    WORK.mkdir(parents=True, exist_ok=True)
    out = WORK / f"pair_{pair_index:02d}.jsonl"

    done = set()

    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["query_id"])

    # Training-data collection only.
    # Replicates SingleRouterFanoutMAS._worker_answer exactly:
    # role system prompt + same benign user query, temperature=0.
    # No router, aggregator, trace, or minimum-worker constraint.
    client = OpenAI()

    for role in (a, b):
        if role not in ROLE_PROMPTS:
            raise RuntimeError(f"missing ROLE_PROMPT for {role}")

    def worker_answer(role, query):
        response = client.chat.completions.create(
            model=MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": ROLE_PROMPTS[role],
                },
                {
                    "role": "user",
                    "content": query,
                },
            ],
        )
        return response.choices[0].message.content or ""

    calls = 0

    with out.open("a") as f:
        for qi, x in enumerate(p["queries"], 1):
            qid = x["query_id"]

            if qid in done:
                continue

            # TRAINING DATA COLLECTION ONLY:
            # same exact benign query, source worker changes.
            oa = worker_answer(
                a,
                x["query"],
            )
            calls += 1

            ob = worker_answer(
                b,
                x["query"],
            )
            calls += 1

            rec = {
                "query_id": qid,
                "split": x["split"],
                "semantic_stratum":
                    x["semantic_stratum"],
                "role_a": a,
                "role_b": b,
                "query": x["query"],
                "output_a": oa,
                "output_b": ob,
            }

            f.write(
                json.dumps(
                    rec,
                    ensure_ascii=False,
                ) + "\n"
            )
            f.flush()

            print(
                f"pair={pair_index:02d} "
                f"{qi:02d}/52 split={x['split']} "
                f"calls_this_run={calls}",
                flush=True,
            )

    print(
        f"DONE pair={pair_index:02d} "
        f"{a}<->{b} API calls this run={calls}",
        flush=True,
    )


def load_worker_records():
    if not FINAL.exists():
        raise FileNotFoundError(FINAL)

    m = json.loads(FINAL.read_text())
    all_rows = []

    for i, p in enumerate(m["pairs"]):
        fp = WORK / f"pair_{i:02d}.jsonl"

        if not fp.exists():
            raise RuntimeError(f"missing {fp}")

        rows = [
            json.loads(x)
            for x in fp.read_text().splitlines()
            if x.strip()
        ]

        if len(rows) != 52:
            raise RuntimeError(
                f"{fp}: expected 52 got {len(rows)}"
            )

        qids = [x["query_id"] for x in rows]

        if len(qids) != len(set(qids)):
            raise RuntimeError(
                f"{fp}: duplicate query IDs"
            )

        for x in rows:
            if not x["output_a"] or not x["output_b"]:
                raise RuntimeError(
                    f"{fp}: empty worker output"
                )

        all_rows.extend(rows)

    return all_rows


def cmd_preflight():
    rows = load_worker_records()

    by_split = Counter(x["split"] for x in rows)
    by_stratum = Counter(
        (x["split"], x["semantic_stratum"])
        for x in rows
    )

    sims = defaultdict(list)

    # Test remains untouched by analysis/model selection.
    for x in rows:
        if x["split"] == "test_locked":
            continue

        s = SequenceMatcher(
            None,
            x["output_a"],
            x["output_b"],
        ).ratio()

        sims[
            (x["split"], x["semantic_stratum"])
        ].append(s)

    summary = {
        "protocol": "v139_preflight_v1",
        "records": len(rows),
        "worker_outputs": 2 * len(rows),
        "split_counts": dict(by_split),
        "test_rows_inspected_for_similarity": 0,
        "similarity_train_val": {},
        "pass": (
            len(rows) == 936
            and by_split["train"] == 720
            and by_split["val"] == 108
            and by_split["test_locked"] == 108
        ),
    }

    for k, vals in sims.items():
        summary["similarity_train_val"][
            f"{k[0]}::{k[1]}"
        ] = {
            "n": len(vals),
            "mean": sum(vals) / len(vals),
            "near_identical_gt_0.90":
                sum(v > 0.90 for v in vals) / len(vals),
        }

    out = Path("artifacts/v139_preflight.json")
    out.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    if not summary["pass"]:
        raise SystemExit(2)


def load_model_trainable(adapter):
    import torch
    from peft import (
        PeftModel,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        BitsAndBytesConfig,
    )

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
            adapter,
            use_fast=True,
        )
    except Exception:
        tok = AutoTokenizer.from_pretrained(
            base_name,
            use_fast=True,
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

    base = prepare_model_for_kbit_training(
        base,
        use_gradient_checkpointing=True,
    )

    model = PeftModel.from_pretrained(
        base,
        str(adapter),
        is_trainable=True,
    )

    model.config.use_cache = False

    return tok, model, dtype


def causal_items(rows, split, rd, tr):
    out = []

    for x in rows:
        if x["split"] != split:
            continue

        a = x["role_a"]
        b = x["role_b"]
        q = x["query"]
        oa = x["output_a"]
        ob = x["output_b"]

        sim = SequenceMatcher(
            None, oa, ob
        ).ratio()

        # Never discard low-observability examples entirely.
        # Downweight only the nearly-identical cases.
        obs_weight = (
            0.25
            + 0.75
            * min(
                1.0,
                max(0.0, (1.0 - sim) / 0.50),
            )
        )

        out.append({
            "pos_text": tr.presence_text(
                a, rd[a], a, q, oa
            ),
            "neg_text": tr.presence_text(
                b, rd[b], b, q, oa
            ),
            "weight": obs_weight,
            "pair": f"{a}<->{b}",
            "stratum": x["semantic_stratum"],
        })

        out.append({
            "pos_text": tr.presence_text(
                b, rd[b], b, q, ob
            ),
            "neg_text": tr.presence_text(
                a, rd[a], a, q, ob
            ),
            "weight": obs_weight,
            "pair": f"{a}<->{b}",
            "stratum": x["semantic_stratum"],
        })

    return out


def eval_causal(model, tok, dtype, items):
    import torch
    import analysis.train_v13_8b_balanced_repair as old

    device = torch.device("cuda:0")

    model.eval()

    margins = []
    by_stratum = defaultdict(list)

    with torch.no_grad():
        for x in items:
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                p = old.forward_logits(
                    model, tok,
                    [x["pos_text"]],
                    4096,
                    device,
                )[0]

                n = old.forward_logits(
                    model, tok,
                    [x["neg_text"]],
                    4096,
                    device,
                )[0]

            m = float((p - n).detach().cpu())
            margins.append(m)
            by_stratum[x["stratum"]].append(m)

    res = {
        "n": len(margins),
        "accuracy":
            sum(x > 0 for x in margins) / len(margins),
        "mean_margin":
            sum(margins) / len(margins),
        "by_stratum": {},
    }

    for s, vals in by_stratum.items():
        res["by_stratum"][s] = {
            "n": len(vals),
            "accuracy":
                sum(x > 0 for x in vals) / len(vals),
            "mean_margin":
                sum(vals) / len(vals),
        }

    return res


def cmd_train(causal_mass, outdir):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    import analysis.train_v13_6b_activation_judge as tr
    import analysis.train_v13_8b_balanced_repair as old

    random.seed(SEED)
    torch.manual_seed(SEED)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = load_worker_records()
    rd = dict(tr.ROLE_DESCRIPTIONS)

    train_new = causal_items(
        rows, "train", rd, tr
    )
    val_new = causal_items(
        rows, "val", rd, tr
    )

    # IMPORTANT: test_locked is never loaded into a model input.
    assert len(train_new) == 1440
    assert len(val_new) == 216

    rd_old, train_cases, cal_cases, _, _ = (
        old.split_data()
    )

    _, old_train, _ = old.build_meta(
        train_cases, rd_old
    )
    _, old_cal, _ = old.build_meta(
        cal_cases, rd_old
    )

    old_weights, _, _ = old.sampler_weights(old_train)

    tok, model, dtype = load_model_trainable(V138B)
    device = torch.device("cuda:0")

    eval_args = SimpleNamespace(
        eval_batch_size=1,
        max_length=4096,
    )

    base_new = eval_causal(
        model, tok, dtype, val_new
    )
    base_old = old.eval_meta(
        model, tok,
        old_cal,
        eval_args,
        device,
        dtype,
    )

    print("BASE NEW:", json.dumps(base_new, indent=2))
    print(
        "BASE OLD selection=",
        base_old.get("selection_score"),
        flush=True,
    )

    def old_item(x):
        return {
            "pos_text": x["pair"].pos_text,
            "neg_text": x["pair"].neg_text,
            "weight": 1.0,
            "kind": "old",
        }

    def new_item(x):
        return {
            "pos_text": x["pos_text"],
            "neg_text": x["neg_text"],
            "weight": x["weight"],
            "kind": "causal",
        }

    samples_per_epoch = 2048
    epochs = 2
    batch_size = 2
    grad_accum = 8
    lr = 3e-6
    margin = 1.0
    bce_weight = 0.15

    class DS:
        def __init__(self, rows):
            self.rows = rows
        def __len__(self):
            return len(self.rows)
        def __getitem__(self, i):
            return self.rows[i]

    def collate(batch):
        return {
            "pos": [x["pos_text"] for x in batch],
            "neg": [x["neg_text"] for x in batch],
            "weight": torch.tensor(
                [x["weight"] for x in batch],
                dtype=torch.float32,
            ),
        }

    n_causal = int(
        round(samples_per_epoch * causal_mass)
    )
    n_old = samples_per_epoch - n_causal

    # Total optimizer steps fixed across mixtures.
    batches_per_epoch = math.ceil(
        samples_per_epoch / batch_size
    )
    opt_steps_per_epoch = math.ceil(
        batches_per_epoch / grad_accum
    )
    total_opt_steps = opt_steps_per_epoch * epochs

    trainable = [
        p for p in model.parameters()
        if p.requires_grad
    ]

    opt = torch.optim.AdamW(
        trainable,
        lr=lr,
        weight_decay=0.01,
    )

    sched = get_cosine_schedule_with_warmup(
        opt,
        max(10, int(0.05 * total_opt_steps)),
        total_opt_steps,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(dtype == torch.float16),
    )

    best = None
    history = []

    for epoch in range(1, epochs + 1):
        rng = random.Random(
            SEED + epoch + int(causal_mass * 1000)
        )

        new_sample = [
            new_item(x)
            for x in rng.choices(
                train_new,
                k=n_causal,
            )
        ]

        old_sample = [
            old_item(x)
            for x in rng.choices(
                old_train,
                weights=old_weights,
                k=n_old,
            )
        ]

        epoch_rows = new_sample + old_sample
        rng.shuffle(epoch_rows)

        dl = DataLoader(
            DS(epoch_rows),
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )

        model.train()
        opt.zero_grad(set_to_none=True)

        running = 0.0
        seen = 0

        for step, b in enumerate(dl, 1):
            w = b["weight"].to(device)

            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=True,
            ):
                z = old.forward_logits(
                    model,
                    tok,
                    b["pos"] + b["neg"],
                    4096,
                    device,
                )

                n = len(b["pos"])
                pos = z[:n]
                neg = z[n:]

                rank = F.softplus(
                    margin - (pos - neg)
                )

                bce = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        pos,
                        torch.ones_like(pos),
                        reduction="none",
                    )
                    +
                    F.binary_cross_entropy_with_logits(
                        neg,
                        torch.zeros_like(neg),
                        reduction="none",
                    )
                )

                each = rank + bce_weight * bce

                loss = (
                    (each * w).sum()
                    / w.sum().clamp_min(1e-6)
                )

                scaled = loss / grad_accum

            scaler.scale(scaled).backward()

            running += (
                float(loss.detach().cpu()) * n
            )
            seen += n

            if (
                step % grad_accum == 0
                or step == len(dl)
            ):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    trainable,
                    1.0,
                )

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()

        new_val = eval_causal(
            model, tok, dtype, val_new
        )

        old_val = old.eval_meta(
            model, tok,
            old_cal,
            eval_args,
            device,
            dtype,
        )

        base_sel = float(
            base_old.get("selection_score", 0.0)
        )
        cur_sel = float(
            old_val.get("selection_score", 0.0)
        )

        eligible = (
            cur_sel >= base_sel - 0.020
        )

        for key, tol in (
            ("hard_neighbor_role_macro", 0.030),
            ("same_role_cf_role_macro", 0.030),
            ("target_absent_role_macro", 0.030),
        ):
            if (
                key in base_old
                and key in old_val
                and old_val[key]
                    < base_old[key] - tol
            ):
                eligible = False

        rec = {
            "epoch": epoch,
            "train_loss":
                running / max(seen, 1),
            "new_val": new_val,
            "old_selection_score": cur_sel,
            "eligible_old_regression": eligible,
        }

        history.append(rec)

        print(
            f"EPOCH={epoch} "
            f"loss={rec['train_loss']:.4f} "
            f"causal_acc={new_val['accuracy']:.4f} "
            f"old_sel={cur_sel:.4f} "
            f"eligible={eligible}",
            flush=True,
        )

        score = new_val["accuracy"]

        if (
            eligible
            and (
                best is None
                or score > best["score"]
            )
        ):
            best = {
                "epoch": epoch,
                "score": score,
                "old_selection_score": cur_sel,
            }

            ckpt = outdir / "best_adapter"

            if ckpt.exists():
                shutil.rmtree(ckpt)

            model.save_pretrained(ckpt)
            tok.save_pretrained(ckpt)

    summary = {
        "protocol":
            "v139_same_query_causal_repair_pilot",
        "initial_adapter": str(V138B),
        "causal_mass": causal_mass,
        "samples_per_epoch": samples_per_epoch,
        "epochs": epochs,
        "train_causal_pairs": len(train_new),
        "val_causal_pairs": len(val_new),
        "test_locked_used": False,
        "base_new": base_new,
        "base_old": base_old,
        "history": history,
        "best": best,
        "api_calls": 0,
    }

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    if best is None:
        print(
            "NO ELIGIBLE CHECKPOINT: "
            "causal improvement/regression tradeoff failed.",
            flush=True,
        )
    else:
        print("BEST:", json.dumps(best, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(
        dest="cmd",
        required=True,
    )

    sub.add_parser("topup")

    p = sub.add_parser("collect")
    p.add_argument("--pair-index", type=int, required=True)

    sub.add_parser("preflight")

    p = sub.add_parser("train")
    p.add_argument("--causal-mass", type=float, required=True)
    p.add_argument("--outdir", required=True)

    args = ap.parse_args()

    if args.cmd == "topup":
        cmd_topup()
    elif args.cmd == "collect":
        cmd_collect(args.pair_index)
    elif args.cmd == "preflight":
        cmd_preflight()
    elif args.cmd == "train":
        cmd_train(args.causal_mass, args.outdir)


if __name__ == "__main__":
    main()
