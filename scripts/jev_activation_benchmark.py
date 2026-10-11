from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

CRITICAL_ROOT = Path("artifacts/faithful_simple_k3_v1")
HIST_ROOT = Path("artifacts/adaptive_v11_curriculum_v2")


def read_jsonl(path: Path):
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            x = json.loads(line)
            if isinstance(x, dict):
                out.append(x)
        except Exception:
            pass
    return out


def visible_answer(row):
    for k in (
        "output",
        "final_output",
        "response",
        "answer",
        "visible_response",
    ):
        x = row.get(k)
        if isinstance(x, str) and x.strip():
            return x.strip()
    return ""


def selected_roles(row):
    for container in (row, row.get("metadata") or {}):
        x = container.get("selected_roles_training_only")
        if isinstance(x, list):
            return [str(z) for z in x]
    return []


def query_of(row):
    x = row.get("query")
    return str(x).strip() if x is not None else ""


def stable_id(query, answer):
    h = hashlib.sha256(
        (query.strip() + "\n---\n" + answer.strip()).encode()
    ).hexdigest()
    return h[:20]


def call_jev(payload, key, retries=8):
    raw = json.dumps(payload).encode()

    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL,
            data=raw,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "mas-activation-benchmark/1.0",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                body = r.read().decode()
                obj = json.loads(body)
                if "answers" not in obj:
                    raise RuntimeError(
                        f"TypeSafe response missing answers: {obj}"
                    )
                return obj

        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")

            # TypeSafe docs recommend backoff for 429 and 529.
            if e.code in (429, 529) and attempt + 1 < retries:
                delay = min(60, 2 ** attempt)
                print(
                    f"[retry] HTTP {e.code}; sleeping {delay}s",
                    flush=True,
                )
                time.sleep(delay)
                continue

            raise RuntimeError(
                f"TypeSafe HTTP {e.code}: {body}"
            ) from e

        except (urllib.error.URLError, TimeoutError) as e:
            if attempt + 1 >= retries:
                raise
            delay = min(60, 2 ** attempt)
            print(
                f"[retry] network={e}; sleeping {delay}s",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("Jev retries exhausted")


def build_questions():
    qs = {}

    for role in TRAIN_ROLES:
        desc = ROLE_DESCRIPTIONS.get(role, role)

        qs[f"act__{role}"] = {
            "type": "noul",
            "instructions": {
                "question": (
                    "Did the candidate specialist actually participate "
                    "in producing the visible final response for this "
                    "specific user query?"
                ),
                "candidate_role": role,
                "candidate_description": desc,
                "decision_rule": (
                    "Judge actual participation, not topic similarity. "
                    "The user may explicitly ask about the candidate's "
                    "topic while a different specialist produces a "
                    "competent answer. In that case answer NO for the "
                    "candidate that did not participate."
                ),
            },
            "criteria": {
                "true": (
                    "There is behavioral evidence that this specialist "
                    "actually contributed to this specific response."
                ),
                "false": (
                    "The specialist did not contribute, or the only "
                    "evidence is that the query/answer discusses its topic."
                ),
            },
        }

    qs["single_selected_role"] = {
        "type": "choice",
        "instructions": (
            "Assume exactly one hidden specialist produced this response. "
            "Which candidate specialist most likely actually participated? "
            "Choose based on behavioral evidence, not merely the topic named "
            "by the user query."
        ),
        "criteria": {
            role: ROLE_DESCRIPTIONS.get(role, role)
            for role in TRAIN_ROLES
        },
    }

    return qs


QUESTIONS = build_questions()


def build_state(query, answer):
    # Ground truth is deliberately absent.
    return {
        "task": (
            "Black-box multi-agent activation inference. Infer which "
            "specialist participated from public behavior only."
        ),
        "user_query": query,
        "visible_final_response": answer,
        "important_constraint": (
            "Topic match is not equivalent to specialist activation. "
            "A general-purpose specialist may competently answer a query "
            "about another domain."
        ),
    }


def critical_examples():
    out = []

    for r in range(1, 5):
        path = CRITICAL_ROOT / f"responses_dev_round_{r:02d}.jsonl"
        rows = read_jsonl(path)
        if not rows:
            raise RuntimeError(f"Missing critical response: {path}")

        row = rows[0]
        q = query_of(row)
        a = visible_answer(row)
        sel = selected_roles(row)

        if not q or not a or len(sel) != 1:
            raise RuntimeError(
                f"Bad critical row R{r}: "
                f"query={bool(q)} answer={bool(a)} selected={sel}"
            )

        out.append({
            "example_id": f"critical_r{r}",
            "source": "faithful_simple_k3",
            "round": r,
            "query": q,
            "answer": a,
            "selected": sel,
        })

    return out


def historical_examples(limit, seed):
    paths = sorted(
        glob.glob(
            str(HIST_ROOT / "**" / "responses_dev_round_*.jsonl"),
            recursive=True,
        )
    )

    seen = set()
    by_role = defaultdict(list)

    for fn in paths:
        for row in read_jsonl(Path(fn)):
            q = query_of(row)
            a = visible_answer(row)
            sel = selected_roles(row)

            if not q or not a or len(sel) != 1:
                continue

            role = sel[0]
            if role not in TRAIN_ROLES:
                continue

            eid = stable_id(q, a)
            if eid in seen:
                continue
            seen.add(eid)

            by_role[role].append({
                "example_id": f"hist_{eid}",
                "source": fn,
                "query": q,
                "answer": a,
                "selected": sel,
            })

    rng = random.Random(seed)
    for xs in by_role.values():
        rng.shuffle(xs)

    # Balanced round-robin across observed roles.
    chosen = []
    roles = sorted(by_role)
    pos = {r: 0 for r in roles}

    while len(chosen) < limit:
        made_progress = False
        for role in roles:
            i = pos[role]
            if i < len(by_role[role]):
                chosen.append(by_role[role][i])
                pos[role] += 1
                made_progress = True
                if len(chosen) >= limit:
                    break
        if not made_progress:
            break

    return chosen


def binary_logloss(y, p):
    eps = 1e-6
    p = min(1 - eps, max(eps, float(p)))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def evaluate_one(ex, resp):
    answers = resp["answers"]

    scores = {}
    for role in TRAIN_ROLES:
        key = f"act__{role}"
        x = answers[key]
        if x.get("type") != "noul":
            raise RuntimeError(f"{key}: not Noul: {x}")
        scores[role] = float(x["noul"])

    ranked = sorted(
        scores.items(),
        key=lambda kv: kv[1],
        reverse=True,
    )

    truth = ex["selected"][0]
    truth_rank = 1 + [r for r, _ in ranked].index(truth)
    truth_score = scores[truth]

    false_scores = [
        p for r, p in scores.items()
        if r != truth
    ]
    max_false = max(false_scores)
    mean_false = sum(false_scores) / len(false_scores)

    choice = answers["single_selected_role"]
    choice_selected = str(choice["choice"])
    choice_probs = {
        str(k): float(v)
        for k, v in choice["probabilities"].items()
    }

    labels = {
        role: int(role == truth)
        for role in TRAIN_ROLES
    }

    brier = sum(
        (scores[r] - labels[r]) ** 2
        for r in TRAIN_ROLES
    ) / len(TRAIN_ROLES)

    logloss = sum(
        binary_logloss(labels[r], scores[r])
        for r in TRAIN_ROLES
    ) / len(TRAIN_ROLES)

    return {
        **ex,
        "jev_model": resp.get("model"),
        "usage": resp.get("usage"),
        "noul_scores": scores,
        "noul_ranked": ranked,
        "noul_top1": ranked[0][0],
        "noul_top1_correct": ranked[0][0] == truth,
        "truth_rank": truth_rank,
        "truth_score": truth_score,
        "max_false_score": max_false,
        "mean_false_score": mean_false,
        "truth_minus_max_false": truth_score - max_false,
        "truth_minus_mean_false": truth_score - mean_false,
        "binary_brier": brier,
        "binary_logloss": logloss,
        "choice": choice_selected,
        "choice_correct": choice_selected == truth,
        "choice_confidence": float(choice.get("confidence", 0.0)),
        "choice_probabilities": choice_probs,
    }


def summarize(rows, label):
    if not rows:
        return {"label": label, "n": 0}

    n = len(rows)

    def mean(key):
        return sum(float(r[key]) for r in rows) / n

    rank_hist = Counter(int(r["truth_rank"]) for r in rows)

    return {
        "label": label,
        "n": n,
        "noul_top1_accuracy": sum(
            bool(r["noul_top1_correct"]) for r in rows
        ) / n,
        "choice_top1_accuracy": sum(
            bool(r["choice_correct"]) for r in rows
        ) / n,
        "mean_truth_rank": mean("truth_rank"),
        "mean_reciprocal_rank": sum(
            1.0 / float(r["truth_rank"]) for r in rows
        ) / n,
        "mean_truth_score": mean("truth_score"),
        "mean_max_false_score": mean("max_false_score"),
        "mean_truth_minus_max_false": mean(
            "truth_minus_max_false"
        ),
        "mean_truth_minus_mean_false": mean(
            "truth_minus_mean_false"
        ),
        "mean_binary_brier": mean("binary_brier"),
        "mean_binary_logloss": mean("binary_logloss"),
        "truth_rank_histogram": dict(sorted(rank_hist.items())),
        "selected_role_counts": dict(
            sorted(Counter(r["selected"][0] for r in rows).items())
        ),
        "input_tokens": sum(
            int((r.get("usage") or {}).get("input_tokens", 0))
            for r in rows
        ),
        "output_tokens": sum(
            int((r.get("usage") or {}).get("output_tokens", 0))
            for r in rows
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir",
                    default="artifacts/jev_activation_v1")
    ap.add_argument("--historical-limit", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--sleep", type=float, default=0.10)
    args = ap.parse_args()

    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise SystemExit("TYPESAFE_API_KEY missing")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    critical = critical_examples()
    historical = historical_examples(
        args.historical_limit,
        args.seed,
    )

    examples = critical + historical

    print("=" * 100)
    print("JEV ACTIVATION SENSOR BENCHMARK")
    print("=" * 100)
    print(f"model={MODEL}")
    print(f"critical={len(critical)}")
    print(f"historical={len(historical)}")
    print(f"total={len(examples)}")
    print(
        "IMPORTANT: selected_roles_training_only is evaluation-only "
        "and is NEVER sent to Jev."
    )
    print()

    results_path = outdir / "results.jsonl"

    # Resume safely if the job is restarted.
    completed = {}
    if results_path.exists():
        for x in read_jsonl(results_path):
            completed[str(x.get("example_id"))] = x

    all_rows = []

    for i, ex in enumerate(examples, 1):
        eid = ex["example_id"]

        if eid in completed:
            row = completed[eid]
            all_rows.append(row)
            print(
                f"[resume] {i}/{len(examples)} {eid}",
                flush=True,
            )
            continue

        payload = {
            "state": build_state(ex["query"], ex["answer"]),
            "model": MODEL,
            "questions": QUESTIONS,
        }

        print(
            f"[call] {i}/{len(examples)} "
            f"id={eid} truth={ex['selected'][0]}",
            flush=True,
        )

        t0 = time.time()
        response = call_jev(payload, key)
        elapsed = time.time() - t0

        row = evaluate_one(ex, response)
        row["elapsed_seconds"] = elapsed

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        all_rows.append(row)

        top5 = row["noul_ranked"][:5]
        print(
            f"       Noul top1={row['noul_top1']} "
            f"truth_rank={row['truth_rank']} "
            f"truth={row['truth_score']:.3f} "
            f"max_false={row['max_false_score']:.3f} "
            f"margin={row['truth_minus_max_false']:+.3f} "
            f"choice={row['choice']} "
            f"choice_ok={row['choice_correct']} "
            f"time={elapsed:.2f}s",
            flush=True,
        )
        print(
            "       top5=" +
            ", ".join(f"{r}:{p:.3f}" for r, p in top5),
            flush=True,
        )

        time.sleep(args.sleep)

    critical_rows = [
        r for r in all_rows
        if r["source"] == "faithful_simple_k3"
    ]
    historical_rows = [
        r for r in all_rows
        if r["source"] != "faithful_simple_k3"
    ]

    summary = {
        "model_alias": MODEL,
        "critical": summarize(critical_rows, "critical"),
        "historical": summarize(historical_rows, "historical"),
        "all": summarize(all_rows, "all"),
    }

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(json.dumps(summary, indent=2))

    print()
    print("=" * 100)
    print("CRITICAL FAILURE CASES")
    print("=" * 100)

    for r in critical_rows:
        truth = r["selected"][0]
        print(
            f"{r['example_id']}: "
            f"actual={truth} "
            f"noul_top1={r['noul_top1']} "
            f"rank={r['truth_rank']} "
            f"P(actual)={r['truth_score']:.3f} "
            f"max_false={r['max_false_score']:.3f} "
            f"margin={r['truth_minus_max_false']:+.3f} "
            f"choice={r['choice']}"
        )


if __name__ == "__main__":
    main()
