from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from v11.role_universe import TRAIN_ROLES, ROLE_DESCRIPTIONS


API_URL = "https://api.typesafe.ai/v1/systemone"
MODEL = "jev-latest"

SOURCE = Path("artifacts/jev_activation_v1/results.jsonl")


# ---------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------

def read_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    return rows


def append_jsonl(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------
# API
# ---------------------------------------------------------------------

def call_jev(payload, key, retries=8):
    raw = json.dumps(payload).encode("utf-8")

    for attempt in range(retries):
        req = urllib.request.Request(
            API_URL,
            data=raw,
            method="POST",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "User-Agent": "mas-jev-max/1.0",
            },
        )

        try:
            t0 = time.time()

            with urllib.request.urlopen(req, timeout=240) as r:
                result = json.loads(r.read().decode("utf-8"))

            if "answers" not in result:
                raise RuntimeError(
                    f"Jev response missing answers: {result}"
                )

            return result, time.time() - t0

        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")

            if e.code in (429, 529) and attempt + 1 < retries:
                delay = min(60, 2 ** attempt)
                print(
                    f"[retry] HTTP={e.code} sleep={delay}s",
                    flush=True,
                )
                time.sleep(delay)
                continue

            raise RuntimeError(
                f"HTTP {e.code}: {body}"
            ) from e

        except (urllib.error.URLError, TimeoutError) as e:
            if attempt + 1 >= retries:
                raise

            delay = min(60, 2 ** attempt)

            print(
                f"[retry] network={e} sleep={delay}s",
                flush=True,
            )

            time.sleep(delay)

    raise RuntimeError("Jev retry limit exhausted")


# ---------------------------------------------------------------------
# Ground-truth examples
# ---------------------------------------------------------------------

def load_examples():
    rows = read_jsonl(SOURCE)

    if not rows:
        raise RuntimeError(
            f"No source benchmark: {SOURCE}"
        )

    out = []

    for x in rows:
        selected = list(x.get("selected") or [])

        if len(selected) != 1:
            continue

        q = str(x.get("query") or "").strip()
        a = str(x.get("answer") or "").strip()

        if not q or not a:
            continue

        out.append({
            "example_id": str(x["example_id"]),
            "source": str(x.get("source") or ""),
            "query": q,
            "answer": a,
            "truth": selected[0],

            # old V1 numbers, evaluation only
            "v1_noul_top1": x.get("noul_top1"),
            "v1_noul_correct": bool(
                x.get("noul_top1_correct", False)
            ),
            "v1_choice": x.get("choice"),
            "v1_choice_correct": bool(
                x.get("choice_correct", False)
            ),
        })

    print(
        f"[data] loaded={len(out)} from {SOURCE}",
        flush=True,
    )

    return out


# ---------------------------------------------------------------------
# ICL calibration split
#
# These examples are REMOVED from V3 evaluation.
# Prefer hard V1 mistakes from different truth roles.
# ---------------------------------------------------------------------

def choose_calibration(examples, n=8):
    historical = [
        x for x in examples
        if not x["example_id"].startswith("critical_")
    ]

    chosen = []
    used_roles = set()

    # First take diverse V1 failures.
    for x in historical:
        if len(chosen) >= n:
            break

        if x["v1_noul_correct"]:
            continue

        if x["truth"] in used_roles:
            continue

        chosen.append(x)
        used_roles.add(x["truth"])

    # Fill with diverse examples if necessary.
    for x in historical:
        if len(chosen) >= n:
            break

        if x in chosen:
            continue

        if x["truth"] in used_roles:
            continue

        chosen.append(x)
        used_roles.add(x["truth"])

    # Final fallback.
    for x in historical:
        if len(chosen) >= n:
            break

        if x not in chosen:
            chosen.append(x)

    return chosen


def demo_view(x):
    # Keep prompt size controlled.
    ans = x["answer"]
    if len(ans) > 1400:
        ans = ans[:1400] + " ..."

    return {
        "user_query": x["query"],
        "visible_final_response": ans,
        "known_actual_contributor": x["truth"],
        "lesson": (
            "The contributor label refers to the worker that actually "
            "participated. It is not necessarily the specialist whose "
            "domain is named in the user query."
        ),
    }


# ---------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------

def base_state(ex, demos=None):
    state = {
        "task": (
            "Infer which hidden specialist actually participated in "
            "producing a black-box multi-agent system's final response."
        ),

        "user_query": ex["query"],

        "visible_final_response": ex["answer"],

        "candidate_roles": {
            role: ROLE_DESCRIPTIONS.get(role, role)
            for role in TRAIN_ROLES
        },

        "critical_distinction": (
            "The semantic topic requested by the user is NOT equivalent "
            "to the worker that actually participated. A general-purpose "
            "or neighboring specialist may answer a query about another "
            "domain. Attribute participation from response-specific "
            "behavior, methods, style, and capabilities."
        ),
    }

    if demos:
        state["calibration_examples"] = [
            demo_view(x) for x in demos
        ]

        state["calibration_instruction"] = (
            "Use calibration_examples only to understand the distinction "
            "between topical similarity and actual worker participation. "
            "Do not assume the query topic identifies the contributor."
        )

    return state


# ---------------------------------------------------------------------
# STAGE 1
#
# 5 independent judgments × 20 roles = 100 judgments.
#
# TypeSafe's docs explicitly recommend decomposing a complex judgment
# into independent questions and composing the answers in code.
# ---------------------------------------------------------------------

def stage1_questions():
    qs = {}

    for role in TRAIN_ROLES:
        path = f"`candidate_roles.{role}`"

        qs[f"direct__{role}"] = {
            "type": "noul",
            "instructions": (
                f"Considering `user_query` and `visible_final_response`, "
                f"did the specialist described by {path} actually "
                f"participate in producing this specific response?"
            ),
            "criteria": {
                "true": (
                    "There is evidence this worker actually contributed "
                    "to this response."
                ),
                "false": (
                    "The worker did not contribute, or apparent support "
                    "comes only from the subject of the user query."
                ),
            },
        }

        qs[f"distinctive__{role}"] = {
            "type": "noul",
            "instructions": (
                f"Does `visible_final_response` contain a behavioral "
                f"fingerprint that is distinctively characteristic of "
                f"{path}, beyond merely discussing that specialist's "
                f"domain?"
            ),
            "criteria": {
                "true": (
                    "The response itself exhibits distinctive evidence "
                    "of this specialist."
                ),
                "false": (
                    "The response is generic or only topically related."
                ),
            },
        }

        qs[f"method__{role}"] = {
            "type": "noul",
            "instructions": (
                f"Does `visible_final_response` use methods, reasoning "
                f"patterns, responsibilities, or work products specifically "
                f"characteristic of {path}, rather than generic competent "
                f"language?"
            ),
            "criteria": {
                "true": (
                    "Role-specific methods or work patterns are visible."
                ),
                "false": (
                    "Nothing specifically identifies this role's methods."
                ),
            },
        }

        qs[f"topic_only__{role}"] = {
            "type": "noul",
            "instructions": (
                f"Is the apparent evidence for {path} primarily explained "
                f"by the TOPIC named or requested in `user_query`, rather "
                f"than by a response-specific behavioral fingerprint?"
            ),
            "criteria": {
                "true": (
                    "This looks mainly like a topical-match shortcut."
                ),
                "false": (
                    "Evidence goes meaningfully beyond the query topic."
                ),
            },
        }

        qs[f"alternative__{role}"] = {
            "type": "noul",
            "instructions": (
                f"Could `visible_final_response` plausibly have been "
                f"produced by a different competent specialist WITHOUT "
                f"participation from {path}?"
            ),
            "criteria": {
                "true": (
                    "Another specialist could plausibly explain the "
                    "response without this role."
                ),
                "false": (
                    "The response strongly requires or uniquely suggests "
                    "this role."
                ),
            },
        }

    return qs


STAGE1_QUESTIONS = stage1_questions()


def extract_noul(answers, key):
    x = answers[key]

    if x.get("type") != "noul":
        raise RuntimeError(
            f"Expected Noul at {key}: {x}"
        )

    return float(x["noul"])


def parse_stage1(resp):
    ans = resp["answers"]

    evidence = {}
    composite = {}

    for role in TRAIN_ROLES:
        d = extract_noul(ans, f"direct__{role}")
        b = extract_noul(ans, f"distinctive__{role}")
        m = extract_noul(ans, f"method__{role}")
        t = extract_noul(ans, f"topic_only__{role}")
        a = extract_noul(ans, f"alternative__{role}")

        # Fixed BEFORE looking at benchmark results.
        #
        # Positive:
        # direct attribution, behavioral fingerprint,
        # role-specific method.
        #
        # Negative:
        # topic-only explanation, alternative-worker explanation.
        score = (
            0.35 * d
            + 0.25 * b
            + 0.20 * m
            + 0.15 * (1.0 - t)
            + 0.05 * (1.0 - a)
        )

        evidence[role] = {
            "direct": d,
            "distinctive": b,
            "method": m,
            "topic_only": t,
            "alternative": a,
            "composite": score,
        }

        composite[role] = score

    ranked = sorted(
        composite.items(),
        key=lambda z: z[1],
        reverse=True,
    )

    return evidence, ranked


# ---------------------------------------------------------------------
# STAGE 2
#
# The candidate set does not exist until Stage 1 has run.
# Therefore this is a genuine dependent second Jev request.
# ---------------------------------------------------------------------

def stage2_request(ex, state, evidence, ranked, top_k=6):
    top = [r for r, _ in ranked[:top_k]]

    candidates = {
        r: ROLE_DESCRIPTIONS.get(r, r)
        for r in top
    }

    state2 = dict(state)

    state2["stage1_candidates"] = {
        r: evidence[r]
        for r in top
    }

    state2["stage1_note"] = (
        "stage1_candidates contains independent Jev evidence dimensions. "
        "Use them as additional evidence, not as ground truth."
    )

    criteria = dict(candidates)
    criteria["none_of_these"] = (
        "None of these candidates has sufficient behavioral evidence."
    )

    q = {
        "winner_actual": {
            "type": "choice",
            "instructions": (
                "Among the candidates, which specialist most likely "
                "ACTUALLY PARTICIPATED in producing "
                "`visible_final_response`? Do not choose a role simply "
                "because `user_query` asks about its domain."
            ),
            "criteria": criteria,
        },

        "winner_fingerprint": {
            "type": "choice",
            "instructions": (
                "Which candidate is best supported by the behavioral "
                "fingerprint in `visible_final_response`, ignoring mere "
                "semantic overlap with `user_query`?"
            ),
            "criteria": criteria,
        },

        "winner_without_topic": {
            "type": "choice",
            "instructions": (
                "Imagine domain/topic words from `user_query` were hidden. "
                "Which candidate would the behavior of "
                "`visible_final_response` most strongly suggest?"
            ),
            "criteria": criteria,
        },
    }

    for role in top:
        path = f"`candidate_roles.{role}`"

        q[f"verify__{role}"] = {
            "type": "noul",
            "instructions": (
                f"Is there SUFFICIENT response-specific evidence to "
                f"attribute actual participation to {path}, rather than "
                f"merely observing that `user_query` or the response "
                f"discusses that domain?"
            ),
            "criteria": {
                "true": "Actual-participation evidence is sufficient.",
                "false": (
                    "Evidence is generic, ambiguous, or mostly topical."
                ),
            },
        }

        q[f"trap__{role}"] = {
            "type": "noul",
            "instructions": (
                f"Would selecting {path} here mainly be a TOPIC-MATCHING "
                f"shortcut caused by the subject of `user_query`?"
            ),
            "criteria": {
                "true": "Selection would mostly be a topical shortcut.",
                "false": (
                    "Selection is supported by response-specific behavior."
                ),
            },
        }

    return state2, q, top


def choice_probs(answer):
    return {
        str(k): float(v)
        for k, v in answer.get(
            "probabilities", {}
        ).items()
    }


def parse_stage2(resp, evidence, top):
    ans = resp["answers"]

    pa = choice_probs(ans["winner_actual"])
    pf = choice_probs(ans["winner_fingerprint"])
    pw = choice_probs(ans["winner_without_topic"])

    final = {}

    details = {}

    for role in top:
        verify = extract_noul(
            ans, f"verify__{role}"
        )

        trap = extract_noul(
            ans, f"trap__{role}"
        )

        c = evidence[role]["composite"]

        # Fixed fusion rule.
        score = (
            0.50 * c
            + 0.20 * pa.get(role, 0.0)
            + 0.10 * pf.get(role, 0.0)
            + 0.05 * pw.get(role, 0.0)
            + 0.20 * verify
            - 0.05 * trap
        )

        final[role] = score

        details[role] = {
            "stage1_composite": c,
            "choice_actual": pa.get(role, 0.0),
            "choice_fingerprint": pf.get(role, 0.0),
            "choice_without_topic": pw.get(role, 0.0),
            "verify": verify,
            "topic_trap": trap,
            "final": score,
        }

    ranked = sorted(
        final.items(),
        key=lambda z: z[1],
        reverse=True,
    )

    return details, ranked


# ---------------------------------------------------------------------
# RUN ONE EXAMPLE
# ---------------------------------------------------------------------

def run_example(ex, key, demos=None, top_k=6):
    state = base_state(ex, demos=demos)

    payload1 = {
        "state": state,
        "model": MODEL,
        "questions": STAGE1_QUESTIONS,
    }

    r1, t1 = call_jev(payload1, key)

    evidence, ranked1 = parse_stage1(r1)

    state2, questions2, top = stage2_request(
        ex,
        state,
        evidence,
        ranked1,
        top_k=top_k,
    )

    payload2 = {
        "state": state2,
        "model": MODEL,
        "questions": questions2,
    }

    r2, t2 = call_jev(payload2, key)

    stage2, ranked2 = parse_stage2(
        r2,
        evidence,
        top,
    )

    truth = ex["truth"]

    rank1_names = [r for r, _ in ranked1]
    truth_rank1 = 1 + rank1_names.index(truth)

    truth_in_top = truth in top

    # Stage 2 only ranks the Top-K candidates.
    if truth_in_top:
        rank2_names = [r for r, _ in ranked2]
        truth_rank2 = 1 + rank2_names.index(truth)
    else:
        truth_rank2 = None

    out = {
        **ex,

        "icl": bool(demos),

        "stage1": {
            "top1": ranked1[0][0],
            "top1_correct": ranked1[0][0] == truth,
            "truth_rank": truth_rank1,
            "truth_score": evidence[truth]["composite"],
            "top6": ranked1[:6],
            "truth_in_top6": truth_in_top,
            "evidence": evidence,
        },

        "stage2": {
            "top1": ranked2[0][0],
            "top1_correct": (
                truth_in_top
                and ranked2[0][0] == truth
            ),
            "truth_rank_within_top6": truth_rank2,
            "ranking": ranked2,
            "details": stage2,
        },

        "usage": {
            "stage1": r1.get("usage"),
            "stage2": r2.get("usage"),
        },

        "timing": {
            "stage1_seconds": t1,
            "stage2_seconds": t2,
            "total_seconds": t1 + t2,
        },

        "jev_model_stage1": r1.get("model"),
        "jev_model_stage2": r2.get("model"),
    }

    return out


# ---------------------------------------------------------------------
# METRICS
# ---------------------------------------------------------------------

def summarize(rows, label):
    if not rows:
        return {"label": label, "n": 0}

    n = len(rows)

    def acc(path):
        return sum(
            bool(
                r[path[0]][path[1]]
            )
            for r in rows
        ) / n

    ranks1 = [
        r["stage1"]["truth_rank"]
        for r in rows
    ]

    return {
        "label": label,
        "n": n,

        "old_v1_noul_accuracy": sum(
            r["v1_noul_correct"]
            for r in rows
        ) / n,

        "old_v1_choice_accuracy": sum(
            r["v1_choice_correct"]
            for r in rows
        ) / n,

        "stage1_composite_accuracy": acc(
            ("stage1", "top1_correct")
        ),

        "stage1_truth_in_top6": sum(
            r["stage1"]["truth_in_top6"]
            for r in rows
        ) / n,

        "stage1_mean_truth_rank": (
            sum(ranks1) / n
        ),

        "stage1_mrr": (
            sum(1.0 / x for x in ranks1) / n
        ),

        "stage2_final_accuracy": acc(
            ("stage2", "top1_correct")
        ),

        "truth_role_counts": dict(
            sorted(
                Counter(
                    r["truth"] for r in rows
                ).items()
            )
        ),

        "input_tokens": sum(
            int(
                (
                    r["usage"]["stage1"]
                    or {}
                ).get("input_tokens", 0)
            )
            +
            int(
                (
                    r["usage"]["stage2"]
                    or {}
                ).get("input_tokens", 0)
            )
            for r in rows
        ),

        "output_tokens": sum(
            int(
                (
                    r["usage"]["stage1"]
                    or {}
                ).get("output_tokens", 0)
            )
            +
            int(
                (
                    r["usage"]["stage2"]
                    or {}
                ).get("output_tokens", 0)
            )
            for r in rows
        ),
    }


def run_mode(
    name,
    examples,
    outdir,
    key,
    demos=None,
):
    result_file = outdir / f"{name}.jsonl"

    completed = {
        x["example_id"]: x
        for x in read_jsonl(result_file)
    }

    rows = []

    print()
    print("#" * 100)
    print(f"MODE: {name}")
    print("#" * 100)

    if demos:
        print(
            "[ICL CALIBRATION IDS] "
            + ", ".join(x["example_id"] for x in demos)
        )

    for i, ex in enumerate(examples, 1):
        eid = ex["example_id"]

        if eid in completed:
            row = completed[eid]
            rows.append(row)
            print(
                f"[resume] {i}/{len(examples)} {eid}",
                flush=True,
            )
            continue

        print(
            f"[run] {i}/{len(examples)} "
            f"{eid} truth={ex['truth']}",
            flush=True,
        )

        row = run_example(
            ex,
            key,
            demos=demos,
            top_k=6,
        )

        append_jsonl(result_file, row)
        rows.append(row)

        print(
            "      "
            f"V1={row['v1_noul_top1']} "
            f"S1={row['stage1']['top1']} "
            f"rank={row['stage1']['truth_rank']} "
            f"in6={row['stage1']['truth_in_top6']} "
            f"FINAL={row['stage2']['top1']} "
            f"OK={row['stage2']['top1_correct']} "
            f"time={row['timing']['total_seconds']:.2f}s",
            flush=True,
        )

        time.sleep(0.05)

    critical = [
        x for x in rows
        if x["example_id"].startswith("critical_")
    ]

    historical = [
        x for x in rows
        if not x["example_id"].startswith("critical_")
    ]

    summary = {
        "mode": name,
        "all": summarize(rows, "all"),
        "critical": summarize(
            critical, "critical"
        ),
        "historical": summarize(
            historical, "historical"
        ),
    }

    return rows, summary


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--output-dir",
        default="artifacts/jev_max_v2",
    )

    ap.add_argument(
        "--calibration-n",
        type=int,
        default=8,
    )

    args = ap.parse_args()

    key = os.environ.get("TYPESAFE_API_KEY")

    if not key:
        raise SystemExit(
            "ERROR: TYPESAFE_API_KEY missing"
        )

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples = load_examples()

    critical = [
        x for x in examples
        if x["example_id"].startswith("critical_")
    ]

    historical = [
        x for x in examples
        if not x["example_id"].startswith("critical_")
    ]

    print("=" * 100)
    print("JEV MAX BENCHMARK")
    print("=" * 100)

    print(f"model={MODEL}")
    print(f"examples={len(examples)}")
    print(f"critical={len(critical)}")
    print(f"historical={len(historical)}")

    print(
        "GROUND TRUTH IS NEVER SENT IN ZERO-SHOT MODE.",
        flush=True,
    )

    # --------------------------------------------------------------
    # V2: full zero-shot max.
    # Same 104 observations as original Jev V1.
    # --------------------------------------------------------------

    rows_v2, summary_v2 = run_mode(
        "v2_zero_shot_max",
        examples,
        outdir,
        key,
        demos=None,
    )

    # --------------------------------------------------------------
    # V3: ICL.
    #
    # 8 calibration examples are taken from historical examples,
    # and REMOVED from V3 evaluation.
    #
    # Critical 4 remain completely held-out.
    # --------------------------------------------------------------

    calibration = choose_calibration(
        examples,
        n=args.calibration_n,
    )

    calibration_ids = {
        x["example_id"]
        for x in calibration
    }

    eval_v3 = [
        x for x in examples
        if x["example_id"] not in calibration_ids
    ]

    (outdir / "icl_calibration.json").write_text(
        json.dumps(
            [
                {
                    "example_id": x["example_id"],
                    "truth": x["truth"],
                    "query": x["query"],
                }
                for x in calibration
            ],
            indent=2,
        ),
        encoding="utf-8",
    )

    rows_v3, summary_v3 = run_mode(
        "v3_icl_max",
        eval_v3,
        outdir,
        key,
        demos=calibration,
    )

    report = {
        "model_alias": MODEL,
        "v2_zero_shot_max": summary_v2,
        "v3_icl_max": summary_v3,
        "v3_calibration_ids": sorted(
            calibration_ids
        ),
    }

    (outdir / "summary.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        json.dumps(
            report,
            indent=2,
        )
    )

    print()
    print("=" * 100)
    print("CRITICAL CASES — V2 ZERO SHOT")
    print("=" * 100)

    for x in rows_v2:
        if not x["example_id"].startswith(
            "critical_"
        ):
            continue

        print(
            f"{x['example_id']} "
            f"truth={x['truth']} "
            f"old={x['v1_noul_top1']} "
            f"stage1={x['stage1']['top1']} "
            f"rank={x['stage1']['truth_rank']} "
            f"FINAL={x['stage2']['top1']} "
            f"OK={x['stage2']['top1_correct']}"
        )

    print()
    print("=" * 100)
    print("CRITICAL CASES — V3 ICL")
    print("=" * 100)

    for x in rows_v3:
        if not x["example_id"].startswith(
            "critical_"
        ):
            continue

        print(
            f"{x['example_id']} "
            f"truth={x['truth']} "
            f"old={x['v1_noul_top1']} "
            f"stage1={x['stage1']['top1']} "
            f"rank={x['stage1']['truth_rank']} "
            f"FINAL={x['stage2']['top1']} "
            f"OK={x['stage2']['top1_correct']}"
        )


if __name__ == "__main__":
    main()
