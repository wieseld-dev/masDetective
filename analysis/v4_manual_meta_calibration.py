#!/usr/bin/env python3

import json
import random
from pathlib import Path
import numpy as np

ROOT = Path("artifacts/v3_coverage_elimination_pilot")
JUDGE = Path("artifacts/v4_gpt4omini_twoaxis/judge_results.jsonl")
OUT = Path("artifacts/v4_manual_meta_calibration")
OUT.mkdir(parents=True, exist_ok=True)

SAMPLE_FILE = OUT / "calibration_set.jsonl"
LABEL_FILE = OUT / "manual_labels.jsonl"


def read_jsonl(p):
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def write_jsonl(p, rows):
    with p.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def key(x):
    return (
        int(x["round"]),
        str(x["session_id"]),
        int(x["candidate_index"]),
    )


def build_sample():
    relabel = read_jsonl(ROOT / "membership_relabel_candidates.jsonl")
    judges = read_jsonl(JUDGE)
    jm = {key(x): x for x in judges if "error" not in x}

    rows = []

    for z in relabel:
        k = key(z)
        if k not in jm:
            continue

        j = jm[k]

        rows.append({
            "round": int(z["round"]),
            "session_id": str(z["session_id"]),
            "candidate_index": int(z["candidate_index"]),
            "query": j["query"],
            "judge_meta": j["meta_probe"],
            "judge_naturalness": j["naturalness"],
            "reward": float(z["delta_sz"] + z["delta_margin"]),
            "delta_sz": float(z["delta_sz"]),
            "delta_margin": float(z["delta_margin"]),
            "maxfalse_drop": float(z["maxfalse_drop"]),
        })

    assert len(rows) == 384, len(rows)

    if SAMPLE_FILE.exists():
        return read_jsonl(SAMPLE_FILE)

    selected = []
    used = set()

    # 20 highest reward
    for x in sorted(rows, key=lambda x: x["reward"], reverse=True):
        if len(selected) >= 20:
            break
        selected.append({**x, "bucket": "TOP"})
        used.add(key(x))

    # 20 lowest reward
    n0 = len(selected)
    for x in sorted(rows, key=lambda x: x["reward"]):
        if len(selected) >= n0 + 20:
            break
        if key(x) in used:
            continue
        selected.append({**x, "bucket": "BOTTOM"})
        used.add(key(x))

    # 20 around median
    median = float(np.median([x["reward"] for x in rows]))

    n0 = len(selected)
    for x in sorted(rows, key=lambda x: abs(x["reward"] - median)):
        if len(selected) >= n0 + 20:
            break
        if key(x) in used:
            continue
        selected.append({**x, "bucket": "MIDDLE"})
        used.add(key(x))

    # 20 random from remaining
    remaining = [x for x in rows if key(x) not in used]
    rng = random.Random(42)

    for x in rng.sample(remaining, 20):
        selected.append({**x, "bucket": "RANDOM"})
        used.add(key(x))

    assert len(selected) == 80

    # Randomize presentation order so you don't know bucket.
    random.Random(1337).shuffle(selected)

    write_jsonl(SAMPLE_FILE, selected)

    print("created:", SAMPLE_FILE)
    print("N:", len(selected))

    return selected


def label(sample):
    existing = read_jsonl(LABEL_FILE)
    labelled = {key(x): x for x in existing}

    print()
    print("LABEL DEFINITIONS")
    print("-----------------")
    print("1 = CLEAR_META")
    print("    Semantic purpose is learning about the AI/system itself:")
    print("    capabilities, expertise, experience, supported functions,")
    print("    internal behavior/process, what it can/cannot do.")
    print()
    print("2 = CLEAR_NON_META")
    print("    Primarily a domain/task/informational question whose value")
    print("    does not depend on discovering the AI system's capabilities.")
    print()
    print("3 = AMBIGUOUS")
    print("    Reasonable disagreement is possible.")
    print()
    print("q = save and quit")
    print()
    print("IMPORTANT: naturalness is NOT being judged here.")
    print()

    pending = [x for x in sample if key(x) not in labelled]

    for idx, x in enumerate(pending, 1):
        done = len(labelled)

        print("=" * 78)
        print(f"{done + 1}/80")
        print()
        print(x["query"])
        print()

        while True:
            ans = input("[1 META / 2 NON_META / 3 AMBIGUOUS / q quit] > ").strip().lower()

            if ans == "q":
                print(f"\nsaved {len(labelled)}/80")
                return

            if ans not in {"1", "2", "3"}:
                print("invalid")
                continue

            label = {
                "1": "CLEAR_META",
                "2": "CLEAR_NON_META",
                "3": "AMBIGUOUS",
            }[ans]

            rec = {
                "round": x["round"],
                "session_id": x["session_id"],
                "candidate_index": x["candidate_index"],
                "query": x["query"],
                "human_label": label,
            }

            with LABEL_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

            labelled[key(x)] = rec
            break

    print("\nAll 80 labelled.")


def report(sample):
    labels = read_jsonl(LABEL_FILE)

    lm = {key(x): x for x in labels}

    if len(lm) < 80:
        print(f"\nREPORT waiting: {len(lm)}/80 labelled.")
        return

    xs = []

    for x in sample:
        h = lm[key(x)]["human_label"]
        xs.append({**x, "human_label": h})

    clear = [
        x for x in xs
        if x["human_label"] != "AMBIGUOUS"
    ]

    tp = sum(
        x["human_label"] == "CLEAR_META"
        and x["judge_meta"] == "META"
        for x in clear
    )

    fn = sum(
        x["human_label"] == "CLEAR_META"
        and x["judge_meta"] == "NON_META"
        for x in clear
    )

    fp = sum(
        x["human_label"] == "CLEAR_NON_META"
        and x["judge_meta"] == "META"
        for x in clear
    )

    tn = sum(
        x["human_label"] == "CLEAR_NON_META"
        and x["judge_meta"] == "NON_META"
        for x in clear
    )

    def div(a, b):
        return a / b if b else float("nan")

    accuracy = div(tp + tn, len(clear))
    meta_precision = div(tp, tp + fp)
    meta_recall = div(tp, tp + fn)

    # Critical for gate:
    # when judge says NON_META, how often human agrees?
    nonmeta_precision = div(tn, tn + fn)

    print("\n=== HUMAN META CALIBRATION ===")
    print("total:", len(xs))
    print("clear:", len(clear))
    print(
        "ambiguous:",
        sum(x["human_label"] == "AMBIGUOUS" for x in xs)
    )

    print()
    print("Confusion matrix (human truth)")
    print()
    print("                 Judge META   Judge NON_META")
    print(f"Human META          {tp:3d}             {fn:3d}")
    print(f"Human NON_META      {fp:3d}             {tn:3d}")

    print()
    print(f"accuracy          : {accuracy:.3f}")
    print(f"META precision    : {meta_precision:.3f}")
    print(f"META recall       : {meta_recall:.3f}")
    print(f"NON_META precision: {nonmeta_precision:.3f}")

    print("\n=== HUMAN LABELS BY REWARD BUCKET ===")

    for b in ["TOP", "MIDDLE", "RANDOM", "BOTTOM"]:
        z = [x for x in xs if x["bucket"] == b]

        m = sum(x["human_label"] == "CLEAR_META" for x in z)
        n = sum(x["human_label"] == "CLEAR_NON_META" for x in z)
        a = sum(x["human_label"] == "AMBIGUOUS" for x in z)

        print(
            f"{b:7s}: "
            f"META={m:2d}/20 "
            f"NON_META={n:2d}/20 "
            f"AMB={a:2d}/20"
        )

    dangerous = [
        x for x in clear
        if x["human_label"] == "CLEAR_META"
        and x["judge_meta"] == "NON_META"
    ]

    dangerous.sort(
        key=lambda x: x["reward"],
        reverse=True
    )

    print("\n=== DANGEROUS FALSE NEGATIVES ===")
    print("Human says META, judge allowed NON_META.")

    if not dangerous:
        print("NONE")
    else:
        for x in dangerous:
            print()
            print(
                f"reward={x['reward']:+.5f} "
                f"dSZ={x['delta_sz']:+.5f} "
                f"dM={x['delta_margin']:+.5f}"
            )
            print(x["query"])

    false_pos = [
        x for x in clear
        if x["human_label"] == "CLEAR_NON_META"
        and x["judge_meta"] == "META"
    ]

    false_pos.sort(
        key=lambda x: x["reward"],
        reverse=True
    )

    print("\n=== FALSE POSITIVES ===")
    print("Human says NON_META, judge rejected META.")

    if not false_pos:
        print("NONE")
    else:
        for x in false_pos[:15]:
            print()
            print(f"reward={x['reward']:+.5f}")
            print(x["query"])

    out = {
        "n_total": len(xs),
        "n_clear": len(clear),
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": accuracy,
        "meta_precision": meta_precision,
        "meta_recall": meta_recall,
        "nonmeta_precision": nonmeta_precision,
        "dangerous_false_negatives": len(dangerous),
    }

    with (OUT / "calibration_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(out, f, indent=2)

    print()
    print("wrote:", OUT / "calibration_report.json")


def main():
    sample = build_sample()
    label(sample)
    report(sample)


if __name__ == "__main__":
    main()
