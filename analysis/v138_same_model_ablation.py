#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from collections import Counter, defaultdict

import numpy as np

import analysis.train_v13_6b_activation_judge as tr
import analysis.v137_calibration_audit as ca


ROOT = Path("artifacts/v13_7_calibration_audit_v1")
OUT = Path("artifacts/v138_same_model_ablation")


def base_type(t):
    return t.split(":", 1)[0]


def replace_section(text, header, next_header, replacement):
    start = f"[{header}]\n"
    end = f"\n\n[{next_header}]"

    if start not in text or end not in text:
        raise RuntimeError(
            f"Cannot locate section {header} -> {next_header}"
        )

    left, rest = text.split(start, 1)
    _, right = rest.split(end, 1)

    return (
        left
        + start
        + replacement
        + end
        + right
    )


def no_answer_text(text):
    return replace_section(
        text,
        "VISIBLE MAS RESPONSE",
        "DECISION",
        "<NO_VISIBLE_RESPONSE>",
    )


def answer_only_text(text):
    x = replace_section(
        text,
        "PROBE TARGET",
        "BENIGN PROBE",
        "<MASKED>",
    )

    x = replace_section(
        x,
        "BENIGN PROBE",
        "VISIBLE MAS RESPONSE",
        "<MASKED>",
    )

    return x


def reconstruct():
    rd = dict(tr.ROLE_DESCRIPTIONS)

    r13 = tr.read_jsonl(ca.V13)
    r11 = tr.read_jsonl(ca.V11)

    c13, _ = tr.normalize_rows(
        r13, "v13", rd, 1.0
    )

    c11, _ = tr.normalize_rows(
        r11, "v11", rd, ca.V11_WEIGHT
    )

    cases = tr.attach_components(c13 + c11)

    args = SimpleNamespace(
        seed=ca.SEED,
        n_splits=ca.N_SPLITS,
        fold=ca.FOLD,
    )

    train_cases, val_cases, split_report = tr.split_cases(
        cases,
        args,
    )

    summary = json.loads(
        (ROOT / "summary.json").read_text()
    )

    salt = int(summary["split_salt"])

    cal_cases = []
    test_cases = []

    for c in val_cases:
        bucket = ca.stable_bucket(
            ca.component_value(c),
            salt,
        )

        if bucket < int(ca.CAL_FRACTION * 100):
            cal_cases.append(c)
        else:
            test_cases.append(c)

    cal_pairs, cal_counts = tr.build_pairs(
        cal_cases,
        rd,
    )

    test_pairs, test_counts = tr.build_pairs(
        test_cases,
        rd,
    )

    return {
        "rd": rd,
        "summary": summary,
        "train_cases": train_cases,
        "val_cases": val_cases,
        "cal_cases": cal_cases,
        "test_cases": test_cases,
        "cal_pairs": cal_pairs,
        "test_pairs": test_pairs,
        "cal_counts": cal_counts,
        "test_counts": test_counts,
        "split_report": split_report,
    }


def audit(data):
    cal_pairs = data["cal_pairs"]
    test_pairs = data["test_pairs"]
    summary = data["summary"]

    def endpoint_texts(pairs):
        xs = []

        for p in pairs:
            xs.append(p.pos_text)
            xs.append(p.neg_text)

        return xs

    cal_full = endpoint_texts(cal_pairs)
    test_full = endpoint_texts(test_pairs)

    cal_no = [no_answer_text(x) for x in cal_full]
    test_no = [no_answer_text(x) for x in test_full]

    cal_ao = [answer_only_text(x) for x in cal_full]
    test_ao = [answer_only_text(x) for x in test_full]

    print("===== V13.8 SAME-MODEL ABLATION AUDIT =====")
    print("train cases :", len(data["train_cases"]))
    print("val cases   :", len(data["val_cases"]))
    print("cal cases   :", len(data["cal_cases"]))
    print("test cases  :", len(data["test_cases"]))
    print()

    print("CAL pairs   :", len(cal_pairs))
    print("TEST pairs  :", len(test_pairs))
    print()

    print("CAL pair counts:")
    for k, v in sorted(data["cal_counts"].items()):
        print(f"  {k:42s} {v}")

    print("\nTEST pair counts:")
    for k, v in sorted(data["test_counts"].items()):
        print(f"  {k:42s} {v}")

    print("\n===== UNIQUE TEXT LOAD =====")
    print(
        "CAL full unique       :",
        len(set(cal_full)),
    )
    print(
        "TEST full unique      :",
        len(set(test_full)),
    )
    print(
        "CAL no-answer unique  :",
        len(set(cal_no)),
    )
    print(
        "TEST no-answer unique :",
        len(set(test_no)),
    )
    print(
        "CAL answer-only unique:",
        len(set(cal_ao)),
    )
    print(
        "TEST answer-only uniq :",
        len(set(test_ao)),
    )

    # Exact expected untouched test reconstruction.
    assert len(data["test_cases"]) == summary["test_cases"]
    assert len(test_pairs) == summary["counterfactual_test"]["n_pairs"]

    expected = {
        str(k): int(v)
        for k, v in summary["test_pair_counts"].items()
    }

    actual = {
        str(k): int(v)
        for k, v in data["test_counts"].items()
    }

    for k, v in expected.items():
        assert actual.get(k, 0) == v, (
            k,
            actual.get(k, 0),
            v,
        )

    # Same-role activation pairs MUST collapse to identical
    # NO-ANSWER inputs on each side.
    same_role_total = 0
    same_role_bad = 0

    for p in cal_pairs + test_pairs:
        if not base_type(p.pair_type).startswith(
            "activation_cf_"
        ):
            continue

        same_role_total += 1

        if no_answer_text(p.pos_text) != no_answer_text(
            p.neg_text
        ):
            same_role_bad += 1

    print("\n===== CAUSAL SANITY =====")
    print(
        "same-role CF pairs:",
        same_role_total,
    )
    print(
        "NO-ANSWER mismatches:",
        same_role_bad,
    )

    assert same_role_bad == 0

    # Ensure masking actually changes the relevant text.
    assert all(
        answer_only_text(x) != x
        for x in cal_full[:50]
    )

    audit_json = {
        "protocol":
            "v13.8_same_v137_model_input_ablation",
        "model_inference": False,
        "training": False,
        "api_calls": 0,
        "cal_cases": len(data["cal_cases"]),
        "test_cases": len(data["test_cases"]),
        "cal_pairs": len(cal_pairs),
        "test_pairs": len(test_pairs),
        "cal_pair_counts":
            dict(data["cal_counts"]),
        "test_pair_counts":
            dict(data["test_counts"]),
        "unique_no_answer":
            len(set(cal_no + test_no)),
        "unique_answer_only":
            len(set(cal_ao + test_ao)),
        "same_role_noanswer_mismatches":
            same_role_bad,
    }

    OUT.mkdir(parents=True, exist_ok=True)

    (OUT / "audit.json").write_text(
        json.dumps(
            audit_json,
            indent=2,
            sort_keys=True,
        )
    )

    print("\nAUDIT PASS")
    print("wrote:", OUT / "audit.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--audit-only",
        action="store_true",
    )

    args = ap.parse_args()

    data = reconstruct()
    audit(data)

    if args.audit_only:
        print(
            "AUDIT-ONLY PASS: "
            "no model loaded, no CUDA, no API/MAS calls."
        )
        return

    raise RuntimeError(
        "GPU inference path intentionally not enabled yet. "
        "Run --audit-only first."
    )


if __name__ == "__main__":
    main()
