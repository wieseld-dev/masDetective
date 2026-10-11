#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("artifacts/general_role_sensor_v1")
OUT = Path("artifacts/general_role_sensor_trainset_v1")

PUBLIC = ROOT / "target_public_responses.jsonl"
TRUTH  = ROOT / "target_truth_training_eval_only.jsonl"


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(
            f"{OUT} is non-empty; refusing overwrite."
        )

    OUT.mkdir(parents=True, exist_ok=True)

    public = read_jsonl(PUBLIC)
    truth = read_jsonl(TRUTH)

    assert len(public) == 3888
    assert len(truth) == 3888

    pub_by = {
        str(x["execution_id"]): x
        for x in public
    }
    tru_by = {
        str(x["execution_id"]): x
        for x in truth
    }

    assert len(pub_by) == 3888
    assert len(tru_by) == 3888
    assert set(pub_by) == set(tru_by)

    clean = []

    for eid in sorted(pub_by):
        p = pub_by[eid]
        t = tru_by[eid]

        assert p["split"] == t["split"]
        assert p["candidate_role"] == t["target_role"]
        assert p["candidate_family"] == t["target_family"]

        split = str(p["split"])
        role = str(p["candidate_role"])
        family = str(p["candidate_family"])
        query = str(p["query"])
        output = str(p["final_output"])

        label = int(t["installed_label"])
        branch = str(t["branch"])
        pair_index = int(t["pair_index"])

        if branch == "PLUS":
            assert label == 1
        elif branch == "MINUS":
            assert label == 0
        else:
            raise RuntimeError(
                f"unexpected branch {branch}"
            )

        if not query.strip():
            raise RuntimeError(f"empty query: {eid}")

        if not output.strip():
            raise RuntimeError(f"empty output: {eid}")

        clean.append({
            # Only identifier for pairing/audit.
            "execution_id": eid,

            # Public inference-time conditioning.
            "target_role": role,
            "target_family": family,
            "probe_id": str(p["probe_id"]),
            "probe_source": str(p["probe_source"]),
            "query": query,
            "final_output": output,

            # TRAIN / VAL metadata.
            "split": split,
            "pair_index": pair_index,
            "branch": branch,

            # TRAINING / EVALUATION LABEL ONLY.
            "label": label,
        })

    train = [
        x for x in clean
        if x["split"] == "train"
    ]
    val = [
        x for x in clean
        if x["split"] == "val"
    ]

    assert len(train) == 3240
    assert len(val) == 648

    train_roles = sorted({
        x["target_role"] for x in train
    })
    val_roles = sorted({
        x["target_role"] for x in val
    })

    assert len(train_roles) == 60
    assert len(val_roles) == 12
    assert not (set(train_roles) & set(val_roles))

    # ----------------------------------------------------------
    # Exact PLUS/MINUS pairing:
    # one PLUS and one MINUS for every
    # role × pair_index × probe_id.
    # ----------------------------------------------------------

    for split_name, rows in [
        ("train", train),
        ("val", val),
    ]:
        groups = defaultdict(list)

        for x in rows:
            key = (
                x["target_role"],
                x["pair_index"],
                x["probe_id"],
            )
            groups[key].append(x)

        expected_pairs = (
            1620 if split_name == "train"
            else 324
        )

        assert len(groups) == expected_pairs

        for key, g in groups.items():
            if len(g) != 2:
                raise RuntimeError(
                    f"{split_name} bad pair size "
                    f"{key}: {len(g)}"
                )

            assert {x["branch"] for x in g} == {
                "PLUS", "MINUS"
            }
            assert {x["label"] for x in g} == {
                0, 1
            }
            assert len({
                x["query"] for x in g
            }) == 1

    # ----------------------------------------------------------
    # Strong query-only leakage sanity:
    # each exact target/probe query occurs 3 PLUS + 3 MINUS.
    # ----------------------------------------------------------

    for split_name, rows in [
        ("train", train),
        ("val", val),
    ]:
        qgroups = defaultdict(list)

        for x in rows:
            key = (
                x["target_role"],
                x["probe_id"],
                x["query"],
            )
            qgroups[key].append(x)

        for key, g in qgroups.items():
            labels = Counter(
                x["label"] for x in g
            )

            if labels != Counter({0: 3, 1: 3}):
                raise RuntimeError(
                    f"query balance failure "
                    f"{split_name} {key}: {labels}"
                )

    write_jsonl(
        OUT / "train.jsonl",
        train,
    )
    write_jsonl(
        OUT / "val.jsonl",
        val,
    )

    audit = {
        "version":
            "general_role_sensor_trainset_v1",

        "source":
            "general_role_sensor_v1",

        "train_rows":
            len(train),

        "val_rows":
            len(val),

        "train_roles":
            train_roles,

        "val_roles":
            val_roles,

        "train_role_count":
            len(train_roles),

        "val_role_count":
            len(val_roles),

        "role_overlap":
            sorted(
                set(train_roles)
                & set(val_roles)
            ),

        "train_pairs":
            1620,

        "val_pairs":
            324,

        "train_labels":
            dict(Counter(
                x["label"]
                for x in train
            )),

        "val_labels":
            dict(Counter(
                x["label"]
                for x in val
            )),

        "features_allowed": [
            "target_role",
            "target_family",
            "query",
            "final_output",
        ],

        "forbidden_not_present": [
            "system_id",
            "k",
            "execution_mode",
            "style",
            "installed_roles",
            "selected_roles",
            "target_activated",
        ],

        "internal_test_used":
            False,
    }

    (OUT / "audit.json").write_text(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    for name in [
        "train.jsonl",
        "val.jsonl",
        "audit.json",
    ]:
        p = OUT / name
        (OUT / f"{name}.sha256").write_text(
            f"{sha256(p)}  {name}\n"
        )

    print("=" * 90)
    print("GENERAL ROLE SENSOR CLEAN DATASET")
    print("=" * 90)
    print("TRAIN rows :", len(train))
    print("VAL rows   :", len(val))
    print("TRAIN roles:", len(train_roles))
    print("VAL roles  :", len(val_roles))
    print("TRAIN pairs:", 1620)
    print("VAL pairs  :", 324)
    print("TRAIN label:", Counter(
        x["label"] for x in train
    ))
    print("VAL label  :", Counter(
        x["label"] for x in val
    ))
    print("role overlap:", [])
    print("INTERNAL TEST USED: FALSE")
    print("PASS")


if __name__ == "__main__":
    main()
