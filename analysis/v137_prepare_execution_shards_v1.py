#!/usr/bin/env python3

import copy
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

PLAN = Path(
    "artifacts/v137_presence_v1/execution_plan_v1/"
    "execute_now_plan.jsonl"
)

PLAN_SHA = Path(
    "artifacts/v137_presence_v1/execution_plan_v1/"
    "execute_now_plan.sha256"
)

OUT = Path(
    "artifacts/v137_presence_v1/execution_shards_v1"
)

OUT.mkdir(parents=True, exist_ok=True)

FORBIDDEN_CANDIDATE_KEYS = {
    "label_target_installed",
    "K_training_only",
    "mode_training_only",
    "truth_roles_training_only",
    "selected_roles_training_only",
    "pair_id",
    "pair_side",
    "match_overlap_other_roles",
    "match_other_role_jaccard",
    "match_exact_one_role_swap",
}


def read_jsonl(path):
    out = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))

    return out


def write_jsonl(path, rows):
    with path.open("w") as f:
        for row in rows:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + "\n"
            )


def sha256_file(path):
    return hashlib.sha256(
        path.read_bytes()
    ).hexdigest()


# ------------------------------------------------------------
# Verify frozen execute-now plan.
# ------------------------------------------------------------

expected_plan_sha = (
    PLAN_SHA.read_text()
    .strip()
    .split()[0]
)

actual_plan_sha = sha256_file(PLAN)

if expected_plan_sha != actual_plan_sha:
    raise RuntimeError(
        "EXECUTION PLAN SHA MISMATCH\n"
        f"expected={expected_plan_sha}\n"
        f"actual={actual_plan_sha}"
    )


plan_rows = read_jsonl(PLAN)

if len(plan_rows) != 950:
    raise RuntimeError(
        f"Expected 950 execute-now rows, got {len(plan_rows)}"
    )

if not all(x.get("execute_now") is True for x in plan_rows):
    raise RuntimeError(
        "execute_now_plan contains held-out rows"
    )

if any(
    x["dataset_split"] == "internal_test_small"
    for x in plan_rows
):
    raise RuntimeError(
        "INTERNAL TEST LEAKED INTO EXECUTE-NOW PLAN"
    )


# ------------------------------------------------------------
# Unique cache keys.
#
# Executor cache key = (session_id, query).
# Duplicate here would silently collapse experiments.
# ------------------------------------------------------------

seen_keys = {}

for i, row in enumerate(plan_rows):
    key = (
        row["system_id"],
        row["query"],
    )

    if key in seen_keys:
        raise RuntimeError(
            "Duplicate executor cache key:\n"
            f"first={seen_keys[key]}\n"
            f"second={i}\n"
            f"key={key}"
        )

    seen_keys[key] = i


# ------------------------------------------------------------
# Cache source candidate templates and manifests.
# ------------------------------------------------------------

candidate_cache = {}
manifest_cache = {}


def candidate_templates(path):
    path = str(path)

    if path not in candidate_cache:
        rows = read_jsonl(Path(path))

        by_sid = {}

        for row in rows:
            sid = (
                row.get("session_id")
                or row.get("system_id")
            )

            if sid and sid not in by_sid:
                by_sid[str(sid)] = row

        candidate_cache[path] = by_sid

    return candidate_cache[path]


def manifest_systems(path):
    path = str(path)

    if path not in manifest_cache:
        obj = json.loads(
            Path(path).read_text()
        )

        if "systems" not in obj:
            raise RuntimeError(
                f"Manifest lacks systems: {path}"
            )

        by_sid = {
            str(x["session_id"]): x
            for x in obj["systems"]
        }

        manifest_cache[path] = (
            obj,
            by_sid,
        )

    return manifest_cache[path]


# ------------------------------------------------------------
# Group by the exact original manifest + candidate source.
#
# One executor invocation receives ONE manifest.
# ------------------------------------------------------------

groups = defaultdict(list)

for row in plan_rows:
    key = (
        row["manifest"],
        row["candidate_path"],
        row["source_root"],
    )

    groups[key].append(row)


group_keys = sorted(groups)

shard_meta = []
all_example_ids = set()


for shard_index, key in enumerate(group_keys):
    manifest_path, candidate_path, source_root = key

    root = OUT / f"shard_{shard_index:03d}"
    root.mkdir(
        parents=True,
        exist_ok=True,
    )

    template_by_sid = candidate_templates(
        candidate_path
    )

    manifest_obj, manifest_by_sid = (
        manifest_systems(manifest_path)
    )

    candidates = []
    sidecar = []

    rows = sorted(
        groups[key],
        key=lambda x: (
            x["dataset_split"],
            x["system_id"],
            x["target_role"],
            x["query"],
        ),
    )

    for local_index, plan_row in enumerate(rows):
        sid = str(
            plan_row["system_id"]
        )

        if sid not in template_by_sid:
            raise RuntimeError(
                f"Missing candidate template: "
                f"sid={sid} path={candidate_path}"
            )

        if sid not in manifest_by_sid:
            raise RuntimeError(
                f"Missing manifest system: "
                f"sid={sid} manifest={manifest_path}"
            )

        m = manifest_by_sid[sid]

        # Important:
        # all source systems came from TRAIN artifacts.
        if str(m.get("split")) != "train":
            raise RuntimeError(
                f"Source system is not manifest split=train: "
                f"sid={sid} split={m.get('split')}"
            )

        query = str(
            plan_row["query"]
        ).strip()

        if not query:
            raise RuntimeError(
                f"Empty query: {sid}"
            )

        fingerprint = (
            f"{plan_row['dataset_split']}\0"
            f"{plan_row['probe_family']}\0"
            f"{sid}\0"
            f"{plan_row['target_role']}\0"
            f"{query}"
        )

        example_id = (
            "v137_"
            + hashlib.sha256(
                fingerprint.encode()
            ).hexdigest()[:20]
        )

        if example_id in all_example_ids:
            raise RuntimeError(
                f"Duplicate example id: {example_id}"
            )

        all_example_ids.add(
            example_id
        )

        # Reuse existing candidate format exactly as the
        # canonical experiment did.
        c = copy.deepcopy(
            template_by_sid[sid]
        )

        # Remove anything that could be mistaken for truth.
        for forbidden in FORBIDDEN_CANDIDATE_KEYS:
            c.pop(forbidden, None)

        # Also remove any old execution output if present.
        for k in [
            "final_output",
            "output",
            "answer",
            "execution_status",
            "cache_hit",
            "retries_used",
            "action",
        ]:
            c.pop(k, None)

        # Replace only execution/probe metadata.
        c["session_id"] = sid
        c["split"] = "train"
        c["round"] = 1
        c["candidate_index"] = local_index

        c["group_id"] = (
            f"{sid}::v137::{example_id}"
        )

        c["query"] = query
        c["valid"] = True
        c["invalid_reason"] = ""

        c["prompt"] = (
            "[V13.7_TARGETED_BENIGN_PROBE "
            f"family={plan_row['probe_family']} "
            f"target={plan_row['target_role']}]"
        )

        c["completion"] = query
        c["raw_completion"] = query
        c["local_attempts"] = 1
        c["chosen_for_continuation"] = False

        # Neutralize old selector metadata.
        if "selector_predicted_utility" in c:
            c["selector_predicted_utility"] = 0.0

        if "selector_model_samples" in c:
            c["selector_model_samples"] = 0

        if "selector_mode" in c:
            c["selector_mode"] = (
                "v137_targeted_presence"
            )

        if "selector_epsilon" in c:
            c["selector_epsilon"] = 0.0

        # Safe identifier only.
        # This is NOT shown to the MAS; executor only sends query.
        c["v137_example_id"] = example_id

        # Assert no truth fields reached the candidate.
        bad = (
            set(c)
            & FORBIDDEN_CANDIDATE_KEYS
        )

        if bad:
            raise RuntimeError(
                f"Forbidden fields in candidate: {bad}"
            )

        candidates.append(c)

        s = dict(plan_row)
        s["v137_example_id"] = example_id
        s["shard_index"] = shard_index
        s["shard_candidate_index"] = local_index

        sidecar.append(s)

    candidate_out = (
        root
        / "candidates_train_round_01.jsonl"
    )

    sidecar_out = (
        root
        / "v137_sidecar_training_only.jsonl"
    )

    write_jsonl(
        candidate_out,
        candidates,
    )

    write_jsonl(
        sidecar_out,
        sidecar,
    )

    # Ensure no forbidden field got serialized.
    for c in read_jsonl(candidate_out):
        bad = (
            set(c)
            & FORBIDDEN_CANDIDATE_KEYS
        )

        if bad:
            raise RuntimeError(
                f"Serialized candidate leakage: {bad}"
            )

    shard_meta.append({
        "index": shard_index,
        "shard_id": f"shard_{shard_index:03d}",
        "root": str(root),
        "manifest": manifest_path,
        "source_candidate_path": candidate_path,
        "source_root": source_root,
        "rows": len(candidates),
        "candidate_sha256": sha256_file(
            candidate_out
        ),
    })


# ------------------------------------------------------------
# Global shard integrity.
# ------------------------------------------------------------

if sum(x["rows"] for x in shard_meta) != 950:
    raise RuntimeError(
        "Shard row total != 950"
    )


# ------------------------------------------------------------
# Save shard registry.
# ------------------------------------------------------------

registry = {
    "version": "v137_execution_shards_v1",
    "execute_plan_sha256": actual_plan_sha,
    "execution_split": "train",
    "round": 1,
    "total_rows": 950,
    "num_shards": len(shard_meta),
    "shards": shard_meta,
}

registry_path = OUT / "shards.json"

registry_text = json.dumps(
    registry,
    indent=2,
    sort_keys=True,
)

registry_path.write_text(
    registry_text
)

registry_sha = hashlib.sha256(
    registry_text.encode()
).hexdigest()

(OUT / "shards.sha256").write_text(
    registry_sha
    + "  shards.json\n"
)


# TSV is convenient for future SLURM array.
with (OUT / "shards.tsv").open("w") as f:
    f.write(
        "index\tshard_id\troot\tmanifest\trows\n"
    )

    for x in shard_meta:
        f.write(
            f"{x['index']}\t"
            f"{x['shard_id']}\t"
            f"{x['root']}\t"
            f"{x['manifest']}\t"
            f"{x['rows']}\n"
        )


# ------------------------------------------------------------
# Create a four-example pilot from shard 0.
#
# Separate root, same original manifest.
# The four API calls will be duplicated later; deliberately cheap.
# ------------------------------------------------------------

first = shard_meta[0]
first_root = Path(first["root"])

pilot_root = OUT / "pilot_4"
pilot_root.mkdir(
    parents=True,
    exist_ok=True,
)

full_candidates = read_jsonl(
    first_root
    / "candidates_train_round_01.jsonl"
)

full_sidecar = read_jsonl(
    first_root
    / "v137_sidecar_training_only.jsonl"
)

if len(full_candidates) < 4:
    raise RuntimeError(
        "First shard has fewer than 4 candidates"
    )

pilot_candidates = full_candidates[:4]
pilot_sidecar = full_sidecar[:4]

# Reindex only candidate_index for clean pilot.
for i, c in enumerate(pilot_candidates):
    c["candidate_index"] = i

for i, s in enumerate(pilot_sidecar):
    s["pilot_candidate_index"] = i

write_jsonl(
    pilot_root
    / "candidates_train_round_01.jsonl",
    pilot_candidates,
)

write_jsonl(
    pilot_root
    / "v137_sidecar_training_only.jsonl",
    pilot_sidecar,
)

pilot_meta = {
    "root": str(pilot_root),
    "manifest": first["manifest"],
    "rows": 4,
    "source_shard": first["shard_id"],
    "execution_split": "train",
    "round": 1,
}

(
    pilot_root
    / "meta.json"
).write_text(
    json.dumps(
        pilot_meta,
        indent=2,
        sort_keys=True,
    )
)


# ------------------------------------------------------------
# Report.
# ------------------------------------------------------------

print("=" * 100)
print("V13.7 EXECUTION SHARDS V1")
print("=" * 100)

print("execute-plan SHA:", actual_plan_sha)
print("shards          :", len(shard_meta))
print("total rows      :", sum(x["rows"] for x in shard_meta))

sizes = [x["rows"] for x in shard_meta]

print("min shard rows  :", min(sizes))
print("max shard rows  :", max(sizes))
print(
    "mean shard rows :",
    round(sum(sizes) / len(sizes), 2),
)

print()
print("CANDIDATE LEAKAGE CHECK: PASS")
print("MANIFEST SESSION CHECK: PASS")
print("SOURCE SPLIT=train CHECK: PASS")
print("CACHE-KEY UNIQUENESS: PASS")
print("INTERNAL TEST INCLUDED: NO")

print()
print("PILOT ROOT     :", pilot_root)
print("PILOT MANIFEST :", first["manifest"])
print("PILOT ROWS     : 4")

print()
print("registry:", registry_path)
print("registry SHA:", registry_sha)

print()
print("PREPARATION INTEGRITY: PASS")
