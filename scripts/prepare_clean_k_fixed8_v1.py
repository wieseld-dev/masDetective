from __future__ import annotations

import hashlib
import itertools
import json
import random
from collections import Counter
from pathlib import Path

from adaptive_v11_active.belief import FrozenBeliefBundle
from adaptive_v11_active.schema import PublicBeliefState
from v11.role_universe import TRAIN_ROLES


SEED = 20260923
N_PER_STYLE = 14

SRC = Path("artifacts/adaptive_v11_curriculum_v2")
OUT = Path("artifacts/clean_k_fixed8_v1")
BELIEF = Path(
    "artifacts/adaptive_v11_active_v1/frozen_belief.joblib"
)

STYLES = ("s1", "s2", "s3")
KS = range(3, 9)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def old_compositions():
    used = set()

    paths = list(
        SRC.glob("stage_*/manifest_train_cycle_*.json")
    )
    paths += list(
        SRC.glob("stage_*/manifest_dev.json")
    )

    for p in paths:
        obj = json.loads(p.read_text())
        for s in obj.get("systems", []):
            roles = tuple(sorted(map(str, s["roles"])))
            used.add(roles)

    return used


def choose_new_composition(k, counts, forbidden, chosen, rng):
    best = None

    # Search many possible compositions and choose one
    # that balances role exposure.
    for _ in range(4000):
        roles = tuple(rng.sample(list(TRAIN_ROLES), k))
        key = tuple(sorted(roles))

        if key in forbidden or key in chosen:
            continue

        score = (
            sum(counts[r] for r in roles),
            max(counts[r] for r in roles),
            rng.random(),
        )

        if best is None or score < best[0]:
            best = (score, roles)

    if best is None:
        raise RuntimeError(f"Could not generate fresh K={k}")

    return best[1]


def main():
    rng = random.Random(SEED)

    OUT.mkdir(parents=True, exist_ok=True)

    forbidden = old_compositions()
    chosen_compositions = set()

    systems = []
    split_sidecar = {}

    role_counts_by_k = {
        k: Counter({r: 0 for r in TRAIN_ROLES})
        for k in KS
    }

    seq = 0

    for k in KS:
        counts = role_counts_by_k[k]

        for style in STYLES:
            for j in range(N_PER_STYLE):

                roles = choose_new_composition(
                    k,
                    counts,
                    forbidden,
                    chosen_compositions,
                    rng,
                )

                key = tuple(sorted(roles))
                chosen_compositions.add(key)
                counts.update(roles)

                # Within every style:
                # 8 train + 3 val + 3 test.
                if j < 8:
                    k_split = "train"
                elif j < 11:
                    k_split = "val"
                else:
                    k_split = "test"

                sid = (
                    f"cleank8_k{k}_{style}_{j:02d}_"
                    f"{seq:04d}"
                )
                seq += 1

                system = {
                    "session_id": sid,

                    # Generation/execution pipeline uses DEV behavior:
                    # selector epsilon=0 and executes only chosen query.
                    "split": "dev",

                    # Critical controls.
                    "mode": "fanout2",
                    "style": style,

                    "roles": list(roles),
                }

                systems.append(system)

                split_sidecar[sid] = {
                    "k_split": k_split,
                    "K": k,
                    "style": style,
                    "mode": "fanout2",
                    "roles": list(roles),
                }

    base_manifest = {
        "experiment": "clean_k_fixed8_v1",
        "systems": systems,
    }

    payload = json.dumps(
        base_manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    manifest = dict(base_manifest)
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()

    (OUT / "manifest_privileged.json").write_text(
        json.dumps(manifest, indent=2)
    )

    (OUT / "K_SPLIT.json").write_text(
        json.dumps(split_sidecar, indent=2, sort_keys=True)
    )

    # ----------------------------------------------------------
    # Identical initial public state for every K.
    # No truth/K enters the generator state.
    # ----------------------------------------------------------

    belief = FrozenBeliefBundle.load(BELIEF)
    kp, rp = belief.initial()

    states = []

    for s in systems:
        st = PublicBeliefState(
            session_id=s["session_id"],
            split="dev",
            round_index=0,
            k_probs=dict(kp),
            role_probs=dict(rp),
            history=[],
        )

        st.validate(TRAIN_ROLES, belief.ks)
        states.append(st.to_public_dict())

    write_jsonl(
        OUT / "states_dev_round_00.jsonl",
        states,
    )

    # ----------------------------------------------------------
    # Audit
    # ----------------------------------------------------------

    by_k = Counter(
        x["K"] for x in split_sidecar.values()
    )

    by_split_k = Counter(
        (x["k_split"], x["K"])
        for x in split_sidecar.values()
    )

    by_style_k = Counter(
        (x["style"], x["K"])
        for x in split_sidecar.values()
    )

    overlap = sum(
        tuple(sorted(s["roles"])) in forbidden
        for s in systems
    )

    print("=" * 100)
    print("CLEAN K FIXED-8 BENCHMARK")
    print("=" * 100)

    print("systems:", len(systems))
    print("by K:", dict(sorted(by_k.items())))
    print("old-composition overlap:", overlap)

    print("\nSPLIT x K")
    for x, n in sorted(by_split_k.items()):
        print(x, n)

    print("\nSTYLE x K")
    for x, n in sorted(by_style_k.items()):
        print(x, n)

    print("\nROLE BALANCE PER K")
    for k in KS:
        vals = list(role_counts_by_k[k].values())
        print(
            f"K={k}: "
            f"min={min(vals)} "
            f"max={max(vals)} "
            f"mean={sum(vals)/len(vals):.2f}"
        )

    assert len(systems) == 252
    assert overlap == 0

    for k in KS:
        assert by_k[k] == 42

        assert by_split_k[("train", k)] == 24
        assert by_split_k[("val", k)] == 9
        assert by_split_k[("test", k)] == 9

        for style in STYLES:
            assert by_style_k[(style, k)] == 14

    assert all(
        s["mode"] == "fanout2"
        for s in systems
    )

    print("\nFIXED QUERY BUDGET: 8")
    print("EXECUTION MODE: fanout2 only")
    print("ROUTER STYLES: perfectly balanced")
    print("FINAL INTERNAL TEST TOUCHED: NO")
    print("PREPARATION: PASS")


if __name__ == "__main__":
    main()
