#!/usr/bin/env python3

import json
import os
from pathlib import Path

import numpy as np

# Registers the extended V11 role prompts.
import targets.single_router_zoo_v11

from targets.single_router_fanout_mas import (
    SingleRouterFanoutMAS,
    SingleRouterSystemSpec,
)

import analysis.train_v13_6b_activation_judge as tr
import analysis.build_visible_realized_gain_v3 as vg


# ---------------------------------------------------------------------
# DELIBERATELY EASY / FAVORABLE DEMO
# ---------------------------------------------------------------------

TRUE_ROLES = (
    "accessibility_review",
    "risk_assessment",
    "translation",
)

FAKE_ROLES = (
    "math",
    "qa_testing",
    "customer_support_triage",
)

CANDIDATES = TRUE_ROLES + FAKE_ROLES

# IMPORTANT:
# Sensor still scores all 20 roles in one same-response batch.
ALL_ROLES = list(tr.ROLE_DESCRIPTIONS)
ROLE_DESC = dict(tr.ROLE_DESCRIPTIONS)

assert len(ALL_ROLES) == 20
assert len(CANDIDATES) == 6
assert len(TRUE_ROLES) == 3
assert set(CANDIDATES).issubset(ALL_ROLES)

SENSOR = Path(
    "artifacts/v13_8b_balanced_repair_2048_run1/best_adapter"
)

CAL = Path(
    "artifacts/visible_realized_gain_v3/channel_calibration.json"
)

OUT = Path(
    "artifacts/cheat_easy_k3_once/result.json"
)

PROBES = [
    (
        "accessibility_review",
        "Review this signup form for accessibility problems: "
        "the icon buttons have no text labels, placeholder text has "
        "low contrast, and the confirmation modal traps keyboard focus. "
        "Suggest concrete fixes."
    ),
    (
        "accessibility_review",
        "Review a checkout page for basic accessibility issues. "
        "Product images have no alt text, visible focus indicators were "
        "removed, and validation errors are communicated only with red color. "
        "What should be changed?"
    ),

    (
        "risk_assessment",
        "Assess the main risks of migrating a customer billing database "
        "during one weekend. For each major risk, describe likelihood, "
        "impact, and a practical mitigation."
    ),
    (
        "risk_assessment",
        "A startup plans to launch a critical service using only one cloud "
        "region. Identify the important operational and business risks and "
        "suggest mitigations."
    ),

    (
        "translation",
        "Translate this sentence into Spanish while preserving a polite "
        "professional tone: 'Please send the revised contract by Friday "
        "afternoon so we can review it before the meeting.'"
    ),
    (
        "translation",
        "Translate this sentence into French naturally: "
        "'The train was delayed, but we still arrived in time for dinner.'"
    ),
]


def top2_mean(xs):
    xs = sorted(
        [float(x) for x in xs],
        reverse=True,
    )
    return float(np.mean(xs[:2]))


def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    # -----------------------------------------
    # Hidden target MAS: exactly three workers.
    # -----------------------------------------
    spec = SingleRouterSystemSpec(
        system_id="cheat_easy_k3_once",
        split="demo",
        execution_mode="single",
        style="s1",
        worker_roles=TRUE_ROLES,
    )

    target = SingleRouterFanoutMAS(
        spec,
        model=os.getenv(
            "MAS_MODEL",
            "gpt-4o-mini",
        ),
    )

    # -----------------------------------------
    # Stable V13.8b sensor.
    # -----------------------------------------
    tok, sensor, dtype = vg.load_sensor(
        SENSOR
    )

    calibration = json.loads(
        CAL.read_text()
    )

    evidence = {
        r: []
        for r in CANDIDATES
    }

    records = []

    print("=" * 100)
    print("CHEAT EASY K=3 DEMO")
    print("=" * 100)
    print("Inference candidate universe:", CANDIDATES)
    print("Known K: 3")
    print(
        "Inference sees ONLY query + final output."
    )
    print(
        "Routing trace below is EVAL ONLY "
        "and is never used for prediction."
    )

    for i, (probe_target, query) in enumerate(
        PROBES,
        1,
    ):
        print("\n" + "=" * 100)
        print(
            f"PROBE {i}/{len(PROBES)} "
            f"(probe intent={probe_target})"
        )
        print("QUERY:")
        print(query)

        # This is the actual black-box interaction.
        output = str(
            target.query(query)
        )

        # Evaluation-only ground truth.
        trace = getattr(
            target,
            "_last_trace",
            None,
        )

        actual = (
            list(
                map(
                    str,
                    getattr(
                        trace,
                        "selected_roles",
                        (),
                    ),
                )
            )
            if trace
            else []
        )

        # --------------------------------------------------
        # Deployment-visible sensor:
        # score ALL 20 roles in the SAME response batch.
        # --------------------------------------------------
        scores = vg.sensor_score_row(
            model=sensor,
            tok=tok,
            dtype=dtype,
            roles=ALL_ROLES,
            role_desc=ROLE_DESC,
            query=query,
            answer=output,
            max_length=4096,
        )

        loglr = vg.sensor_log_lr(
            scores,
            ALL_ROLES,
            calibration,
        )

        score_map = {
            r: float(loglr[j])
            for j, r in enumerate(
                ALL_ROLES
            )
        }

        candidate_rank = sorted(
            CANDIDATES,
            key=lambda r: score_map[r],
            reverse=True,
        )

        for r in CANDIDATES:
            evidence[r].append(
                score_map[r]
            )

        print("\nVISIBLE OUTPUT:")
        print(output)

        print(
            "\nSENSOR RANK AMONG SIX CANDIDATES:"
        )

        for rank, r in enumerate(
            candidate_rank,
            1,
        ):
            mark = (
                " <-- EVAL TRUE SOURCE"
                if r in actual
                else ""
            )

            print(
                f"{rank}. "
                f"{r:<26} "
                f"logLR={score_map[r]:+.4f}"
                f"{mark}"
            )

        print(
            "\nACTUAL ROUTE [EVAL ONLY]:",
            actual,
        )

        records.append({
            "probe_index": i,
            "probe_intent":
                probe_target,
            "query": query,
            "output": output,
            "actual_selected_eval_only":
                actual,
            "candidate_scores": {
                r: score_map[r]
                for r in CANDIDATES
            },
            "candidate_rank":
                candidate_rank,
        })

    # ------------------------------------------------------
    # Since K=3 is GIVEN in this cheat test:
    #
    # evidence(role) = mean of its two strongest observations.
    #
    # Requiring two strong observations makes one accidental
    # fake-role spike less dangerous than simple max().
    # ------------------------------------------------------
    aggregate = {
        r: top2_mean(vals)
        for r, vals in evidence.items()
    }

    final_rank = sorted(
        CANDIDATES,
        key=lambda r: aggregate[r],
        reverse=True,
    )

    predicted = final_rank[:3]

    exact = (
        set(predicted)
        == set(TRUE_ROLES)
    )

    print("\n" + "#" * 100)
    print("FINAL INSTALLATION INFERENCE")
    print("#" * 100)

    for rank, r in enumerate(
        final_rank,
        1,
    ):
        mark = (
            " <-- TRUE INSTALLED"
            if r in TRUE_ROLES
            else " <-- FAKE"
        )

        print(
            f"{rank}. "
            f"{r:<26} "
            f"top2_mean_logLR="
            f"{aggregate[r]:+.4f}"
            f"{mark}"
        )

    print("\nKNOWN K = 3")
    print("PREDICTED INSTALLED:", predicted)

    # Print truth only after prediction.
    print(
        "TRUE INSTALLED [EVAL ONLY]:",
        list(TRUE_ROLES),
    )

    print("EXACT SET:", exact)

    result = {
        "protocol":
            "cheat_easy_k3_once_v1",
        "warning":
            "Deliberately favorable one-off demonstration; "
            "known K=3, easy installed roles, easy distractors, "
            "and probes intentionally expose true roles.",
        "sensor": str(SENSOR),
        "candidate_roles":
            list(CANDIDATES),
        "known_k": 3,
        "aggregate":
            aggregate,
        "final_rank":
            final_rank,
        "predicted_installed":
            predicted,
        "true_installed_eval_only":
            list(TRUE_ROLES),
        "exact_set":
            bool(exact),
        "records":
            records,
    }

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nARTIFACT:", OUT)


if __name__ == "__main__":
    main()
