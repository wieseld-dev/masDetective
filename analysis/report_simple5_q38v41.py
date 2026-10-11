import json
from pathlib import Path

NEW = Path(
    "artifacts/v14_R7_q38selectorV41_G8_simple5_v1"
)

OLD = Path(
    "artifacts/v14_cleanR7_selectorV2_simple5_v1"
)


def auc(root, r):
    if r == 0:
        # Initial Simple5 state is exactly 0.5.
        return 0.5

    p = (
        root
        / f"advance_dev_round_{r:02d}_summary.json"
    )

    if not p.exists():
        return None

    x = json.loads(
        p.read_text()
    )

    return float(
        x["mean_role_auc"]
    )


print()
print("=" * 72)
print("SIMPLE5 PROSPECTIVE SELECTOR COMPARISON")
print("=" * 72)

print(
    f"{'ROUND':<8}"
    f"{'V2 G8':>12}"
    f"{'Q38 V4.1':>14}"
    f"{'DELTA':>12}"
)

print("-" * 46)

new_vals = []

for r in range(0, 9):
    a = auc(OLD, r)
    b = auc(NEW, r)

    if b is not None:
        new_vals.append(b)

    if a is None or b is None:
        continue

    print(
        f"R{r:<7}"
        f"{a:>12.6f}"
        f"{b:>14.6f}"
        f"{b-a:>+12.6f}"
    )

if new_vals:
    print()
    print(
        "Q38 best AUC  =",
        f"{max(new_vals):.6f}"
    )

    print(
        "Q38 final AUC =",
        f"{new_vals[-1]:.6f}"
    )

if auc(NEW, 8) is not None:
    d = (
        auc(NEW, 8)
        - auc(OLD, 8)
    )

    print()
    print(
        "FINAL DELTA vs V2 G8 =",
        f"{d:+.6f}"
    )

    print(
        "RESULT =",
        (
            "Q38 SELECTOR BETTER"
            if d > 0
            else
            "Q38 SELECTOR NOT BETTER"
        )
    )
