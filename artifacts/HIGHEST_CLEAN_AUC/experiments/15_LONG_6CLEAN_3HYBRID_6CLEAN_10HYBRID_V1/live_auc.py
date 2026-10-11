import json
from pathlib import Path

root = Path(
    "artifacts/HIGHEST_CLEAN_AUC/"
    "experiments/15_LONG_6CLEAN_3HYBRID_6CLEAN_10HYBRID_V1/"
    "data/run"
)


def generator(r):
    if r <= 6:
        return "CLEAN"
    if r <= 9:
        return "HYBRID"
    if r <= 15:
        return "CLEAN"
    return "HYBRID"


vals = []

for r in range(1, 26):
    p = root / f"advance_dev_round_{r:02d}_summary.json"

    if not p.exists():
        continue

    try:
        auc = float(
            json.loads(
                p.read_text()
            )["mean_role_auc"]
        )
    except Exception:
        continue

    vals.append(
        (r, generator(r), auc)
    )


print("===================================================")
print(" 6 CLEAN -> 3 HYBRID -> 6 CLEAN -> 10 HYBRID")
print("===================================================")
print()

print(
    f"{'R':>3} "
    f"{'GEN':>7} "
    f"{'AUC':>10} "
    f"{'DELTA':>10} "
    f"{'BEST':>10}"
)

print("-" * 46)

best = -1.0
prev = None

for r, gen, auc in vals:

    if prev is None:
        dtxt = "-"
    else:
        dtxt = f"{auc - prev:+.6f}"

    best = max(best, auc)

    print(
        f"{r:3d} "
        f"{gen:>7} "
        f"{auc:10.6f} "
        f"{dtxt:>10} "
        f"{best:10.6f}"
    )

    prev = auc


print()

if vals:
    br, bg, ba = max(
        vals,
        key=lambda x: x[2]
    )

    lr, lg, la = vals[-1]

    print(f"COMPLETED = {len(vals)}/25")
    print(f"LATEST    = R{lr} {la:.6f}")
    print(f"BEST      = R{br} {ba:.6f} ({bg})")

    print()
    print("PHASES:")
    print("R01-R06 = CLEAN-R7")
    print("R07-R09 = HYBRID")
    print("R10-R15 = CLEAN-R7")
    print("R16-R25 = HYBRID")

else:
    print("WAITING FOR R1...")
