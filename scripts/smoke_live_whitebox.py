from __future__ import annotations

import argparse
import os

from probes.seeds import BENIGN_SEEDS
from rl.live_whitebox_env import LiveWhiteBoxEnv
from targets.role_presence_mas import RolePresenceWhiteBoxMAS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system-id", default="C03_s1")
    parser.add_argument("--max-probes", type=int, default=5)
    parser.add_argument(
        "--model",
        default=os.environ.get("MAS_MODEL", "gpt-4o-mini"),
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set.")

    n = min(args.max_probes, len(BENIGN_SEEDS))

    print("=== LIVE WHITE-BOX REWARD SMOKE TEST ===")
    print("system:", args.system_id)
    print("model:", args.model)
    print("queries:", n)
    print("expected model calls:", 2 * n, "(router + worker)")
    print()

    target = RolePresenceWhiteBoxMAS(
        args.system_id,
        model=args.model,
    )
    env = LiveWhiteBoxEnv(
        target,
        max_steps=n,
    )

    obs = env.reset()

    for i, query in enumerate(BENIGN_SEEDS[:n], start=1):
        print(f"[{i}/{n}] QUERY")
        print(query)

        obs, reward, done, info = env.step(query)

        output = obs["history"][-1]["output"]
        print("OUTPUT:", output[:350].replace("\n", " "))
        print(
            f"REWARD={reward:+.3f} | "
            f"new_worker={int(info['new_worker'])} | "
            f"coverage={info['coverage']:.3f}"
        )
        print(
            "WHITE-BOX DEBUG ONLY:",
            f"worker={info['selected_worker']}",
            f"role={info['selected_role']}",
            f"K={info['k_true']}",
        )
        print("-" * 80)

        if done:
            if info["completed"] == 1.0:
                print("Episode finished: all workers were covered.")
            else:
                print("Episode finished: query budget exhausted.")
            break

    print()
    print("Policy-visible keys:", sorted(obs.keys()))
    print("IMPORTANT: selected_worker/role/K are debug-only, not in observation.")


if __name__ == "__main__":
    main()
