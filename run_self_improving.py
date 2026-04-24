"""
Self-Improving Environment driver.

Runs N episodes against MeverseEnvironment. Between every episode the
EnvController ingests the episode result and produces a new config that
biases bot-type weights toward the agent's misses and advances the
curriculum when score thresholds are cleared.

This script is self-contained — it exercises the SIE loop directly against
the in-process environment (no Docker / HTTP needed). An LLM-driven run can
pass the same ``sie_config`` kwarg through ``env.reset(...)``.
"""

from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List

from meverse import EnvController, MeverseAction
from meverse.server.meverse_environment import MeverseEnvironment


def heuristic_policy(obs, rng: random.Random) -> Dict[str, Any]:
    """Tiny stand-in policy for wiring tests.

    Behaviour:
      - small swap with private RPC if mempool looks busy
      - split swap when large swap desired
      - hold otherwise
    """
    mempool = getattr(obs, "mempool", []) or []
    active_liq = getattr(obs, "active_liquidity", 0.0) or 0.0

    if len(mempool) >= 2:
        return {
            "action_type": "swap_exact_in",
            "params": {
                "amount_in": 0.2,
                "zero_for_one": True,
                "use_private_rpc": True,
            },
        }

    if active_liq > 0 and rng.random() < 0.6:
        return {
            "action_type": "split_swap",
            "params": {
                "total_amount": 0.8,
                "num_splits": 4,
                "zero_for_one": rng.random() < 0.5,
            },
        }

    if rng.random() < 0.25:
        return {"action_type": "hold", "params": {}}

    return {
        "action_type": "swap_exact_in",
        "params": {
            "amount_in": round(rng.uniform(0.3, 1.2), 3),
            "zero_for_one": rng.random() < 0.5,
            "use_private_rpc": rng.random() < 0.3,
        },
    }


def run_episode(env: MeverseEnvironment, policy, seed: int, sie_config: Dict, task: str):
    env.reset(seed=seed, task=task, sie_config=sie_config)
    rng = random.Random(seed + 7919)

    done = False
    steps = 0
    total_reward = 0.0
    while not done:
        obs = env._build_observation(done=False, reward=0.0)  # current snapshot
        action_dict = policy(obs, rng)
        action = MeverseAction(
            action_type=action_dict["action_type"],
            params=action_dict.get("params", {}),
        )
        next_obs = env.step(action)
        total_reward += next_obs.reward or 0.0
        done = next_obs.done
        steps += 1
        if steps >= env._config.max_steps + 5:
            break
    return env.build_sie_result(), total_reward


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="medium", choices=["easy", "medium", "hard"])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    controller = EnvController(task_name=args.task)
    env = MeverseEnvironment(task=args.task)

    print(
        f"[SIE] task={args.task} episodes={args.episodes} starting curriculum_phase=0"
    )
    for ep in range(1, args.episodes + 1):
        sie_config = controller.next_config()
        result, total_reward = run_episode(
            env,
            heuristic_policy,
            seed=args.seed + ep,
            sie_config=sie_config,
            task=args.task,
        )
        controller.ingest(result)

        dominant = max(controller.bot_weights, key=controller.bot_weights.get)
        print(
            f"[EP {ep:02d}] score={result.final_score:.3f} "
            f"mev_avoid={result.mev_avoidance_score:.3f} "
            f"fails={len(result.failures)} "
            f"phase={controller.curriculum_phase} "
            f"dominant={dominant} "
            f"weights="
            + ",".join(
                f"{k}:{v:.2f}" for k, v in controller.bot_weights.items()
            )
        )
        if args.verbose:
            print(
                "        bot_performance="
                + json.dumps({k: round(v, 3) for k, v in result.bot_type_performance.items()})
            )
            print(
                "        exposures="
                + json.dumps(result.bot_type_exposures)
            )

    print("\n[SIE] final evolution snapshots:")
    for snap in controller.evolution_history():
        print(
            f"  ep={snap['episode_id']} score={snap['final_score']:.3f} "
            f"phase={snap['curriculum_phase']} dom={snap['dominant_bot_type']} "
            f"fails={snap['num_failures']}"
        )


if __name__ == "__main__":
    main()
