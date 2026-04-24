"""
Inference Script — MEVerse (RLiquidity V3) + Self-Improving Environment
======================================================================

Runs MEVerse and prints the standard OpenEnv [START]/[STEP]/[END] log
format. By default it drives the in-process ``MeverseEnvironment`` so you
can see the environment working without Docker or any LLM credentials.

Run modes (pick one):

  # default: SIE loop, heuristic policy, in-process env (no Docker, no LLM)
  python inference.py

  # same but a single episode
  python inference.py --episodes 1

  # use an LLM (requires HF_TOKEN or OPENAI_API_KEY set)
  python inference.py --llm

  # talk to a remote OpenEnv server (requires MEVERSE_BASE_URL)
  python inference.py --remote --llm

  # talk to a local Docker image (requires LOCAL_IMAGE_NAME)
  python inference.py --docker --llm

Env vars used when --llm is passed:
    API_BASE_URL   LLM endpoint (default https://router.huggingface.co/v1)
    MODEL_NAME     Model id (default Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN       API key (also accepts OPENAI_API_KEY / API_KEY)

STDOUT FORMAT (per episode)
    [START] task=<task> env=meverse model=<model> episode=<n>
    [STEP]  step=<n> action=<json> reward=<r> done=<bool> error=<msg|null>
    [END]   success=<bool> steps=<n> rewards=<r1,r2,...>
    [SIE]   episode=<n> score=<s> mev_avoid=<m> phase=<p> dominant=<bot> weights=<...>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import textwrap
from typing import Any, Dict, List, Optional

from meverse import EnvController, MeverseAction, MeverseObservation
from meverse.server.meverse_environment import MeverseEnvironment

API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
MEVERSE_BASE_URL = os.getenv("MEVERSE_BASE_URL")
TEMPERATURE = 0.7
MAX_TOKENS = 300
SUCCESS_SCORE_THRESHOLD = 0.3
BENCHMARK = "meverse"

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert DeFi trader in a simulated Uniswap V3 ETH/USDC pool.
    Maximize portfolio value while dodging MEV attacks.

    Signals:
    - jit_liquidity > 0 at current tick → JIT bot staged. Use private RPC or split_swap.
    - Multiple mempool txs → sandwich risk. Use split_swap or private RPC.
    - Low active_liquidity → price impact high. Reduce size or hold.

    Actions (JSON only, no markdown):
      swap_exact_in     {"amount_in": float, "zero_for_one": bool, "use_private_rpc": bool}
      split_swap        {"total_amount": float, "num_splits": int, "zero_for_one": bool}
      add_liquidity     {"tick_lower": int, "tick_upper": int, "amount0_desired": float, "amount1_desired": float}
      remove_liquidity  {"position_id": str}
      range_order       {"tick_lower": int, "tick_upper": int, "token": "token0"|"token1", "amount": float}
      jit_liquidity     {"tick_lower": int, "tick_upper": int, "amount0": float, "amount1": float}
      hold              {}
      close_episode     {}

    Respond with exactly one JSON object: {"action_type": "...", "params": {...}}
    """
)


# ──────────────────────────────────────────────
#  Log helpers
# ──────────────────────────────────────────────

def log_start(task: str, model: str, episode: int) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model} episode={episode}", flush=True)


def log_step(step: int, action_json: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action_json} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def log_sie(ep: int, controller: EnvController, result) -> None:
    dominant = max(controller.bot_weights, key=controller.bot_weights.get)
    weights = ",".join(f"{k}:{v:.2f}" for k, v in controller.bot_weights.items())
    print(
        f"[SIE] episode={ep} score={result.final_score:.3f} "
        f"mev_avoid={result.mev_avoidance_score:.3f} "
        f"profit={result.profit_score:.3f} "
        f"fails={len(result.failures)} "
        f"phase={controller.curriculum_phase} "
        f"dominant={dominant} weights={weights}",
        flush=True,
    )


# ──────────────────────────────────────────────
#  Policies
# ──────────────────────────────────────────────

def heuristic_policy(obs: MeverseObservation, rng: random.Random) -> Dict[str, Any]:
    """Cheap defensive policy for wiring tests and no-LLM runs."""
    mempool = getattr(obs, "mempool", []) or []
    active_liq = getattr(obs, "active_liquidity", 0.0) or 0.0

    if len(mempool) >= 2:
        return {
            "action_type": "swap_exact_in",
            "params": {"amount_in": 0.2, "zero_for_one": True, "use_private_rpc": True},
        }
    if active_liq > 0 and rng.random() < 0.5:
        return {
            "action_type": "split_swap",
            "params": {
                "total_amount": 0.8,
                "num_splits": 4,
                "zero_for_one": rng.random() < 0.5,
            },
        }
    if rng.random() < 0.2:
        return {"action_type": "hold", "params": {}}
    return {
        "action_type": "swap_exact_in",
        "params": {
            "amount_in": round(rng.uniform(0.3, 1.2), 3),
            "zero_for_one": rng.random() < 0.5,
            "use_private_rpc": rng.random() < 0.3,
        },
    }


def build_user_prompt(obs: Any, step: int, last_reward: float, history: List[str]) -> str:
    history_block = "\n".join(history[-5:]) if history else "None"
    obs_dict = {
        "current_tick": obs.current_tick,
        "current_price": obs.current_price,
        "active_liquidity": obs.active_liquidity,
        "tick_distribution": obs.tick_distribution[:5],
        "agent_token0_ETH": obs.agent_token0,
        "agent_token1_USDC": obs.agent_token1,
        "agent_positions": obs.agent_positions,
        "mempool": obs.mempool,
        "last_mev_loss": obs.last_mev_loss,
        "step_num": obs.step_num,
        "max_steps": obs.max_steps,
        "task": obs.task_name,
    }
    return (
        f"Step: {step}/{obs.max_steps}\n"
        f"Last reward: {last_reward:.2f}\n\n"
        f"Current pool state:\n{json.dumps(obs_dict, indent=2)}\n\n"
        f"Recent history:\n{history_block}\n\n"
        f"Choose your next action. Respond with JSON only."
    )


def llm_policy_factory(client):
    def _policy(obs, rng: random.Random, step: int, last_reward: float, history: List[str]) -> Dict[str, Any]:
        fallback = {"action_type": "hold", "params": {}}
        text = ""
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(obs, step, last_reward, history)},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            text = (completion.choices[0].message.content or "").strip()
            if text.startswith("```"):
                text = "\n".join(
                    line for line in text.split("\n") if not line.strip().startswith("```")
                ).strip()
            action = json.loads(text)
            if "action_type" not in action:
                print(f"[DEBUG] missing action_type | raw={text}", file=sys.stderr, flush=True)
                return fallback
            action.setdefault("params", {})
            return action
        except json.JSONDecodeError as exc:
            print(f"[DEBUG] JSON parse fail: {exc} | raw={text}", file=sys.stderr, flush=True)
            return fallback
        except Exception as exc:
            print(f"[DEBUG] LLM request failed: {exc}", file=sys.stderr, flush=True)
            return fallback
    return _policy


# ──────────────────────────────────────────────
#  Local in-process runner (default)
# ──────────────────────────────────────────────

def run_local(args: argparse.Namespace) -> None:
    controller = EnvController(task_name=args.task)
    env = MeverseEnvironment(task=args.task)

    llm_call = None
    model_label = "heuristic"
    if args.llm:
        if not API_KEY:
            print(
                "[DEBUG] --llm requested but no HF_TOKEN/OPENAI_API_KEY set. Falling back to heuristic.",
                file=sys.stderr,
                flush=True,
            )
        else:
            try:
                from openai import OpenAI
                client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
                llm_call = llm_policy_factory(client)
                model_label = MODEL_NAME
            except Exception as exc:
                print(f"[DEBUG] OpenAI client init failed: {exc}", file=sys.stderr, flush=True)

    print(
        f"[SIE] task={args.task} episodes={args.episodes} policy={model_label} "
        f"starting curriculum_phase=0",
        flush=True,
    )

    for ep in range(1, args.episodes + 1):
        sie_config = controller.next_config()
        seed = args.seed + ep
        env.reset(seed=seed, task=args.task, sie_config=sie_config)
        rng = random.Random(seed + 7919)

        log_start(task=args.task, model=model_label, episode=ep)
        rewards: List[float] = []
        history: List[str] = []
        last_reward = 0.0
        steps_taken = 0
        done = False

        while not done:
            obs = env._build_observation(done=False, reward=0.0)
            if llm_call is not None:
                action_dict = llm_call(obs, rng, steps_taken + 1, last_reward, history)
            else:
                action_dict = heuristic_policy(obs, rng)

            action = MeverseAction(
                action_type=action_dict.get("action_type", "hold"),
                params=action_dict.get("params", {}),
            )
            next_obs = env.step(action)
            reward = next_obs.reward or 0.0
            done = next_obs.done
            steps_taken += 1

            err = None
            if getattr(next_obs, "metadata", None):
                err = next_obs.metadata.get("last_action_error")

            log_step(
                step=steps_taken,
                action_json=json.dumps(action_dict, separators=(",", ":"), sort_keys=True),
                reward=reward,
                done=done,
                error=err,
            )
            rewards.append(reward)
            history.append(
                f"Step {steps_taken}: {action.action_type} -> reward {reward:+.2f}, "
                f"price={next_obs.current_price}, mev_loss={next_obs.last_mev_loss}"
            )
            last_reward = reward
            if steps_taken >= env._config.max_steps + 5:
                break

        result = env.build_sie_result()
        controller.ingest(result)
        success = result.final_score >= SUCCESS_SCORE_THRESHOLD
        log_end(success=success, steps=steps_taken, rewards=rewards)
        log_sie(ep, controller, result)

    print("\n[SIE] final evolution snapshots:", flush=True)
    for snap in controller.evolution_history():
        print(
            f"  ep={snap['episode_id']} score={snap['final_score']:.3f} "
            f"phase={snap['curriculum_phase']} dom={snap['dominant_bot_type']} "
            f"fails={snap['num_failures']}",
            flush=True,
        )


# ──────────────────────────────────────────────
#  Remote / Docker runner (single episode, LLM only)
# ──────────────────────────────────────────────

async def run_remote(args: argparse.Namespace) -> None:
    from meverse import MeverseEnv

    if not API_KEY:
        raise RuntimeError("Remote mode requires --llm with HF_TOKEN/OPENAI_API_KEY")

    from openai import OpenAI
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    llm_call = llm_policy_factory(client)

    if args.docker:
        if not LOCAL_IMAGE_NAME:
            raise RuntimeError("--docker requires LOCAL_IMAGE_NAME env var")
        env = await MeverseEnv.from_docker_image(LOCAL_IMAGE_NAME)
    else:
        if not MEVERSE_BASE_URL:
            raise RuntimeError("--remote requires MEVERSE_BASE_URL env var")
        env = MeverseEnv(base_url=MEVERSE_BASE_URL)
        await env.connect()

    rng = random.Random(args.seed)
    log_start(task=args.task, model=MODEL_NAME, episode=1)

    rewards: List[float] = []
    history: List[str] = []
    steps_taken = 0
    success = False

    try:
        result = await env.reset(task=args.task)
        obs = result.observation
        last_reward = 0.0
        max_steps = max(1, int(obs.max_steps or 1))

        for step in range(1, max_steps + 1):
            if result.done:
                break
            action_dict = llm_call(obs, rng, step, last_reward, history)
            action = MeverseAction(
                action_type=action_dict.get("action_type", "hold"),
                params=action_dict.get("params", {}),
            )
            result = await env.step(action)
            obs = result.observation
            reward = result.reward or 0.0
            done = result.done
            err = obs.metadata.get("last_action_error") if getattr(obs, "metadata", None) else None

            log_step(
                step=step,
                action_json=json.dumps(action_dict, separators=(",", ":"), sort_keys=True),
                reward=reward,
                done=done,
                error=err,
            )
            rewards.append(reward)
            last_reward = reward
            steps_taken = step
            history.append(
                f"Step {step}: {action.action_type} -> reward {reward:+.2f}"
            )
            if done:
                break

        max_total_reward = max(1.0, max_steps * 1.0)
        score = min(1.0, max(0.0, sum(rewards) / max_total_reward))
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        try:
            await env.close()
        except Exception as exc:
            print(f"[DEBUG] env.close error: {exc}", file=sys.stderr, flush=True)
        log_end(success=success, steps=steps_taken, rewards=rewards)


# ──────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEVerse inference runner (SIE-aware)")
    p.add_argument("--task", default=os.getenv("MEVERSE_TASK", "easy"),
                   choices=["easy", "medium", "hard"])
    p.add_argument("--episodes", type=int, default=5, help="SIE loop length (local mode)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--llm", action="store_true", help="Use OpenAI-compatible LLM policy")
    p.add_argument("--remote", action="store_true", help="Talk to MEVERSE_BASE_URL (HTTP)")
    p.add_argument("--docker", action="store_true", help="Talk to LOCAL_IMAGE_NAME (Docker)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.remote or args.docker:
        asyncio.run(run_remote(args))
    else:
        run_local(args)


if __name__ == "__main__":
    main()
