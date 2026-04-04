"""Competition-style inference runner for the surveillance benchmark."""

from __future__ import annotations

import json
import os
from typing import Optional

from openai import OpenAI

from meverse.env import load_repo_env
from meverse import SurveillanceAction, choose_surveillance_action, list_task_names
from meverse.server.meverse_environment import MarketSurveillanceEnvironment

load_repo_env()

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN", "")
TASK_NAME = os.getenv("MEVERSE_TASK") or os.getenv("TASK_NAME") or "full_market_surveillance"
BENCHMARK = "amm-market-surveillance"


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}", flush=True)


def log_end(success: bool, steps: int, rewards: list[float]) -> None:
    reward_text = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={reward_text}", flush=True)


def llm_action(client: OpenAI, observation) -> str:
    prompt = {
        "task": observation.task_name,
        "step_num": observation.step_num,
        "max_steps": observation.max_steps,
        "current_amm_price": observation.current_amm_price,
        "liquidity_snapshot": observation.liquidity_snapshot,
        "recent_trade_count": observation.recent_trade_count,
        "trades_in_window": observation.trades_in_window,
        "trade_frequency": observation.trade_frequency,
        "average_trade_size": observation.average_trade_size,
        "maximum_trade_size": observation.maximum_trade_size,
        "recent_slippage_impact": observation.recent_slippage_impact,
        "time_gap_mean": observation.time_gap_mean,
        "time_gap_min": observation.time_gap_min,
        "burst_indicator": observation.burst_indicator,
        "pattern_indicator": observation.pattern_indicator,
        "suspiciousness_score": observation.suspiciousness_score,
        "manipulation_score": observation.manipulation_score,
    }
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=12,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a market surveillance controller in a simulated AMM market. "
                    "Your job: detect suspicious bot-like activity and respond decisively, "
                    "but NEVER penalize normal healthy trading.\n"
                    "Actions: ALLOW, FLAG, BLOCK, MONITOR.\n"
                    "Rules (apply in order, stop at first match):\n"
                    "1. manipulation_score >= 0.55 → BLOCK\n"
                    "2. trade_frequency >= 7.0 AND time_gap_min < 0.5 → BLOCK\n"
                    "3. burst_indicator >= 0.50 AND suspiciousness_score >= 0.45 → BLOCK\n"
                    "4. suspiciousness_score >= 0.50 AND pattern_indicator >= 0.35 → FLAG\n"
                    "5. trade_frequency >= 5.0 AND average_trade_size > 18 → FLAG\n"
                    "6. burst_indicator >= 0.35 OR suspiciousness_score >= 0.40 → MONITOR\n"
                    "7. Otherwise → ALLOW\n"
                    "Return JSON only: {\"action\": \"ALLOW\"}"
                ),
            },
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    try:
        parsed = json.loads(content)
        action = str(parsed.get("action", "")).strip().upper()
        if action in {"ALLOW", "FLAG", "BLOCK", "MONITOR"}:
            return action
    except Exception:
        pass
    return choose_surveillance_action(observation)


def select_action(observation) -> str:
    if HF_TOKEN:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
            return llm_action(client, observation)
        except Exception:
            return choose_surveillance_action(observation)
    return choose_surveillance_action(observation)


def main() -> None:
    task_name = TASK_NAME if TASK_NAME in list_task_names() else "full_market_surveillance"
    demo_mode = env_flag("DEMO_MODE", False)
    eval_mode = False if demo_mode else env_flag("EVAL_MODE", True)
    env = MarketSurveillanceEnvironment(task=task_name, eval_mode=eval_mode, demo_mode=demo_mode)
    observation = env.reset(task=task_name)
    rewards: list[float] = []
    steps = 0

    log_start(task_name, BENCHMARK, MODEL_NAME if HF_TOKEN else "heuristic-fallback")

    try:
        while not observation.done:
            final_action = select_action(observation)
            observation = env.step(SurveillanceAction(action_type=final_action))
            steps += 1
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            log_step(step=steps, action=final_action, reward=reward, done=observation.done, error=observation.metadata.get("last_action_error"))
        grade = env.grade()
        success = bool(grade["score"] >= 0.6)
    except Exception:
        success = False
        raise
    finally:
        log_end(success=success, steps=steps, rewards=rewards)


if __name__ == "__main__":
    main()
