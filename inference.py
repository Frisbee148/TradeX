"""Competition-style inference runner for the surveillance benchmark."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

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


def build_signal_snapshot(observation) -> dict[str, Any]:
    return {
        "task_name": observation.task_name,
        "step_num": observation.step_num,
        "max_steps": observation.max_steps,
        "done": observation.done,
        "reward": float(observation.reward or 0.0),
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
        "recent_time_gaps": observation.recent_time_gaps,
        "recent_price_impacts": observation.recent_price_impacts,
        "burst_indicator": observation.burst_indicator,
        "pattern_indicator": observation.pattern_indicator,
        "suspiciousness_score": observation.suspiciousness_score,
        "manipulation_score": observation.manipulation_score,
        "metadata": {
            "episode_id": observation.metadata.get("episode_id"),
            "seed": observation.metadata.get("seed"),
            "eval_mode": observation.metadata.get("eval_mode"),
            "demo_mode": observation.metadata.get("demo_mode"),
            "scenario_note": observation.metadata.get("scenario_note"),
            "amm_price": observation.metadata.get("amm_price"),
            "amm_liquidity": observation.metadata.get("amm_liquidity"),
            "bot_confidence": observation.metadata.get("bot_confidence"),
            "last_action_error": observation.metadata.get("last_action_error"),
        },
    }


class DebugTelemetryWriter:
    """Write detailed episode telemetry to JSONL without touching stdout logs."""

    def __init__(self, enabled: bool, task_name: str):
        self.enabled = enabled
        self.path: Optional[Path] = None
        if not enabled:
            return
        configured_path = os.getenv("DEBUG_TELEMETRY_PATH", "").strip()
        if configured_path:
            self.path = Path(configured_path)
        else:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            self.path = Path("telemetry") / f"{task_name}-{stamp}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: str, payload: dict[str, Any]) -> None:
        if not self.enabled or self.path is None:
            return
        record = {
            "event": event,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            # Telemetry must never interfere with the benchmark run.
            pass


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
    telemetry = DebugTelemetryWriter(enabled=env_flag("DEBUG_TELEMETRY", False), task_name=task_name)
    rewards: list[float] = []
    steps = 0

    log_start(task_name, BENCHMARK, MODEL_NAME if HF_TOKEN else "heuristic-fallback")
    telemetry.write(
        "episode_start",
        {
            "task": task_name,
            "benchmark": BENCHMARK,
            "model": MODEL_NAME if HF_TOKEN else "heuristic-fallback",
            "initial_observation": build_signal_snapshot(observation),
            "environment": env.debug_snapshot(),
        },
    )

    try:
        while not observation.done:
            decision_observation = observation
            pre_action_debug = env.debug_snapshot()
            final_action = select_action(observation)
            observation = env.step(SurveillanceAction(action_type=final_action))
            steps += 1
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            telemetry.write(
                "step",
                {
                    "step": steps,
                    "action": final_action,
                    "reward": reward,
                    "done": observation.done,
                    "decision_observation": build_signal_snapshot(decision_observation),
                    "returned_observation": build_signal_snapshot(observation),
                    "pre_action_environment": pre_action_debug,
                    "post_action_environment": env.debug_snapshot(),
                },
            )
            log_step(step=steps, action=final_action, reward=reward, done=observation.done, error=observation.metadata.get("last_action_error"))
        grade = env.grade()
        success = bool(grade["score"] >= 0.6)
    except Exception:
        success = False
        telemetry.write(
            "episode_error",
            {
                "steps_completed": steps,
                "rewards": rewards,
            },
        )
        raise
    finally:
        final_grade = env.grade() if steps > 0 else None
        telemetry.write(
            "episode_end",
            {
                "success": success,
                "steps": steps,
                "rewards": rewards,
                "grade": final_grade,
                "telemetry_path": str(telemetry.path) if telemetry.path else None,
            },
        )
        log_end(success=success, steps=steps, rewards=rewards)


if __name__ == "__main__":
    main()
