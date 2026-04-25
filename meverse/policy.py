"""Policy helpers for LLM-driven inference and rule-based fallback.

The competition baseline is the LLM policy. The heuristic policy exists for two
operational reasons only:

1. comparison testing, so we can measure whether the environment has headroom
2. crash fallback, so inference still completes if the model response is invalid
   or the upstream API errors mid-run
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI

from .baseline_policy import choose_surveillance_action
from .models import SurveillanceObservation

VALID_ACTIONS = {"ALLOW", "FLAG", "BLOCK", "MONITOR"}


@dataclass(frozen=True)
class PolicyConfig:
    api_base_url: str
    model_name: str
    api_token: str


def load_policy_config() -> PolicyConfig:
    """Load model config from the submission-required environment variables."""

    return PolicyConfig(
        api_base_url=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"),
        model_name=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        api_token=os.getenv("HF_TOKEN", ""),
    )


def llm_action(client: OpenAI, observation: SurveillanceObservation, model_name: str) -> str:
    """Query the baseline LLM policy and normalize the returned action."""

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
        model=model_name,
        temperature=0.0,
        max_tokens=8,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a market surveillance controller in a simulated AMM market. "
                    "Choose exactly one action from ALLOW, FLAG, BLOCK, MONITOR. "
                    "Return JSON only: {\"action\": \"ALLOW\"}."
                ),
            },
            {"role": "user", "content": json.dumps(prompt, separators=(",", ":"))},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    if content.startswith("```"):
        content = content.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(content)
    action = str(parsed.get("action", "")).strip().upper()
    if action not in VALID_ACTIONS:
        raise ValueError(f"Invalid LLM action: {action!r}")
    return action


def build_llm_client(config: Optional[PolicyConfig] = None) -> Optional[OpenAI]:
    """Create the OpenAI client used by the official LLM baseline."""

    config = config or load_policy_config()
    if not config.api_token:
        return None
    return OpenAI(base_url=config.api_base_url, api_key=config.api_token)


def select_action(
    observation: SurveillanceObservation,
    *,
    client: Optional[OpenAI] = None,
    config: Optional[PolicyConfig] = None,
    allow_fallback: bool = True,
) -> str:
    """Run the primary LLM baseline and optionally fall back to the heuristic policy."""

    config = config or load_policy_config()
    if client is None:
        client = build_llm_client(config)

    if client is None:
        if allow_fallback:
            return choose_surveillance_action(observation)
        raise RuntimeError("HF_TOKEN is required for the primary LLM baseline.")

    try:
        return llm_action(client, observation, config.model_name)
    except Exception:
        if allow_fallback:
            return choose_surveillance_action(observation)
        raise


def policy_label(*, client: Optional[OpenAI], config: Optional[PolicyConfig] = None) -> str:
    """Return a human-readable label for logs and diagnostics."""

    config = config or load_policy_config()
    if client is not None:
        return config.model_name
    return "heuristic-fallback"
