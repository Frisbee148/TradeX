"""Typed models for the bot-aware AMM market surveillance benchmark."""

from __future__ import annotations

import json
from typing import Any, List, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator

SurveillanceActionType = Literal["ALLOW", "FLAG", "BLOCK", "MONITOR"]


class SurveillanceAction(Action):
    """Market surveillance response selected by the agent."""

    action_type: SurveillanceActionType = Field(
        ...,
        description="Final surveillance response to the current market activity.",
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_action(cls, data: Any) -> Any:
        if isinstance(data, dict) and "action_type" in data:
            value = data["action_type"]
            if isinstance(value, str):
                data["action_type"] = value.strip().upper()
            params = data.get("params")
            if isinstance(params, str):
                try:
                    parsed = json.loads(params)
                    if isinstance(parsed, dict) and "action_type" in parsed:
                        data["action_type"] = str(parsed["action_type"]).strip().upper()
                except json.JSONDecodeError:
                    pass
        return data


class SurveillanceObservation(Observation):
    """Fixed-size surveillance features for a recent AMM trading window."""

    current_amm_price: float = Field(default=0.0)
    liquidity_snapshot: float = Field(default=0.0)
    recent_trade_count: int = Field(default=0)
    trades_in_window: List[float] = Field(default_factory=list)
    trade_frequency: float = Field(default=0.0)
    average_trade_size: float = Field(default=0.0)
    maximum_trade_size: float = Field(default=0.0)
    recent_slippage_impact: float = Field(default=0.0)
    time_gap_mean: float = Field(default=0.0)
    time_gap_min: float = Field(default=0.0)
    recent_time_gaps: List[float] = Field(default_factory=list)
    recent_price_impacts: List[float] = Field(default_factory=list)
    burst_indicator: float = Field(default=0.0)
    pattern_indicator: float = Field(default=0.0)
    suspiciousness_score: float = Field(default=0.0)
    manipulation_score: float = Field(default=0.0)
    step_num: int = Field(default=0)
    max_steps: int = Field(default=0)
    task_name: str = Field(default="burst_detection")


MeverseAction = SurveillanceAction
MeverseObservation = SurveillanceObservation
