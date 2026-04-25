"""OpenEnv client for the market surveillance benchmark."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import SurveillanceAction, SurveillanceObservation


class MeverseEnv(EnvClient[SurveillanceAction, SurveillanceObservation, State]):
    """Client for the AMM market surveillance environment."""

    def _step_payload(self, action: SurveillanceAction) -> Dict:
        return {"action_type": action.action_type}

    def _parse_result(self, payload: Dict) -> StepResult[SurveillanceObservation]:
        obs_data = payload.get("observation", {})
        observation = SurveillanceObservation(
            current_amm_price=obs_data.get("current_amm_price", 0.0),
            liquidity_snapshot=obs_data.get("liquidity_snapshot", 0.0),
            recent_trade_count=obs_data.get("recent_trade_count", 0),
            trades_in_window=obs_data.get("trades_in_window", []),
            trade_frequency=obs_data.get("trade_frequency", 0.0),
            average_trade_size=obs_data.get("average_trade_size", 0.0),
            maximum_trade_size=obs_data.get("maximum_trade_size", 0.0),
            recent_slippage_impact=obs_data.get("recent_slippage_impact", 0.0),
            time_gap_mean=obs_data.get("time_gap_mean", 0.0),
            time_gap_min=obs_data.get("time_gap_min", 0.0),
            recent_time_gaps=obs_data.get("recent_time_gaps", []),
            recent_price_impacts=obs_data.get("recent_price_impacts", []),
            burst_indicator=obs_data.get("burst_indicator", 0.0),
            pattern_indicator=obs_data.get("pattern_indicator", 0.0),
            suspiciousness_score=obs_data.get("suspiciousness_score", 0.0),
            manipulation_score=obs_data.get("manipulation_score", 0.0),
            step_num=obs_data.get("step_num", 0),
            max_steps=obs_data.get("max_steps", 0),
            task_name=obs_data.get("task_name", "burst_detection"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(observation=observation, reward=payload.get("reward"), done=payload.get("done", False))

    def _parse_state(self, payload: Dict) -> State:
        return State(episode_id=payload.get("episode_id"), step_count=payload.get("step_count", 0))
