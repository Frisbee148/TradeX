"""MEVerse Environment Client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import MeverseAction, MeverseObservation


class MeverseEnv(EnvClient[MeverseAction, MeverseObservation, State]):
    """
    Client for the MEVerse Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example:
        >>> client = MeverseEnv.from_docker_image("meverse:latest")
        >>> result = await client.reset()
        >>> obs = result.observation
        >>> print(obs.current_price, obs.agent_token0)
        >>> result = await client.step(MeverseAction(
        ...     action_type="swap_exact_in",
        ...     params={"amount_in": 1.0, "zero_for_one": True}
        ... ))
    """

    def _step_payload(self, action: MeverseAction) -> Dict:
        return {
            "action_type": action.action_type,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict) -> StepResult[MeverseObservation]:
        obs_data = payload.get("observation", {})
        observation = MeverseObservation(
            current_tick=obs_data.get("current_tick", 0),
            current_price=obs_data.get("current_price", 0.0),
            sqrt_price=obs_data.get("sqrt_price", 0.0),
            active_liquidity=obs_data.get("active_liquidity", 0.0),
            tick_distribution=obs_data.get("tick_distribution", []),
            agent_token0=obs_data.get("agent_token0", 0.0),
            agent_token1=obs_data.get("agent_token1", 0.0),
            agent_positions=obs_data.get("agent_positions", []),
            mempool=obs_data.get("mempool", []),
            last_mev_loss=obs_data.get("last_mev_loss", 0.0),
            step_num=obs_data.get("step_num", 0),
            max_steps=obs_data.get("max_steps", 30),
            task_name=obs_data.get("task_name", "easy"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
