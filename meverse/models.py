"""
Data models for the MEVerse (RLiquidity V3) Environment.

MEV-Aware Reinforcement Learning environment for Uniswap V3 pool simulation.
Agents learn to trade, provide liquidity, and survive adversarial MEV attacks.
"""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ──────────────────────────────────────────────
#  Action Models
# ──────────────────────────────────────────────

ActionType = Literal[
    "swap_exact_in",
    "split_swap",
    "add_liquidity",
    "remove_liquidity",
    "range_order",
    "jit_liquidity",
    "hold",
    "close_episode",
]


class MeverseAction(Action):
    """Action for the MEVerse environment."""

    action_type: ActionType = Field(..., description="One of the 8 strategic action types")
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Action-specific parameters (amount_in, zero_for_one, use_private_rpc, etc.)",
    )


# ──────────────────────────────────────────────
#  Observation Sub-Models
# ──────────────────────────────────────────────

class TickInfo(Dict[str, Any]):
    """Tick distribution entry — used as plain dict in observation."""
    pass


class MempoolTx(Dict[str, Any]):
    """Mempool transaction entry — used as plain dict in observation."""
    pass


class LPPositionInfo(Dict[str, Any]):
    """LP position entry — used as plain dict in observation."""
    pass


class MeverseObservation(Observation):
    """Observation from the MEVerse environment — full V3 pool state."""

    # Pool state
    current_tick: int = Field(default=0, description="Active tick index")
    current_price: float = Field(default=0.0, description="Token1/Token0 price (e.g. USDC per ETH)")
    sqrt_price: float = Field(default=0.0, description="Square root of price, used in V3 math")
    active_liquidity: float = Field(default=0.0, description="Total liquidity at current tick")

    # Tick distribution (±5 ticks window)
    tick_distribution: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="11 entries: tick, price, net_liquidity, jit_liquidity",
    )

    # Agent portfolio
    agent_token0: float = Field(default=0.0, description="ETH balance")
    agent_token1: float = Field(default=0.0, description="USDC balance")
    agent_positions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Agent LP positions with tick_lower, tick_upper, liquidity, fees_earned",
    )

    # Mempool
    mempool: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Pending transactions visible in mempool",
    )

    # Step info
    last_mev_loss: float = Field(default=0.0, description="MEV extracted in previous step")
    step_num: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=30, description="Maximum steps in this episode")

    # Task info
    task_name: str = Field(default="easy", description="Current task: easy/medium/hard")
