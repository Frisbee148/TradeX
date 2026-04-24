"""
MEVerse (RLiquidity V3) Environment Implementation.

A self-contained RL environment simulating a Uniswap V3 concentrated-liquidity
pool with adversarial MEV bots. Zero external dependencies beyond Python 3.10 stdlib.

Modules (all inline):
  - TickManager: per-tick liquidity map
  - SwapEngine: V3 sqrt-price swap math
  - LPManager: agent LP positions + fee accrual
  - MEVEngine: sandwich / JIT / front-run / adaptive strategies
  - RewardEngine: dense per-step + terminal rewards
  - Grader: deterministic normalized scoring
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import MeverseAction, MeverseObservation
    from ..env_controller import (
        BOT_TYPES,
        EpisodeFailure,
        build_episode_result,
    )
except ImportError:
    from models import MeverseAction, MeverseObservation
    from env_controller import (
        BOT_TYPES,
        EpisodeFailure,
        build_episode_result,
    )


# ════════════════════════════════════════════════
#  Constants & Config
# ════════════════════════════════════════════════

TICK_SPACING = 60  # standard for 0.3% fee tier
BASE = 1.0001


@dataclass
class EnvConfig:
    mev_strategy: str = "passive"      # passive | jit | sandwich | adaptive
    max_steps: int = 30
    price_volatility: float = 0.01
    slippage_threshold: float = 0.01
    num_lp_positions: int = 8
    initial_price: float = 1800.0
    initial_token0: float = 10.0       # ETH
    initial_token1: float = 18_000.0   # USDC
    fee_rate: float = 0.003            # 0.3% fee tier
    w_profit: float = 0.40
    w_mev_avoidance: float = 0.30
    w_efficiency: float = 0.20
    w_lp_yield: float = 0.10


TASK_CONFIGS = {
    "easy": EnvConfig(
        mev_strategy="passive",
        max_steps=30,
        price_volatility=0.01,
        num_lp_positions=8,
        initial_price=1800.0,
        initial_token0=10.0,
        initial_token1=18_000.0,
    ),
    "medium": EnvConfig(
        mev_strategy="jit",
        max_steps=40,
        price_volatility=0.025,
        slippage_threshold=0.008,
        num_lp_positions=12,
        initial_price=1800.0,
        initial_token0=10.0,
        initial_token1=18_000.0,
    ),
    "hard": EnvConfig(
        mev_strategy="adaptive",
        max_steps=50,
        price_volatility=0.045,
        slippage_threshold=0.005,
        num_lp_positions=16,
        initial_price=1800.0,
        initial_token0=10.0,
        initial_token1=18_000.0,
        w_mev_avoidance=0.35,
        w_efficiency=0.15,
    ),
}


# ════════════════════════════════════════════════
#  Data Structures
# ════════════════════════════════════════════════

def price_to_tick(price: float) -> int:
    return int(math.log(price) / math.log(BASE))


def tick_to_price(tick: int) -> float:
    return BASE ** tick


def price_to_sqrt(price: float) -> float:
    return math.sqrt(price)


@dataclass
class LPPosition:
    position_id: str
    tick_lower: int
    tick_upper: int
    liquidity: float
    fees_earned: float = 0.0
    is_agent: bool = False


@dataclass
class TickState:
    net_liquidity: float = 0.0
    jit_liquidity: float = 0.0


# ════════════════════════════════════════════════
#  Tick Manager
# ════════════════════════════════════════════════

class TickManager:
    def __init__(self):
        self.ticks: Dict[int, TickState] = {}

    def add_liquidity(self, tick_lower: int, tick_upper: int, liquidity: float, is_jit: bool = False):
        for t in range(tick_lower, tick_upper + 1, TICK_SPACING):
            if t not in self.ticks:
                self.ticks[t] = TickState()
            if is_jit:
                self.ticks[t].jit_liquidity += liquidity
            else:
                self.ticks[t].net_liquidity += liquidity

    def remove_liquidity(self, tick_lower: int, tick_upper: int, liquidity: float, is_jit: bool = False):
        for t in range(tick_lower, tick_upper + 1, TICK_SPACING):
            if t in self.ticks:
                if is_jit:
                    self.ticks[t].jit_liquidity = max(0, self.ticks[t].jit_liquidity - liquidity)
                else:
                    self.ticks[t].net_liquidity = max(0, self.ticks[t].net_liquidity - liquidity)

    def get_active_liquidity(self, current_tick: int) -> float:
        nearest = self._nearest_tick(current_tick)
        if nearest in self.ticks:
            ts = self.ticks[nearest]
            return ts.net_liquidity + ts.jit_liquidity
        return 0.0

    def get_distribution(self, current_tick: int) -> List[Dict[str, Any]]:
        """Return ±5 tick window around current tick."""
        result = []
        center = self._nearest_tick(current_tick)
        for i in range(-5, 6):
            t = center + i * TICK_SPACING
            price = tick_to_price(t)
            ts = self.ticks.get(t, TickState())
            result.append({
                "tick": t,
                "price": round(price, 4),
                "net_liquidity": round(ts.net_liquidity, 4),
                "jit_liquidity": round(ts.jit_liquidity, 4),
            })
        return result

    def _nearest_tick(self, tick: int) -> int:
        return round(tick / TICK_SPACING) * TICK_SPACING


# ════════════════════════════════════════════════
#  Swap Engine (V3 sqrt-price math)
# ════════════════════════════════════════════════

class SwapEngine:
    def __init__(self, fee_rate: float = 0.003):
        self.fee_rate = fee_rate

    def execute_swap(
        self,
        amount_in: float,
        zero_for_one: bool,
        sqrt_price: float,
        active_liquidity: float,
    ) -> Tuple[float, float, float]:
        """
        Execute a V3-style swap.
        Returns: (amount_out, new_sqrt_price, fee_paid)
        """
        if active_liquidity <= 0 or amount_in <= 0:
            return 0.0, sqrt_price, 0.0

        fee = amount_in * self.fee_rate
        amount_after_fee = amount_in - fee

        if zero_for_one:
            # token0 -> token1: Δ(1/√P) = Δx / L
            delta_inv_sqrt = amount_after_fee / active_liquidity
            new_inv_sqrt = (1.0 / sqrt_price) + delta_inv_sqrt
            new_sqrt_price = 1.0 / new_inv_sqrt
            # amount_out in token1
            amount_out = active_liquidity * (sqrt_price - new_sqrt_price)
        else:
            # token1 -> token0: Δ√P = Δy / L
            delta_sqrt = amount_after_fee / active_liquidity
            new_sqrt_price = sqrt_price + delta_sqrt
            # amount_out in token0
            amount_out = active_liquidity * (1.0 / sqrt_price - 1.0 / new_sqrt_price)

        amount_out = max(0.0, amount_out)
        return amount_out, new_sqrt_price, fee


# ════════════════════════════════════════════════
#  LP Manager
# ════════════════════════════════════════════════

class LPManager:
    def __init__(self):
        self.positions: List[LPPosition] = []

    def add_position(
        self,
        tick_lower: int,
        tick_upper: int,
        liquidity: float,
        is_agent: bool = False,
    ) -> LPPosition:
        pos = LPPosition(
            position_id=str(uuid4())[:8],
            tick_lower=tick_lower,
            tick_upper=tick_upper,
            liquidity=liquidity,
            is_agent=is_agent,
        )
        self.positions.append(pos)
        return pos

    def remove_position(self, position_id: str) -> Optional[LPPosition]:
        for i, pos in enumerate(self.positions):
            if pos.position_id == position_id:
                return self.positions.pop(i)
        return None

    def accrue_fees(self, current_tick: int, fee_amount: float):
        """Distribute fees pro-rata to in-range positions."""
        in_range = [p for p in self.positions if p.tick_lower <= current_tick <= p.tick_upper]
        total_liq = sum(p.liquidity for p in in_range)
        if total_liq <= 0:
            return
        for p in in_range:
            share = p.liquidity / total_liq
            p.fees_earned += fee_amount * share

    def get_agent_positions(self) -> List[Dict[str, Any]]:
        return [
            {
                "position_id": p.position_id,
                "tick_lower": p.tick_lower,
                "tick_upper": p.tick_upper,
                "liquidity": round(p.liquidity, 4),
                "fees_earned": round(p.fees_earned, 6),
            }
            for p in self.positions
            if p.is_agent
        ]

    def get_agent_total_fees(self) -> float:
        return sum(p.fees_earned for p in self.positions if p.is_agent)


# ════════════════════════════════════════════════
#  MEV Engine
# ════════════════════════════════════════════════

class MEVEngine:
    """
    Samples a bot personality per step from `strategy_weights`.
    Reports the sampled bot + attack success so the EnvController can learn.
    """

    def __init__(
        self,
        strategy: str,
        slippage_threshold: float,
        rng: random.Random,
        strategy_weights: Optional[Dict[str, float]] = None,
        initial_aggression: float = 0.5,
        stealth_acceleration: float = 0.012,
        noise_spike_probability: float = 0.0,
        slippage_threshold_tightness: float = 1.0,
    ):
        self.strategy = strategy
        self.base_slippage_threshold = slippage_threshold
        self.slippage_threshold = slippage_threshold * max(0.1, slippage_threshold_tightness)
        self.slippage_threshold_tightness = slippage_threshold_tightness
        self.rng = rng
        self.aggression = initial_aggression
        self.initial_aggression = initial_aggression
        self.stealth_acceleration = stealth_acceleration
        self.noise_spike_probability = noise_spike_probability
        self.agent_pattern_count = 0

        # Per-step telemetry — reset each step
        self.last_bot_type: Optional[str] = None
        self.last_attack_success: bool = False

        # Aggregate exposure + success counters (reset per episode)
        self.exposure_counts: Dict[str, int] = {b: 0 for b in BOT_TYPES}
        self.attack_success_counts: Dict[str, int] = {b: 0 for b in BOT_TYPES}

        # Bot-type sampling weights
        default_weights = {b: 1.0 for b in BOT_TYPES}
        self.strategy_weights = strategy_weights or default_weights
        self._prime_weights_from_strategy()

    def _prime_weights_from_strategy(self) -> None:
        """If controller didn't supply weights, bias toward configured strategy."""
        if sum(self.strategy_weights.values()) <= 0:
            self.strategy_weights = {b: 1.0 for b in BOT_TYPES}
        if self.strategy in BOT_TYPES and self.strategy != "passive":
            # only nudge when the env was started with a single strategy profile
            bias = 2.0
            nudged = dict(self.strategy_weights)
            nudged[self.strategy] = nudged.get(self.strategy, 1.0) * bias
            self.strategy_weights = nudged

    def _sample_bot_type(self) -> str:
        items = list(self.strategy_weights.items())
        total = sum(w for _, w in items)
        if total <= 0:
            return "passive"
        pick = self.rng.random() * total
        cum = 0.0
        for bot, w in items:
            cum += w
            if pick <= cum:
                return bot
        return items[-1][0]

    def reset_episode_counters(self) -> None:
        self.exposure_counts = {b: 0 for b in BOT_TYPES}
        self.attack_success_counts = {b: 0 for b in BOT_TYPES}
        self.aggression = self.initial_aggression

    def pre_step(
        self,
        action: Dict[str, Any],
        tick_manager: TickManager,
        current_tick: int,
        active_liquidity: float,
        step_num: int = 0,
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Sample a bot for this step, let it attempt an attack, return
        (mev_loss, mempool_txs). Records per-bot exposure + success.
        """
        action_type = action.get("action_type", "hold")
        params = action.get("params", {})
        mempool_txs: List[Dict[str, Any]] = []
        mev_loss = 0.0

        bot = self._sample_bot_type()
        self.last_bot_type = bot
        self.last_attack_success = False

        # Stealth ramps agression over the episode (harder to detect late-game)
        stealth_bonus = min(0.6, step_num * self.stealth_acceleration)

        # Private RPC defeats every bot except "hybrid" (which does LP-level attacks)
        used_private_rpc = bool(params.get("use_private_rpc", False))

        if bot == "passive":
            if self.rng.random() < self.noise_spike_probability:
                mempool_txs.append(self._noise_tx())
            return 0.0, mempool_txs

        amount = params.get("amount_in", 0) or params.get("total_amount", 0)
        attackable = action_type in ("swap_exact_in", "split_swap") and amount and active_liquidity > 0

        if attackable and not used_private_rpc:
            mempool_txs.append({
                "tx_id": str(uuid4())[:8],
                "action_type": action_type,
                "amount": amount,
                "zero_for_one": params.get("zero_for_one", True),
                "bot_hint": bot,
            })

        if self.rng.random() < self.noise_spike_probability:
            mempool_txs.append(self._noise_tx())

        if not attackable:
            return 0.0, mempool_txs

        # Only count exposures on attackable actions
        self.exposure_counts[bot] = self.exposure_counts.get(bot, 0) + 1

        if used_private_rpc and bot != "hybrid":
            return 0.0, mempool_txs

        ratio = amount / active_liquidity
        effective_threshold = self.slippage_threshold * (1.0 - stealth_bonus * 0.5)

        if bot == "jit" and action_type == "swap_exact_in":
            if ratio > effective_threshold:
                jit_liq = amount * 2.0
                tick_manager.add_liquidity(
                    current_tick - TICK_SPACING,
                    current_tick + TICK_SPACING,
                    jit_liq,
                    is_jit=True,
                )
                mev_loss = amount * 0.003 * 0.8

        elif bot == "sandwich" and action_type == "swap_exact_in":
            if ratio > effective_threshold:
                impact = ratio * 0.3
                mev_loss = amount * min(impact, 0.05)

        elif bot == "adaptive" and action_type == "swap_exact_in":
            adapt_threshold = effective_threshold * (1.0 - self.aggression * 0.5)
            if ratio > adapt_threshold:
                impact = ratio * 0.3 * (1.0 + self.aggression)
                mev_loss = amount * min(impact, 0.07)

        elif bot == "hybrid":
            # Hybrid: split-swap or sandwich-style, harder to detect
            if action_type == "split_swap":
                mev_loss = amount * 0.004 * (1.0 + stealth_bonus)
            elif ratio > effective_threshold * 0.7:
                impact = ratio * 0.25 * (1.0 + stealth_bonus)
                mev_loss = amount * min(impact, 0.06)

        if mev_loss > 0:
            self.last_attack_success = True
            self.attack_success_counts[bot] = self.attack_success_counts.get(bot, 0) + 1

        return mev_loss, mempool_txs

    def _noise_tx(self) -> Dict[str, Any]:
        return {
            "tx_id": str(uuid4())[:8],
            "action_type": "swap_exact_in",
            "amount": round(self.rng.uniform(0.1, 2.0), 3),
            "zero_for_one": self.rng.random() < 0.5,
            "bot_hint": "noise",
        }

    def post_step(self, tick_manager: TickManager, current_tick: int):
        """Clean up JIT liquidity after step."""
        for t, ts in tick_manager.ticks.items():
            ts.jit_liquidity = 0.0

    def update_adaptive(self, action_type: str, used_private_rpc: bool):
        """Adaptive bot learns from agent behavior across the episode."""
        self.agent_pattern_count += 1
        if used_private_rpc:
            self.aggression = min(1.0, self.aggression + 0.05)
        if action_type == "hold":
            self.aggression = min(1.0, self.aggression + 0.03)
        if action_type == "split_swap":
            self.aggression = max(0.2, self.aggression - 0.02)


# ════════════════════════════════════════════════
#  Main Environment
# ════════════════════════════════════════════════

class MeverseEnvironment(Environment):
    """
    MEVerse: MEV-Aware RL Environment for Uniswap V3.

    Agents trade, provide liquidity, and defend against adversarial MEV bots
    in a simulated concentrated-liquidity pool.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    # Mev-loss threshold that counts a step as a "failure" for SIE
    FAILURE_MEV_LOSS_THRESHOLD: float = 0.01

    def __init__(self, task: str = "easy", transform=None, rubric=None):
        super().__init__(transform=transform, rubric=rubric)
        self._task = task if task in TASK_CONFIGS else "easy"
        self._config = TASK_CONFIGS[self._task]
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._rng = random.Random(42)

        # SIE config (set by controller via reset kwargs)
        self._sie_config: Dict[str, Any] = {}
        self._episode_counter: int = 0

        # Pool state
        self._current_price = self._config.initial_price
        self._sqrt_price = price_to_sqrt(self._current_price)
        self._current_tick = price_to_tick(self._current_price)

        # Managers
        self._tick_manager = TickManager()
        self._swap_engine = SwapEngine(self._config.fee_rate)
        self._lp_manager = LPManager()
        self._mev_engine = self._build_mev_engine()

        # Agent wallet
        self._agent_token0 = self._config.initial_token0
        self._agent_token1 = self._config.initial_token1
        self._initial_value = self._portfolio_value()

        # Episode tracking
        self._step_num = 0
        self._terminated = False
        self._mev_losses: List[float] = []
        self._last_mev_loss = 0.0
        self._rewards: List[float] = []
        self._private_rpc_uses = 0
        self._split_swap_uses = 0
        self._invalid_actions = 0
        self._total_fees_distributed = 0.0
        self._last_action_error: Optional[str] = None
        self._failures: List[EpisodeFailure] = []
        self._price_volatility_multiplier: float = 1.0

    def _set_task(self, task: Optional[str]) -> None:
        selected = task if task in TASK_CONFIGS else "easy"
        self._task = selected
        self._config = TASK_CONFIGS[selected]

    def _build_mev_engine(self) -> "MEVEngine":
        cfg = self._sie_config
        return MEVEngine(
            strategy=self._config.mev_strategy,
            slippage_threshold=self._config.slippage_threshold,
            rng=self._rng,
            strategy_weights=cfg.get("bot_weights"),
            initial_aggression=float(cfg.get("initial_aggression", 0.5)),
            stealth_acceleration=float(cfg.get("stealth_acceleration", 0.012)),
            noise_spike_probability=float(cfg.get("noise_spike_probability", 0.0)),
            slippage_threshold_tightness=float(cfg.get("slippage_threshold_tightness", 1.0)),
        )

    def _invalid_action(self, message: str, penalty: float = -0.2) -> float:
        self._invalid_actions += 1
        self._last_action_error = message
        return penalty

    def _portfolio_value(self) -> float:
        """Portfolio value in token1 (USDC)."""
        val = self._agent_token1 + self._agent_token0 * self._current_price
        # Add LP position value (simplified: liquidity * 2 / sqrt(price) equivalent)
        for pos in self._lp_manager.positions:
            if pos.is_agent:
                val += pos.fees_earned
                # Approximate LP value based on whether in range
                if pos.tick_lower <= self._current_tick <= pos.tick_upper:
                    val += pos.liquidity * 0.01  # simplified value proxy
        return val

    def _seed_background_lps(self):
        """Seed background LP positions in a bell curve around current tick."""
        center = self._tick_manager._nearest_tick(self._current_tick)
        for i in range(self._config.num_lp_positions):
            offset = self._rng.gauss(0, 3) * TICK_SPACING
            tick_lower = int(center + offset - 2 * TICK_SPACING)
            tick_upper = int(center + offset + 2 * TICK_SPACING)
            # Snap to tick spacing
            tick_lower = round(tick_lower / TICK_SPACING) * TICK_SPACING
            tick_upper = round(tick_upper / TICK_SPACING) * TICK_SPACING
            if tick_upper <= tick_lower:
                tick_upper = tick_lower + TICK_SPACING

            liq = self._rng.uniform(500, 5000)
            self._tick_manager.add_liquidity(tick_lower, tick_upper, liq)
            self._lp_manager.add_position(tick_lower, tick_upper, liq, is_agent=False)

    def reset(
        self,
        seed: Optional[int] = 42,
        episode_id: Optional[str] = None,
        task: Optional[str] = None,
        task_name: Optional[str] = None,
        sie_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> MeverseObservation:
        """Reset the environment with deterministic seed.

        ``sie_config`` is the per-episode override injected by the EnvController
        (bot weights, curriculum phase, stealth acceleration, etc.).
        """
        del kwargs
        self._set_task(task_name or task or self._task)
        self._rng = random.Random(seed)

        self._sie_config = dict(sie_config or {})
        self._price_volatility_multiplier = float(
            self._sie_config.get("price_volatility_multiplier", 1.0)
        )

        self._current_price = self._config.initial_price
        self._sqrt_price = price_to_sqrt(self._current_price)
        self._current_tick = price_to_tick(self._current_price)

        self._tick_manager = TickManager()
        self._swap_engine = SwapEngine(self._config.fee_rate)
        self._lp_manager = LPManager()
        self._mev_engine = self._build_mev_engine()

        self._agent_token0 = self._config.initial_token0
        self._agent_token1 = self._config.initial_token1

        self._seed_background_lps()

        self._initial_value = self._portfolio_value()
        self._step_num = 0
        self._terminated = False
        self._mev_losses = []
        self._last_mev_loss = 0.0
        self._rewards = []
        self._private_rpc_uses = 0
        self._split_swap_uses = 0
        self._invalid_actions = 0
        self._total_fees_distributed = 0.0
        self._last_action_error = None
        self._failures = []
        self._episode_counter += 1

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)

        return self._build_observation(done=False, reward=0.0)

    def step(
        self,
        action: MeverseAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MeverseObservation:
        """Execute one step."""
        del timeout_s, kwargs
        if self._terminated:
            return self._build_observation(done=True, reward=0.0)

        self._step_num += 1
        self._state.step_count = self._step_num

        action_dict = {"action_type": action.action_type, "params": action.params}
        params = action.params
        self._last_action_error = None

        # ── PRE-STEP: MEV bot inspects ──
        active_liq = self._tick_manager.get_active_liquidity(self._current_tick)
        mev_loss, mempool_txs = self._mev_engine.pre_step(
            action_dict, self._tick_manager, self._current_tick, active_liq,
            step_num=self._step_num,
        )

        # ── EXECUTE ACTION ──
        step_reward = -0.05  # time pressure cost
        action_type = action.action_type
        used_private_rpc = params.get("use_private_rpc", False)
        fee_this_step = 0.0

        if action_type == "swap_exact_in":
            step_reward += self._execute_swap(params)
            fee_this_step = params.get("_fee", 0)
            if used_private_rpc:
                self._private_rpc_uses += 1

        elif action_type == "split_swap":
            total = params.get("total_amount", 0)
            splits = max(1, int(params.get("num_splits", 3)))
            zero_for_one = params.get("zero_for_one", True)
            per_split = total / splits
            for _ in range(splits):
                r = self._execute_swap({
                    "amount_in": per_split,
                    "zero_for_one": zero_for_one,
                })
                step_reward += r
            self._split_swap_uses += 1
            step_reward += 0.1  # strategic action bonus

        elif action_type == "add_liquidity":
            step_reward += self._execute_add_liquidity(params)

        elif action_type == "remove_liquidity":
            step_reward += self._execute_remove_liquidity(params)

        elif action_type == "range_order":
            step_reward += self._execute_range_order(params)

        elif action_type == "jit_liquidity":
            step_reward += self._execute_jit(params)

        elif action_type == "hold":
            pass  # only pays time cost

        elif action_type == "close_episode":
            self._terminated = True

        else:
            step_reward += self._invalid_action(f"Unknown action_type: {action_type}")

        if self._last_action_error:
            # Invalid actions should not also trigger simulated MEV extraction.
            mev_loss = 0.0
            mempool_txs = []

        # ── POST-STEP: MEV cleanup ──
        self._mev_engine.post_step(self._tick_manager, self._current_tick)
        self._mev_engine.update_adaptive(action_type, used_private_rpc)

        # ── Fee accrual ──
        if fee_this_step > 0:
            self._lp_manager.accrue_fees(self._current_tick, fee_this_step)
            self._total_fees_distributed += fee_this_step

        agent_fees = self._lp_manager.get_agent_total_fees()
        step_reward += agent_fees * 0.01  # LP fee reward component

        # ── MEV loss ──
        step_reward -= mev_loss
        self._mev_losses.append(mev_loss)
        self._last_mev_loss = mev_loss

        # ── Record failure for SIE ──
        if mev_loss > self.FAILURE_MEV_LOSS_THRESHOLD and self._mev_engine.last_bot_type:
            self._failures.append(
                EpisodeFailure(
                    episode_id=self._episode_counter,
                    task_name=self._task,
                    step_num=self._step_num,
                    bot_type=self._mev_engine.last_bot_type,
                    mev_loss=round(mev_loss, 6),
                    action_taken=action_type,
                    agent_used_private_rpc=used_private_rpc,
                    agent_used_split_swap=(action_type == "split_swap"),
                    stealth_level=min(
                        0.6, self._step_num * self._mev_engine.stealth_acceleration
                    ),
                    aggression_level=self._mev_engine.aggression,
                )
            )

        # ── Price walk (geometric Brownian motion) ──
        drift = self._rng.gauss(
            0, self._config.price_volatility * self._price_volatility_multiplier
        )
        self._current_price *= math.exp(drift)
        self._current_price = max(100.0, min(50000.0, self._current_price))
        self._sqrt_price = price_to_sqrt(self._current_price)
        self._current_tick = price_to_tick(self._current_price)

        # ── Terminal reward ──
        if self._step_num >= self._config.max_steps:
            self._terminated = True

        if self._terminated:
            final_val = self._portfolio_value()
            terminal_reward = (final_val - self._initial_value) / self._initial_value * 5.0
            step_reward += terminal_reward

        self._rewards.append(step_reward)

        return self._build_observation(
            done=self._terminated,
            reward=round(step_reward, 4),
            mempool=mempool_txs,
            mev_loss=mev_loss,
        )

    def _execute_swap(self, params: Dict[str, Any]) -> float:
        amount_in = params.get("amount_in", 0)
        zero_for_one = params.get("zero_for_one", True)

        if amount_in <= 0:
            return self._invalid_action("swap_exact_in requires params.amount_in > 0")

        # Check balance
        if zero_for_one and amount_in > self._agent_token0:
            return self._invalid_action("Insufficient token0 balance for swap_exact_in")
        if not zero_for_one and amount_in > self._agent_token1:
            return self._invalid_action("Insufficient token1 balance for swap_exact_in")

        active_liq = self._tick_manager.get_active_liquidity(self._current_tick)
        amount_out, new_sqrt, fee = self._swap_engine.execute_swap(
            amount_in, zero_for_one, self._sqrt_price, active_liq,
        )

        if amount_out <= 0:
            self._last_action_error = "Swap produced zero output because active liquidity was too low"
            return -0.1

        # Update balances
        if zero_for_one:
            self._agent_token0 -= amount_in
            self._agent_token1 += amount_out
        else:
            self._agent_token1 -= amount_in
            self._agent_token0 += amount_out

        self._sqrt_price = new_sqrt
        self._current_price = new_sqrt ** 2
        self._current_tick = price_to_tick(self._current_price)

        params["_fee"] = fee

        # Swap quality reward
        fair_value = amount_in * (self._current_price if zero_for_one else 1.0 / self._current_price)
        if fair_value > 0:
            quality = (amount_out - fair_value) / fair_value * 0.1
        else:
            quality = 0.0
        return quality

    def _execute_add_liquidity(self, params: Dict[str, Any]) -> float:
        tick_lower = params.get("tick_lower", self._current_tick - 2 * TICK_SPACING)
        tick_upper = params.get("tick_upper", self._current_tick + 2 * TICK_SPACING)
        amount0 = params.get("amount0_desired", 0)
        amount1 = params.get("amount1_desired", 0)

        # Snap to tick spacing
        tick_lower = round(tick_lower / TICK_SPACING) * TICK_SPACING
        tick_upper = round(tick_upper / TICK_SPACING) * TICK_SPACING

        if tick_upper <= tick_lower:
            return self._invalid_action("add_liquidity requires tick_upper > tick_lower")

        if amount0 > self._agent_token0 or amount1 > self._agent_token1:
            return self._invalid_action("Insufficient balance for add_liquidity")

        if amount0 <= 0 and amount1 <= 0:
            return self._invalid_action("add_liquidity requires a positive token amount")

        # Simplified liquidity calculation
        liquidity = (amount0 * self._current_price + amount1) * 0.5

        self._agent_token0 -= amount0
        self._agent_token1 -= amount1

        self._tick_manager.add_liquidity(tick_lower, tick_upper, liquidity)
        self._lp_manager.add_position(tick_lower, tick_upper, liquidity, is_agent=True)

        return 0.05  # small bonus for providing liquidity

    def _execute_remove_liquidity(self, params: Dict[str, Any]) -> float:
        position_id = params.get("position_id", "")
        pos = self._lp_manager.remove_position(position_id)
        if pos is None or not pos.is_agent:
            return self._invalid_action("remove_liquidity requires a valid agent position_id")

        self._tick_manager.remove_liquidity(pos.tick_lower, pos.tick_upper, pos.liquidity)

        # Return capital + fees (simplified)
        value = pos.liquidity * 0.01 + pos.fees_earned
        # Split roughly between tokens
        self._agent_token1 += value * 0.5
        self._agent_token0 += (value * 0.5) / max(self._current_price, 1.0)

        return 0.0

    def _execute_range_order(self, params: Dict[str, Any]) -> float:
        tick_lower = params.get("tick_lower", self._current_tick)
        tick_upper = params.get("tick_upper", self._current_tick + TICK_SPACING)
        token = params.get("token", "token1")
        amount = params.get("amount", 0)

        tick_lower = round(tick_lower / TICK_SPACING) * TICK_SPACING
        tick_upper = round(tick_upper / TICK_SPACING) * TICK_SPACING
        if tick_upper <= tick_lower:
            tick_upper = tick_lower + TICK_SPACING

        if amount <= 0:
            return self._invalid_action("range_order requires params.amount > 0")

        if token == "token0" and amount > self._agent_token0:
            return self._invalid_action("Insufficient token0 balance for range_order")
        if token == "token1" and amount > self._agent_token1:
            return self._invalid_action("Insufficient token1 balance for range_order")

        liquidity = amount * (self._current_price if token == "token0" else 1.0) * 0.5
        if token == "token0":
            self._agent_token0 -= amount
        else:
            self._agent_token1 -= amount

        self._tick_manager.add_liquidity(tick_lower, tick_upper, liquidity)
        self._lp_manager.add_position(tick_lower, tick_upper, liquidity, is_agent=True)
        return 0.03

    def _execute_jit(self, params: Dict[str, Any]) -> float:
        """Agent uses JIT offensively."""
        amount0 = params.get("amount0", 0)
        amount1 = params.get("amount1", 0)
        tick_lower = params.get("tick_lower", self._current_tick - TICK_SPACING)
        tick_upper = params.get("tick_upper", self._current_tick + TICK_SPACING)

        tick_lower = round(tick_lower / TICK_SPACING) * TICK_SPACING
        tick_upper = round(tick_upper / TICK_SPACING) * TICK_SPACING

        if amount0 > self._agent_token0 or amount1 > self._agent_token1:
            return self._invalid_action("Insufficient balance for jit_liquidity")

        if amount0 <= 0 and amount1 <= 0:
            return self._invalid_action("jit_liquidity requires a positive token amount")

        liquidity = (amount0 * self._current_price + amount1) * 0.5
        self._agent_token0 -= amount0
        self._agent_token1 -= amount1

        self._tick_manager.add_liquidity(tick_lower, tick_upper, liquidity, is_jit=True)
        # JIT is ephemeral — capital returned after step with small fee capture
        fee_capture = liquidity * 0.001 * self._rng.uniform(0.5, 1.5)
        self._agent_token1 += amount1 + fee_capture
        self._agent_token0 += amount0

        return fee_capture * 0.1

    def grade(self) -> Dict[str, Any]:
        """Deterministic grading — normalized 0.0 to 1.0."""
        final_value = self._portfolio_value()
        steps_run = max(1, self._step_num)

        # Profit score (40%)
        raw_return = (final_value - self._initial_value) / max(self._initial_value, 1.0)
        profit_score = min(1.0, max(0.0, (raw_return + 1.0) / 1.5))

        # MEV avoidance (30%)
        total_mev = sum(self._mev_losses)
        max_possible_mev = steps_run * 0.05
        mev_score = max(0.0, 1.0 - total_mev / max(max_possible_mev, 0.001))

        # Efficiency (20%)
        strategic = self._private_rpc_uses + self._split_swap_uses
        efficiency_score = min(1.0, strategic / max(steps_run * 0.3, 1))

        # LP yield (10%)
        fees_earned = self._lp_manager.get_agent_total_fees()
        lp_score = min(1.0, fees_earned / max(self._initial_value * 0.05, 0.001))

        w_p = self._config.w_profit
        w_m = self._config.w_mev_avoidance
        w_e = self._config.w_efficiency
        w_l = self._config.w_lp_yield

        final_score = w_p * profit_score + w_m * mev_score + w_e * efficiency_score + w_l * lp_score

        return {
            "final_score": round(final_score, 4),
            "profit_score": round(profit_score, 4),
            "mev_avoidance_score": round(mev_score, 4),
            "efficiency_score": round(efficiency_score, 4),
            "lp_yield_score": round(lp_score, 4),
            "total_mev_loss": round(total_mev, 6),
            "fees_earned": round(fees_earned, 6),
            "final_portfolio_value": round(final_value, 4),
            "initial_portfolio_value": round(self._initial_value, 4),
            "steps_run": steps_run,
            "task": self._task,
            "bot_exposures": dict(self._mev_engine.exposure_counts),
            "bot_successful_attacks": dict(self._mev_engine.attack_success_counts),
            "num_failures": len(self._failures),
            "sie_config_used": dict(self._sie_config),
        }

    def build_sie_result(self):
        """Build an `EpisodeResult` for the EnvController from current state."""
        grade = self.grade()
        return build_episode_result(
            episode_id=self._episode_counter,
            task_name=self._task,
            grade=grade,
            bot_exposures=dict(self._mev_engine.exposure_counts),
            bot_successful_attacks=dict(self._mev_engine.attack_success_counts),
            failures=list(self._failures),
            config_used=dict(self._sie_config),
        )

    def _build_observation(
        self,
        done: bool,
        reward: float,
        mempool: Optional[List[Dict]] = None,
        mev_loss: float = 0.0,
    ) -> MeverseObservation:
        return MeverseObservation(
            current_tick=self._current_tick,
            current_price=round(self._current_price, 4),
            sqrt_price=round(self._sqrt_price, 6),
            active_liquidity=round(
                self._tick_manager.get_active_liquidity(self._current_tick), 4
            ),
            tick_distribution=self._tick_manager.get_distribution(self._current_tick),
            agent_token0=round(self._agent_token0, 6),
            agent_token1=round(self._agent_token1, 4),
            agent_positions=self._lp_manager.get_agent_positions(),
            mempool=mempool or [],
            last_mev_loss=round(self._last_mev_loss, 6),
            step_num=self._step_num,
            max_steps=self._config.max_steps,
            task_name=self._task,
            done=done,
            reward=reward,
            metadata={
                "episode_id": self._state.episode_id,
                "available_tasks": list(TASK_CONFIGS.keys()),
                "last_action_error": self._last_action_error,
            },
        )

    @property
    def state(self) -> State:
        return self._state
