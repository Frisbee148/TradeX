"""
Self-Improving Environment (SIE) controller for MEVerse.

Episodes are immutable. EnvController sits outside the episode loop:
  episode N ends -> ingest(result) -> next_config() -> episode N+1 starts

It maintains a failure buffer, per-bot-type weights, and a curriculum phase.
Between episodes it biases the MEV strategy mix toward bot types the agent
missed most, and escalates difficulty once the agent clears score thresholds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from statistics import mean
from typing import Deque, Dict, List, Optional


BOT_TYPES: List[str] = ["passive", "jit", "sandwich", "adaptive", "hybrid"]

PHASE_CONFIGS: Dict[int, Dict[str, float]] = {
    0: {
        "initial_aggression": 0.20,
        "stealth_acceleration": 0.008,
        "noise_spike_probability": 0.10,
        "price_volatility_multiplier": 0.8,
        "slippage_threshold_tightness": 1.2,
    },
    1: {
        "initial_aggression": 0.30,
        "stealth_acceleration": 0.012,
        "noise_spike_probability": 0.15,
        "price_volatility_multiplier": 1.0,
        "slippage_threshold_tightness": 1.0,
    },
    2: {
        "initial_aggression": 0.45,
        "stealth_acceleration": 0.018,
        "noise_spike_probability": 0.20,
        "price_volatility_multiplier": 1.3,
        "slippage_threshold_tightness": 0.75,
    },
    3: {
        "initial_aggression": 0.55,
        "stealth_acceleration": 0.024,
        "noise_spike_probability": 0.28,
        "price_volatility_multiplier": 1.5,
        "slippage_threshold_tightness": 0.55,
    },
}

PHASE_THRESHOLDS: List[float] = [0.55, 0.70, 0.82]


@dataclass
class EpisodeFailure:
    episode_id: int
    task_name: str
    step_num: int
    bot_type: str
    mev_loss: float
    action_taken: str
    agent_used_private_rpc: bool
    agent_used_split_swap: bool
    stealth_level: float
    aggression_level: float


@dataclass
class EpisodeResult:
    episode_id: int
    task_name: str
    final_score: float
    profit_score: float
    mev_avoidance_score: float
    efficiency_score: float
    lp_yield_score: float
    steps_run: int
    total_mev_loss: float
    bot_type_performance: Dict[str, float]  # bot_type -> detection_rate in [0,1]
    bot_type_exposures: Dict[str, int]      # bot_type -> number of attack attempts
    failures: List[EpisodeFailure] = field(default_factory=list)
    config_used: Dict[str, float] = field(default_factory=dict)


class EnvController:
    """Between-episode meta-controller. Never mutates a running env."""

    def __init__(
        self,
        task_name: str = "easy",
        max_failure_history: int = 200,
        max_episode_history: int = 100,
        weight_lr: float = 0.3,
        min_weight: float = 0.05,
    ) -> None:
        self.task_name = task_name
        self.weight_lr = weight_lr
        self.min_weight = min_weight

        self.bot_weights: Dict[str, float] = {b: 1.0 for b in BOT_TYPES}
        self._normalize_weights()

        self.curriculum_phase: int = 0
        self.episodes_in_phase: int = 0

        self.history: List[EpisodeResult] = []
        self._history_cap = max_episode_history
        self.failure_buffer: Deque[EpisodeFailure] = deque(maxlen=max_failure_history)

        self._snapshots: List[Dict] = []

    # ── Public API ───────────────────────────────────────────────

    def ingest(self, result: EpisodeResult) -> None:
        """Called after every episode. Updates weights + curriculum."""
        self.history.append(result)
        if len(self.history) > self._history_cap:
            self.history = self.history[-self._history_cap :]

        for f in result.failures:
            self.failure_buffer.append(f)

        self._update_bot_weights(result)
        self._advance_curriculum(result)
        self._snapshots.append(self._make_snapshot(result))

    def next_config(self) -> Dict:
        """Config dict for the next episode. Env reads this at reset()."""
        phase_cfg = PHASE_CONFIGS[self.curriculum_phase]
        return {
            "bot_weights": dict(self.bot_weights),
            "initial_aggression": phase_cfg["initial_aggression"],
            "stealth_acceleration": phase_cfg["stealth_acceleration"],
            "noise_spike_probability": phase_cfg["noise_spike_probability"],
            "price_volatility_multiplier": phase_cfg["price_volatility_multiplier"],
            "slippage_threshold_tightness": phase_cfg["slippage_threshold_tightness"],
            "curriculum_phase": self.curriculum_phase,
        }

    def evolution_history(self) -> List[Dict]:
        return list(self._snapshots)

    def recent_failure_summary(self, last_n: int = 20) -> Dict[str, int]:
        recent = list(self.failure_buffer)[-last_n:]
        counts: Dict[str, int] = {b: 0 for b in BOT_TYPES}
        for f in recent:
            counts[f.bot_type] = counts.get(f.bot_type, 0) + 1
        return counts

    # ── Internals ────────────────────────────────────────────────

    def _update_bot_weights(self, result: EpisodeResult) -> None:
        """Bot types with lower detection_rate (more misses) get higher weight."""
        for bot_type, detection_rate in result.bot_type_performance.items():
            if bot_type not in self.bot_weights:
                continue
            miss_rate = max(0.0, 1.0 - detection_rate)
            target = 1.0 + miss_rate * 2.0
            prev = self.bot_weights[bot_type]
            self.bot_weights[bot_type] = (1.0 - self.weight_lr) * prev + self.weight_lr * target
            self.bot_weights[bot_type] = max(self.min_weight, self.bot_weights[bot_type])

        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.bot_weights.values())
        if total <= 0:
            for k in self.bot_weights:
                self.bot_weights[k] = 1.0 / len(self.bot_weights)
            return
        scale = len(self.bot_weights) / total
        for k in self.bot_weights:
            self.bot_weights[k] *= scale
        gross = sum(self.bot_weights.values())
        for k in self.bot_weights:
            self.bot_weights[k] /= gross

    def _advance_curriculum(self, result: EpisodeResult) -> None:
        self.episodes_in_phase += 1
        if self.curriculum_phase >= 3:
            return
        threshold = PHASE_THRESHOLDS[self.curriculum_phase]
        recent = [r.final_score for r in self.history[-5:]]
        if len(recent) >= 3 and mean(recent[-3:]) >= threshold:
            self.curriculum_phase += 1
            self.episodes_in_phase = 0

    def _make_snapshot(self, result: EpisodeResult) -> Dict:
        dominant = max(self.bot_weights, key=self.bot_weights.get)
        return {
            "episode_id": result.episode_id,
            "task": result.task_name,
            "final_score": result.final_score,
            "mev_avoidance_score": result.mev_avoidance_score,
            "profit_score": result.profit_score,
            "curriculum_phase": self.curriculum_phase,
            "bot_weights": dict(self.bot_weights),
            "dominant_bot_type": dominant,
            "num_failures": len(result.failures),
        }


# ── Helpers used by the environment to emit EpisodeResult ──────────

def build_episode_result(
    episode_id: int,
    task_name: str,
    grade: Dict,
    bot_exposures: Dict[str, int],
    bot_successful_attacks: Dict[str, int],
    failures: List[EpisodeFailure],
    config_used: Optional[Dict] = None,
) -> EpisodeResult:
    """Convert raw grade dict + per-bot telemetry into an EpisodeResult."""
    bot_performance: Dict[str, float] = {}
    for bot in BOT_TYPES:
        exposures = bot_exposures.get(bot, 0)
        if exposures <= 0:
            bot_performance[bot] = 1.0  # no attempts -> perfect by default
        else:
            successes = bot_successful_attacks.get(bot, 0)
            bot_performance[bot] = max(0.0, 1.0 - successes / exposures)

    return EpisodeResult(
        episode_id=episode_id,
        task_name=task_name,
        final_score=float(grade.get("final_score", 0.0)),
        profit_score=float(grade.get("profit_score", 0.0)),
        mev_avoidance_score=float(grade.get("mev_avoidance_score", 0.0)),
        efficiency_score=float(grade.get("efficiency_score", 0.0)),
        lp_yield_score=float(grade.get("lp_yield_score", 0.0)),
        steps_run=int(grade.get("steps_run", 0)),
        total_mev_loss=float(grade.get("total_mev_loss", 0.0)),
        bot_type_performance=bot_performance,
        bot_type_exposures=dict(bot_exposures),
        failures=list(failures),
        config_used=dict(config_used or {}),
    )
