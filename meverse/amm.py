"""Lightweight AMM state machine with constant-product dynamics."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AMMState:
    """Mutable state for a simulated constant-product AMM pool."""

    reserve_x: float = 1000.0
    reserve_y: float = 100000.0
    bot_confidence: float = 0.3
    volatility: float = 0.05
    health_index: float = 0.95
    step: int = 0

    @property
    def k(self) -> float:
        return self.reserve_x * self.reserve_y

    @property
    def price(self) -> float:
        return self.reserve_y / self.reserve_x

    @property
    def liquidity(self) -> float:
        return 2.0 * math.sqrt(self.k)


def apply_trade(state: AMMState, dx: float) -> float:
    """Execute a trade of dx on reserve_x, return price impact."""
    price_before = state.price
    k = state.k
    state.reserve_x += dx
    state.reserve_y = k / state.reserve_x
    price_after = state.price
    return abs(price_after - price_before) / max(price_before, 1e-9)


def apply_action_effects(state: AMMState, action: str, was_suspicious: bool) -> None:
    """Update AMM state based on the surveillance action taken."""
    if was_suspicious:
        if action == "BLOCK":
            state.bot_confidence = max(0.0, state.bot_confidence - 0.25)
            state.volatility = max(0.01, state.volatility * 0.85)
        elif action == "FLAG":
            state.bot_confidence = max(0.0, state.bot_confidence - 0.10)
            state.volatility = max(0.01, state.volatility * 0.92)
        elif action == "MONITOR":
            state.bot_confidence = min(1.0, state.bot_confidence + 0.03)
        else:  # ALLOW
            state.bot_confidence = min(1.0, state.bot_confidence + 0.15)
            state.volatility = min(0.5, state.volatility * 1.12)
    else:
        if action == "ALLOW":
            state.health_index = min(1.0, state.health_index + 0.03)
        elif action == "MONITOR":
            state.health_index = max(0.0, state.health_index - 0.01)
        elif action == "FLAG":
            state.health_index = max(0.0, state.health_index - 0.04)
            state.volatility = min(0.5, state.volatility * 1.05)
        else:  # BLOCK on normal
            state.health_index = max(0.0, state.health_index - 0.08)
            state.volatility = min(0.5, state.volatility * 1.10)

    state.step += 1


def generate_step_from_state(
    state: AMMState,
    rng: random.Random,
    task_profile: str,
) -> dict:
    """Generate a single observation step from the current AMM state.

    Returns a dict with all fields needed to build a ScenarioStep.
    The 'label' is emergent from bot_confidence + random draws.
    """
    # Determine if this step is suspicious based on bot_confidence + randomness
    suspicion_roll = rng.random()
    bot_active = suspicion_roll < state.bot_confidence

    # Task profile controls bot behavior style
    if task_profile == "burst_detection":
        burst_bias, pattern_bias = 0.8, 0.2
    elif task_profile == "pattern_manipulation_detection":
        burst_bias, pattern_bias = 0.2, 0.8
    else:  # full_market_surveillance
        burst_bias, pattern_bias = 0.5, 0.5

    if bot_active:
        # Simulate bot trades hitting the AMM
        num_bot_trades = rng.randint(3, 6)
        for _ in range(num_bot_trades):
            dx = rng.uniform(5.0, 25.0) * (1.0 + state.bot_confidence)
            if rng.random() < 0.5:
                dx = -dx
            apply_trade(state, dx)

        # Stealth factor: bots that have been blocked learn to hide
        # More steps survived = smarter bots with lower visible signals
        stealth = min(0.6, state.step * 0.012)
        is_stealthy = rng.random() < (0.25 + stealth)

        intensity = 0.4 + 0.6 * state.bot_confidence
        if is_stealthy:
            # Stealthy bot: deliberately suppressed indicators
            burst = min(1.0, rng.uniform(0.18, 0.48) * (0.5 + burst_bias))
            pattern = min(1.0, rng.uniform(0.15, 0.45) * (0.5 + pattern_bias))
            suspicious = min(1.0, rng.uniform(0.20, 0.50) + state.bot_confidence * 0.15)
            manipulation = min(1.0, rng.uniform(0.10, 0.40))
            severity = min(1.0, 0.35 + 0.4 * state.bot_confidence * rng.uniform(0.6, 1.0))
        else:
            # Overt bot: clear signals
            burst = min(1.0, rng.uniform(0.55, 0.95) * intensity * (0.5 + burst_bias))
            pattern = min(1.0, rng.uniform(0.50, 0.95) * intensity * (0.5 + pattern_bias))
            suspicious = min(1.0, 0.3 + 0.7 * state.bot_confidence + rng.uniform(-0.05, 0.05))
            manipulation = min(1.0, pattern * rng.uniform(0.85, 1.0))
            severity = min(1.0, 0.3 + 0.7 * state.bot_confidence * rng.uniform(0.8, 1.0))

        health = max(0.0, min(1.0, state.health_index - 0.2 * state.bot_confidence))
        label = "suspicious"

        if is_stealthy:
            trades = [round(rng.uniform(10, 22) * (1.0 + 0.2 * state.bot_confidence), 4) for _ in range(5)]
            gaps = [round(max(0.3, rng.uniform(1.0, 4.0)), 4) for _ in range(5)]
            impacts = [round(min(0.1, rng.uniform(0.008, 0.025)), 4) for _ in range(5)]
        else:
            trades = [round(rng.uniform(15, 32) * (1.0 + 0.5 * state.bot_confidence), 4) for _ in range(5)]
            gaps = [round(max(0.1, rng.uniform(0.2, 1.5) / (1.0 + state.bot_confidence)), 4) for _ in range(5)]
            impacts = [round(min(0.1, rng.uniform(0.015, 0.05) * (1.0 + state.bot_confidence)), 4) for _ in range(5)]
    else:
        # Normal organic trading
        num_trades = rng.randint(1, 3)
        for _ in range(num_trades):
            dx = rng.uniform(1.0, 8.0)
            if rng.random() < 0.5:
                dx = -dx
            apply_trade(state, dx)

        # Noisy normal: real markets have spikes that look suspicious
        noise_spike = rng.random() < 0.15
        if noise_spike:
            burst = min(1.0, max(0.0, rng.uniform(0.25, 0.55) + state.volatility * 0.4))
            pattern = min(1.0, max(0.0, rng.uniform(0.15, 0.40)))
            suspicious = min(1.0, max(0.0, rng.uniform(0.25, 0.50) + state.volatility * 0.3))
            manipulation = min(1.0, max(0.0, rng.uniform(0.10, 0.30)))
        else:
            burst = min(1.0, max(0.0, rng.uniform(0.02, 0.30) + state.volatility * 0.3))
            pattern = min(1.0, max(0.0, rng.uniform(0.02, 0.25)))
            suspicious = min(1.0, max(0.0, rng.uniform(0.05, 0.35) + state.volatility * 0.2))
            manipulation = min(1.0, max(0.0, rng.uniform(0.02, 0.20)))

        severity = rng.uniform(0.02, 0.18)
        health = min(1.0, state.health_index + rng.uniform(-0.02, 0.02))
        label = "normal"

        trades = [round(rng.uniform(8, 18), 4) for _ in range(5)]
        gaps = [round(rng.uniform(3.0, 9.0), 4) for _ in range(5)]
        impacts = [round(rng.uniform(0.002, 0.010), 4) for _ in range(5)]

    note = _generate_note(label, burst, pattern, state.bot_confidence)

    return {
        "price": round(state.price, 4),
        "liquidity": round(state.liquidity, 4),
        "trades": trades,
        "gaps": gaps,
        "impacts": impacts,
        "burst": round(burst, 4),
        "pattern": round(pattern, 4),
        "suspicious": round(suspicious, 4),
        "manipulation": round(manipulation, 4),
        "label": label,
        "severity": round(severity, 4),
        "health": round(health, 4),
        "note": note,
    }


def _generate_note(label: str, burst: float, pattern: float, bot_conf: float) -> str:
    if label == "normal":
        if burst > 0.35:
            return "Noisy but organic spike — false positive risk."
        if burst > 0.20:
            return "Slightly elevated but organic activity."
        return "Routine market flow."
    if burst > 0.80:
        return f"Aggressive burst attack (bot confidence {bot_conf:.0%})."
    if pattern > 0.75:
        return f"Coordinated pattern manipulation (bot confidence {bot_conf:.0%})."
    if burst < 0.35 and pattern < 0.35:
        return f"Stealthy bot activity — low signal footprint (bot confidence {bot_conf:.0%})."
    return f"Suspicious activity detected (bot confidence {bot_conf:.0%})."


# Task configuration: initial bot_confidence and episode length per difficulty
TASK_CONFIGS = {
    "burst_detection": {
        "initial_bot_confidence": 0.25,
        "num_steps": 50,
        "profile": "burst_detection",
    },
    "pattern_manipulation_detection": {
        "initial_bot_confidence": 0.35,
        "num_steps": 50,
        "profile": "pattern_manipulation_detection",
    },
    "full_market_surveillance": {
        "initial_bot_confidence": 0.30,
        "num_steps": 60,
        "profile": "full_market_surveillance",
    },
}
