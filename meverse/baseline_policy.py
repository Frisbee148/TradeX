"""Baseline policy for the surveillance benchmark."""

from __future__ import annotations

from .models import SurveillanceObservation


def choose_surveillance_action(observation: SurveillanceObservation) -> str:
    """Simple threshold policy aligned with the benchmark instructions."""

    if observation.pattern_indicator >= 0.72 and observation.recent_slippage_impact >= 0.055:
        return "BLOCK"
    if observation.manipulation_score >= 0.78:
        return "BLOCK"
    if observation.burst_indicator >= 0.70 or observation.trade_frequency >= 7.5:
        return "FLAG"
    if observation.suspiciousness_score >= 0.52:
        return "MONITOR"
    return "ALLOW"
