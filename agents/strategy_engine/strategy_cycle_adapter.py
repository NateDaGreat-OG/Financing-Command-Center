"""Cycle-aware parameter adjustment engine.

Maps the market cycle state produced by CycleAnalyzer to per-strategy parameter
overrides, confidence multipliers, and favored-strategy rankings.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Per-strategy parameter overrides indexed by cycle regime
# ---------------------------------------------------------------------------

_CYCLE_PARAM_OVERRIDES: Dict[str, Dict[str, Dict[str, Any]]] = {
    "trend_continuation": {
        "bull": {
            "ema_short": 9, "ema_long": 20,
            "rsi_low": 35, "rsi_high": 75, "atr_mult": 2.5, "size": 60,
        },
        "bear": {
            "ema_short": 5, "ema_long": 20,
            "rsi_low": 30, "rsi_high": 60, "atr_mult": 1.5, "size": 30,
        },
        "sideways": {
            "ema_short": 9, "ema_long": 20,
            "rsi_low": 40, "rsi_high": 60, "atr_mult": 1.8, "size": 40,
        },
    },
    "volatility_compression": {
        "high": {
            "bb_mult": 2.0, "squeeze_threshold": 0.025,
            "atr_mult": 2.0, "vol_expansion_ratio": 1.3, "size": 50,
        },
        "low": {
            "bb_mult": 2.5, "squeeze_threshold": 0.040,
            "atr_mult": 1.5, "vol_expansion_ratio": 1.1, "size": 30,
        },
    },
    "scalping_premarket": {
        "open_drive": {
            "gap_min": 0.008, "vol_ratio_min": 1.5,
            "stop_pct": 0.002, "target_pct": 0.005, "size": 120,
        },
        "power_hour": {
            "gap_min": 0.012, "vol_ratio_min": 2.0,
            "stop_pct": 0.003, "target_pct": 0.006, "size": 100,
        },
        "chop": {
            "gap_min": 0.015, "vol_ratio_min": 2.5,
            "stop_pct": 0.002, "target_pct": 0.004, "size": 60,
        },
        "fade": {
            "gap_min": 0.010, "vol_ratio_min": 1.8,
            "stop_pct": 0.002, "target_pct": 0.004, "size": 70,
        },
    },
    "liquidity_window": {
        "expanding": {"vwap_drift_pct": 0.003, "size_mult": 1.2, "size": 60},
        "contracting": {"vwap_drift_pct": 0.005, "size_mult": 0.8, "size": 35},
    },
    "magic_formula": {
        "bull": {"min_score": 0.45, "stop_pct": 0.08, "target_pct": 0.20, "size": 25},
        "bear": {"min_score": 0.65, "stop_pct": 0.06, "target_pct": 0.12, "size": 15},
        "sideways": {"min_score": 0.55, "stop_pct": 0.07, "target_pct": 0.15, "size": 20},
    },
}

# ---------------------------------------------------------------------------
# Confidence multipliers per cycle dimension and regime
# ---------------------------------------------------------------------------

_CONFIDENCE_TABLE: Dict[str, Dict[str, float]] = {
    "trend":     {"bull": 1.15, "bear": 0.90, "sideways": 0.95},
    "volatility": {"high": 1.05, "low": 0.95},
    "liquidity": {"expanding": 1.10, "contracting": 0.85},
    "macro":     {"risk_on": 1.05, "risk_off": 0.80},
    "intraday":  {"open_drive": 1.15, "power_hour": 1.10, "chop": 0.80, "fade": 0.90},
}

# Map each strategy to the cycle dimension that drives its parameter overrides
_STRATEGY_CYCLE_KEY: Dict[str, str] = {
    "trend_continuation":    "trend",
    "volatility_compression": "volatility",
    "scalping_premarket":    "intraday",
    "liquidity_window":      "liquidity",
    "magic_formula":         "trend",
}

# Strategies favored by compound regime key "<trend>_<volatility>"
_FAVORED_STRATEGIES: Dict[str, List[str]] = {
    "bull_high":    ["trend_continuation", "magic_formula"],
    "bull_low":     ["trend_continuation", "magic_formula", "liquidity_window"],
    "bear_high":    ["scalping_premarket", "volatility_compression"],
    "bear_low":     ["magic_formula"],
    "sideways_high": ["volatility_compression", "scalping_premarket"],
    "sideways_low":  ["liquidity_window"],
}


class CycleAdapter:
    """Adjusts strategy parameters and signal confidence based on market cycle state."""

    def adapt_params(
        self,
        strategy_name: str,
        base_params: Optional[Dict[str, Any]] = None,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a merged parameter dict combining *base_params* with cycle overrides.

        The cycle dimension used to look up overrides is determined by
        ``_STRATEGY_CYCLE_KEY``; falls back to ``trend`` when unknown.
        """
        params = dict(base_params or {})
        cycle_state = cycle_state or {}

        overrides_for_strategy = _CYCLE_PARAM_OVERRIDES.get(strategy_name, {})
        if not overrides_for_strategy:
            return params

        cycle_key = _STRATEGY_CYCLE_KEY.get(strategy_name, "trend")
        regime = cycle_state.get(cycle_key, "sideways")
        cycle_overrides = overrides_for_strategy.get(regime, {})
        params.update(cycle_overrides)
        return params

    def adapt_signal_confidence(
        self,
        signal: Dict[str, Any],
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Return a confidence multiplier in [0.10, 1.50] for a signal given the cycle state.

        Each active cycle dimension contributes a multiplicative adjustment from
        ``_CONFIDENCE_TABLE``.
        """
        cycle_state = cycle_state or {}
        multiplier = 1.0
        for dimension, table in _CONFIDENCE_TABLE.items():
            regime = cycle_state.get(dimension)
            if regime:
                multiplier *= table.get(regime, 1.0)
        return float(np.clip(multiplier, 0.10, 1.50))

    def get_favored_strategies(
        self, cycle_state: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Return strategy names best suited to the current cycle regime.

        The lookup key is ``"<trend>_<volatility>"``.  Falls back to all known
        strategies when the combination is not found.
        """
        cycle_state = cycle_state or {}
        trend = cycle_state.get("trend", "sideways")
        volatility = cycle_state.get("volatility", "low")
        key = f"{trend}_{volatility}"
        return list(_FAVORED_STRATEGIES.get(key, list(_CYCLE_PARAM_OVERRIDES.keys())))

    def scale_size_for_cycle(
        self,
        size: float,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Scale a position size down in adverse macro/liquidity/volatility conditions."""
        cycle_state = cycle_state or {}
        mult = 1.0
        if cycle_state.get("macro") == "risk_off":
            mult *= 0.75
        if cycle_state.get("liquidity") == "contracting":
            mult *= 0.85
        if cycle_state.get("volatility") == "high":
            mult *= 0.90
        return float(max(size * mult, 1.0))
