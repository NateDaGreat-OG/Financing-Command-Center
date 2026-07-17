"""Risk-aware position sizing and execution logic.

Wraps RiskManager with cycle-sensitive stop/target computation, Kelly-criterion
sizing, and signal-level risk-constraint filtering.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from core.risk_manager import RiskManager


class RiskAdapter:
    """Extends RiskManager with cycle-aware and signal-level risk controls."""

    def __init__(
        self,
        risk_manager: Optional[RiskManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        self.risk_manager = risk_manager or RiskManager(config)
        self.max_risk_per_trade = float(config.get("MAX_RISK_PER_TRADE", 0.01))
        self.max_concurrent = int(config.get("MAX_CONCURRENT_POSITIONS", 5))
        self.min_rr_ratio = float(config.get("MIN_RR_RATIO", 1.5))

    # ------------------------------------------------------------------
    # Per-signal risk sizing
    # ------------------------------------------------------------------

    def size_signal(
        self,
        signal: Dict[str, Any],
        capital: float,
        cycle_state: Optional[Dict[str, Any]] = None,
        volatility_scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Return a copy of *signal* with a risk-adjusted ``size`` field.

        Risk percent is reduced in high-volatility and risk-off environments.
        """
        result = signal.copy()
        cycle_state = cycle_state or {}

        entry = float(signal.get("entry_price", 0.0))
        stop = float(signal.get("stop_loss", entry))
        if entry <= 0:
            return result

        risk_pct = self.max_risk_per_trade
        if cycle_state.get("volatility") == "high":
            risk_pct *= 0.75
        if cycle_state.get("macro") == "risk_off":
            risk_pct *= 0.80

        risk_amount = capital * risk_pct
        risk_per_share = abs(entry - stop) * max(volatility_scale, 0.01)
        if risk_per_share <= 0:
            return result

        size = math.floor(risk_amount / risk_per_share)
        result["size"] = max(size, 1)
        result["risk_pct_used"] = round(risk_pct, 4)
        return result

    # ------------------------------------------------------------------
    # Constraint filtering
    # ------------------------------------------------------------------

    def apply_risk_constraints(
        self,
        signals: List[Dict[str, Any]],
        current_positions: int,
        capital: float,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Filter out signals that breach position limits or minimum R:R.

        At most ``max_concurrent - current_positions`` new signals are returned.
        """
        available_slots = self.max_concurrent - current_positions
        if available_slots <= 0:
            return []

        filtered: List[Dict[str, Any]] = []
        for signal in signals:
            if len(filtered) >= available_slots:
                break
            if self._passes_rr_check(signal):
                filtered.append(signal)

        return filtered

    # ------------------------------------------------------------------
    # Dynamic stop / target
    # ------------------------------------------------------------------

    def compute_stop_and_target(
        self,
        signal: Dict[str, Any],
        atr: float,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a copy of *signal* with dynamically computed stop_loss and target.

        ATR multiples are tightened in high-volatility or risk-off environments.
        The original stop/target fields are only overwritten when ATR is available.
        """
        cycle_state = cycle_state or {}
        result = signal.copy()
        entry = float(signal.get("entry_price", 0.0))
        side = signal.get("side", "long")

        if entry <= 0 or atr <= 0:
            return result

        stop_mult = 1.5
        target_mult = 3.0
        if cycle_state.get("volatility") == "high":
            stop_mult = 1.2
            target_mult = 2.5
        if cycle_state.get("macro") == "risk_off":
            stop_mult = 1.0
            target_mult = 2.0

        if side == "long":
            result["stop_loss"] = round(entry - atr * stop_mult, 4)
            result["target"] = round(entry + atr * target_mult, 4)
        else:
            result["stop_loss"] = round(entry + atr * stop_mult, 4)
            result["target"] = round(entry - atr * target_mult, 4)

        return result

    # ------------------------------------------------------------------
    # Kelly sizing
    # ------------------------------------------------------------------

    def kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """Return a half-Kelly fraction capped within a safe range.

        Returns the raw ``max_risk_per_trade`` when inputs are degenerate.
        """
        if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return self.max_risk_per_trade
        b = avg_win / avg_loss
        kelly = (b * win_rate - (1.0 - win_rate)) / b
        return float(np.clip(kelly * 0.5, 0.001, self.max_risk_per_trade * 3))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _passes_rr_check(self, signal: Dict[str, Any]) -> bool:
        entry = float(signal.get("entry_price", 0.0))
        stop = float(signal.get("stop_loss", entry))
        target = float(signal.get("target", entry))
        risk = abs(entry - stop)
        if entry <= 0 or risk <= 0:
            return True  # cannot assess — let through
        rr = abs(target - entry) / risk
        return rr >= self.min_rr_ratio
