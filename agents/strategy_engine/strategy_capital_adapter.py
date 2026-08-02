"""Capital-aware position sizing and allocation.

Applies CapitalManager allocations to individual signals, enforces portfolio
exposure limits, and supports Kelly-fraction sizing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from core.capital_manager import CapitalManager


class CapitalAdapter:
    """Applies CapitalManager allocations to individual signals."""

    def __init__(
        self,
        capital_manager: Optional[CapitalManager] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        config = config or {}
        self.capital_manager = capital_manager or CapitalManager(config)
        self.max_position_size_pct = float(config.get("MAX_POSITION_SIZE_PCT", 0.20))
        self.min_allocation_pct = float(config.get("MIN_ALLOCATION_PCT", 0.01))

    # ------------------------------------------------------------------
    # Signal sizing from allocation map
    # ------------------------------------------------------------------

    def size_from_allocation(
        self,
        signal: Dict[str, Any],
        allocation_map: Dict[str, Dict[str, Dict[str, float]]],
        strategy_name: str,
        symbol: str,
    ) -> Dict[str, Any]:
        """Return a copy of *signal* with ``size`` derived from the allocation map.

        Also attaches ``allocated_capital`` and ``risk_budget`` fields for
        downstream audit logging.
        """
        result = signal.copy()
        strategy_alloc = allocation_map.get(strategy_name, {})
        symbol_alloc = strategy_alloc.get(symbol, {})
        allocated_capital = float(symbol_alloc.get("allocated_capital", 0.0))
        max_pos = float(symbol_alloc.get("max_position_size", 0.0))

        entry = float(signal.get("entry_price", 0.0))
        if entry <= 0 or allocated_capital <= 0:
            return result

        size = int(allocated_capital / entry)
        if max_pos > 0:
            size = min(size, int(max_pos / entry))

        result["size"] = max(size, 1)
        result["allocated_capital"] = allocated_capital
        result["risk_budget"] = float(symbol_alloc.get("risk_budget", 0.0))
        return result

    # ------------------------------------------------------------------
    # Exposure cap enforcement
    # ------------------------------------------------------------------

    def enforce_exposure_limits(
        self,
        signals: List[Dict[str, Any]],
        total_capital: float,
        current_exposure: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """Cap total notional exposure at ``max_position_size_pct`` of capital.

        Signals are processed in order.  When the remaining budget is
        insufficient for the full size, the size is trimmed to fit.
        When budget is exhausted the remaining signals are dropped.
        """
        max_exposure = total_capital * self.max_position_size_pct
        remaining = max(max_exposure - current_exposure, 0.0)

        result: List[Dict[str, Any]] = []
        for signal in signals:
            entry = float(signal.get("entry_price", 0.0))
            size = int(signal.get("size", 1))
            notional = entry * size
            if entry <= 0 or remaining <= 0:
                continue
            if notional > remaining:
                trimmed = max(int(remaining / entry), 1)
                result.append({**signal, "size": trimmed})
                remaining = 0.0
            else:
                result.append(signal)
                remaining -= notional

        return result

    # ------------------------------------------------------------------
    # Kelly-fraction sizing
    # ------------------------------------------------------------------

    def compute_kelly_size(
        self,
        signal: Dict[str, Any],
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
    ) -> Dict[str, Any]:
        """Return a copy of *signal* with ``size`` computed using a half-Kelly fraction.

        The Kelly fraction is capped at 25 % of capital to limit over-sizing.
        Falls back to the original ``size`` when inputs are degenerate.
        """
        result = signal.copy()
        entry = float(signal.get("entry_price", 0.0))
        if entry <= 0 or avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
            return result

        b = avg_win / avg_loss
        kelly = (b * win_rate - (1.0 - win_rate)) / b
        half_kelly = float(np.clip(kelly * 0.5, 0.001, 0.25))
        size = max(int(capital * half_kelly / entry), 1)
        result["size"] = size
        result["kelly_fraction"] = round(half_kelly, 4)
        return result

    # ------------------------------------------------------------------
    # Allocation map builder (delegates to CapitalManager)
    # ------------------------------------------------------------------

    def build_allocation_map(
        self,
        strategies: List[str],
        symbols: List[str],
        strategy_metrics: Dict[str, Any],
        rl_metrics: Optional[Dict[str, Any]] = None,
        cycle_state: Optional[Dict[str, Any]] = None,
        portfolio_capital: Optional[float] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Delegate capital allocation to the underlying CapitalManager."""
        return self.capital_manager.allocate(
            strategies=strategies,
            symbols=symbols,
            strategy_metrics=strategy_metrics,
            rl_metrics=rl_metrics,
            cycle_state=cycle_state,
            portfolio_capital=portfolio_capital,
        )
