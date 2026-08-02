"""Execution intelligence layer.

Provides VWAP/TWAP order scheduling, slippage modelling, partial-fill
simulation, and RL-timed execution windows for live and simulated trading.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_MINUTES_PER_DAY = 390  # 9:30 – 16:00 US/Eastern


class ExecutionIntelligence:
    """Smart order routing using VWAP/TWAP scheduling and slippage modelling.

    Parameters
    ----------
    config:
        Application configuration dict.  Recognised keys:

        ``SLIPPAGE_BPS`` (float, default 5) – base slippage in basis points.
        ``COMMISSION_PCT`` (float, default 0.001) – fractional commission rate.
        ``TWAP_SLICES`` (int, default 5) – default number of TWAP child orders.
        ``PARTIAL_FILL_MIN_PCT`` (float, default 0.50) – minimum fill fraction.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.slippage_bps = float(cfg.get("SLIPPAGE_BPS", 5))
        self.commission_pct = float(cfg.get("COMMISSION_PCT", 0.001))
        self.twap_slices = int(cfg.get("TWAP_SLICES", 5))
        self.partial_fill_min_pct = float(cfg.get("PARTIAL_FILL_MIN_PCT", 0.50))

    # ------------------------------------------------------------------
    # Slippage modelling
    # ------------------------------------------------------------------

    def estimate_slippage(
        self,
        price: float,
        size: float,
        avg_volume: float,
        side: str = "long",
        volatility: float = 0.0,
    ) -> float:
        """Return the estimated slippage cost in dollars.

        Slippage scales with participation rate (size / avg_volume) and
        realised volatility.

        Parameters
        ----------
        price:
            Execution price per share.
        size:
            Order size in shares.
        avg_volume:
            Average daily volume for the symbol.
        side:
            ``"long"`` or ``"short"``.
        volatility:
            Normalised volatility factor (e.g. ATR / price).
        """
        if price <= 0 or size <= 0:
            return 0.0
        base_slip = self.slippage_bps / 10_000.0
        participation = min(size / max(avg_volume, 1.0), 1.0)
        vol_factor = 1.0 + volatility * 2.0
        slip_pct = base_slip * (1.0 + participation) * vol_factor
        cost = price * size * slip_pct
        return round(abs(cost), 4)

    def apply_slippage(
        self,
        signal: Dict[str, Any],
        avg_volume: float = 1_000_000.0,
        volatility: float = 0.0,
    ) -> Dict[str, Any]:
        """Return a copy of *signal* with slippage factored into ``entry_price``."""
        sig = signal.copy()
        price = float(sig.get("entry_price", 0.0))
        size = float(sig.get("size", 0.0))
        side = sig.get("side", "long")

        slip = self.estimate_slippage(price, size, avg_volume, side, volatility)
        slip_per_share = slip / max(size, 1.0)

        if side == "long":
            sig["entry_price"] = round(price + slip_per_share, 6)
        else:
            sig["entry_price"] = round(price - slip_per_share, 6)

        sig["slippage_cost"] = slip
        return sig

    # ------------------------------------------------------------------
    # VWAP execution
    # ------------------------------------------------------------------

    def vwap_schedule(
        self,
        symbol: str,
        total_size: float,
        intraday_data: pd.DataFrame,
        start_bar: int = 0,
        end_bar: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Decompose an order into VWAP-weighted child orders.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        total_size:
            Total shares to execute.
        intraday_data:
            Intraday OHLCV DataFrame (bars as rows).
        start_bar / end_bar:
            Bar index window to schedule execution over.

        Returns
        -------
        List of child order dicts with keys ``bar``, ``size``, ``price``.
        """
        if intraday_data.empty or total_size <= 0:
            return []

        df = intraday_data.copy()
        end_bar = end_bar if end_bar is not None else len(df)
        window = df.iloc[start_bar:end_bar]

        if window.empty or "volume" not in window.columns:
            return [{"bar": start_bar, "size": total_size, "price": float(df["close"].iloc[-1])}]

        vol = window["volume"].clip(lower=0)
        total_vol = vol.sum()
        if total_vol <= 0:
            weights = np.ones(len(window)) / len(window)
        else:
            weights = (vol / total_vol).values

        orders = []
        for i, (idx, row) in enumerate(window.iterrows()):
            child_size = math.floor(total_size * weights[i])
            if child_size <= 0:
                continue
            orders.append({
                "bar": start_bar + i,
                "timestamp": str(idx),
                "symbol": symbol,
                "size": child_size,
                "price": float(row.get("close", row.get("c", 0.0))),
            })

        # Remainder shares go into last child order
        filled = sum(o["size"] for o in orders)
        remainder = int(total_size - filled)
        if remainder > 0 and orders:
            orders[-1]["size"] += remainder

        return orders

    # ------------------------------------------------------------------
    # TWAP execution
    # ------------------------------------------------------------------

    def twap_schedule(
        self,
        symbol: str,
        total_size: float,
        price: float,
        start_time: Optional[datetime] = None,
        slices: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Decompose an order into equal time-weighted child orders.

        Parameters
        ----------
        symbol:
            Ticker symbol.
        total_size:
            Total shares to execute.
        price:
            Reference price for each slice.
        start_time:
            Execution window start (defaults to now).
        slices:
            Number of child orders (defaults to ``self.twap_slices``).
        """
        slices = max(slices or self.twap_slices, 1)
        base_size = int(total_size // slices)
        remainder = int(total_size - base_size * slices)
        start_time = start_time or datetime.utcnow()

        orders = []
        for i in range(slices):
            child_size = base_size + (remainder if i == slices - 1 else 0)
            if child_size <= 0:
                continue
            orders.append({
                "slice": i + 1,
                "symbol": symbol,
                "size": child_size,
                "price": price,
                "scheduled_time": start_time.isoformat(),
            })
        return orders

    # ------------------------------------------------------------------
    # Partial fill simulation
    # ------------------------------------------------------------------

    def simulate_partial_fill(
        self,
        order: Dict[str, Any],
        available_liquidity: float,
        rng_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Simulate a (possibly partial) fill of an order.

        Parameters
        ----------
        order:
            Order dict with at least ``size`` and ``price`` keys.
        available_liquidity:
            Available shares in the book at the order price level.
        rng_seed:
            Optional RNG seed for reproducible simulation.

        Returns
        -------
        Order dict enriched with ``filled_size``, ``unfilled_size``, and
        ``fill_status`` (``"full"`` / ``"partial"`` / ``"none"``).
        """
        rng = np.random.default_rng(rng_seed)
        requested = float(order.get("size", 0))
        if requested <= 0 or available_liquidity <= 0:
            return {**order, "filled_size": 0.0, "unfilled_size": requested, "fill_status": "none"}

        # Random fill fraction biased toward the liquidity ratio
        max_fill = min(requested, available_liquidity)
        fill_pct = float(rng.uniform(self.partial_fill_min_pct, 1.0))
        filled = math.floor(max_fill * fill_pct)
        unfilled = requested - filled
        status = "full" if unfilled == 0 else ("partial" if filled > 0 else "none")

        return {
            **order,
            "filled_size": filled,
            "unfilled_size": unfilled,
            "fill_status": status,
        }

    # ------------------------------------------------------------------
    # RL-timed execution windows
    # ------------------------------------------------------------------

    def rl_execution_window(
        self,
        cycle_state: Optional[Dict[str, Any]] = None,
        rl_action: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Recommend an execution timing window based on cycle state and RL action.

        Parameters
        ----------
        cycle_state:
            Current market regime dict from CycleAnalyzer.
        rl_action:
            Integer RL action (0 = hold, 1 = buy, 2 = sell).

        Returns
        -------
        Dict with ``"execute_now"`` (bool), ``"urgency"`` ("high"/"medium"/"low"),
        and ``"reason"`` string.
        """
        cycle_state = cycle_state or {}
        intraday = cycle_state.get("intraday", "chop")
        volatility = cycle_state.get("volatility", "low")

        execute_now = False
        urgency = "low"
        reason = "default – wait for better window"

        if rl_action in (1, 2):  # buy or sell signal
            if intraday in ("open_drive", "power_hour"):
                execute_now = True
                urgency = "high"
                reason = f"RL action {rl_action} aligned with {intraday} session"
            elif volatility == "high":
                execute_now = True
                urgency = "medium"
                reason = "high-volatility window – execute promptly"
            else:
                execute_now = True
                urgency = "low"
                reason = "standard execution during chop/fade session"

        return {
            "execute_now": execute_now,
            "urgency": urgency,
            "reason": reason,
            "intraday_session": intraday,
            "volatility": volatility,
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(
        self,
        signals: List[Dict[str, Any]],
        avg_volume: float = 1_000_000.0,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return execution diagnostics for a list of signals."""
        total_slippage = 0.0
        enriched = []
        for sig in signals:
            enriched_sig = self.apply_slippage(sig, avg_volume=avg_volume)
            total_slippage += float(enriched_sig.get("slippage_cost", 0.0))
            enriched.append(enriched_sig)

        window = self.rl_execution_window(cycle_state)
        return {
            "signal_count": len(signals),
            "total_estimated_slippage": round(total_slippage, 4),
            "execution_window": window,
            "slippage_bps": self.slippage_bps,
            "commission_pct": self.commission_pct,
        }
