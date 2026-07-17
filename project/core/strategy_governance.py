"""Strategy governance engine.

Monitors live and backtested strategy performance for decay, anomalies,
and rule violations.  Automatically disables underperforming strategies,
reduces capital allocation, and tightens risk parameters when the cycle
shifts to an adverse regime.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_DEFAULT_DECAY_WINDOW = 20           # bars / trades
_DEFAULT_MIN_WIN_RATE = 0.35         # 35 %
_DEFAULT_MIN_SHARPE = 0.20
_DEFAULT_MAX_CONSECUTIVE_LOSSES = 6
_DEFAULT_ANOMALY_ZSCORE = 3.0        # Z-score threshold
_DEFAULT_MAX_DRAWDOWN_PCT = 0.15     # 15 % per-strategy drawdown before action


class StrategyGovernance:
    """Governance layer that monitors and enforces strategy health rules.

    Parameters
    ----------
    config:
        Application configuration dict.  Recognised keys:

        ``DECAY_WINDOW`` (int) – rolling window for decay detection.
        ``MIN_WIN_RATE`` (float) – below this win-rate a strategy is flagged.
        ``MIN_SHARPE`` (float) – below this rolling Sharpe a strategy is flagged.
        ``MAX_CONSECUTIVE_LOSSES`` (int) – consecutive loss threshold.
        ``ANOMALY_ZSCORE`` (float) – Z-score threshold for anomaly detection.
        ``GOVERNANCE_MAX_DRAWDOWN_PCT`` (float) – per-strategy drawdown ceiling.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.decay_window = int(cfg.get("DECAY_WINDOW", _DEFAULT_DECAY_WINDOW))
        self.min_win_rate = float(cfg.get("MIN_WIN_RATE", _DEFAULT_MIN_WIN_RATE))
        self.min_sharpe = float(cfg.get("MIN_SHARPE", _DEFAULT_MIN_SHARPE))
        self.max_consecutive_losses = int(cfg.get("MAX_CONSECUTIVE_LOSSES", _DEFAULT_MAX_CONSECUTIVE_LOSSES))
        self.anomaly_zscore = float(cfg.get("ANOMALY_ZSCORE", _DEFAULT_ANOMALY_ZSCORE))
        self.max_drawdown_pct = float(cfg.get("GOVERNANCE_MAX_DRAWDOWN_PCT", _DEFAULT_MAX_DRAWDOWN_PCT))

        # In-memory registry: strategy_name -> governance state
        self._state: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def _get_state(self, strategy: str) -> Dict[str, Any]:
        if strategy not in self._state:
            self._state[strategy] = {
                "enabled": True,
                "trade_history": [],
                "flags": [],
                "consecutive_losses": 0,
                "capital_reduction_factor": 1.0,
                "last_updated": datetime.utcnow().isoformat(),
            }
        return self._state[strategy]

    # ------------------------------------------------------------------
    # Decay detection
    # ------------------------------------------------------------------

    def detect_decay(self, strategy: str, trade_pnls: List[float]) -> Dict[str, Any]:
        """Detect performance decay over the most recent ``decay_window`` trades.

        Parameters
        ----------
        strategy:
            Strategy name.
        trade_pnls:
            Ordered list of trade net-PnL values (newest last).

        Returns
        -------
        Dict with ``"decaying"`` bool and diagnostic details.
        """
        recent = trade_pnls[-self.decay_window:]
        if len(recent) < 3:
            return {"decaying": False, "reason": "insufficient data"}

        wins = sum(1 for p in recent if p > 0)
        win_rate = wins / len(recent)
        mean_pnl = float(np.mean(recent))
        ret_arr = np.array(recent, dtype=float)
        std = float(np.std(ret_arr))
        sharpe = mean_pnl / std if std > 0 else 0.0

        # Consecutive losses
        losses_run = 0
        for pnl in reversed(recent):
            if pnl < 0:
                losses_run += 1
            else:
                break

        state = self._get_state(strategy)
        state["consecutive_losses"] = losses_run

        decaying = (
            win_rate < self.min_win_rate
            or sharpe < self.min_sharpe
            or losses_run >= self.max_consecutive_losses
        )

        return {
            "decaying": decaying,
            "win_rate": round(win_rate, 4),
            "rolling_sharpe": round(sharpe, 4),
            "consecutive_losses": losses_run,
            "window": len(recent),
        }

    # ------------------------------------------------------------------
    # Anomaly detection
    # ------------------------------------------------------------------

    def detect_anomalies(
        self,
        strategy: str,
        trade_pnls: List[float],
    ) -> List[Dict[str, Any]]:
        """Identify statistically anomalous trades via Z-score.

        Returns a list of anomaly records for trades exceeding the threshold.
        """
        if len(trade_pnls) < 5:
            return []

        arr = np.array(trade_pnls, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std == 0:
            return []

        anomalies = []
        for i, pnl in enumerate(trade_pnls):
            z = abs((pnl - mean) / std)
            if z >= self.anomaly_zscore:
                anomalies.append({"index": i, "pnl": round(pnl, 4), "z_score": round(z, 4)})
        return anomalies

    # ------------------------------------------------------------------
    # Auto-disable
    # ------------------------------------------------------------------

    def evaluate_strategy(
        self,
        strategy: str,
        trade_pnls: List[float],
        equity_curve: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        """Evaluate a strategy and apply governance actions when needed.

        Actions applied (mutates internal state):
        - Disable the strategy when decaying *and* drawdown exceeds ceiling.
        - Reduce capital allocation factor when decaying but within drawdown limits.
        - Flag anomalies for review.

        Returns
        -------
        Governance action dict with ``"action"`` key:
        ``"disabled"`` / ``"reduce_capital"`` / ``"ok"``.
        """
        state = self._get_state(strategy)
        decay_info = self.detect_decay(strategy, trade_pnls)
        anomalies = self.detect_anomalies(strategy, trade_pnls)

        # Drawdown check
        drawdown = 0.0
        if equity_curve and len(equity_curve) > 1:
            arr = np.array(equity_curve, dtype=float)
            peak = np.maximum.accumulate(arr)
            dd = (arr - peak) / np.where(peak == 0, 1.0, peak)
            drawdown = float(abs(dd.min()))

        action = "ok"
        reason = "strategy within acceptable parameters"

        if decay_info["decaying"]:
            if drawdown >= self.max_drawdown_pct:
                state["enabled"] = False
                state["capital_reduction_factor"] = 0.0
                action = "disabled"
                reason = f"decay detected + drawdown {drawdown:.1%} >= ceiling {self.max_drawdown_pct:.1%}"
                state["flags"].append({"type": "auto_disabled", "ts": datetime.utcnow().isoformat(), "reason": reason})
            else:
                new_factor = max(state["capital_reduction_factor"] * 0.75, 0.10)
                state["capital_reduction_factor"] = new_factor
                action = "reduce_capital"
                reason = f"decay detected – capital factor reduced to {new_factor:.2f}"
                state["flags"].append({"type": "capital_reduced", "ts": datetime.utcnow().isoformat(), "reason": reason})

        for anom in anomalies:
            state["flags"].append({"type": "anomaly", "ts": datetime.utcnow().isoformat(), **anom})

        state["last_updated"] = datetime.utcnow().isoformat()

        return {
            "strategy": strategy,
            "action": action,
            "reason": reason,
            "enabled": state["enabled"],
            "capital_reduction_factor": state["capital_reduction_factor"],
            "decay": decay_info,
            "anomalies": anomalies,
            "drawdown": round(drawdown, 4),
        }

    # ------------------------------------------------------------------
    # Cycle-driven tightening
    # ------------------------------------------------------------------

    def cycle_tighten_risk(
        self,
        signals: List[Dict[str, Any]],
        cycle_state: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Tighten stop-loss distances and reduce sizes when cycle is adverse.

        Applies:
        - Bear trend → reduce size 25 %, tighten stop 20 %.
        - High volatility → tighten stop 10 %.
        - Risk-off macro → reduce size 15 %.
        """
        trend = cycle_state.get("trend", "sideways")
        volatility = cycle_state.get("volatility", "low")
        macro = cycle_state.get("macro", "risk_on")

        size_mult = 1.0
        stop_mult = 1.0

        if trend == "bear":
            size_mult *= 0.75
            stop_mult *= 0.80
        if volatility == "high":
            stop_mult *= 0.90
        if macro == "risk_off":
            size_mult *= 0.85

        if size_mult == 1.0 and stop_mult == 1.0:
            return signals

        tightened = []
        for sig in signals:
            t = sig.copy()
            t["size"] = round(float(t.get("size", 0.0)) * size_mult, 4)
            entry = float(t.get("entry_price", 0.0))
            stop = float(t.get("stop_loss", entry))
            dist = abs(entry - stop) * stop_mult
            side = t.get("side", "long")
            t["stop_loss"] = round(entry - dist if side == "long" else entry + dist, 6)
            t["governance_cycle_tightened"] = True
            tightened.append(t)
        return tightened

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def is_enabled(self, strategy: str) -> bool:
        """Return True when the strategy is not governance-disabled."""
        return self._get_state(strategy).get("enabled", True)

    def capital_factor(self, strategy: str) -> float:
        """Return the current capital reduction factor for a strategy (0–1)."""
        return float(self._get_state(strategy).get("capital_reduction_factor", 1.0))

    def get_flags(self, strategy: str) -> List[Dict[str, Any]]:
        """Return the governance event log for a strategy."""
        return list(self._get_state(strategy).get("flags", []))

    def reenable(self, strategy: str) -> None:
        """Manually re-enable a previously disabled strategy and reset factors."""
        state = self._get_state(strategy)
        state["enabled"] = True
        state["capital_reduction_factor"] = 1.0
        state["flags"].append({"type": "reenabled", "ts": datetime.utcnow().isoformat()})

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self, strategies: List[str]) -> Dict[str, Any]:
        """Return a governance diagnostics snapshot for a list of strategies."""
        return {
            strategy: {
                "enabled": self.is_enabled(strategy),
                "capital_factor": self.capital_factor(strategy),
                "flags": self.get_flags(strategy),
                "consecutive_losses": self._get_state(strategy).get("consecutive_losses", 0),
            }
            for strategy in strategies
        }
