"""Intelligence-aware backtesting adapter.

Wraps the existing :class:`~backtest.backtester.Backtester` and injects the
full AI Strategy Intelligence Layer (StrategyIntelligence) so that every bar
uses:

- Cycle-aware parameter overrides
- AI-enriched and risk-sized signals
- Capital-allocation-aware position sizing
- Optional RL-agent signal blending

Returns expanded metrics including annualised Sharpe, Sortino, profit factor,
expectancy, max drawdown, and equity/drawdown curves.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.backtester import Backtester
from core.capital_manager import CapitalManager
from core.cycle_analyzer import CycleAnalyzer
from core.diagnostics_layer import DiagnosticsLayer
from core.risk_manager import RiskManager
from core.strategy_intelligence import StrategyIntelligence
from core.strategy_registry import create_intelligence_layer
from core.trade_logger import TradeLogger

logger = logging.getLogger(__name__)

_TRADING_DAYS_PER_YEAR = 252


class BacktesterIntelAdapter:
    """Intelligence-aware backtesting adapter.

    Parameters
    ----------
    strategy_module:
        Loaded strategy module (must expose ``generate_signals``).
    config:
        Application configuration dict.
    capital:
        Starting capital in dollars.
    slippage:
        Per-trade slippage fraction.
    commission:
        Per-trade commission fraction.
    intelligence:
        Pre-built :class:`StrategyIntelligence` instance.  When omitted one is
        constructed from *config*.
    logger:
        :class:`~core.trade_logger.TradeLogger` instance.
    """

    def __init__(
        self,
        strategy_module: Any,
        config: Dict[str, Any],
        capital: float = 100_000.0,
        slippage: float = 0.0005,
        commission: float = 0.001,
        intelligence: Optional[StrategyIntelligence] = None,
        logger: Optional[TradeLogger] = None,
    ):
        self.strategy = strategy_module
        self.config = config
        self.capital = float(capital)
        self.slippage = float(slippage)
        self.commission = float(commission)

        self.intelligence = intelligence or create_intelligence_layer(config=config)
        self.trade_logger = logger or TradeLogger(log_dir=config.get("LOG_DIR", "logs"))
        self.diagnostics = DiagnosticsLayer(config=config)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(
        self,
        historical_data: Dict[str, Any],
        allocation_map: Optional[Dict[str, Any]] = None,
        strategy_name: str = "",
    ) -> Dict[str, Any]:
        """Run an intelligence-enriched backtest.

        Parameters
        ----------
        historical_data:
            Mapping of ``symbol -> raw bar data`` (same format as
            :class:`~backtest.backtester.Backtester`).
        allocation_map:
            Optional capital allocation map from :class:`~core.capital_manager.CapitalManager`.
        strategy_name:
            Human-readable name forwarded to the intelligence layer.

        Returns
        -------
        Dict with keys:

        - ``metrics`` – expanded performance metrics
        - ``trade_log`` – list of simulated trades with AI metadata
        - ``equity_curve`` – list of cumulative equity values
        - ``drawdown_curve`` – list of drawdown fractions
        - ``trade_distribution`` – histogram of trade returns (20 buckets)
        - ``intelligence_diagnostics`` – AI / cycle / risk snapshot
        """
        equity = self.capital
        trade_log: List[Dict[str, Any]] = []
        returns: List[float] = []
        equity_curve: List[float] = [self.capital]
        intel_snapshots: List[Dict[str, Any]] = []

        strategy_name = strategy_name or getattr(self.strategy, "__name__", "unknown")

        for symbol, raw_data in historical_data.items():
            df = _normalize_bars(raw_data)
            if df.empty:
                continue

            # Detect cycle once per symbol
            try:
                cycle_state = self.intelligence.cycle_analyzer.analyze(df, symbol=symbol)
            except Exception:
                cycle_state = {}

            # Run intelligence pipeline for signals
            intel_result = self.intelligence.run(
                strategy_module=self.strategy,
                data=df,
                symbol=symbol,
                strategy_name=strategy_name,
                cycle_state=cycle_state,
                capital=equity,
                allocation_map=allocation_map,
            )
            intel_snapshots.append(intel_result)

            signals = intel_result.get("signals", [])
            if not signals:
                # Fall back to raw strategy if intelligence produces nothing
                try:
                    signals = self.strategy.generate_signals(df)
                except Exception:
                    signals = []

            for signal in signals:
                trade = _simulate_trade(symbol, signal, df, self.slippage, self.commission)
                if trade is None:
                    continue
                trade["intelligence_metadata"] = {
                    "ai_score": signal.get("ai_score"),
                    "cycle_state": cycle_state,
                    "cycle_params": intel_result.get("cycle_params"),
                    "rl_signal": intel_result.get("rl_signal"),
                }
                trade_log.append(trade)
                equity += trade["net_pnl"]
                equity_curve.append(equity)
                returns.append(trade["net_pnl"] / self.capital)
                self.trade_logger.log_trade(trade)

        # Compute expanded metrics
        metrics = _expanded_metrics(equity, self.capital, returns, trade_log)

        # Build curves
        eq_arr = np.array(equity_curve, dtype=float)
        peak = np.maximum.accumulate(eq_arr)
        dd_curve = (eq_arr - peak) / np.where(peak == 0, 1.0, peak)

        # Trade PnL distribution
        pnl_values = [t["net_pnl"] for t in trade_log]
        hist, edges = (np.histogram(pnl_values, bins=20) if pnl_values else ([], []))

        # Aggregate diagnostics
        combined_intel = {
            "signals": [],
            "cycle_state": {},
            "cycle_params": {},
            "rl_signal": None,
            "metadata": {"strategy": strategy_name, "capital": self.capital},
        }
        if intel_snapshots:
            last = intel_snapshots[-1]
            combined_intel.update(last)

        diag = self.diagnostics.build(
            intelligence_result=combined_intel,
            cycle_state=combined_intel.get("cycle_state", {}),
            cycle_params=combined_intel.get("cycle_params"),
            rl_signal=combined_intel.get("rl_signal"),
        )

        result = {
            "metrics": metrics,
            "trade_log": trade_log,
            "equity_curve": eq_arr.round(2).tolist(),
            "drawdown_curve": dd_curve.round(4).tolist(),
            "trade_distribution": {
                "counts": hist.tolist() if len(hist) else [],
                "edges": [round(float(e), 4) for e in edges] if len(edges) else [],
            },
            "intelligence_diagnostics": diag,
        }
        self.trade_logger.log_backtest(result)
        return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_bars(raw_data: Any) -> pd.DataFrame:
    """Convert raw Alpaca bar data or a DataFrame to a normalised DataFrame."""
    if isinstance(raw_data, pd.DataFrame):
        df = raw_data.copy()
    else:
        rows = raw_data.get("bars", raw_data.get("data", raw_data)) if isinstance(raw_data, dict) else raw_data
        df = pd.DataFrame(rows)

    if df.empty:
        return df

    col_map = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=col_map)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df


def _simulate_trade(
    symbol: str,
    signal: Dict[str, Any],
    df: pd.DataFrame,
    slippage: float,
    commission: float,
) -> Optional[Dict[str, Any]]:
    """Simulate a single trade from a signal, returning a trade record or None."""
    if not signal or "entry_price" not in signal or "size" not in signal:
        return None

    entry_price = float(signal["entry_price"])
    exit_price = float(signal.get("exit_price", entry_price))
    size = float(signal["size"])
    if size <= 0:
        return None

    side = signal.get("side", "long")
    gross_pnl = (exit_price - entry_price) * size if side == "long" else (entry_price - exit_price) * size
    fees = abs(gross_pnl) * commission
    slip_cost = abs(gross_pnl) * slippage
    net_pnl = gross_pnl - fees - slip_cost

    return {
        "timestamp": datetime.utcnow().isoformat(),
        "symbol": symbol,
        "side": side,
        "entry_price": round(entry_price, 6),
        "exit_price": round(exit_price, 6),
        "size": round(size, 4),
        "gross_pnl": round(gross_pnl, 4),
        "fees": round(fees, 4),
        "slippage": round(slip_cost, 4),
        "net_pnl": round(net_pnl, 4),
        "signal": signal,
    }


def _expanded_metrics(
    final_equity: float,
    initial_capital: float,
    returns: List[float],
    trade_log: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute expanded backtest performance metrics."""
    if not returns:
        return {
            "cagr": 0.0,
            "annualized_sharpe": 0.0,
            "sortino": 0.0,
            "profit_factor": 0.0,
            "expectancy": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "trade_count": 0,
        }

    arr = np.array(returns, dtype=float)
    n = max(len(arr), 1)

    mean_ret = float(arr.mean())
    std_ret = float(arr.std()) if n > 1 else 0.0

    # Annualised Sharpe (assumes returns are per-trade, scale to daily equivalent)
    ann_factor = np.sqrt(_TRADING_DAYS_PER_YEAR / n)
    sharpe = mean_ret / std_ret * ann_factor if std_ret > 0 else 0.0

    # Sortino uses downside deviation
    downside = arr[arr < 0]
    down_std = float(downside.std()) if len(downside) > 1 else 0.0
    sortino = mean_ret / down_std * ann_factor if down_std > 0 else sharpe

    # Profit factor
    gross_wins = sum(t["net_pnl"] for t in trade_log if t["net_pnl"] > 0)
    gross_losses = abs(sum(t["net_pnl"] for t in trade_log if t["net_pnl"] < 0))
    profit_factor = round(gross_wins / gross_losses, 4) if gross_losses > 0 else float("inf")

    # Expectancy
    win_count = sum(1 for t in trade_log if t["net_pnl"] > 0)
    loss_count = len(trade_log) - win_count
    win_rate_frac = win_count / max(len(trade_log), 1)
    avg_win = gross_wins / max(win_count, 1)
    avg_loss = gross_losses / max(loss_count, 1)
    expectancy = win_rate_frac * avg_win - (1 - win_rate_frac) * avg_loss

    # Max drawdown
    equity_curve = np.cumsum(arr) * initial_capital + initial_capital
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve)
    max_drawdown = float(dd.max())

    return {
        "cagr": round(((final_equity / initial_capital) - 1) * 100, 2),
        "annualized_sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "profit_factor": round(profit_factor, 4),
        "expectancy": round(expectancy, 4),
        "max_drawdown": round(max_drawdown, 2),
        "win_rate": round(win_rate_frac * 100, 2),
        "trade_count": len(trade_log),
        "final_equity": round(final_equity, 2),
    }
