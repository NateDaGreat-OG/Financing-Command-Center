"""Live trading orchestrator.

Provides a unified live-trading engine that coordinates:

- Live data fetching from Alpaca
- AI Strategy Intelligence signal generation
- Execution intelligence (VWAP/TWAP/slippage/RL timing)
- Portfolio risk engine constraints
- Strategy governance rules (disable/reduce capital)
- Order submission to Alpaca
- Diagnostics logging

Designed for paper and live trading.  No orders are submitted in dry-run mode.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.capital_manager import CapitalManager
from core.cycle_analyzer import CycleAnalyzer
from core.diagnostics_layer import DiagnosticsLayer
from core.execution_intelligence import ExecutionIntelligence
from core.portfolio_risk_engine import PortfolioRiskEngine
from core.risk_manager import RiskManager
from core.strategy_governance import StrategyGovernance
from core.strategy_intelligence import StrategyIntelligence
from core.strategy_registry import create_intelligence_layer
from core.trade_logger import TradeLogger
from services.alpaca_client import AlpacaClient

logger = logging.getLogger(__name__)


class LiveTradingOrchestrator:
    """Unified live trading engine using the AI Intelligence Layer.

    Parameters
    ----------
    strategy_module:
        Loaded strategy module (must expose ``generate_signals``).
    alpaca:
        :class:`~services.alpaca_client.AlpacaClient` instance.
    config:
        Application configuration dict.
    intelligence:
        Pre-built :class:`StrategyIntelligence` (constructed from *config* if omitted).
    portfolio_risk:
        Pre-built :class:`PortfolioRiskEngine`.
    execution:
        Pre-built :class:`ExecutionIntelligence`.
    governance:
        Pre-built :class:`StrategyGovernance`.
    trade_logger:
        :class:`~core.trade_logger.TradeLogger` instance.
    dry_run:
        When True, signals are generated but no orders are submitted.
    """

    def __init__(
        self,
        strategy_module: Any,
        alpaca: AlpacaClient,
        config: Dict[str, Any],
        intelligence: Optional[StrategyIntelligence] = None,
        portfolio_risk: Optional[PortfolioRiskEngine] = None,
        execution: Optional[ExecutionIntelligence] = None,
        governance: Optional[StrategyGovernance] = None,
        trade_logger: Optional[TradeLogger] = None,
        dry_run: bool = True,
    ):
        self.strategy = strategy_module
        self.alpaca = alpaca
        self.config = config
        self.dry_run = dry_run

        self.intelligence = intelligence or create_intelligence_layer(config=config)
        self.portfolio_risk = portfolio_risk or PortfolioRiskEngine(config=config)
        self.execution = execution or ExecutionIntelligence(config=config)
        self.governance = governance or StrategyGovernance(config=config)
        self.trade_logger = trade_logger or TradeLogger(log_dir=config.get("LOG_DIR", "logs"))
        self.diagnostics_layer = DiagnosticsLayer(config=config)

        self._strategy_name = getattr(strategy_module, "__name__", "unknown").split(".")[-1]

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(
        self,
        symbols: List[str],
        allocation_map: Optional[Dict[str, Any]] = None,
        rl_agent: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute one live trading cycle for all *symbols*.

        Parameters
        ----------
        symbols:
            List of ticker symbols to trade.
        allocation_map:
            Optional capital allocation map.
        rl_agent:
            Optional RL agent for signal blending.

        Returns
        -------
        Dict with ``"results"`` (per-symbol), ``"account"``, ``"positions"``,
        ``"diagnostics"``, and ``"orders_submitted"``.
        """
        if rl_agent is not None:
            self.intelligence.rl_agent = rl_agent

        account_info = self._safe_get_account()
        positions = self._safe_get_positions()
        current_positions = len(positions) if isinstance(positions, list) else 0
        capital = self._account_equity(account_info)

        results: Dict[str, Any] = {}
        all_signals: List[Dict[str, Any]] = []
        all_intel_results: List[Dict[str, Any]] = []
        orders_submitted: List[Dict[str, Any]] = []

        for symbol in symbols:
            symbol_result = self._run_symbol(
                symbol=symbol,
                capital=capital,
                current_positions=current_positions,
                allocation_map=allocation_map,
            )
            results[symbol] = symbol_result
            all_signals.extend(symbol_result.get("signals", []))
            if symbol_result.get("intel_result"):
                all_intel_results.append(symbol_result["intel_result"])

        # --- Portfolio risk gate ---
        if all_signals:
            all_signals = self.portfolio_risk.cap_position_sizes(all_signals, capital)

        # --- Governance filter ---
        all_signals = self._governance_filter(all_signals)

        # --- Execution intelligence ---
        exec_diag = self.execution.diagnostics(all_signals, cycle_state=self._latest_cycle(all_intel_results))

        # --- Submit orders ---
        for sig in all_signals:
            order = self._submit_order(sig)
            if order:
                orders_submitted.append(order)

        # --- Diagnostics ---
        last_intel = all_intel_results[-1] if all_intel_results else {}
        diag = self.diagnostics_layer.build(
            intelligence_result=last_intel,
            cycle_state=last_intel.get("cycle_state", {}),
            cycle_params=last_intel.get("cycle_params"),
            rl_signal=last_intel.get("rl_signal"),
            execution_info=exec_diag,
            governance_snapshot=self.governance.diagnostics([self._strategy_name]),
        )

        return {
            "status": "dry_run" if self.dry_run else "live",
            "timestamp": datetime.utcnow().isoformat(),
            "account": account_info,
            "positions": positions,
            "results": results,
            "total_signals": len(all_signals),
            "orders_submitted": orders_submitted,
            "diagnostics": diag,
        }

    # ------------------------------------------------------------------
    # Per-symbol execution
    # ------------------------------------------------------------------

    def _run_symbol(
        self,
        symbol: str,
        capital: float,
        current_positions: int,
        allocation_map: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Fetch live data, generate intelligence signals, and apply governance."""
        if not self.governance.is_enabled(self._strategy_name):
            logger.info("Strategy %s is governance-disabled; skipping %s", self._strategy_name, symbol)
            return {"symbol": symbol, "signals": [], "skipped": "governance_disabled"}

        try:
            raw = self.alpaca.get_intraday(symbol, interval=self.config.get("LIVE_INTERVAL", "1Min"))
            df = _normalize_bars(raw)
        except Exception as exc:
            logger.warning("Failed to fetch data for %s: %s", symbol, exc)
            return {"symbol": symbol, "signals": [], "error": str(exc)}

        if df.empty:
            return {"symbol": symbol, "signals": [], "error": "no data"}

        # Intelligence pipeline
        intel_result = self.intelligence.run(
            strategy_module=self.strategy,
            data=df,
            symbol=symbol,
            strategy_name=self._strategy_name,
            current_positions=current_positions,
            capital=capital,
            allocation_map=allocation_map,
        )

        signals = intel_result.get("signals", [])
        cycle_state = intel_result.get("cycle_state", {})

        # Governance tightening
        signals = self.governance.cycle_tighten_risk(signals, cycle_state)

        # Execution window check
        rl_action = None
        if intel_result.get("rl_signal"):
            rl_action = intel_result["rl_signal"].get("action")
        exec_window = self.execution.rl_execution_window(cycle_state, rl_action)

        if not exec_window["execute_now"]:
            logger.debug("Execution deferred for %s – %s", symbol, exec_window["reason"])
            signals = []

        # Apply slippage estimates
        signals = [self.execution.apply_slippage(sig) for sig in signals]

        self.trade_logger.log_signals(symbol, signals)

        return {
            "symbol": symbol,
            "signals": signals,
            "intel_result": intel_result,
            "execution_window": exec_window,
            "cycle_state": cycle_state,
        }

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def _submit_order(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Submit a single order to Alpaca, or log it in dry-run mode."""
        symbol = signal.get("symbol", "")
        size = float(signal.get("size", 0))
        side = signal.get("side", "long")
        alpaca_side = "buy" if side == "long" else "sell"

        if size <= 0 or not symbol:
            return None

        order_record = {
            "symbol": symbol,
            "qty": size,
            "side": alpaca_side,
            "type": "market",
            "time_in_force": "day",
            "dry_run": self.dry_run,
            "timestamp": datetime.utcnow().isoformat(),
        }

        if self.dry_run:
            logger.info("[DRY RUN] Would submit: %s", order_record)
            return {**order_record, "status": "simulated"}

        try:
            resp = self.alpaca.submit_order(
                symbol=symbol,
                qty=size,
                side=alpaca_side,
                type="market",
                time_in_force="day",
            )
            return {**order_record, "status": "submitted", "response": resp}
        except Exception as exc:
            logger.error("Order submission failed for %s: %s", symbol, exc)
            return {**order_record, "status": "error", "error": str(exc)}

    # ------------------------------------------------------------------
    # Governance filter
    # ------------------------------------------------------------------

    def _governance_filter(self, signals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove or scale-down signals from governed strategies."""
        if not signals:
            return signals
        factor = self.governance.capital_factor(self._strategy_name)
        if factor >= 1.0:
            return signals
        scaled = []
        for sig in signals:
            t = sig.copy()
            t["size"] = round(float(t.get("size", 0)) * factor, 4)
            scaled.append(t)
        return scaled

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def _safe_get_account(self) -> Dict[str, Any]:
        try:
            return self.alpaca.get_account()
        except Exception as exc:
            logger.warning("Could not fetch account: %s", exc)
            return {}

    def _safe_get_positions(self) -> Any:
        try:
            return self.alpaca.get_positions()
        except Exception as exc:
            logger.warning("Could not fetch positions: %s", exc)
            return []

    def _account_equity(self, account_info: Dict[str, Any]) -> float:
        try:
            return float(account_info.get("equity", self.config.get("DEFAULT_CAPITAL", 100_000)))
        except (TypeError, ValueError):
            return float(self.config.get("DEFAULT_CAPITAL", 100_000))

    def _latest_cycle(self, intel_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not intel_results:
            return {}
        return intel_results[-1].get("cycle_state", {})


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normalize_bars(raw_data: Any) -> pd.DataFrame:
    if isinstance(raw_data, pd.DataFrame):
        return raw_data.copy()
    rows = raw_data.get("bars", raw_data) if isinstance(raw_data, dict) else raw_data
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    col_map = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    df = df.rename(columns=col_map)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    return df
