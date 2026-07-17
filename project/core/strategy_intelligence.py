"""Main AI intelligence layer.

Orchestrates AI-enhanced signal generation through all adapter layers:

  1. Generate raw signals from the strategy module.
  2. Apply cycle-aware parameter overrides (CycleAdapter).
  3. Enrich each signal with ``ai_score`` (strategy_ai_utils).
  4. Compute dynamic stop/target and size with RiskAdapter.
  5. Apply capital allocation with CapitalAdapter.
  6. Optionally inject/blend an RL agent signal via RLAdapter.
  7. Filter by minimum AI-score quality threshold.
  8. Return enriched, ranked signal list with full diagnostic metadata.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from core.capital_manager import CapitalManager
from core.cycle_analyzer import CycleAnalyzer
from core.risk_manager import RiskManager
from core.strategy_ai_utils import enrich_signal, filter_signals_by_quality
from core.strategy_capital_adapter import CapitalAdapter
from core.strategy_cycle_adapter import CycleAdapter
from core.strategy_risk_adapter import RiskAdapter
from core.strategy_rl_adapter import RLAdapter

logger = logging.getLogger(__name__)


class StrategyIntelligence:
    """Orchestrates AI-enhanced signal generation through all adapter layers.

    Parameters
    ----------
    capital_manager:
        CapitalManager instance used for capital allocation; created from
        *config* when omitted.
    cycle_analyzer:
        CycleAnalyzer instance; created from *config* when omitted.
    risk_manager:
        RiskManager instance; created from *config* when omitted.
    rl_agent:
        Optional trained DQNAgent (or any object with an ``act`` method).
    config:
        Configuration dict (same shape as ``project/config.py``).
    rl_blend_weight:
        Blend weight passed to RLAdapter (0 = pure strategy, 1 = pure RL).
    min_signal_score:
        Minimum ``ai_score`` required for a signal to be included in output.
    """

    def __init__(
        self,
        capital_manager: Optional[CapitalManager] = None,
        cycle_analyzer: Optional[CycleAnalyzer] = None,
        risk_manager: Optional[RiskManager] = None,
        rl_agent: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None,
        rl_blend_weight: float = 0.4,
        min_signal_score: float = 0.40,
    ):
        self.config = config or {}
        self.capital_manager = capital_manager or CapitalManager(self.config)
        self.cycle_analyzer = cycle_analyzer or CycleAnalyzer(config=self.config)
        self.risk_manager = risk_manager or RiskManager(self.config)
        self.rl_agent = rl_agent

        self.cycle_adapter = CycleAdapter()
        self.rl_adapter = RLAdapter(blend_weight=rl_blend_weight)
        self.risk_adapter = RiskAdapter(risk_manager=self.risk_manager, config=self.config)
        self.capital_adapter = CapitalAdapter(capital_manager=self.capital_manager, config=self.config)

        self.min_signal_score = float(min_signal_score)

    # ------------------------------------------------------------------
    # Single-strategy run
    # ------------------------------------------------------------------

    def run(
        self,
        strategy_module: Any,
        data: pd.DataFrame,
        symbol: str,
        strategy_name: str = "",
        cycle_state: Optional[Dict[str, Any]] = None,
        current_positions: int = 0,
        capital: Optional[float] = None,
        allocation_map: Optional[Dict[str, Any]] = None,
        position: int = 0,
        equity: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the full intelligence pipeline for one strategy + symbol.

        Returns
        -------
        dict with keys:
          - ``signals``      : list of enriched, sized, filtered signal dicts
          - ``cycle_state``  : cycle state used during the run
          - ``cycle_params`` : adapted parameter overrides for this strategy
          - ``rl_signal``    : raw RL signal dict (or ``None``)
          - ``metadata``     : diagnostic info
        """
        capital = float(capital or self.config.get("DEFAULT_CAPITAL", 100_000.0))
        equity = float(equity if equity is not None else capital)
        strategy_name = strategy_name or _module_short_name(strategy_module)

        # 1. Detect or accept cycle state
        if cycle_state is None:
            cycle_state = self._detect_cycle(data, symbol)

        # 2. Adapt strategy parameters for the current cycle
        cycle_params = self.cycle_adapter.adapt_params(strategy_name, cycle_state=cycle_state)
        if cycle_params and hasattr(strategy_module, "set_params"):
            strategy_module.set_params(cycle_params)

        # 3. Generate raw strategy signals
        try:
            raw_signals: List[Dict[str, Any]] = strategy_module.generate_signals(data)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Strategy %s signal generation failed: %s", strategy_name, exc)
            raw_signals = []

        # 4. Compute ATR for dynamic risk targets
        atr = self._latest_atr(data)

        # 5. Enrich, risk-size, and apply capital allocation per signal
        enriched: List[Dict[str, Any]] = []
        for sig in raw_signals:
            sig = {**sig, "symbol": sig.get("symbol", symbol)}
            sig = self.risk_adapter.compute_stop_and_target(sig, atr, cycle_state)
            sig = self.risk_adapter.size_signal(sig, capital, cycle_state)
            if allocation_map:
                sig = self.capital_adapter.size_from_allocation(
                    sig, allocation_map, strategy_name, symbol
                )
            sig = enrich_signal(sig, data, cycle_state, source_strategy=strategy_name)
            confidence = self.cycle_adapter.adapt_signal_confidence(sig, cycle_state)
            sig["confidence"] = round(confidence, 4)
            enriched.append(sig)

        # 6. Apply position-count and R:R constraints
        enriched = self.risk_adapter.apply_risk_constraints(
            enriched, current_positions, capital, cycle_state
        )

        # 7. Enforce portfolio exposure limits
        enriched = self.capital_adapter.enforce_exposure_limits(enriched, capital)

        # 8. RL signal generation and hybrid blending
        rl_signal: Optional[Dict[str, Any]] = None
        if self.rl_agent is not None:
            state = self.rl_adapter.build_env_state(
                data, position=position, equity=equity, capital=capital
            )
            if state is not None:
                current_price = float(data["close"].iloc[-1]) if not data.empty else 0.0
                rl_signal = self.rl_adapter.get_rl_signal(
                    self.rl_agent, state, symbol, current_price, capital, cycle_state
                )
                if rl_signal is not None:
                    rl_signal = enrich_signal(
                        rl_signal, data, cycle_state, source_strategy="rl_agent"
                    )
                enriched = self.rl_adapter.hybrid_decision(
                    enriched, rl_signal, data, cycle_state
                )

        # 9. Filter by minimum AI score and sort
        final_signals = filter_signals_by_quality(enriched, min_score=self.min_signal_score)
        final_signals.sort(key=lambda s: float(s.get("ai_score", 0.0)), reverse=True)

        logger.debug(
            "Intelligence run | strategy=%s symbol=%s | raw=%d enriched=%d final=%d",
            strategy_name, symbol, len(raw_signals), len(enriched), len(final_signals),
        )

        return {
            "signals": final_signals,
            "cycle_state": cycle_state,
            "cycle_params": cycle_params,
            "rl_signal": rl_signal,
            "metadata": {
                "strategy": strategy_name,
                "symbol": symbol,
                "raw_signal_count": len(raw_signals),
                "final_signal_count": len(final_signals),
                "capital": capital,
                "atr": atr,
            },
        }

    # ------------------------------------------------------------------
    # Multi-strategy ensemble run
    # ------------------------------------------------------------------

    def run_ensemble(
        self,
        strategy_modules: Dict[str, Any],
        data: pd.DataFrame,
        symbol: str,
        cycle_state: Optional[Dict[str, Any]] = None,
        current_positions: int = 0,
        capital: Optional[float] = None,
        allocation_map: Optional[Dict[str, Any]] = None,
        strategy_weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Run the intelligence pipeline across multiple strategies and aggregate.

        Only strategies favoured by the current cycle regime are executed
        (unless the cycle state is empty, in which case all are run).

        Returns a deduplicated, ranked signal list together with per-strategy
        intermediate results.
        """
        capital = float(capital or self.config.get("DEFAULT_CAPITAL", 100_000.0))
        if cycle_state is None:
            cycle_state = self._detect_cycle(data, symbol)

        favored = self.cycle_adapter.get_favored_strategies(cycle_state)
        all_results: Dict[str, Dict[str, Any]] = {}
        all_signals: List[Dict[str, Any]] = []

        for name, module in strategy_modules.items():
            if favored and name not in favored:
                logger.debug(
                    "Skipping non-favoured strategy %s for cycle %s", name, cycle_state
                )
                continue
            result = self.run(
                module, data, symbol,
                strategy_name=name,
                cycle_state=cycle_state,
                current_positions=current_positions + len(all_signals),
                capital=capital,
                allocation_map=allocation_map,
            )
            all_results[name] = result
            all_signals.extend(result.get("signals", []))

        # Deduplicate (symbol, side) — keep highest ai_score
        deduped: Dict[tuple, Dict[str, Any]] = {}
        for sig in all_signals:
            key = (sig.get("symbol", symbol), sig.get("side", "long"))
            existing = deduped.get(key)
            if existing is None or float(sig.get("ai_score", 0.0)) > float(existing.get("ai_score", 0.0)):
                deduped[key] = sig

        final = sorted(deduped.values(), key=lambda s: float(s.get("ai_score", 0.0)), reverse=True)

        return {
            "signals": final,
            "cycle_state": cycle_state,
            "strategy_results": all_results,
            "metadata": {
                "symbol": symbol,
                "strategies_run": list(all_results.keys()),
                "total_signals": len(final),
                "capital": capital,
            },
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _detect_cycle(self, data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        if data.empty:
            return {}
        try:
            return self.cycle_analyzer.analyze(data, symbol=symbol)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CycleAnalyzer failed for %s: %s", symbol, exc)
            return {}

    def _latest_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        if data.empty or len(data) < 2:
            return 0.0
        try:
            tr1 = data["high"] - data["low"]
            tr2 = (data["high"] - data["close"].shift()).abs()
            tr3 = (data["low"] - data["close"].shift()).abs()
            atr_series = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(period).mean()
            val = float(atr_series.iloc[-1])
            return val if np.isfinite(val) else 0.0
        except Exception:  # noqa: BLE001
            return 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _module_short_name(module: Any) -> str:
    name = getattr(module, "__name__", "") or getattr(module, "__class__.__name__", "unknown")
    return name.split(".")[-1]
