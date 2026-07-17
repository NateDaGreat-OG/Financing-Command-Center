"""Portfolio-level capital allocation engine.

This module provides a CapitalManager that distributes capital across strategies,
symbols, and RL agents while enforcing risk constraints and cycle-aware sizing.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
try:
    from .risk_manager import RiskManager
except ImportError:
    from core.risk_manager import RiskManager


class CapitalManager:
    def __init__(self, config: dict, risk_manager: Optional[RiskManager] = None):
        self.config = config or {}
        self.total_capital = float(self.config.get("DEFAULT_CAPITAL", 100000.0))
        self.max_portfolio_drawdown = float(self.config.get("MAX_PORTFOLIO_DRAWDOWN", 0.2))
        self.max_strategy_drawdown = float(self.config.get("MAX_STRATEGY_DRAWDOWN", 0.1))
        self.max_symbol_exposure = float(self.config.get("MAX_SYMBOL_EXPOSURE", 0.15))
        self.min_allocation_pct = float(self.config.get("MIN_ALLOCATION_PCT", 0.01))
        self.max_position_size_pct = float(self.config.get("MAX_POSITION_SIZE_PCT", 0.2))
        self.risk_manager = risk_manager or RiskManager(config)

    def allocate(
        self,
        strategies: List[str],
        symbols: List[str],
        strategy_metrics: Dict[str, Any],
        rl_metrics: Optional[Dict[str, Any]] = None,
        cycle_state: Optional[Dict[str, Any]] = None,
        portfolio_capital: Optional[float] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """Compute a capital allocation map for strategies and symbols."""
        capital = float(portfolio_capital or self.total_capital)
        cycle_state = cycle_state or {}
        rl_metrics = rl_metrics or {}

        strategies = strategies or list(strategy_metrics.keys())
        symbols = symbols or []

        strategy_scores = self._strategy_scores(strategies, strategy_metrics, cycle_state)
        strategy_weights = self._normalize_weights(strategy_scores)
        symbol_weights = self._symbol_weights(symbols, rl_metrics, cycle_state)

        allocation_map: Dict[str, Dict[str, Dict[str, float]]] = {}
        for strategy, weight in zip(strategies, strategy_weights):
            strategy_allocation = max(weight * capital, capital * self.min_allocation_pct)
            allocation_map[strategy] = {}
            for symbol, symbol_weight in zip(symbols, symbol_weights):
                allocated_capital = float(round(strategy_allocation * symbol_weight, 2))
                risk_budget = float(round(allocated_capital * self.max_strategy_drawdown, 2))
                max_position_size = float(round(allocated_capital * self.max_position_size_pct, 2))
                allocation_map[strategy][symbol] = {
                    "allocated_capital": allocated_capital,
                    "max_position_size": max_position_size,
                    "risk_budget": risk_budget,
                }

        return allocation_map

    def _strategy_scores(
        self,
        strategies: List[str],
        strategy_metrics: Dict[str, Any],
        cycle_state: Dict[str, Any],
    ) -> List[float]:
        scores: List[float] = []
        for strategy in strategies:
            metrics = strategy_metrics.get(strategy, {})
            sharpe = float(metrics.get("sharpe", 0.0))
            win_rate = float(metrics.get("win_rate", 0.0)) / 100.0
            drawdown = float(metrics.get("max_drawdown", 0.0))
            score = (sharpe * 1.5) + (win_rate * 0.5) - (drawdown * 0.1)
            score *= self._cycle_multiplier(strategy, cycle_state)
            score = max(score, 0.01)
            scores.append(score)
        return scores

    def _symbol_weights(
        self,
        symbols: List[str],
        rl_metrics: Dict[str, Any],
        cycle_state: Dict[str, Any],
    ) -> List[float]:
        base_weights: List[float] = []
        for symbol in symbols:
            metrics = rl_metrics.get(symbol, {})
            reward = float(metrics.get("average_reward", 0.0))
            stability = float(metrics.get("stability", 0.5))
            weight = 1.0 + reward * 0.1 + stability * 0.2
            weight *= self._symbol_cycle_adjustment(symbol, cycle_state)
            base_weights.append(max(weight, 0.01))
        return self._normalize_weights(base_weights)

    def _cycle_multiplier(self, strategy: str, cycle_state: Dict[str, Any]) -> float:
        trend = cycle_state.get("trend", "sideways")
        volatility = cycle_state.get("volatility", "low")
        if "trend" in strategy.lower():
            multiplier = 1.2 if trend == "bull" else 0.8 if trend == "bear" else 1.0
        elif "volatility" in strategy.lower():
            multiplier = 1.2 if volatility == "high" else 0.9
        else:
            multiplier = 1.0

        if cycle_state.get("macro") == "risk_off":
            multiplier *= 0.9
        return float(np.clip(multiplier, 0.5, 1.5))

    def _symbol_cycle_adjustment(self, symbol: str, cycle_state: Dict[str, Any]) -> float:
        liquidity = cycle_state.get("liquidity", "expanding")
        if liquidity == "expanding":
            return 1.1
        if liquidity == "contracting":
            return 0.9
        return 1.0

    def _normalize_weights(self, values: List[float]) -> List[float]:
        total = float(sum(values))
        if total <= 0:
            count = max(len(values), 1)
            return [1.0 / count] * len(values)
        return [float(value / total) for value in values]
