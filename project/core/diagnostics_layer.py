"""Unified diagnostics layer.

Aggregates diagnostic information from every subsystem — AI signals,
cycle analysis, RL agent, risk management, capital allocation, execution,
and governance — into a single JSON-serialisable output suitable for the
strategy dashboard UI.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DiagnosticsLayer:
    """Collects and structures diagnostics from all intelligence subsystems.

    Parameters
    ----------
    config:
        Optional application configuration dict (reserved for future use).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    # ------------------------------------------------------------------
    # Per-subsystem formatters
    # ------------------------------------------------------------------

    def ai_diagnostics(
        self,
        intelligence_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Extract AI signal diagnostics from a StrategyIntelligence result dict."""
        signals = intelligence_result.get("signals", [])
        metadata = intelligence_result.get("metadata", {})
        return {
            "signal_count": len(signals),
            "raw_signal_count": metadata.get("raw_signal_count", 0),
            "avg_ai_score": round(
                sum(float(s.get("ai_score", 0)) for s in signals) / max(len(signals), 1), 4
            ),
            "top_signals": [
                {
                    "symbol": s.get("symbol", ""),
                    "side": s.get("side", ""),
                    "ai_score": s.get("ai_score", 0),
                    "entry_price": s.get("entry_price", 0),
                    "size": s.get("size", 0),
                }
                for s in signals[:5]
            ],
            "capital": metadata.get("capital"),
            "atr": metadata.get("atr"),
            "strategy": metadata.get("strategy", ""),
        }

    def cycle_diagnostics(
        self,
        cycle_state: Dict[str, Any],
        cycle_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format cycle state for the diagnostics payload."""
        return {
            "trend": cycle_state.get("trend", "sideways"),
            "volatility": cycle_state.get("volatility", "low"),
            "liquidity": cycle_state.get("liquidity", "expanding"),
            "macro": cycle_state.get("macro", "risk_on"),
            "intraday": cycle_state.get("intraday", "chop"),
            "sector_rotation": cycle_state.get("sector_rotation", {}),
            "cycle_params": cycle_params or {},
        }

    def rl_diagnostics(
        self,
        rl_signal: Optional[Dict[str, Any]],
        rl_action_history: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Format RL signal and action history for the diagnostics payload."""
        return {
            "rl_signal": rl_signal or {},
            "rl_action": rl_signal.get("action") if rl_signal else None,
            "rl_confidence": rl_signal.get("confidence") if rl_signal else None,
            "rl_ai_score": rl_signal.get("ai_score") if rl_signal else None,
            "action_history_count": len(rl_action_history or []),
            "recent_actions": (rl_action_history or [])[-5:],
        }

    def risk_diagnostics(
        self,
        portfolio_risk: Optional[Dict[str, Any]] = None,
        signal_risk_info: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Format portfolio risk engine output for the diagnostics payload."""
        portfolio_risk = portfolio_risk or {}
        return {
            "dynamic_leverage": portfolio_risk.get("dynamic_leverage"),
            "current_drawdown": portfolio_risk.get("current_drawdown"),
            "drawdown_breached": portfolio_risk.get("drawdown_breached", False),
            "target_vol": portfolio_risk.get("target_vol"),
            "max_portfolio_drawdown": portfolio_risk.get("max_portfolio_drawdown"),
            "beta": portfolio_risk.get("beta", {}),
            "signal_count": len(signal_risk_info or []),
        }

    def capital_diagnostics(
        self,
        allocation_map: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format capital allocation map for the diagnostics payload."""
        allocation_map = allocation_map or {}
        strategies = list(allocation_map.keys())
        total_allocated = 0.0
        for strategy_alloc in allocation_map.values():
            for symbol_alloc in strategy_alloc.values() if isinstance(strategy_alloc, dict) else []:
                if isinstance(symbol_alloc, dict):
                    total_allocated += float(symbol_alloc.get("allocated_capital", 0.0))
        return {
            "strategies": strategies,
            "total_allocated": round(total_allocated, 2),
            "allocation_summary": {
                strategy: {
                    sym: alloc.get("allocated_capital", 0.0)
                    for sym, alloc in symbols.items()
                }
                for strategy, symbols in allocation_map.items()
                if isinstance(symbols, dict)
            },
        }

    def execution_diagnostics(
        self,
        execution_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format execution intelligence output for the diagnostics payload."""
        execution_info = execution_info or {}
        return {
            "signal_count": execution_info.get("signal_count", 0),
            "total_estimated_slippage": execution_info.get("total_estimated_slippage", 0.0),
            "execution_window": execution_info.get("execution_window", {}),
            "slippage_bps": execution_info.get("slippage_bps"),
            "commission_pct": execution_info.get("commission_pct"),
        }

    def governance_diagnostics(
        self,
        governance_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Format strategy governance state for the diagnostics payload."""
        governance_snapshot = governance_snapshot or {}
        disabled = [k for k, v in governance_snapshot.items() if not v.get("enabled", True)]
        reduced = [
            k for k, v in governance_snapshot.items()
            if v.get("capital_factor", 1.0) < 1.0 and v.get("enabled", True)
        ]
        return {
            "strategy_count": len(governance_snapshot),
            "disabled_strategies": disabled,
            "capital_reduced_strategies": reduced,
            "governance_state": governance_snapshot,
        }

    # ------------------------------------------------------------------
    # Unified output
    # ------------------------------------------------------------------

    def build(
        self,
        intelligence_result: Optional[Dict[str, Any]] = None,
        cycle_state: Optional[Dict[str, Any]] = None,
        cycle_params: Optional[Dict[str, Any]] = None,
        rl_signal: Optional[Dict[str, Any]] = None,
        portfolio_risk: Optional[Dict[str, Any]] = None,
        allocation_map: Optional[Dict[str, Any]] = None,
        execution_info: Optional[Dict[str, Any]] = None,
        governance_snapshot: Optional[Dict[str, Any]] = None,
        rl_action_history: Optional[List[Dict[str, Any]]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build and return the unified diagnostics JSON payload.

        All arguments are optional; omitted subsystems produce empty dicts.
        """
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "ai": self.ai_diagnostics(intelligence_result or {}),
            "cycle": self.cycle_diagnostics(cycle_state or {}, cycle_params),
            "rl": self.rl_diagnostics(rl_signal, rl_action_history),
            "risk": self.risk_diagnostics(portfolio_risk),
            "capital": self.capital_diagnostics(allocation_map),
            "execution": self.execution_diagnostics(execution_info),
            "governance": self.governance_diagnostics(governance_snapshot),
            **(extra or {}),
        }
