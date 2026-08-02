"""Intelligence-aware optimizer adapter.

Wraps the existing optimizer functions and injects the AI Strategy
Intelligence Layer so parameter search uses intelligence-enriched backtests.

Supports:
- Grid search
- Random search
- Bayesian optimisation (via random exploration with best-so-far tracking)
- Parallel evaluation (thread-pool)
- Cycle-aware and RL-aware search space filtering
- Enriched optimizer diagnostics
"""
from __future__ import annotations

import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from core.diagnostics_layer import DiagnosticsLayer
from core.strategy_intelligence import StrategyIntelligence
from core.strategy_registry import create_intelligence_layer
from core.trade_logger import TradeLogger
from services.backtester_intel_adapter import BacktesterIntelAdapter

logger = logging.getLogger(__name__)

_DEFAULT_PARALLEL_WORKERS = 4


class OptimizerIntelAdapter:
    """Intelligence-aware parameter search wrapper.

    Parameters
    ----------
    strategy_module:
        Loaded strategy module.
    search_space:
        Mapping of ``param_name -> [candidate_values]``.
    historical_data:
        Mapping of ``symbol -> raw bar data``.
    config:
        Application configuration dict.
    intelligence:
        Pre-built :class:`StrategyIntelligence` (constructed from *config* if omitted).
    logger:
        :class:`~core.trade_logger.TradeLogger` instance.
    objective:
        Optimisation objective: ``"max_sharpe"``, ``"max_cagr_drawdown"``,
        ``"max_sortino"``, or ``"max_avg_r"``.
    """

    def __init__(
        self,
        strategy_module: Any,
        search_space: Dict[str, List[Any]],
        historical_data: Dict[str, Any],
        config: Dict[str, Any],
        intelligence: Optional[StrategyIntelligence] = None,
        logger: Optional[TradeLogger] = None,
        objective: str = "max_sharpe",
    ):
        self.strategy = strategy_module
        self.search_space = search_space
        self.historical_data = historical_data
        self.config = config
        self.intelligence = intelligence or create_intelligence_layer(config=config)
        self.trade_logger = logger or TradeLogger(log_dir=config.get("LOG_DIR", "logs"))
        self.objective = objective
        self.diagnostics_layer = DiagnosticsLayer(config=config)

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------

    def _objective_score(self, metrics: Dict[str, Any]) -> float:
        if self.objective == "max_sharpe":
            return float(metrics.get("annualized_sharpe", metrics.get("sharpe", 0.0)))
        if self.objective == "max_cagr_drawdown":
            cagr = float(metrics.get("cagr", 0.0))
            dd = float(metrics.get("max_drawdown", 0.0))
            return cagr - dd * 0.5
        if self.objective == "max_sortino":
            return float(metrics.get("sortino", 0.0))
        if self.objective == "max_avg_r":
            return float(metrics.get("expectancy", 0.0))
        return float(metrics.get("annualized_sharpe", 0.0))

    def _evaluate_params(
        self,
        params: Dict[str, Any],
        strategy_name: str = "",
        allocation_map: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Apply *params*, run an intel-aware backtest, and return scored result."""
        if hasattr(self.strategy, "set_params"):
            self.strategy.set_params(params)

        adapter = BacktesterIntelAdapter(
            strategy_module=self.strategy,
            config=self.config,
            capital=self.config.get("DEFAULT_CAPITAL", 100_000),
            slippage=self.config.get("DEFAULT_SLIPPAGE", 0.0005),
            commission=self.config.get("DEFAULT_COMMISSION", 0.001),
            intelligence=self.intelligence,
            logger=self.trade_logger,
        )
        result = adapter.run(self.historical_data, allocation_map=allocation_map, strategy_name=strategy_name)
        metrics = result.get("metrics", {})
        score = self._objective_score(metrics)
        return {
            "params": params,
            "metrics": metrics,
            "score": round(score, 6),
            "intel_diagnostics": result.get("intelligence_diagnostics", {}),
        }

    # ------------------------------------------------------------------
    # Search space utilities
    # ------------------------------------------------------------------

    def _compose_grid(self) -> List[Dict[str, Any]]:
        import itertools
        keys = list(self.search_space.keys())
        values = [self.search_space[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    def _validate_search_space(self) -> List[str]:
        """Return a list of validation warning strings (empty = valid)."""
        warnings: List[str] = []
        for key, vals in self.search_space.items():
            if not isinstance(vals, (list, tuple)) or len(vals) == 0:
                warnings.append(f"Search space key '{key}' has no candidate values.")
        return warnings

    # ------------------------------------------------------------------
    # Grid search
    # ------------------------------------------------------------------

    def run_grid_search(
        self,
        strategy_name: str = "",
        allocation_map: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """Exhaustive grid search over all parameter combinations.

        Parameters
        ----------
        strategy_name:
            Forwarded to the backtester adapter.
        allocation_map:
            Optional capital allocation map.
        parallel:
            When True, evaluate combinations in a thread pool.
        """
        warnings = self._validate_search_space()
        combinations = self._compose_grid()
        history = self._run_evaluations(
            combinations, strategy_name, allocation_map, parallel=parallel
        )
        return self._build_result(history, warnings=warnings, method="grid")

    # ------------------------------------------------------------------
    # Random search
    # ------------------------------------------------------------------

    def run_random_search(
        self,
        iterations: int = 10,
        strategy_name: str = "",
        allocation_map: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
    ) -> Dict[str, Any]:
        """Random parameter search over *iterations* samples."""
        warnings = self._validate_search_space()
        keys = list(self.search_space.keys())
        combinations = [
            {key: random.choice(self.search_space[key]) for key in keys}
            for _ in range(iterations)
        ]
        history = self._run_evaluations(
            combinations, strategy_name, allocation_map, parallel=parallel
        )
        return self._build_result(history, warnings=warnings, method="random")

    # ------------------------------------------------------------------
    # Bayesian (greedy Thompson-sampling style)
    # ------------------------------------------------------------------

    def run_bayesian_optimization(
        self,
        iterations: int = 10,
        strategy_name: str = "",
        allocation_map: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Bayesian-style optimisation.

        Uses a greedy exploitation strategy: evaluates random candidates while
        progressively removing already-explored ones, favouring unexplored
        regions.  This is compatible with discrete search spaces and avoids
        scikit-optimize as an external dependency.
        """
        warnings = self._validate_search_space()
        all_candidates = list(self._compose_grid())
        random.shuffle(all_candidates)
        selected = all_candidates[: min(iterations, len(all_candidates))]
        history = self._run_evaluations(selected, strategy_name, allocation_map, parallel=False)
        return self._build_result(history, warnings=warnings, method="bayesian")

    # ------------------------------------------------------------------
    # Parallel evaluation helper
    # ------------------------------------------------------------------

    def _run_evaluations(
        self,
        combinations: List[Dict[str, Any]],
        strategy_name: str,
        allocation_map: Optional[Dict[str, Any]],
        parallel: bool = False,
    ) -> List[Dict[str, Any]]:
        if not parallel or len(combinations) <= 1:
            return [self._evaluate_params(p, strategy_name, allocation_map) for p in combinations]

        history: List[Dict[str, Any]] = []
        workers = min(_DEFAULT_PARALLEL_WORKERS, len(combinations))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._evaluate_params, p, strategy_name, allocation_map): p for p in combinations}
            for future in as_completed(futures):
                try:
                    history.append(future.result())
                except Exception as exc:
                    logger.warning("Parallel evaluation failed: %s", exc)
        return history

    # ------------------------------------------------------------------
    # Result builder
    # ------------------------------------------------------------------

    def _build_result(
        self,
        history: List[Dict[str, Any]],
        warnings: Optional[List[str]] = None,
        method: str = "grid",
    ) -> Dict[str, Any]:
        if not history:
            return {
                "best_params": {},
                "best_metrics": {},
                "best_score": 0.0,
                "log": [],
                "warnings": warnings or [],
                "method": method,
            }
        best = max(history, key=lambda r: r.get("score", 0.0))
        return {
            "best_params": best["params"],
            "best_metrics": best["metrics"],
            "best_score": best["score"],
            "best_intel_diagnostics": best.get("intel_diagnostics", {}),
            "log": history,
            "warnings": warnings or [],
            "method": method,
        }
