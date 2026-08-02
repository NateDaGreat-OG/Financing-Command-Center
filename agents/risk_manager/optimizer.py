"""Hyperparameter optimization utilities for strategy tuning."""
import itertools
import importlib
import random
from typing import Any, Dict, List

from core.search_spaces import DEFAULT_SEARCH_SPACES
try:
    from ..backtest.backtester import Backtester
    from .risk_manager import RiskManager
    from .trade_logger import TradeLogger
    from ..services.alpaca_client import AlpacaClient
except ImportError:
    from backtest.backtester import Backtester
    from core.risk_manager import RiskManager
    from core.trade_logger import TradeLogger
    from services.alpaca_client import AlpacaClient


def _objective_score(metrics: Dict[str, Any], objective: str) -> float:
    if objective == "max_sharpe":
        return metrics.get("sharpe", 0.0)
    if objective == "max_cagr_drawdown":
        drawdown = metrics.get("max_drawdown", 0.0)
        cagr = metrics.get("cagr", 0.0)
        return cagr - (drawdown * 0.5)
    if objective == "max_avg_r":
        return metrics.get("avg_r", 0.0)
    return metrics.get("sharpe", 0.0)


def _compose_search_space(search_space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]
    combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    return combinations


def evaluate_params(strategy_module, params: Dict[str, Any], data: dict, config: dict, logger: TradeLogger) -> dict:
    if hasattr(strategy_module, "set_params"):
        strategy_module.set_params(params)
    backtester = Backtester(
        strategy_module=strategy_module,
        capital=config.get("DEFAULT_CAPITAL", 100000),
        slippage=config.get("DEFAULT_SLIPPAGE", 0.0005),
        commission=config.get("DEFAULT_COMMISSION", 0.001),
        risk_manager=RiskManager(config),
        logger=logger,
    )
    results = backtester.run(data)
    trades = results.get("trade_log", [])
    avg_r = _calculate_avg_r(trades)
    metrics = results["metrics"].copy()
    metrics["avg_r"] = avg_r
    return {"params": params, "metrics": metrics}


def run_grid_search(strategy_module, search_space: Dict[str, List[Any]], data: dict, config: dict, logger: TradeLogger, objective: str = "max_sharpe") -> dict:
    combinations = _compose_search_space(search_space)
    best = None
    history = []

    for params in combinations:
        result = evaluate_params(strategy_module, params, data, config, logger)
        score = _objective_score(result["metrics"], objective)
        result["score"] = score
        history.append(result)

        if best is None or score > best["score"]:
            best = result

    # SAFETY CHECK AFTER LOOP
    if best is None:
        return {
            "best_params": None,
            "best_metrics": None,
            "log": history,
            "error": "No valid parameter combination produced metrics."
        }

    return {"best_params": best["params"], "best_metrics": best["metrics"], "log": history}

def run_random_search(strategy_module, search_space: Dict[str, List[Any]], data: dict, config: dict, logger: TradeLogger, iterations: int = 10, objective: str = "max_sharpe") -> dict:
    keys = list(search_space.keys())
    best = None
    history = []

    for _ in range(iterations):
        params = {key: random.choice(search_space[key]) for key in keys}
        result = evaluate_params(strategy_module, params, data, config, logger)
        score = _objective_score(result["metrics"], objective)
        result["score"] = score
        history.append(result)

        if best is None or score > best["score"]:
            best = result

    # SAFETY CHECK AFTER LOOP
    if best is None:
        return {
            "best_params": None,
            "best_metrics": None,
            "log": history,
            "error": "Random search produced no valid results."
        }

    return {"best_params": best["params"], "best_metrics": best["metrics"], "log": history}


def run_bayesian_optimization(strategy_module, search_space: Dict[str, List[Any]], data: dict, config: dict, logger: TradeLogger, iterations: int = 10, objective: str = "max_sharpe") -> dict:
    best = None
    history = []
    choices = _compose_search_space(search_space)

    for _ in range(min(iterations, len(choices))):
        params = random.choice(choices)
        result = evaluate_params(strategy_module, params, data, config, logger)
        score = _objective_score(result["metrics"], objective)
        result["score"] = score
        history.append(result)

        if best is None or score > best["score"]:
            best = result

        choices.remove(params)

    # SAFETY CHECK AFTER LOOP
    if best is None:
        return {
            "best_params": None,
            "best_metrics": None,
            "log": history,
            "error": "Bayesian optimization produced no valid results."
        }

    return {"best_params": best["params"], "best_metrics": best["metrics"], "log": history}


def optimize_strategy(strategy_name: str, symbols: list, objective: str, search_space: dict, config: dict, logger: TradeLogger, alpaca_client: AlpacaClient) -> dict:
    strategy_module = None
    module_names = []
    if __package__:
        module_names.append(f"{__package__.rsplit('.', 1)[0]}.strategies.{strategy_name}")
    module_names.append(f"strategies.{strategy_name}")
    for module_name in module_names:
        try:
            strategy_module = importlib.import_module(module_name)
            break
        except ImportError:
            continue
    if not strategy_module:
        return {"error": "strategy not found"}

    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = alpaca_client.get_historical(symbol, timeframe=config.get("BACKTEST_TIMEFRAME", "1D"))

    result = run_grid_search(strategy_module, search_space, historical_data, config, logger, objective=objective)
    return result


def _calculate_avg_r(trades: list) -> float:
    if not trades:
        return 0.0
    returns = []
    for trade in trades:
        entry = trade.get("entry_price", 0)
        exit = trade.get("exit_price", 0)
        size = trade.get("size", 1)
        if entry:
            rtn = (exit - entry) / entry if trade.get("side") == "long" else (entry - exit) / entry
            returns.append(rtn)
    return round(sum(returns) / len(returns) * 100, 2) if returns else 0.0
