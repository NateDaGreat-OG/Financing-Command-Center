import os
import re
import json
import numpy as np
import pandas as pd
from project.news.news_intel import analyze_news
from project.intel.ticker_selector import get_best_tickers
from project.data.massive_today_minutes import load_today_minute_bars
from flask import Flask, render_template, request, jsonify
from typing import Any, Dict, List, Optional
from project.news.news_intel import analyze_news
from project.events.event_aggregator import aggregate_events
from project.events.event_filter import filter_next_3_months
try:
    from .services.alpaca_client import AlpacaClient
    from .services.massive_client import MassiveClient
    from .services.cycle_data_client import CycleDataClient
    from .core.strategy_registry import list_styles, list_strategies_for_style, load_strategy, create_intelligence_layer
    from .core.optimizer import optimize_strategy, run_grid_search, run_random_search, run_bayesian_optimization
    from .core.search_spaces import DEFAULT_SEARCH_SPACES
    from .core.capital_manager import CapitalManager
    from .core.cycle_analyzer import CycleAnalyzer
    from .core.diagnostics_layer import DiagnosticsLayer
    from .core.portfolio_risk_engine import PortfolioRiskEngine
    from .core.execution_intelligence import ExecutionIntelligence
    from .core.strategy_governance import StrategyGovernance
    from .backtest.backtester import Backtester
    from .core.risk_manager import RiskManager
    from .core.trade_logger import TradeLogger
    from .services.backtester_intel_adapter import BacktesterIntelAdapter
    from .services.optimizer_intel_adapter import OptimizerIntelAdapter
    from .services.live_trading_orchestrator import LiveTradingOrchestrator
    _CONFIG_OBJECT = "project.config"

except ImportError:
    from services.alpaca_client import AlpacaClient
    from services.massive_client import MassiveClient
    from services.cycle_data_client import CycleDataClient
    from core.strategy_registry import list_styles, list_strategies_for_style, load_strategy, create_intelligence_layer
    from project.core.optimizer import run_grid_search, run_random_search, run_bayesian_optimization
    from core.search_spaces import DEFAULT_SEARCH_SPACES
    from core.capital_manager import CapitalManager
    from core.cycle_analyzer import CycleAnalyzer
    from core.diagnostics_layer import DiagnosticsLayer
    from core.portfolio_risk_engine import PortfolioRiskEngine
    from core.execution_intelligence import ExecutionIntelligence
    from core.strategy_governance import StrategyGovernance
    from backtest.backtester import Backtester
    from core.risk_manager import RiskManager
    from core.trade_logger import TradeLogger
    from services.backtester_intel_adapter import BacktesterIntelAdapter
    from services.optimizer_intel_adapter import OptimizerIntelAdapter
    from services.live_trading_orchestrator import LiveTradingOrchestrator
    _CONFIG_OBJECT = "config"

try:
    from .rl.trading_env import TradingEnv
    from .rl.dqn_agent import DQNAgent
    from .rl.rl_utils import save_model, load_model
except ImportError:
    try:
        from project.rl.trading_env import TradingEnv
        from project.rl.dqn_agent import DQNAgent
        from project.rl.rl_utils import save_model, load_model
    except ImportError:
        from rl.trading_env import TradingEnv
        from rl.dqn_agent import DQNAgent
        from rl.rl_utils import save_model, load_model

# Only allow symbols that look like real tickers (e.g. AAPL, BRK.B, SPY, GM).
# Length 1: single alphanum. Length 2-20: starts + ends with alphanum, dots allowed for
# share classes like BRK.B. Middle group {0,18} means 2-char symbols (GM, FB) are valid.
_SYMBOL_RE = re.compile(r'^[A-Za-z0-9]([A-Za-z0-9.]{0,18}[A-Za-z0-9])?$')
UNIVERSE = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "AMD", "GOOGL", "NFLX", "CRM"]

def _validate_symbol(symbol: str) -> bool:
    """Return True only when the symbol contains safe, ticker-like characters."""
    if not symbol or ".." in symbol:
        return False
    return bool(_SYMBOL_RE.match(symbol))


def _normalize_bar_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename Alpaca's single-letter bar columns to human-readable names."""
    return df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})

def data_loader(symbol):
    # simple wrapper around your existing loader
    data = load_today_minute_bars([symbol])
    return data.get(symbol, [])

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object(_CONFIG_OBJECT)

alpaca = AlpacaClient(
    api_key=app.config["ALPACA_API_KEY"],
    api_secret=app.config["ALPACA_API_SECRET"],
    base_url=app.config["ALPACA_BASE_URL"],
)

massive = MassiveClient(api_key=app.config["MASSIVE_API_KEY"], base_url=app.config["MASSIVE_BASE_URL"])
logger = TradeLogger(log_dir=app.config["LOG_DIR"])

cycle_client = None
if app.config.get("CYCLE_DATA_API_KEY") and app.config.get("CYCLE_DATA_BASE_URL"):
    cycle_client = CycleDataClient(
        api_key=app.config["CYCLE_DATA_API_KEY"],
        base_url=app.config["CYCLE_DATA_BASE_URL"],
    )

cycle_analyzer = CycleAnalyzer(macro_client=massive, config=app.config)
capital_manager = CapitalManager(config=app.config, risk_manager=RiskManager(app.config))

intelligence_layer = create_intelligence_layer(
    config=app.config,
    cycle_analyzer=cycle_analyzer,
    capital_manager=capital_manager,
    risk_manager=RiskManager(app.config),
)
portfolio_risk_engine = PortfolioRiskEngine(config=app.config)
execution_intelligence = ExecutionIntelligence(config=app.config)
strategy_governance = StrategyGovernance(config=app.config)
diagnostics_layer = DiagnosticsLayer(config=app.config)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/styles", methods=["GET"])
def api_styles():
    return jsonify(list_styles())

@app.route("/api/strategies/<style>", methods=["GET"])
def api_strategies(style):
    strategies = list_strategies_for_style(style)
    if strategies is None:
        return jsonify({"error": "Unknown style"}), 404
    return jsonify(strategies)

@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    payload = request.get_json() or {}
    strategy_name = payload.get("strategy")
    symbols = payload.get("symbols", [])
    capital = payload.get("capital", app.config["DEFAULT_CAPITAL"])
    slippage = payload.get("slippage", app.config["DEFAULT_SLIPPAGE"])
    commission = payload.get("commission", app.config["DEFAULT_COMMISSION"])

    if not strategy_name or not symbols:
        return jsonify({"error": "strategy and symbols are required"}), 400

    strategy = load_strategy(strategy_name)
    if not strategy:
        return jsonify({"error": "strategy not found"}), 404

    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = alpaca.get_historical(symbol, timeframe=app.config["BACKTEST_TIMEFRAME"])

    backtester = Backtester(
        strategy_module=strategy,
        capital=capital,
        slippage=slippage,
        commission=commission,
        risk_manager=RiskManager(app.config),
        logger=logger,
    )

    results = backtester.run(historical_data)
    return jsonify(results)

@app.route("/api/optimize", methods=["POST"])
def api_optimize():
    payload = request.get_json() or {}
    strategy_name = payload.get("strategy")
    symbols = payload.get("symbols", [])
    objective = payload.get("objective", "max_sharpe")
    method = payload.get("method", "grid")

    if not strategy_name or not symbols:
        return jsonify({"error": "strategy and symbols are required"}), 400

    search_space = DEFAULT_SEARCH_SPACES.get(strategy_name)
    if not search_space:
        return jsonify({"error": "search space not found for strategy"}), 404

    strategy_module = load_strategy(strategy_name)
    if not strategy_module:
        return jsonify({"error": "strategy not found"}), 404

    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = alpaca.get_historical(symbol, timeframe=app.config["BACKTEST_TIMEFRAME"])

    if method == "random":
        result = run_random_search(strategy_module, search_space, historical_data, app.config, logger, objective=objective)
    elif method == "bayesian":
        result = run_bayesian_optimization(strategy_module, search_space, historical_data, app.config, logger, objective=objective)
    else:
        result = run_grid_search(strategy_module, search_space, historical_data, app.config, logger, objective=objective)

    return jsonify(result)

@app.route("/api/cycles/analyze", methods=["POST"])
def api_cycles_analyze():
    payload = request.get_json() or {}
    symbols = payload.get("symbols", [])
    timeframe = payload.get("timeframe", app.config["BACKTEST_TIMEFRAME"])

    if not symbols:
        return jsonify({"error": "symbols are required"}), 400

    aggregated_cycle: Dict[str, Any] = {
        "trend": "sideways",
        "volatility": "low",
        "liquidity": "expanding",
        "macro": "risk_on",
        "intraday": "chop",
        "sector_rotation": {},
    }
    cycle_states = {}
    for symbol in symbols:
        raw_data = alpaca.get_historical(symbol, timeframe=timeframe)
        data = pd.DataFrame(raw_data.get("bars", []))
        if data.empty:
            continue
        data["t"] = pd.to_datetime(data["t"])
        data = data.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        cycle_state = cycle_analyzer.analyze(data, symbol=symbol)
        cycle_states[symbol] = cycle_state

    if cycle_states:
        aggregated_cycle = _aggregate_cycle_states(cycle_states)

    return jsonify({"cycle_state": aggregated_cycle, "symbol_states": cycle_states})


def _fetch_cycle_state(symbols: List[str], timeframe: str) -> Dict[str, Any]:
    cycle_states: Dict[str, Any] = {}
    for symbol in symbols:
        raw_data = alpaca.get_historical(symbol, timeframe=timeframe)
        data = pd.DataFrame(raw_data.get("bars", []))
        if data.empty:
            continue
        data["t"] = pd.to_datetime(data["t"])
        data = data.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        cycle_states[symbol] = cycle_analyzer.analyze(data, symbol=symbol)

    aggregated_cycle = _aggregate_cycle_states(cycle_states) if cycle_states else {
        "trend": "sideways",
        "volatility": "low",
        "liquidity": "expanding",
        "macro": "risk_on",
        "intraday": "chop",
        "sector_rotation": {},
    }
    return {"cycle_state": aggregated_cycle, "symbol_states": cycle_states}


def _aggregate_cycle_states(cycle_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    trends = [state.get("trend", "sideways") for state in cycle_states.values()]
    volatility = [state.get("volatility", "low") for state in cycle_states.values()]
    liquidity = [state.get("liquidity", "expanding") for state in cycle_states.values()]
    macros = [state.get("macro", "risk_on") for state in cycle_states.values()]
    intraday = [state.get("intraday", "chop") for state in cycle_states.values()]

    def majority(values):
        if not values:
            return values
        return max(set(values), key=values.count)

    return {
        "trend": majority(trends),
        "volatility": majority(volatility),
        "liquidity": majority(liquidity),
        "macro": majority(macros),
        "intraday": majority(intraday),
        "sector_rotation": {"symbols": list(cycle_states.keys())},
    }


@app.route("/api/capital/allocate", methods=["POST"])
def api_capital_allocate():
    payload = request.get_json() or {}
    strategies = payload.get("strategies", [])
    symbols = payload.get("symbols", [])
    timeframe = payload.get("timeframe", app.config["BACKTEST_TIMEFRAME"])

    if not strategies or not symbols:
        return jsonify({"error": "strategies and symbols are required"}), 400

    cycle_state = {}
    if payload.get("cycle_state"):
        cycle_state = payload.get("cycle_state")
    else:
        cycle_resp = _fetch_cycle_state(symbols, timeframe)
        cycle_state = cycle_resp.get("cycle_state", {})

    strategy_metrics = {}
    for strategy_name in strategies:
        strategy_module = load_strategy(strategy_name)
        if not strategy_module:
            strategy_metrics[strategy_name] = {"sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0}
            continue
        historical_data = {}
        for symbol in symbols:
            raw_data = alpaca.get_historical(symbol, timeframe=timeframe)
            historical_data[symbol] = raw_data
        backtester = Backtester(
            strategy_module=strategy_module,
            capital=app.config["DEFAULT_CAPITAL"],
            slippage=app.config.get("DEFAULT_SLIPPAGE", 0.0005),
            commission=app.config.get("DEFAULT_COMMISSION", 0.001),
            risk_manager=RiskManager(app.config),
            logger=logger,
        )
        results = backtester.run(historical_data)
        strategy_metrics[strategy_name] = results.get("metrics", {})

    rl_metrics = {}
    for symbol in symbols:
        raw_data = alpaca.get_historical(symbol, timeframe=timeframe)
        df = pd.DataFrame(raw_data.get("bars", []))
        if not df.empty:
            rl_metrics[symbol] = _derive_rl_metrics(df)
        else:
            rl_metrics[symbol] = {"average_reward": 0.0, "stability": 0.5}

    allocation_map = capital_manager.allocate(
        strategies=strategies,
        symbols=symbols,
        strategy_metrics=strategy_metrics,
        rl_metrics=rl_metrics,
        cycle_state=cycle_state,
    )

    return jsonify({"allocation": allocation_map, "strategy_metrics": strategy_metrics, "rl_metrics": rl_metrics, "cycle_state": cycle_state})

@app.route("/news")
def news_feed():
    # Your universe of tickers
    ticker_list = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]

    # Load performance history (optional for RL refinement)
    performance_history = {}  # or load from backtest_results

    intel = analyze_news(ticker_list, performance_history)

    return render_template("news.html", intel=intel)

@app.route("/auto_select_tickers", methods=["POST"])
def auto_select_tickers():
    strategy_name = request.form.get("strategy")
    
    if not strategy_name:
        return jsonify({"error": "strategy is required"}), 400

    # TODO: load performance history from your backtest_results
    performance_history = {}  # { "AAPL": sharpe, ... }

    best = get_best_tickers(
        strategy_name=strategy_name,
        universe=UNIVERSE,
        data_loader=data_loader,
        performance_history=performance_history,
        top_n=3,
    )

    # Return JSON or re-render template with symbols pre-filled
    return jsonify({"symbols": best})

def _derive_rl_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    df = _normalize_bar_columns(data.copy())
    if df.empty:
        return {"average_reward": 0.0, "stability": 0.5}
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean().ffill()
    avg_atr = float(df["atr"].iloc[-1])
    volatility = float(df["atr"].std())
    average_reward = float(np.tanh(avg_atr / max(df["close"].iloc[-1], 1.0)))
    stability = float(1.0 / (1.0 + volatility))
    return {"average_reward": average_reward, "stability": float(np.clip(stability, 0.1, 1.0))}

_RL_MODEL_DIR = os.path.join("instance", "rl_models")
# Resolved once at import time so containment checks are stable regardless of cwd changes.
_RL_MODEL_DIR_ABS = os.path.realpath(os.path.abspath(_RL_MODEL_DIR))


def _safe_model_path(symbol: str) -> Optional[str]:
    """Return the absolute model path for *symbol*, or None if it would escape the model dir.

    This is defense-in-depth on top of ``_validate_symbol``: even if a symbol somehow
    passed validation, ``os.path.realpath`` resolves any remaining traversal sequences
    and the containment check prevents writes/reads outside ``_RL_MODEL_DIR``.
    """
    candidate = os.path.realpath(os.path.abspath(os.path.join(_RL_MODEL_DIR, f"dqn_{symbol}.pth")))
    if not candidate.startswith(_RL_MODEL_DIR_ABS + os.sep):
        return None
    return candidate


@app.route("/api/rl/train", methods=["POST"])
def api_rl_train():
    payload = request.get_json() or {}
    agent_type = payload.get("agent", "dqn")
    symbols = payload.get("symbols", [])
    episodes = int(payload.get("episodes", 50))
    timeframe = payload.get("timeframe", app.config["BACKTEST_TIMEFRAME"])

    if agent_type != "dqn":
        return jsonify({"error": "Only dqn agent is supported"}), 400
    if not symbols:
        return jsonify({"error": "symbols is required"}), 400

    os.makedirs(_RL_MODEL_DIR, exist_ok=True)
    results: Dict[str, Any] = {}
    for symbol in symbols:
        if not _validate_symbol(symbol):
            results[symbol] = {"error": "invalid symbol"}
            continue
        model_path = _safe_model_path(symbol)
        if model_path is None:
            results[symbol] = {"error": "invalid symbol"}
            continue
        raw_data = alpaca.get_historical(symbol, timeframe=timeframe)
        df = pd.DataFrame(raw_data.get("bars", []))
        if df.empty:
            results[symbol] = {"error": "no data available"}
            continue
        df = _normalize_bar_columns(df)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        env = TradingEnv(data=df, capital=app.config["DEFAULT_CAPITAL"])
        agent = DQNAgent(state_dim=17, action_dim=3)
        train_result = agent.train(env, episodes=episodes, checkpoint_interval=0)
        agent.save_model(model_path)
        results[symbol] = {
            "episodes_trained": train_result["episodes"],
            "best_reward": train_result["best_reward"],
            "history": train_result["history"],
            "model_saved": model_path,
        }

    return jsonify(results)


@app.route("/api/rl/models", methods=["GET"])
def api_rl_models():
    registry: Dict[str, Any] = {}
    if os.path.isdir(_RL_MODEL_DIR):
        for filename in os.listdir(_RL_MODEL_DIR):
            if filename.startswith("dqn_") and filename.endswith(".pth"):
                symbol = filename[4:-4]
                filepath = os.path.join(_RL_MODEL_DIR, filename)
                stat = os.stat(filepath)
                registry[symbol] = {
                    "size_bytes": stat.st_size,
                    "modified": stat.st_mtime,
                }
    return jsonify(registry)


@app.route("/api/rl/models/<symbol>", methods=["DELETE"])
def api_rl_model_delete(symbol: str):
    if not _validate_symbol(symbol):
        return jsonify({"error": "invalid symbol"}), 400
    model_path = _safe_model_path(symbol)
    if model_path is None:
        return jsonify({"error": "invalid symbol"}), 400
    if not os.path.exists(model_path):
        return jsonify({"error": "model not found"}), 404
    os.remove(model_path)
    return jsonify({"deleted": symbol})


@app.route("/api/rl/run", methods=["POST"])
def api_rl_run():
    payload = request.get_json() or {}
    agent_type = payload.get("agent")
    symbol = payload.get("symbol")

    if agent_type != "dqn":
        return jsonify({"error": "Only dqn agent is supported currently"}), 400
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400
    if not _validate_symbol(symbol):
        return jsonify({"error": "invalid symbol"}), 400

    model_path = _safe_model_path(symbol)
    if model_path is None:
        return jsonify({"error": "invalid symbol"}), 400
    if not os.path.exists(model_path):
        return jsonify({"error": "model not found"}), 404

    agent = DQNAgent(state_dim=17, action_dim=3)
    agent.load_model(model_path)

    raw_data = alpaca.get_intraday(symbol, interval=app.config["LIVE_INTERVAL"])
    data = _normalize_bar_columns(pd.DataFrame(raw_data.get("bars", [])))
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.set_index("timestamp")

    env = TradingEnv(data=data, capital=app.config["DEFAULT_CAPITAL"])
    state = env.reset()
    actions = []
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        actions.append({"step": env.current_step, "action": action, "info": info})
        state = next_state

    return jsonify({"symbol": symbol, "actions": actions, "final_equity": env.equity})

@app.route("/api/live", methods=["POST"])
def api_live():
    payload = request.get_json() or {}
    strategy_name = payload.get("strategy")
    symbols = payload.get("symbols", [])
    if not strategy_name or not symbols:
        return jsonify({"error": "strategy and symbols are required"}), 400

    strategy = load_strategy(strategy_name)
    if not strategy:
        return jsonify({"error": "strategy not found"}), 404

    live_signals = []
    for symbol in symbols:
        data = alpaca.get_intraday(symbol, interval=app.config["LIVE_INTERVAL"])
        candidates = strategy.scan_candidates([symbol])
        signals = strategy.generate_signals(data)
        orders = strategy.execute_signals(signals)
        live_signals.append({"symbol": symbol, "signals": signals, "orders": orders})
        logger.log_signals(symbol, signals)

    account = alpaca.get_account()
    positions = alpaca.get_positions()
    return jsonify({"status": "live started", "account": account, "positions": positions, "results": live_signals})

@app.route("/api/intel/backtest", methods=["POST"])
def api_intel_backtest():
    """Intelligence-enriched backtest endpoint.

    Accepts the same payload as /api/backtest but runs signals through the
    full StrategyIntelligence pipeline and returns expanded metrics plus
    equity/drawdown curves and diagnostics.
    """
    payload = request.get_json() or {}
    strategy_name = payload.get("strategy")
    symbols = payload.get("symbols", [])
    capital = float(payload.get("capital", app.config["DEFAULT_CAPITAL"]))
    slippage = float(payload.get("slippage", app.config.get("DEFAULT_SLIPPAGE", 0.0005)))
    commission = float(payload.get("commission", app.config.get("DEFAULT_COMMISSION", 0.001)))

    if not strategy_name or not symbols:
        return jsonify({"error": "strategy and symbols are required"}), 400

    strategy = load_strategy(strategy_name)
    if not strategy:
        return jsonify({"error": "strategy not found"}), 404

    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = alpaca.get_historical(symbol, timeframe=app.config["BACKTEST_TIMEFRAME"])

    adapter = BacktesterIntelAdapter(
        strategy_module=strategy,
        config=app.config,
        capital=capital,
        slippage=slippage,
        commission=commission,
        intelligence=intelligence_layer,
        logger=logger,
    )
    result = adapter.run(historical_data, strategy_name=strategy_name)
    return jsonify(result)


@app.route("/api/intel/optimize", methods=["POST"])
def api_intel_optimize():
    """Intelligence-enriched optimizer endpoint.

    Supports ``method`` = ``"grid"``, ``"random"``, or ``"bayesian"``.
    Uses intelligence-aware backtests for all evaluations.
    """
    payload = request.get_json() or {}
    strategy_name = payload.get("strategy")
    symbols = payload.get("symbols", [])
    objective = payload.get("objective", "max_sharpe")
    method = payload.get("method", "grid")
    iterations = int(payload.get("iterations", 10))
    parallel = bool(payload.get("parallel", False))

    if not strategy_name or not symbols:
        return jsonify({"error": "strategy and symbols are required"}), 400

    search_space = DEFAULT_SEARCH_SPACES.get(strategy_name)
    if not search_space:
        return jsonify({"error": "search space not found for strategy"}), 404

    strategy = load_strategy(strategy_name)
    if not strategy:
        return jsonify({"error": "strategy not found"}), 404

    historical_data = {}
    for symbol in symbols:
        historical_data[symbol] = alpaca.get_historical(symbol, timeframe=app.config["BACKTEST_TIMEFRAME"])

    optimizer = OptimizerIntelAdapter(
        strategy_module=strategy,
        search_space=search_space,
        historical_data=historical_data,
        config=app.config,
        intelligence=intelligence_layer,
        logger=logger,
        objective=objective,
    )

    if method == "random":
        result = optimizer.run_random_search(iterations=iterations, strategy_name=strategy_name, parallel=parallel)
    elif method == "bayesian":
        result = optimizer.run_bayesian_optimization(iterations=iterations, strategy_name=strategy_name)
    else:
        result = optimizer.run_grid_search(strategy_name=strategy_name, parallel=parallel)

    return jsonify(result)


@app.route("/api/intel/live", methods=["POST"])
def api_intel_live():
    """Intelligence-enriched live trading endpoint.

    Runs the full intelligence pipeline (AI signals + execution intelligence +
    portfolio risk + governance) for each symbol and returns signals and diagnostics.
    Orders are only submitted when ``dry_run`` is False in the payload.
    """
    payload = request.get_json() or {}
    strategy_name = payload.get("strategy")
    symbols = payload.get("symbols", [])
    dry_run = bool(payload.get("dry_run", True))

    if not strategy_name or not symbols:
        return jsonify({"error": "strategy and symbols are required"}), 400

    strategy = load_strategy(strategy_name)
    if not strategy:
        return jsonify({"error": "strategy not found"}), 404

    orchestrator = LiveTradingOrchestrator(
        strategy_module=strategy,
        alpaca=alpaca,
        config=app.config,
        intelligence=intelligence_layer,
        portfolio_risk=portfolio_risk_engine,
        execution=execution_intelligence,
        governance=strategy_governance,
        trade_logger=logger,
        dry_run=dry_run,
    )

    result = orchestrator.run(symbols=symbols)
    return jsonify(result)


@app.route("/api/intel/diagnostics", methods=["POST"])
def api_intel_diagnostics():
    """Unified diagnostics endpoint.

    Loads strategy + intelligence state, runs the intelligence pipeline, and
    returns a full diagnostics JSON payload covering AI signals, cycle state,
    RL, risk, capital, execution, and governance.
    """
    payload = request.get_json() or {}
    strategy_name = payload.get("strategy")
    symbols = payload.get("symbols", [])
    timeframe = payload.get("timeframe", app.config["BACKTEST_TIMEFRAME"])

    if not strategy_name or not symbols:
        return jsonify({"error": "strategy and symbols are required"}), 400

    strategy = load_strategy(strategy_name)
    if not strategy:
        return jsonify({"error": "strategy not found"}), 404

    # Fetch cycle state and data
    cycle_resp = _fetch_cycle_state(symbols, timeframe)
    cycle_state = cycle_resp.get("cycle_state", {})

    # Run intelligence pipeline on the first symbol for diagnostics
    symbol = symbols[0]
    raw_data = alpaca.get_historical(symbol, timeframe=timeframe)
    df = pd.DataFrame(raw_data.get("bars", []))
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"])
        df = _normalize_bar_columns(df)

    intel_result: Dict[str, Any] = {}
    if not df.empty:
        intel_result = intelligence_layer.run(
            strategy_module=strategy,
            data=df,
            symbol=symbol,
            strategy_name=strategy_name,
            cycle_state=cycle_state,
            capital=app.config["DEFAULT_CAPITAL"],
        )

    # Execution diagnostics
    signals = intel_result.get("signals", [])
    exec_diag = execution_intelligence.diagnostics(signals, cycle_state=cycle_state)

    # Capital allocation diagnostics
    allocation_map: Dict[str, Any] = {}

    # Governance diagnostics
    gov_snap = strategy_governance.diagnostics([strategy_name])

    # Portfolio risk (placeholder – no price history available here)
    port_risk_diag: Dict[str, Any] = {
        "dynamic_leverage": portfolio_risk_engine.dynamic_leverage(pd.Series(dtype=float), cycle_state),
        "target_vol": portfolio_risk_engine.target_vol,
        "max_portfolio_drawdown": portfolio_risk_engine.max_portfolio_drawdown,
    }

    diag = diagnostics_layer.build(
        intelligence_result=intel_result,
        cycle_state=intel_result.get("cycle_state", cycle_state),
        cycle_params=intel_result.get("cycle_params"),
        rl_signal=intel_result.get("rl_signal"),
        portfolio_risk=port_risk_diag,
        allocation_map=allocation_map,
        execution_info=exec_diag,
        governance_snapshot=gov_snap,
    )

    return jsonify(diag)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

@app.route("/events")
def events():
    events = aggregate_events()
    upcoming = filter_next_3_months(events)
    return render_template("events.html", events=upcoming)