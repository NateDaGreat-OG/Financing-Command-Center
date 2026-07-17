import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from typing import Any, Dict, List, Optional
from services.alpaca_client import AlpacaClient
from services.massive_client import MassiveClient
from services.cycle_data_client import CycleDataClient
from core.strategy_registry import list_styles, list_strategies_for_style, load_strategy
from core.optimizer import optimize_strategy, run_grid_search, run_random_search, run_bayesian_optimization
from core.search_spaces import DEFAULT_SEARCH_SPACES
from core.capital_manager import CapitalManager
from core.cycle_analyzer import CycleAnalyzer
from backtest.backtester import Backtester
from core.risk_manager import RiskManager
from core.trade_logger import TradeLogger
from project.rl.trading_env import TradingEnv
from project.rl.dqn_agent import DQNAgent
from project.rl.rl_utils import save_model, load_model

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object("config")

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


def _derive_rl_metrics(data: pd.DataFrame) -> Dict[str, Any]:
    df = data.copy()
    if df.empty:
        return {"average_reward": 0.0, "stability": 0.5}
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean().ffill()
    avg_atr = float(df["atr"].iloc[-1])
    volatility = float(df["atr"].std())
    average_reward = float(np.tanh(avg_atr / max(df["close"].iloc[-1], 1.0)))
    stability = float(1.0 / (1.0 + volatility))
    return {"average_reward": average_reward, "stability": float(np.clip(stability, 0.1, 1.0))}

_RL_MODEL_DIR = os.path.join("instance", "rl_models")


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
        raw_data = alpaca.get_historical(symbol, timeframe=timeframe)
        df = pd.DataFrame(raw_data.get("bars", []))
        if df.empty:
            results[symbol] = {"error": "no data available"}
            continue
        df = df.rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
        env = TradingEnv(data=df, capital=app.config["DEFAULT_CAPITAL"])
        agent = DQNAgent(state_dim=17, action_dim=3)
        model_path = os.path.join(_RL_MODEL_DIR, f"dqn_{symbol}.pth")
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
    model_path = os.path.join(_RL_MODEL_DIR, f"dqn_{symbol}.pth")
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

    model_path = os.path.join(_RL_MODEL_DIR, f"dqn_{symbol}.pth")
    if not os.path.exists(model_path):
        return jsonify({"error": "model not found"}), 404

    agent = DQNAgent(state_dim=17, action_dim=3)
    agent.load_model(model_path)

    raw_data = alpaca.get_intraday(symbol, interval=app.config["LIVE_INTERVAL"])
    data = pd.DataFrame(raw_data.get("bars", []))
    data["t"] = pd.to_datetime(data["t"])
    data = data.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}).set_index("timestamp")

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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
