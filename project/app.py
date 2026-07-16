import os
import json
import pandas as pd
from flask import Flask, render_template, request, jsonify
from services.alpaca_client import AlpacaClient
from services.massive_client import MassiveClient
from core.strategy_registry import list_styles, list_strategies_for_style, load_strategy
from core.optimizer import optimize_strategy, run_grid_search, run_random_search, run_bayesian_optimization
from core.search_spaces import DEFAULT_SEARCH_SPACES
from backtest.backtester import Backtester
from core.risk_manager import RiskManager
from core.trade_logger import TradeLogger
from rl.trading_env import TradingEnv
from rl.dqn_agent import DQNAgent
from rl.rl_utils import save_model, load_model

app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object("config")

alpaca = AlpacaClient(
    api_key=app.config["ALPACA_API_KEY"],
    api_secret=app.config["ALPACA_API_SECRET"],
    base_url=app.config["ALPACA_BASE_URL"],
)

massive = MassiveClient(api_key=app.config["MASSIVE_API_KEY"], base_url=app.config["MASSIVE_BASE_URL"])
logger = TradeLogger(log_dir=app.config["LOG_DIR"])

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

@app.route("/api/rl/train", methods=["POST"])
def api_rl_train():
    payload = request.get_json() or {}
    agent_type = payload.get("agent")
    symbols = payload.get("symbols", [])
    episodes = int(payload.get("episodes", 50))
    timeframe = payload.get("timeframe", app.config["BACKTEST_TIMEFRAME"])

    if agent_type != "dqn":
        return jsonify({"error": "Only dqn agent is supported currently"}), 400
    if not symbols:
        return jsonify({"error": "symbols are required"}), 400

    symbol = symbols[0]
    raw_data = alpaca.get_historical(symbol, timeframe=timeframe)
    data = pd.DataFrame(raw_data.get("bars", []))
    data["t"] = pd.to_datetime(data["t"])
    data = data.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}).set_index("timestamp")

    env = TradingEnv(data=data, capital=app.config["DEFAULT_CAPITAL"])
    agent = DQNAgent(state_dim=14, action_dim=3)
    agent.train(env, episodes=episodes)
    model_path = os.path.join("instance", "rl_models", f"dqn_{symbol}.pkl")
    agent.save_model(model_path)

    return jsonify({"status": "trained", "symbol": symbol, "episodes": episodes, "model_path": model_path})

@app.route("/api/rl/run", methods=["POST"])
def api_rl_run():
    payload = request.get_json() or {}
    agent_type = payload.get("agent")
    symbol = payload.get("symbol")

    if agent_type != "dqn":
        return jsonify({"error": "Only dqn agent is supported currently"}), 400
    if not symbol:
        return jsonify({"error": "symbol is required"}), 400

    model_path = os.path.join("instance", "rl_models", f"dqn_{symbol}.pkl")
    if not os.path.exists(model_path):
        return jsonify({"error": "model not found"}), 404

    agent = DQNAgent(state_dim=14, action_dim=3)
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
