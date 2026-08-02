import os
import re
import json
import numpy as np
from project.events.ipo_scraper import (
    fetch_massive_ipos,
    fetch_benzinga_ipos,
    fetch_marketbeat_ipos
)
import pandas as pd
from flask import Flask, render_template, request, jsonify
from typing import Any, Dict, List, Optional

# --- EXISTING IMPORTS (unchanged) ---
from project.news.news_intel import analyze_news
from project.intel.ticker_selector import get_best_tickers
from project.data.massive_today_minutes import load_today_minute_bars
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

# RL imports
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

# --- VALIDATION ---
_SYMBOL_RE = re.compile(r'^[A-Za-z0-9]([A-Za-z0-9.]{0,18}[A-Za-z0-9])?$')
UNIVERSE = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "META", "AMD", "GOOGL", "NFLX", "CRM"]

def _validate_symbol(symbol: str) -> bool:
    if not symbol or ".." in symbol:
        return False
    return bool(_SYMBOL_RE.match(symbol))

def _normalize_bar_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})

def data_loader(symbol):
    data = load_today_minute_bars([symbol])
    return data.get(symbol, [])

# --- FLASK APP ---
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config.from_object(_CONFIG_OBJECT)

# --- CLIENTS ---
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

# ============================================================
#   MAIN UI ROUTES (Sidebar Pages)
# ============================================================

@app.route("/")
def dashboard_home():
    return render_template("dashboard.html")

@app.route("/news")
def news_home():
    ticker_list = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
    intel = analyze_news(ticker_list, {})
    return render_template("news.html", intel=intel)

@app.route("/events")
def events_page():
    massive_ipos = fetch_massive_ipos()
    benzinga_ipos = fetch_benzinga_ipos()
    marketbeat_ipos = fetch_marketbeat_ipos()

    return render_template(
        "events.html",
        massive=massive_ipos,
        benzinga=benzinga_ipos,
        marketbeat=marketbeat_ipos
    )

@app.route("/ai")
def ai_home():
    return render_template("ai.html")

@app.route("/strategies")
def strategies_home():
    return render_template("strategies.html")

@app.route("/rl")
def rl_home():
    return render_template("rl.html")

@app.route("/intel")
def intel_home():
    return render_template("intel.html")

@app.route("/diagnostics")
def diagnostics_home():
    return render_template("diagnostics.html")

@app.route("/settings")
def settings_home():
    return render_template("settings.html")

@app.route("/strategies/emerging_shotgun")
def strategies_emerging_shotgun():
    return render_template("strategies_emerging_shotgun.html")


# ============================================================
#   DASHBOARD SUB‑PAGE ROUTES
# ============================================================

@app.route("/dashboard/overview")
def dashboard_overview():
    return render_template("dashboard_overview.html")

@app.route("/dashboard/strategies")
def dashboard_strategies():
    return render_template("dashboard_strategies.html")

@app.route("/dashboard/combined")
def dashboard_combined():
    return render_template("dashboard_combined.html")

@app.route("/dashboard/risk")
def dashboard_risk():
    return render_template("dashboard_risk.html")

@app.route("/dashboard/cycles")
def dashboard_cycles():
    return render_template("dashboard_cycles.html")

@app.route("/dashboard/rl")
def dashboard_rl():
    return render_template("dashboard_rl.html")


# ============================================================
#   NEWS SUB‑PAGE ROUTES
# ============================================================

@app.route("/news/tickers")
def news_tickers():
    intel = analyze_news(UNIVERSE, {})
    return render_template("news_tickers.html", intel=intel)

@app.route("/news/ai")
def news_ai():
    intel = analyze_news(UNIVERSE, {})
    return render_template("news_ai.html", intel=intel)

@app.route("/news/sentiment")
def news_sentiment():
    intel = analyze_news(UNIVERSE, {})
    return render_template("news_sentiment.html", intel=intel)


# ============================================================
#   AI LEARNING SUB‑PAGE ROUTES
# ============================================================

@app.route("/ai/train_rl")
def ai_train_rl():
    return render_template("ai_train_rl.html")

@app.route("/ai/intel_backtest")
def ai_intel_backtest():
    return render_template("ai_intel_backtest.html")

@app.route("/ai/intel_optimize")
def ai_intel_optimize():
    return render_template("ai_intel_optimize.html")

@app.route("/train_news_rl")
def train_news_rl():
    return render_template("train_news_rl.html")


# ============================================================
#   INTELLIGENCE SUB‑PAGE ROUTES
# ============================================================

@app.route("/intel/signals")
def intel_signals():
    return render_template("intel_signals.html")

@app.route("/intel/live")
def intel_live_page():
    return render_template("intel_live.html")


# ============================================================
#   DIAGNOSTICS SUB‑PAGE ROUTES
# ============================================================

@app.route("/diagnostics/system")
def diagnostics_system():
    return render_template("diagnostics_system.html")


# ============================================================
#   SETTINGS SUB‑PAGE ROUTES
# ============================================================

@app.route("/settings/api")
def settings_api():
    return render_template("settings_api.html")
