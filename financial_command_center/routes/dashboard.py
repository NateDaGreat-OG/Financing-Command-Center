import os
import json
from flask import Blueprint, render_template
from project.core.strategy_registry import list_styles, list_strategies_for_style

dashboard_bp = Blueprint("dashboard", __name__)

BACKTEST_DIR = "backtest_results"

def load_backtest(strategy_name):
    """Load backtest JSON for a strategy, or return empty defaults."""
    path = os.path.join(BACKTEST_DIR, f"{strategy_name}.json")
    if not os.path.exists(path):
        return {
            "metrics": {},
            "trade_log": [],
            "equity_curve": [],
            "drawdown_curve": [],
            "trade_distribution": {"counts": [], "edges": []},
            "intelligence_diagnostics": {},
        }
    with open(path, "r") as f:
        return json.load(f)

@dashboard_bp.route("/")
def dashboard():
    # Load strategies from registry
    strategies = []
    for style in list_styles() or []:
        for strat in list_strategies_for_style(style) or []:
            name = strat

            # Load backtest results
            backtest = load_backtest(name)

            strategies.append({
                "name": name,
                "style": style,
                "module": f"project.strategies.{name}",
                "metrics": backtest.get("metrics", {}),
                "trade_log": backtest.get("trade_log", []),
                "equity_curve": backtest.get("equity_curve", []),
                "drawdown_curve": backtest.get("drawdown_curve", []),
                "trade_distribution": backtest.get("trade_distribution", {}),
            })

    return render_template("dashboard.html", strategies=strategies)
