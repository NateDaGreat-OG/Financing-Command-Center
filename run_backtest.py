import os
import importlib
import json

import project.config as Config
from project.services.backtester_intel_adapter import BacktesterIntelAdapter

# Use 5-call loader (Massive Basic compatible)
from project.data.massive_recent_loader import load_daily_bars_5call

STRATEGY_FOLDER = "project/strategies"
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
DAYS_BACK = 5   # Limited by Massive Basic plan


def get_strategy_names():
    files = os.listdir(STRATEGY_FOLDER)
    return [
        f.replace(".py", "")
        for f in files
        if f.endswith(".py") and not f.startswith("__")
    ]


def run_backtest_for_strategy(strategy_name: str):
    print(f"\n=== Running Backtest for {strategy_name} ===")

    strategy_module = importlib.import_module(f"project.strategies.{strategy_name}")

    config = {
        "LOG_DIR": Config.LOG_DIR,
        "DEFAULT_CAPITAL": Config.DEFAULT_CAPITAL,
        "DEFAULT_SLIPPAGE": Config.DEFAULT_SLIPPAGE,
        "DEFAULT_COMMISSION": Config.DEFAULT_COMMISSION,
        "MASSIVE_API_KEY": Config.MASSIVE_API_KEY,
        "MASSIVE_BASE_URL": Config.MASSIVE_BASE_URL,
        "BACKTEST_TIMEFRAME": Config.BACKTEST_TIMEFRAME,
        "MAX_RISK_PER_TRADE": Config.MAX_RISK_PER_TRADE,
        "MAX_CONCURRENT_POSITIONS": Config.MAX_CONCURRENT_POSITIONS,
        "MAX_PORTFOLIO_DRAWDOWN": Config.MAX_PORTFOLIO_DRAWDOWN,
        "MAX_STRATEGY_DRAWDOWN": Config.MAX_STRATEGY_DRAWDOWN,
        "MAX_SYMBOL_EXPOSURE": Config.MAX_SYMBOL_EXPOSURE,
        "MIN_ALLOCATION_PCT": Config.MIN_ALLOCATION_PCT,
        "MAX_POSITION_SIZE_PCT": Config.MAX_POSITION_SIZE_PCT,
    }

    adapter = BacktesterIntelAdapter(strategy_module, config)

    # Load recent daily data for all tickers in 5 API calls
    historical_data = load_daily_bars_5call(TICKERS, days_back=DAYS_BACK)

    results = adapter.run(
        historical_data=historical_data,
        strategy_name=strategy_name,
    )

    print("Metrics:", results.get("metrics"))
    print("Trades:", results.get("trade_log", [])[:5])
    print("Equity Curve (first 10):", results.get("equity_curve", [])[:10])

    os.makedirs("backtest_results", exist_ok=True)
    out_path = f"backtest_results/{strategy_name}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Saved results → {out_path}")


def main():
    strategy_names = get_strategy_names()
    print(f"Found strategies: {strategy_names}")

    for name in strategy_names:
        run_backtest_for_strategy(name)


if __name__ == "__main__":
    main()
