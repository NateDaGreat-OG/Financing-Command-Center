import os
import importlib
import json

import project.config as Config
from project.services.backtester_intel_adapter import BacktesterIntelAdapter
from project.data.massive_recent_loader import load_recent_daily_bars as load_historical_data

# Folder containing all strategy modules
STRATEGY_FOLDER = "project/strategies"

# Backtest parameters
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
DAYS_BACK = 7  # Number of days of historical data to load for backtesting


def get_strategy_names():
    """Return all strategy module names (without .py)."""
    files = os.listdir(STRATEGY_FOLDER)
    return [
        f.replace(".py", "")
        for f in files
        if f.endswith(".py") and not f.startswith("__")
    ]


def run_backtest_for_strategy(strategy_name: str):
    """Run a backtest for a single strategy across all tickers."""
    print(f"\n=== Running Backtest for {strategy_name} ===")

    # Load strategy module dynamically
    strategy_module = importlib.import_module(f"project.strategies.{strategy_name}")

    # Wrap config module into a dict for the adapter
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

    # Create adapter
    adapter = BacktesterIntelAdapter(strategy_module, config)

    # Load historical data for all tickers from Massive
    historical_data = {
        symbol: load_historical_data(symbol, days_back=DAYS_BACK)
        for symbol in TICKERS
    }

    # Run backtest
    results = adapter.run(
        historical_data=historical_data,
        strategy_name=strategy_name,
    )

    # Print summary
    print("Metrics:", results.get("metrics"))
    print("Trades:", results.get("trade_log", [])[:5])
    print("Equity Curve (first 10):", results.get("equity_curve", [])[:10])

    # Save results for dashboard
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
