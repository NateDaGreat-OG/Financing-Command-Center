import os
import importlib
import json
import project.config as Config
from project.services.backtester_intel_adapter import BacktesterIntelAdapter
from project.data.massive_loader import load_historical_data   # adjust if your loader is elsewhere

api_key = Config.MASSIVE_API_KEY
base_url = Config.MASSIVE_BASE_URL

# Folder containing all strategy modules
STRATEGY_FOLDER = "project/strategies"

# Backtest parameters
TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN"]
START_DATE = "2023-01-01"
END_DATE = "2024-01-01"


def get_strategy_names():
    """Return all strategy module names (without .py)."""
    files = os.listdir(STRATEGY_FOLDER)
    return [
        f.replace(".py", "")
        for f in files
        if f.endswith(".py") and not f.startswith("__")
    ]


def run_backtest_for_strategy(strategy_name):
    """Run a backtest for a single strategy across ALL tickers."""
    print(f"\n=== Running Backtest for {strategy_name} ===")

    # Load strategy module dynamically
    strategy_module = importlib.import_module(f"project.strategies.{strategy_name}")

    # Load config module and convert to dict (adapter requires dict)
    config = {
        "LOG_DIR": Config.LOG_DIR,
        "DEFAULT_CAPITAL": Config.DEFAULT_CAPITAL,
        "DEFAULT_SLIPPAGE": Config.DEFAULT_SLIPPAGE,
        "DEFAULT_COMMISSION": Config.DEFAULT_COMMISSION,
        "MASSIVE_API_KEY": Config.MASSIVE_API_KEY,
        "MASSIVE_BASE_URL": Config.MASSIVE_BASE_URL,
        "BACKTEST_TIMEFRAME": Config.BACKTEST_TIMEFRAME,}



    # Create adapter
    adapter = BacktesterIntelAdapter(strategy_module, config)

    # Load historical data for ALL tickers
    historical_data = {
        symbol: load_historical_data(symbol, START_DATE, END_DATE)
        for symbol in TICKERS
    }

    # Run backtest using correct adapter signature
    results = adapter.run(
        historical_data=historical_data,
        strategy_name=strategy_name
    )

    # Print summary
    print("Metrics:", results["metrics"])
    print("Trades:", results["trade_log"][:5])
    print("Equity Curve (first 10):", results["equity_curve"][:10])

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
