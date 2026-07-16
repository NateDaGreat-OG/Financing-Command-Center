import os

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "your-alpaca-key")
ALPACA_API_SECRET = os.environ.get("ALPACA_API_SECRET", "your-alpaca-secret")
ALPACA_BASE_URL = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")

MASSIVE_API_KEY = os.environ.get("MASSIVE_API_KEY", "your-massive-key")
MASSIVE_BASE_URL = os.environ.get("MASSIVE_BASE_URL", "https://api.massive.com")

LOG_DIR = os.environ.get("LOG_DIR", "./logs")
DEFAULT_CAPITAL = float(os.environ.get("DEFAULT_CAPITAL", 100000))
DEFAULT_SLIPPAGE = float(os.environ.get("DEFAULT_SLIPPAGE", 0.0005))
DEFAULT_COMMISSION = float(os.environ.get("DEFAULT_COMMISSION", 0.001))
BACKTEST_TIMEFRAME = os.environ.get("BACKTEST_TIMEFRAME", "1D")
LIVE_INTERVAL = os.environ.get("LIVE_INTERVAL", "5Min")
MAX_RISK_PER_TRADE = float(os.environ.get("MAX_RISK_PER_TRADE", 0.01))
MAX_CONCURRENT_POSITIONS = int(os.environ.get("MAX_CONCURRENT_POSITIONS", 5))
