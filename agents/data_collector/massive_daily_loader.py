import requests
import project.config as Config
from datetime import datetime, timedelta

def load_daily_bars(symbol, start, end):
    """
    Load daily OHLCV bars using Massive Basic plan.
    Uses the 'open-close' endpoint which is fully supported.
    """

    headers = {
        "Authorization": f"Bearer {Config.MASSIVE_API_KEY}"
    }

    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    all_bars = []

    current = start_dt
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")

        url = f"{Config.MASSIVE_BASE_URL}/v1/open-close/{symbol}/{date_str}"

        response = requests.get(url, headers=headers)

        # Skip days with no trading (weekends, holidays)
        if response.status_code == 404:
            current += timedelta(days=1)
            continue

        response.raise_for_status()
        data = response.json()

        # Massive returns daily OHLC like:
        # {
        #   "open": 123.45,
        #   "close": 125.67,
        #   "high": 126.00,
        #   "low": 122.50,
        #   "volume": 54321000,
        #   "from": "2023-01-03",
        #   "symbol": "AAPL"
        # }

        bar = {
            "t": date_str,
            "o": data.get("open"),
            "h": data.get("high"),
            "l": data.get("low"),
            "c": data.get("close"),
            "v": data.get("volume"),
        }

        all_bars.append(bar)

        current += timedelta(days=1)

    return all_bars
