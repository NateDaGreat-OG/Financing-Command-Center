import requests
import project.config as Config
from datetime import datetime, timedelta

def load_daily_bars_5call(symbols, days_back=5):
    """
    Load daily OHLCV bars for multiple symbols using the grouped endpoint.
    Limited to 5 API calls per day (Massive Basic limit).
    """

    headers = {
        "Authorization": f"Bearer {Config.MASSIVE_API_KEY}"
    }

    today = datetime.utcnow().date()
    start_date = today - timedelta(days=days_back)

    # Output structure: { "AAPL": [...], "MSFT": [...], ... }
    all_data = {s: [] for s in symbols}

    current = start_date
    calls_used = 0

    while current <= today and calls_used < 5:
        date_str = current.strftime("%Y-%m-%d")

        url = f"{Config.MASSIVE_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{date_str}"

        response = requests.get(url, headers=headers)

        # Skip weekends / holidays
        if response.status_code == 404:
            current += timedelta(days=1)
            continue

        # Forbidden → skip
        if response.status_code == 403:
            current += timedelta(days=1)
            continue

        response.raise_for_status()
        calls_used += 1

        data = response.json()
        results = data.get("results", [])

        # Filter only the tickers we care about
        for bar in results:
            ticker = bar.get("T")
            if ticker in symbols:
                all_data[ticker].append({
                    "t": date_str,
                    "o": bar.get("o"),
                    "h": bar.get("h"),
                    "l": bar.get("l"),
                    "c": bar.get("c"),
                    "v": bar.get("v"),
                })

        current += timedelta(days=1)

    return all_data
