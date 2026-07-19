import requests
import project.config as Config
from datetime import datetime

def load_today_daily_bars(symbols):
    """
    Load today's daily OHLCV bars for multiple symbols.
    Works on Massive Basic with only 1 API call.
    """

    headers = {
        "Authorization": f"Bearer {Config.MASSIVE_API_KEY}"
    }

    today = datetime.utcnow().date().strftime("%Y-%m-%d")

    url = f"{Config.MASSIVE_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{today}"

    response = requests.get(url, headers=headers)
    response.raise_for_status()

    data = response.json()
    results = data.get("results", [])

    all_data = {s: [] for s in symbols}

    for bar in results:
        ticker = bar.get("T")
        if ticker in symbols:
            all_data[ticker].append({
                "t": today,
                "o": bar.get("o"),
                "h": bar.get("h"),
                "l": bar.get("l"),
                "c": bar.get("c"),
                "v": bar.get("v"),
            })

    return all_data
