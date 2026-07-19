import requests
import project.config as Config
from datetime import datetime

def load_today_minute_bars(symbols):
    """
    Load today's minute OHLCV bars for multiple symbols.
    Works on Massive Basic with one API call per symbol.
    """

    headers = {
        "Authorization": f"Bearer {Config.MASSIVE_API_KEY}"
    }

    today = datetime.utcnow().date().strftime("%Y-%m-%d")

    all_data = {}

    for symbol in symbols:
        url = (
            f"{Config.MASSIVE_BASE_URL}/v2/aggs/ticker/"
            f"{symbol}/range/1/minute/{today}/{today}"
        )

        response = requests.get(url, headers=headers)

        # If forbidden or rate-limited, skip symbol
        if response.status_code in (403, 429):
            all_data[symbol] = []
            continue

        response.raise_for_status()
        data = response.json()

        bars = data.get("results", [])
        normalized = []

        for b in bars:
            normalized.append({
                "t": b.get("t"),
                "o": b.get("o"),
                "h": b.get("h"),
                "l": b.get("l"),
                "c": b.get("c"),
                "v": b.get("v"),
            })

        all_data[symbol] = normalized

    return all_data
