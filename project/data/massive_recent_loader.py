import requests
import project.config as Config
from datetime import datetime, timedelta

def load_recent_daily_bars(symbol, days_back=7):
    """
    Load daily OHLCV bars for the last N days.
    Works on Massive Basic because it only requests recent data.
    """

    headers = {
        "Authorization": f"Bearer {Config.MASSIVE_API_KEY}"
    }

    today = datetime.utcnow().date()
    start_date = today - timedelta(days=days_back)

    all_bars = []

    current = start_date
    while current <= today:
        date_str = current.strftime("%Y-%m-%d")
        url = f"{Config.MASSIVE_BASE_URL}/v1/open-close/{symbol}/{date_str}"

        response = requests.get(url, headers=headers)

        # Skip weekends / holidays
        if response.status_code == 404:
            current += timedelta(days=1)
            continue

        # If forbidden, skip (Basic plan sometimes blocks older days)
        if response.status_code == 403:
            current += timedelta(days=1)
            continue

        response.raise_for_status()
        data = response.json()

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
