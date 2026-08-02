import requests
import project.config as Config
from datetime import datetime, timedelta

def load_historical_data(symbol, start, end, multiplier=1, timespan="minute"):
    """
    Load minute bars from Massive Basic plan using automatic chunking.
    Massive Basic cannot return large multi-month minute ranges.
    """

    headers = {
        "Authorization": f"Bearer {Config.MASSIVE_API_KEY}"
    }

    # Convert to datetime objects
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)

    # Massive Basic usually allows 7-day minute windows
    CHUNK_DAYS = 7

    all_bars = []

    current = start_dt
    while current < end_dt:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS), end_dt)

        url = (
            f"{Config.MASSIVE_BASE_URL}/v2/aggs/ticker/"
            f"{symbol}/range/{multiplier}/{timespan}/"
            f"{current.date()}/{chunk_end.date()}"
        )

        response = requests.get(url, headers=headers)
        if response.status_code == 403:
            print(f"403 Forbidden for chunk {current.date()} → {chunk_end.date()}")
            current = chunk_end
            continue

        response.raise_for_status()
        data = response.json()

        bars = data.get("results", [])
        for b in bars:
            all_bars.append({
                "t": b.get("t"),
                "o": b.get("o"),
                "h": b.get("h"),
                "l": b.get("l"),
                "c": b.get("c"),
                "v": b.get("v"),
            })

        current = chunk_end

    return all_bars
