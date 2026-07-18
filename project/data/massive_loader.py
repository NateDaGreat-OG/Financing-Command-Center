import requests
import project.config as config

def load_historical_data(symbol, start, end):
    url = f"{config.MASSIVE_BASE_URL}/v1/bars"
    params = {
        "symbol": symbol,
        "start": start,
        "end": end,
        "apikey": config.MASSIVE_API_KEY,
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Massive.com bar format → adapter format
    bars = data.get("bars", data)

    normalized = []
    for b in bars:
        normalized.append({
            "t": b.get("timestamp"),
            "o": b.get("open"),
            "h": b.get("high"),
            "l": b.get("low"),
            "c": b.get("close"),
            "v": b.get("volume"),
        })

    return normalized
