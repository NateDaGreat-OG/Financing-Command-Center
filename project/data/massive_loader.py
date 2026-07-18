import requests
import project.config as Config

def load_historical_data(symbol, start, end, multiplier=1, timespan="minute"):
    """
    Load OHLCV bars from Massive.com using minute aggregates.
    This works on the Massive Stocks Basic plan.
    """

    url = (
        f"{Config.MASSIVE_BASE_URL}/v2/aggs/ticker/"
        f"{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    )

    headers = {
        "Authorization": f"Bearer {Config.MASSIVE_API_KEY}"
    }

    response = requests.get(url, headers=headers)
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

    return normalized

