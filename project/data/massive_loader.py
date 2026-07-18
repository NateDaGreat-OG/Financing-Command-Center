import requests
import project.config as Config

def load_historical_data(symbol, start, end, multiplier=1, timespan="day"):
    """
    Load OHLCV bars from Massive.com using the official v2 aggs endpoint.
    """

    url = (
        f"{Config.MASSIVE_BASE_URL}/v2/aggs/ticker/"
        f"{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
    )

    params = {
        "apiKey": Config.MASSIVE_API_KEY  # MUST be apiKey (case-sensitive)
    }

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    # Massive returns bars under "results"
    bars = data.get("results", [])

    normalized = []
    for b in bars:
        normalized.append({
            "t": b.get("t"),          # timestamp
            "o": b.get("o"),          # open
            "h": b.get("h"),          # high
            "l": b.get("l"),          # low
            "c": b.get("c"),          # close
            "v": b.get("v"),          # volume
        })

    return normalized

