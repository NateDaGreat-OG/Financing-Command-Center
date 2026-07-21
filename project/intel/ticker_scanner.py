# project/intel/ticker_scanner.py
from typing import List, Dict

def scan_universe(universe: List[str], data_loader) -> Dict[str, dict]:
    """
    Returns basic features per ticker:
    {
      "AAPL": {"avg_volume": ..., "atr": ..., "trend_score": ...},
      ...
    }
    """
    features = {}

    for symbol in universe:
        bars = data_loader(symbol)  # recent daily or minute bars

        if not bars:
            continue

        # Example: compute simple stats
        avg_volume = sum(b["v"] for b in bars) / len(bars)
        atr = _compute_atr(bars)
        trend_score = _compute_trend_strength(bars)

        features[symbol] = {
            "avg_volume": avg_volume,
            "atr": atr,
            "trend_score": trend_score,
        }

    return features


def _compute_atr(bars):
    # placeholder: true range average
    return 0.0


def _compute_trend_strength(bars):
    # placeholder: slope / ADX / etc.
    return 0.0
