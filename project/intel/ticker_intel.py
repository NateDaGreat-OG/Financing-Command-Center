# project/intel/ticker_intel.py
from typing import List
from .strategy_profiles import STRATEGY_PROFILE
from .ticker_scanner import scan_universe

def score_ticker_for_strategy(symbol_features, profile) -> float:
    """
    Returns a score (0–1) for how well this ticker matches the strategy profile.
    """
    vol = symbol_features["avg_volume"]
    atr = symbol_features["atr"]
    trend = symbol_features["trend_score"]

    score = 0.0

    # Volume
    if vol >= profile["min_volume"]:
        score += 0.3

    # ATR range
    atr_min, atr_max = profile["atr_range"]
    if atr_min <= atr <= atr_max:
        score += 0.3

    # Trend strength
    if trend >= profile["min_trend_score"]:
        score += 0.4

    return score


def get_ranked_tickers_for_strategy(
    strategy_name: str,
    universe: List[str],
    data_loader,
    top_n: int = 10,
) -> List[str]:
    """
    Returns top_n tickers ranked for the given strategy.
    """
    profile = STRATEGY_PROFILE.get(strategy_name)
    if profile is None:
        return []

    features = scan_universe(universe, data_loader)

    scored = []
    for symbol, f in features.items():
        s = score_ticker_for_strategy(f, profile)
        scored.append((symbol, s))

    scored.sort(key=lambda x: x[1], reverse=True)

    return [sym for sym, s in scored[:top_n]]
