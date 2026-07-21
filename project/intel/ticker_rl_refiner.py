# project/intel/ticker_rl_refiner.py
from typing import List, Dict

def refine_with_rl(
    strategy_name: str,
    ranked_tickers: List[str],
    performance_history: Dict[str, float],
) -> List[str]:
    """
    Adjusts ranked tickers using RL / performance feedback.
    performance_history: { "AAPL": sharpe, "MSFT": sharpe, ... }
    For now, treat it as a simple re-weighting placeholder.
    """
    adjusted = []

    for symbol in ranked_tickers:
        base_rank = ranked_tickers.index(symbol)
        perf = performance_history.get(symbol, 0.0)

        # Example: combine base rank and performance
        score = (1.0 / (1 + base_rank)) + perf
        adjusted.append((symbol, score))

    adjusted.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, s in adjusted]
