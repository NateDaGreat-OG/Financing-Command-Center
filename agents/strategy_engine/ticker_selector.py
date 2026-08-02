# project/intel/ticker_selector.py
from typing import List
from .ticker_intel import get_ranked_tickers_for_strategy
from .ticker_rl_refiner import refine_with_rl

def get_best_tickers(
    strategy_name: str,
    universe: List[str],
    data_loader,
    performance_history,
    top_n: int = 5,
) -> List[str]:
    """
    Hybrid:
    1) Fast scanner + intelligence scoring
    2) RL refinement based on past performance
    """
    ranked = get_ranked_tickers_for_strategy(
        strategy_name=strategy_name,
        universe=universe,
        data_loader=data_loader,
        top_n=top_n * 2,  # get a wider set first
    )

    refined = refine_with_rl(
        strategy_name=strategy_name,
        ranked_tickers=ranked,
        performance_history=performance_history,
    )

    return refined[:top_n]
