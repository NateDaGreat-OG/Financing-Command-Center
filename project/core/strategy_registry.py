import importlib

STRATEGY_MAP = {
    "scalping": ["scalping_premarket"],
    "day_trading": ["liquidity_window", "volatility_compression"],
    "swing_trading": ["trend_continuation", "volatility_compression"],
    "short_term_investing": ["magic_formula"],
    "long_term_investing": ["magic_formula"],
}

def list_styles():
    return list(STRATEGY_MAP.keys())

def list_strategies_for_style(style: str):
    return STRATEGY_MAP.get(style)

def load_strategy(name: str):
    try:
        module = importlib.import_module(f"strategies.{name}")
        return module
    except ImportError:
        return None
