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
    module_names = []
    if __package__:
        module_names.append(f"{__package__.rsplit('.', 1)[0]}.strategies.{name}")
    module_names.append(f"strategies.{name}")

    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None
