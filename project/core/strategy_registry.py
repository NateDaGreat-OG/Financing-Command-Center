import importlib
from typing import Any, Dict, Optional

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
        # rsplit('.', 1)[0] strips the last component (e.g. 'project.core' → 'project').
        # For a single-segment package (no dot) it safely returns the whole string.
        module_names.append(f"{__package__.rsplit('.', 1)[0]}.strategies.{name}")
    module_names.append(f"strategies.{name}")

    for module_name in module_names:
        try:
            return importlib.import_module(module_name)
        except ImportError:
            continue
    return None


def load_all_strategies() -> Dict[str, Any]:
    """Return a dict mapping every registered strategy name to its module."""
    all_names = {name for names in STRATEGY_MAP.values() for name in names}
    modules: Dict[str, Any] = {}
    for name in sorted(all_names):
        module = load_strategy(name)
        if module is not None:
            modules[name] = module
    return modules


def create_intelligence_layer(
    config: Optional[Dict[str, Any]] = None,
    capital_manager: Optional[Any] = None,
    cycle_analyzer: Optional[Any] = None,
    risk_manager: Optional[Any] = None,
    rl_agent: Optional[Any] = None,
    rl_blend_weight: float = 0.4,
    min_signal_score: float = 0.40,
) -> Any:
    """Factory that instantiates and returns a :class:`StrategyIntelligence` layer.

    All arguments are optional; when omitted each dependency is constructed from
    *config* (or an empty dict when *config* is also omitted).
    """
    from core.strategy_intelligence import StrategyIntelligence  # lazy import avoids circularity

    return StrategyIntelligence(
        capital_manager=capital_manager,
        cycle_analyzer=cycle_analyzer,
        risk_manager=risk_manager,
        rl_agent=rl_agent,
        config=config or {},
        rl_blend_weight=rl_blend_weight,
        min_signal_score=min_signal_score,
    )


def load_strategy_with_intelligence(name: str, intelligence: Any) -> Optional[Any]:
    """Load a strategy module and return a thin wrapper that routes
    ``generate_signals`` through the intelligence layer.

    The wrapper preserves the ``scan_candidates`` and ``execute_signals``
    interface so it can be used as a drop-in replacement everywhere a raw
    strategy module is expected.
    """
    module = load_strategy(name)
    if module is None:
        return None
    return _IntelligenceWrappedStrategy(module, name, intelligence)


class _IntelligenceWrappedStrategy:
    """Drop-in strategy wrapper that enriches signals via StrategyIntelligence."""

    def __init__(self, module: Any, name: str, intelligence: Any):
        self._module = module
        self._name = name
        self._intelligence = intelligence

    def scan_candidates(self, symbols):
        return self._module.scan_candidates(symbols)

    def generate_signals(self, data, symbol: str = "", **kwargs):
        result = self._intelligence.run(
            self._module, data, symbol=symbol, strategy_name=self._name, **kwargs
        )
        return result.get("signals", [])

    def execute_signals(self, signals):
        return self._module.execute_signals(signals)
