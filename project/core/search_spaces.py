"""Defines hyperparameter search spaces for each trading strategy."""

SCALPING_SPACE = {
    "min_gap": [1, 2, 3],
    "min_premarket_volume": [50000, 100000, 200000],
    "stop_pct": [0.2, 0.3, 0.5],
    "target_pct": [0.3, 0.5, 1.0],
}

TREND_CONTINUATION_SPACE = {
    "ema_fast": [5, 10, 15],
    "ema_slow": [20, 30, 40],
    "atr_multiplier": [1.5, 2.0, 2.5],
    "rsi_lower": [45, 50, 55],
    "rsi_upper": [60, 65, 70],
}

VOLATILITY_COMPRESSION_SPACE = {
    "bb_period": [10, 20, 30],
    "bb_width_threshold": [0.02, 0.03, 0.05],
    "consolidation_days": [3, 5, 7],
}

LIQUIDITY_WINDOW_SPACE = {
    "vwap_window": [20, 30, 40],
    "volume_threshold": [1.2, 1.5, 2.0],
    "time_windows": ["morning", "afternoon", "both"],
}

MAGIC_FORMULA_SPACE = {
    "top_n": [10, 20, 30],
    "rebalance_days": [30, 60, 90],
    "min_market_cap": [500e6, 1e9, 5e9],
}

DEFAULT_SEARCH_SPACES = {
    "scalping_premarket": SCALPING_SPACE,
    "trend_continuation": TREND_CONTINUATION_SPACE,
    "volatility_compression": VOLATILITY_COMPRESSION_SPACE,
    "liquidity_window": LIQUIDITY_WINDOW_SPACE,
    "magic_formula": MAGIC_FORMULA_SPACE,
}
