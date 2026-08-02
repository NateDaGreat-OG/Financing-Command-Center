# project/intel/strategy_profiles.py

STRATEGY_PROFILE = {
    "scalping_premarket": {
        "min_volume": 5_000_000,
        "atr_range": (0.5, 3.0),
        "min_trend_score": 0.0,
    },
    "trend_continuation": {
        "min_volume": 2_000_000,
        "atr_range": (1.0, 5.0),
        "min_trend_score": 0.7,
    },
    "volatility_compression": {
        "min_volume": 1_000_000,
        "atr_range": (0.5, 2.5),
        "min_trend_score": 0.3,
    },
    "liquidity_window": {
        "min_volume": 3_000_000,
        "atr_range": (0.8, 4.0),
        "min_trend_score": 0.5,
    },
    "magic_formula": {
        # likely more fundamental, but keep placeholder
        "min_volume": 500_000,
        "atr_range": (0.5, 3.0),
        "min_trend_score": 0.2,
    },
}
