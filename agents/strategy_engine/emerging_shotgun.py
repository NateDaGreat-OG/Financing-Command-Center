"""
Emerging Shotgun strategy.

Multi-timeframe momentum + volume burst + range expansion.
Designed for intraday or short-term breakout trading on liquid symbols.
"""

from typing import Dict, Any, List
import pandas as pd


DEFAULT_PARAMS: Dict[str, Any] = {
    "vol_mult": 2.0,          # volume burst multiplier vs 20-bar average
    "stop_pct": 0.01,         # 1% stop
    "target_pct": 0.02,       # 2% target
    "min_range_mult": 1.5,    # range expansion vs 10-bar average
    "min_price": 5.0,         # avoid penny stocks
    "max_spread_pct": 0.5,    # avoid wide spreads (if you track them)
}


_params: Dict[str, Any] = DEFAULT_PARAMS.copy()


def set_params(params: Dict[str, Any]) -> None:
    """
    Called by optimizer/backtester to configure strategy hyperparameters.
    """
    global _params
    _params = {**DEFAULT_PARAMS, **params}


def _compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute simple multi-timeframe trend proxies using rolling means.
    Assumes df has 'close' column.
    """
    df = df.copy()
    df["trend_1m"] = df["close"].rolling(3, min_periods=3).mean()
    df["trend_5m"] = df["close"].rolling(15, min_periods=15).mean()
    df["trend_15m"] = df["close"].rolling(45, min_periods=45).mean()

    df["trend_align"] = (
        (df["trend_1m"] > df["trend_1m"].shift(1)) &
        (df["trend_5m"] > df["trend_5m"].shift(1)) &
        (df["trend_15m"] > df["trend_15m"].shift(1))
    )

    return df


def _compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute volume burst relative to 20-bar average.
    Assumes df has 'volume' column.
    """
    df = df.copy()
    vol_ma = df["volume"].rolling(20, min_periods=20).mean()
    df["vol_burst"] = df["volume"] > vol_ma * _params["vol_mult"]
    return df


def _compute_range_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute range expansion relative to 10-bar average.
    Assumes df has 'high' and 'low' columns.
    """
    df = df.copy()
    df["range"] = df["high"] - df["low"]
    range_ma = df["range"].rolling(10, min_periods=10).mean()
    df["range_expansion"] = df["range"] > range_ma * _params["min_range_mult"]
    return df


def _apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic filters: min price, etc.
    """
    df = df.copy()
    df["price_ok"] = df["close"] >= _params["min_price"]
    return df


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Core signal generator for a single symbol.

    Input:
        df: DataFrame with at least ['open', 'high', 'low', 'close', 'volume'].

    Output:
        df with added columns:
            'shotgun' (bool) - entry signal
    """
    df = _compute_trend_features(df)
    df = _compute_volume_features(df)
    df = _compute_range_features(df)
    df = _apply_filters(df)

    df["shotgun"] = (
        df["trend_align"] &
        df["vol_burst"] &
        df["range_expansion"] &
        df["price_ok"]
    )

    return df


def generate_portfolio_signals(data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Multi-symbol interface for Backtester.

    Input:
        data: dict[symbol] -> DataFrame

    Output:
        dict[symbol] -> DataFrame with 'shotgun' column.
    """
    out: Dict[str, pd.DataFrame] = {}
    for symbol, df in data.items():
        out[symbol] = generate_signals(df)
    return out


def should_enter(row: pd.Series) -> bool:
    """
    Entry condition for a single bar.
    """
    return bool(row.get("shotgun", False))


def compute_stop_and_target(entry_price: float) -> Dict[str, float]:
    """
    Compute stop and target levels based on params.
    """
    stop = entry_price * (1 - _params["stop_pct"])
    target = entry_price * (1 + _params["target_pct"])
    return {"stop": stop, "target": target}


def should_exit(row: pd.Series, entry_price: float) -> bool:
    """
    Exit condition for a single bar.
    Uses static stop/target for simplicity.
    """
    levels = compute_stop_and_target(entry_price)
    close = float(row["close"])

    if close <= levels["stop"]:
        return True
    if close >= levels["target"]:
        return True

    return False

def emerging_shotgun_strategy(ticker, data, intel):
    return {
        "ticker": ticker,
        "signal": "emerging_shotgun",
        "score": 0.0,
        "details": "Placeholder — logic not implemented yet."
    }