"""Trend continuation swing trading strategy.

This strategy uses EMA crossovers, RSI, ATR, and volume confirmation to buy pullbacks in strong trends.
"""

import pandas as pd
from typing import Any, Dict, List

# Module-level parameters — overridable via set_params() by the intelligence layer.
_PARAMS: Dict[str, Any] = {
    "ema_short": 9,
    "ema_long": 20,
    "rsi_low": 40,
    "rsi_high": 70,
    "atr_mult": 2.0,
    "size": 50,
}


def set_params(params: Dict[str, Any]) -> None:
    """Update module-level strategy parameters (used by CycleAdapter)."""
    _PARAMS.update(params)


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "trend": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if data.empty or len(data) < 30:
        return signals

    ema_short = int(_PARAMS["ema_short"])
    ema_long = int(_PARAMS["ema_long"])
    rsi_low = float(_PARAMS["rsi_low"])
    rsi_high = float(_PARAMS["rsi_high"])
    atr_mult = float(_PARAMS["atr_mult"])
    size = int(_PARAMS["size"])

    df = data.copy()
    df["ema_s"] = df["close"].ewm(span=ema_short, adjust=False).mean()
    df["ema_l"] = df["close"].ewm(span=ema_long, adjust=False).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["rsi"] = _rsi(df["close"], 14)
    df["vol_avg"] = df["volume"].rolling(20).mean()

    latest = df.iloc[-1]
    is_uptrend = latest["ema_s"] > latest["ema_l"]
    pullback = latest["close"] < latest["ema_s"]
    strong_volume = latest["volume"] > latest["vol_avg"]
    rsi_ok = rsi_low < latest["rsi"] < rsi_high

    if is_uptrend and pullback and strong_volume and rsi_ok:
        entry_price = latest["close"]
        stop_loss = latest["ema_l"]
        target = entry_price + latest["atr"] * atr_mult
        signals.append({
            "side": "long",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "size": size,
            "signal_type": "trend_continuation",
        })

    return signals


def execute_signals(signals):
    return [{"symbol": s.get("symbol"), "side": s["side"], "qty": s["size"], "status": "submitted"} for s in signals]


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss.replace(0, 1)
    return 100 - (100 / (1 + rs))
