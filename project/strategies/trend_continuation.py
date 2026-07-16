"""Trend continuation swing trading strategy.

This strategy uses EMA crossovers, RSI, ATR, and volume confirmation to buy pullbacks in strong trends.
"""

import pandas as pd
from typing import List


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "trend": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if data.empty or len(data) < 30:
        return signals

    df = data.copy()
    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
    df["rsi"] = _rsi(df["close"], 14)
    df["vol_avg"] = df["volume"].rolling(20).mean()

    latest = df.iloc[-1]
    is_uptrend = latest["ema9"] > latest["ema20"]
    pullback = latest["close"] < latest["ema9"]
    strong_volume = latest["volume"] > latest["vol_avg"]
    rsi_ok = 40 < latest["rsi"] < 70

    if is_uptrend and pullback and strong_volume and rsi_ok:
        entry_price = latest["close"]
        stop_loss = latest["ema20"]
        target = entry_price + latest["atr"] * 2
        size = 50
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
