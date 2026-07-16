"""Volatility compression breakout strategy for day and swing trading.

This strategy identifies Bollinger Band squeezes and buys breakouts on volume expansion.
"""

import pandas as pd
from typing import List


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "squeeze": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if data.empty or len(data) < 30:
        return signals

    df = data.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()
    df["upper"] = df["ma20"] + 2 * df["std20"]
    df["lower"] = df["ma20"] - 2 * df["std20"]
    df["bandwidth"] = (df["upper"] - df["lower"]) / df["ma20"]
    df["vol_avg"] = df["volume"].rolling(20).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    is_squeeze = latest["bandwidth"] < 0.03 and prev["bandwidth"] < latest["bandwidth"]
    breakout = latest["close"] > latest["upper"]
    volume_expansion = latest["volume"] > latest["vol_avg"] * 1.2

    if is_squeeze and breakout and volume_expansion:
        entry_price = latest["close"]
        atr = df["high"].rolling(14).mean().iloc[-1]
        stop_loss = latest["lower"]
        target = entry_price + atr * 1.8
        size = 40
        signals.append({
            "side": "long",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "size": size,
            "signal_type": "volatility_compression",
        })

    return signals


def execute_signals(signals):
    return [{"symbol": s.get("symbol"), "side": s["side"], "qty": s["size"], "status": "submitted"} for s in signals]
