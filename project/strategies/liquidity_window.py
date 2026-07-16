"""Liquidity window day trading strategy.

This strategy trades during high-probability intraday windows using VWAP and time-of-day volatility.
"""

import pandas as pd
from typing import List

TRADE_WINDOWS = [("09:30", "10:15"), ("14:00", "15:30")]


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "intraday": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if data.empty or len(data) < 30:
        return signals

    df = data.copy()
    df["vwap"] = ((df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum())
    df["time"] = df.index.time
    latest = df.iloc[-1]
    timestamp = latest.name
    window_ok = any(start <= timestamp.strftime("%H:%M") <= end for start, end in TRADE_WINDOWS)

    if not window_ok:
        return signals

    if latest["close"] > latest["vwap"] and latest["open"] < latest["vwap"]:
        signals.append(_build_signal("long", latest, 50))
    elif latest["close"] < latest["vwap"] and latest["open"] > latest["vwap"]:
        signals.append(_build_signal("short", latest, 50))

    return signals


def execute_signals(signals):
    return [{"symbol": s.get("symbol"), "side": s["side"], "qty": s["size"], "status": "submitted"} for s in signals]


def _build_signal(side: str, row, size: int):
    entry_price = row["close"]
    stop_loss = row["vwap"] if side == "long" else row["vwap"]
    target = entry_price + (entry_price - stop_loss) * 1.2 if side == "long" else entry_price - (stop_loss - entry_price) * 1.2
    return {
        "side": side,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "size": size,
        "signal_type": "liquidity_window",
    }
