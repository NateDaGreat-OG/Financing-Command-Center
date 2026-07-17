"""Liquidity window day trading strategy.

This strategy trades during high-probability intraday windows using VWAP and time-of-day volatility.
"""

import pandas as pd
from typing import Any, Dict, List

TRADE_WINDOWS = [("09:30", "10:15"), ("14:00", "15:30")]

# Module-level parameters — overridable via set_params() by the intelligence layer.
_PARAMS: Dict[str, Any] = {
    "vwap_drift_pct": 0.003,
    "size_mult": 1.0,
    "size": 50,
}


def set_params(params: Dict[str, Any]) -> None:
    """Update module-level strategy parameters (used by CycleAdapter)."""
    _PARAMS.update(params)


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "intraday": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if data.empty or len(data) < 30:
        return signals

    size_mult = float(_PARAMS["size_mult"])
    base_size = int(_PARAMS["size"])

    df = data.copy()
    df["vwap"] = ((df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum())
    df["time"] = df.index.time
    latest = df.iloc[-1]
    timestamp = latest.name
    window_ok = any(start <= timestamp.strftime("%H:%M") <= end for start, end in TRADE_WINDOWS)

    if not window_ok:
        return signals

    effective_size = max(int(base_size * size_mult), 1)
    if latest["close"] > latest["vwap"] and latest["open"] < latest["vwap"]:
        signals.append(_build_signal("long", latest, effective_size))
    elif latest["close"] < latest["vwap"] and latest["open"] > latest["vwap"]:
        signals.append(_build_signal("short", latest, effective_size))

    return signals


def execute_signals(signals):
    return [{"symbol": s.get("symbol"), "side": s["side"], "qty": s["size"], "status": "submitted"} for s in signals]


def _build_signal(side: str, row, size: int):
    entry_price = row["close"]
    stop_loss = row["vwap"]
    target = entry_price + (entry_price - stop_loss) * 1.2 if side == "long" else entry_price - (stop_loss - entry_price) * 1.2
    return {
        "side": side,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target": target,
        "size": size,
        "signal_type": "liquidity_window",
    }
