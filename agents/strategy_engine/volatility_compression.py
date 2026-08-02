"""Volatility compression breakout strategy for day and swing trading.

This strategy identifies Bollinger Band squeezes and buys breakouts on volume expansion.
"""

import pandas as pd
from typing import Any, Dict, List

# Module-level parameters — overridable via set_params() by the intelligence layer.
_PARAMS: Dict[str, Any] = {
    "bb_mult": 2.0,
    "squeeze_threshold": 0.03,
    "atr_mult": 1.8,
    "vol_expansion_ratio": 1.2,
    "size": 40,
}


def set_params(params: Dict[str, Any]) -> None:
    """Update module-level strategy parameters (used by CycleAdapter)."""
    _PARAMS.update(params)


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "squeeze": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if data.empty or len(data) < 30:
        return signals

    bb_mult = float(_PARAMS["bb_mult"])
    squeeze_threshold = float(_PARAMS["squeeze_threshold"])
    atr_mult = float(_PARAMS["atr_mult"])
    vol_ratio = float(_PARAMS["vol_expansion_ratio"])
    size = int(_PARAMS["size"])

    df = data.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["std20"] = df["close"].rolling(20).std()
    df["upper"] = df["ma20"] + bb_mult * df["std20"]
    df["lower"] = df["ma20"] - bb_mult * df["std20"]
    df["bandwidth"] = (df["upper"] - df["lower"]) / df["ma20"]
    df["vol_avg"] = df["volume"].rolling(20).mean()

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    is_squeeze = latest["bandwidth"] < squeeze_threshold and prev["bandwidth"] < latest["bandwidth"]
    breakout = latest["close"] > latest["upper"]
    volume_expansion = latest["volume"] > latest["vol_avg"] * vol_ratio

    if is_squeeze and breakout and volume_expansion:
        entry_price = latest["close"]
        atr = df["high"].rolling(14).mean().iloc[-1]
        stop_loss = latest["lower"]
        target = entry_price + atr * atr_mult
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
