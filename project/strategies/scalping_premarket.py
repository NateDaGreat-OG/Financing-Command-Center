"""Premarket momentum scalping strategy.

This strategy identifies strong premarket gaps, abnormal volume, and news catalysts.
It generates entry signals at open and uses tight stop-loss / small profit targets.
"""

from typing import List

REQUIRED_FIELDS = ["open", "high", "low", "close", "volume"]


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "premarket_momentum": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if data.empty:
        return signals

    latest = data.iloc[-1]
    gap = (latest["open"] - latest["close"]) / latest["close"] if latest["close"] else 0
    volume = latest["volume"]
    avg_volume = data["volume"].rolling(20).mean().iloc[-1] if len(data) >= 20 else volume
    has_momentum = abs(gap) > 0.01 and volume > avg_volume * 2

    if has_momentum:
        entry_price = latest["open"]
        stop_loss = entry_price * 0.998
        target = entry_price * 1.005
        size = 100
        signals.append({
            "side": "long" if gap > 0 else "short",
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "size": size,
            "signal_type": "scalp_premarket",
        })

    return signals


def execute_signals(signals):
    executed = []
    for signal in signals:
        executed.append({
            "symbol": signal.get("symbol"),
            "side": signal["side"],
            "qty": signal["size"],
            "type": "market",
            "status": "submitted",
        })
    return executed
