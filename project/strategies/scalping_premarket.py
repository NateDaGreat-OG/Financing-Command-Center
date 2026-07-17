"""Premarket momentum scalping strategy.

This strategy identifies strong premarket gaps, abnormal volume, and news catalysts.
It generates entry signals at open and uses tight stop-loss / small profit targets.
"""

from typing import Any, Dict, List

REQUIRED_FIELDS = ["open", "high", "low", "close", "volume"]

# Module-level parameters — overridable via set_params() by the intelligence layer.
_PARAMS: Dict[str, Any] = {
    "gap_min": 0.01,
    "vol_ratio_min": 2.0,
    "stop_pct": 0.002,
    "target_pct": 0.005,
    "size": 100,
}


def set_params(params: Dict[str, Any]) -> None:
    """Update module-level strategy parameters (used by CycleAdapter)."""
    _PARAMS.update(params)


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "premarket_momentum": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if data.empty:
        return signals

    gap_min = float(_PARAMS["gap_min"])
    vol_ratio_min = float(_PARAMS["vol_ratio_min"])
    stop_pct = float(_PARAMS["stop_pct"])
    target_pct = float(_PARAMS["target_pct"])
    size = int(_PARAMS["size"])

    latest = data.iloc[-1]
    gap = (latest["open"] - latest["close"]) / latest["close"] if latest["close"] else 0
    volume = latest["volume"]
    avg_volume = data["volume"].rolling(20).mean().iloc[-1] if len(data) >= 20 else volume
    has_momentum = abs(gap) > gap_min and volume > avg_volume * vol_ratio_min

    if has_momentum:
        entry_price = latest["open"]
        stop_loss = entry_price * (1 - stop_pct)
        target = entry_price * (1 + target_pct)
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
