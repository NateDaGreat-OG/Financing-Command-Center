"""Magic Formula strategy for short-term and long-term investing.

This strategy ranks stocks based on quality and value, using fundamentals from massive.com.
"""

from typing import List


def scan_candidates(symbols: List[str]):
    return [{"symbol": symbol, "fundamentals": True} for symbol in symbols]


def generate_signals(data):
    signals = []
    if not data:
        return signals

    for symbol, fundamentals in data.items():
        score = _score_company(fundamentals)
        if score > 0.5:
            entry_price = fundamentals.get("price", 0)
            stop_loss = entry_price * 0.92
            target = entry_price * 1.15
            size = 20
            signals.append({
                "symbol": symbol,
                "side": "long",
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "target": target,
                "size": size,
                "signal_type": "magic_formula",
"ranking": score,
            })

    return signals


def execute_signals(signals):
    return [{"symbol": s.get("symbol"), "side": s["side"], "qty": s["size"], "status": "submitted"} for s in signals]


def _score_company(fundamentals: dict) -> float:
    roic = fundamentals.get("roic", 0)
    earnings_yield = fundamentals.get("earnings_yield", 0)
    quality = min(max(roic / 20.0, 0), 1)
    value = min(max(earnings_yield / 0.1, 0), 1)
    return round((quality + value) / 2, 2)
