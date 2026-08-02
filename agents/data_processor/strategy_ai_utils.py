"""AI-enhanced signal utility functions.

Provides technical indicator computation, signal scoring, ensemble aggregation,
and signal enrichment helpers used by the intelligence layer.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Technical indicator helpers
# ---------------------------------------------------------------------------

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a standard set of technical indicators on an OHLCV DataFrame.

    Input columns expected: open, high, low, close, volume.
    Returns a new DataFrame with the original columns plus derived indicators.
    """
    if df.empty or len(df) < 2:
        return df.copy()

    out = df.copy()

    # EMAs
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema50"] = out["close"].ewm(span=50, adjust=False).mean()

    # ATR
    tr1 = out["high"] - out["low"]
    tr2 = (out["high"] - out["close"].shift()).abs()
    tr3 = (out["low"] - out["close"].shift()).abs()
    out["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # RSI
    out["rsi"] = _rsi(out["close"], 14)

    # VWAP
    if "volume" in out.columns and out["volume"].sum() > 0:
        typical = (out["high"] + out["low"] + out["close"]) / 3.0
        out["vwap"] = (typical * out["volume"]).cumsum() / out["volume"].cumsum().replace(0, 1)

    # Bollinger Bands
    out["bb_mid"] = out["close"].rolling(20).mean()
    out["bb_std"] = out["close"].rolling(20).std().fillna(0.0)
    out["bb_upper"] = out["bb_mid"] + 2.0 * out["bb_std"]
    out["bb_lower"] = out["bb_mid"] - 2.0 * out["bb_std"]
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0, 1)

    # Volume features
    out["vol_avg"] = out["volume"].rolling(20).mean()
    out["vol_ratio"] = out["volume"] / out["vol_avg"].replace(0, 1)

    # Price momentum (5-period return)
    out["momentum"] = out["close"].pct_change(5)

    out = out.bfill().ffill().replace([np.inf, -np.inf], 0.0)
    return out


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def score_signal(
    signal: Dict[str, Any],
    data: pd.DataFrame,
    cycle_state: Optional[Dict[str, Any]] = None,
) -> float:
    """Score a single signal in [0, 1] based on technical quality and cycle context.

    Higher scores indicate stronger alignment with current conditions.
    """
    cycle_state = cycle_state or {}
    score = 0.5  # neutral baseline

    if data.empty or len(data) < 2:
        return score

    df = compute_technical_indicators(data)
    latest = df.iloc[-1]
    side = signal.get("side", "long")

    # RSI alignment
    rsi = float(latest.get("rsi", 50.0))
    if side == "long":
        if 40 < rsi < 65:
            score += 0.10
        elif rsi >= 70:
            score -= 0.05
    else:
        if 35 < rsi < 60:
            score += 0.10
        elif rsi <= 30:
            score -= 0.05

    # EMA trend alignment
    ema9 = float(latest.get("ema9", latest["close"]))
    ema20 = float(latest.get("ema20", latest["close"]))
    if side == "long" and ema9 > ema20:
        score += 0.10
    elif side == "short" and ema9 < ema20:
        score += 0.10

    # Volume confirmation
    vol_ratio = float(latest.get("vol_ratio", 1.0))
    if vol_ratio > 1.5:
        score += 0.10
    elif vol_ratio < 0.5:
        score -= 0.05

    # Risk/reward ratio
    entry = float(signal.get("entry_price", float(latest["close"])))
    stop = float(signal.get("stop_loss", entry))
    target = float(signal.get("target", entry))
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk > 0:
        rr = reward / risk
        if rr >= 2.0:
            score += 0.15
        elif rr >= 1.5:
            score += 0.07
        elif rr < 1.0:
            score -= 0.10

    # Cycle alignment
    trend = cycle_state.get("trend", "sideways")
    volatility = cycle_state.get("volatility", "low")
    signal_type = signal.get("signal_type", "")

    if "trend" in signal_type:
        if trend == "bull" and side == "long":
            score += 0.10
        elif trend == "bear" and side == "short":
            score += 0.10
    if "volatility" in signal_type and volatility == "high":
        score += 0.05
    if cycle_state.get("macro") == "risk_off" and side == "long":
        score -= 0.10

    return float(np.clip(score, 0.0, 1.0))


def filter_signals_by_quality(
    signals: List[Dict[str, Any]],
    min_score: float = 0.40,
) -> List[Dict[str, Any]]:
    """Return only signals whose pre-computed ``ai_score`` meets the threshold."""
    return [s for s in signals if float(s.get("ai_score", 0.5)) >= min_score]


# ---------------------------------------------------------------------------
# Ensemble aggregation
# ---------------------------------------------------------------------------

def compute_ensemble_signals(
    all_signals: Dict[str, List[Dict[str, Any]]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Merge signals from multiple strategies into a deduplicated ranked list.

    Signals sharing the same (symbol, side) key are merged; the highest-weighted
    version is kept.  Weights default to 1.0 for every strategy.
    """
    weights = weights or {}
    merged: Dict[tuple, Dict[str, Any]] = {}

    for strategy_name, signals in all_signals.items():
        strategy_weight = float(weights.get(strategy_name, 1.0))
        for signal in signals:
            key = (signal.get("symbol", ""), signal.get("side", "long"))
            weighted_score = float(signal.get("ai_score", 0.5)) * strategy_weight
            existing = merged.get(key)
            if existing is None or weighted_score > float(existing.get("ai_score", 0.0)):
                merged[key] = {
                    **signal,
                    "ai_score": round(weighted_score, 4),
                    "source_strategy": strategy_name,
                }

    return list(merged.values())


# ---------------------------------------------------------------------------
# Signal enrichment
# ---------------------------------------------------------------------------

def enrich_signal(
    signal: Dict[str, Any],
    data: pd.DataFrame,
    cycle_state: Optional[Dict[str, Any]] = None,
    source_strategy: str = "",
) -> Dict[str, Any]:
    """Return a copy of *signal* with ``ai_score``, ``cycle_context``, and metadata added."""
    enriched = signal.copy()
    enriched["ai_score"] = round(score_signal(signal, data, cycle_state), 4)
    enriched["cycle_context"] = dict(cycle_state) if cycle_state else {}
    if source_strategy:
        enriched["source_strategy"] = source_strategy
    return enriched


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = -delta.clip(upper=0).rolling(window=period).mean()
    rs = gain / loss.replace(0.0, 1.0)
    return 100.0 - (100.0 / (1.0 + rs))
