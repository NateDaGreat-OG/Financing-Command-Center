"""Market regime and cyclical analysis engine.

This module detects market regimes, volatility cycles, liquidity states,
macro risk sentiment, and intraday patterns from OHLCV and optional external data.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from services.massive_client import MassiveClient


class CycleAnalyzer:
    def __init__(
        self,
        macro_client: Optional[MassiveClient] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.macro_client = macro_client
        self.config = config or {}

    def analyze(
        self,
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        extra_data: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        extra_data = extra_data or {}

        df = self._normalize_ohlcv(data)
        if df.empty:
            return self._empty_cycle_state()

        trend_state = self._trend_state(df)
        volatility_state = self._volatility_state(df)
        liquidity_state = self._liquidity_state(df)
        macro_state = self._macro_state(symbol)
        intraday_state = self._intraday_state(df)
        sector_state = self._sector_rotation_state(extra_data.get("sector"))

        return {
            "trend": trend_state,
            "volatility": volatility_state,
            "liquidity": liquidity_state,
            "macro": macro_state,
            "intraday": intraday_state,
            "sector_rotation": sector_state,
        }

    def _empty_cycle_state(self) -> Dict[str, Any]:
        return {
            "trend": "sideways",
            "volatility": "low",
            "liquidity": "expanding",
            "macro": "risk_on",
            "intraday": "chop",
            "sector_rotation": {},
        }

    def _normalize_ohlcv(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
        df = data.copy()
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")]
        df = df.sort_index()
        return df

    def _trend_state(self, df: pd.DataFrame) -> str:
        ema_short = df["close"].ewm(span=20, adjust=False).mean()
        ema_long = df["close"].ewm(span=50, adjust=False).mean()
        slope = ema_short.iloc[-1] - ema_long.iloc[-1]
        price_relation = df["close"].iloc[-1] / ema_long.iloc[-1] if ema_long.iloc[-1] != 0 else 1.0

        if slope > 0 and price_relation > 1.02:
            return "bull"
        if slope < 0 and price_relation < 0.98:
            return "bear"
        return "sideways"

    def _volatility_state(self, df: pd.DataFrame) -> str:
        atr = self._atr(df["high"], df["low"], df["close"], period=14)
        latest_atr = atr.iloc[-1]
        volatility_percentile = self._percentile(latest_atr, atr)
        return "high" if volatility_percentile > 0.75 else "low"

    def _liquidity_state(self, df: pd.DataFrame) -> str:
        vwap = self._vwap(df)
        if vwap is None:
            return "expanding"
        deviation = abs(df["close"].iloc[-1] - vwap.iloc[-1]) / max(vwap.iloc[-1], 1.0)
        volume_pct = self._percentile(df["volume"].iloc[-1], df["volume"])
        if deviation > 0.02 and volume_pct > 0.6:
            return "expanding"
        if deviation < 0.01 and volume_pct < 0.4:
            return "contracting"
        return "expanding"

    def _macro_state(self, symbol: Optional[str] = None) -> str:
        if not self.macro_client or not symbol:
            return "risk_on"
        try:
            fundamentals = self.macro_client.get_fundamentals(symbol)
            sentiment = fundamentals.get("sentiment", {}).get("score", 0.0)
            return "risk_off" if sentiment < 0 else "risk_on"
        except Exception:
            return "risk_on"

    def _intraday_state(self, df: pd.DataFrame) -> str:
        if df.index.tz is None:
            times = df.index
        else:
            times = df.index.tz_convert("US/Eastern")

        latest = times[-1].time()
        if latest >= pd.to_datetime("09:30").time() and latest <= pd.to_datetime("10:30").time():
            return "open_drive"
        if latest >= pd.to_datetime("14:30").time() and latest <= pd.to_datetime("15:59").time():
            return "power_hour"
        if latest >= pd.to_datetime("10:30").time() and latest <= pd.to_datetime("13:00").time():
            return "chop"
        return "fade"

    def _sector_rotation_state(self, sector_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if sector_data is None or sector_data.empty:
            return {}
        momentum = sector_data.iloc[-1].get("momentum", 0.0)
        rotation = "defensive" if momentum < 0 else "cyclical"
        return {"rotation": rotation, "momentum": float(momentum)}

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean().fillna(method="bfill")

    def _vwap(self, df: pd.DataFrame) -> Optional[pd.Series]:
        if "volume" not in df.columns or df["volume"].sum() <= 0:
            return None
        cum_vol = df["volume"].cumsum()
        cum_vwap = (df["close"] * df["volume"]).cumsum()
        return cum_vwap / cum_vol

    def _percentile(self, value: float, series: pd.Series) -> float:
        values = series.dropna().astype(float)
        if values.empty:
            return 0.0
        rank = float((values < value).sum())
        return float(rank / max(len(values) - 1, 1))
