"""RL override and hybrid decision engine.

Bridges DQNAgent decisions with rule-based strategy signals, providing:
- State vector construction compatible with TradingEnv
- Conversion of RL actions to signal dicts
- Hybrid blending of RL and strategy signals
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

_RL_ACTIONS: Dict[int, str] = {0: "hold", 1: "buy", 2: "sell"}
_ACTION_TO_SIDE: Dict[int, str] = {1: "long", 2: "short"}


class RLAdapter:
    """Bridges a DQNAgent with rule-based strategy signals."""

    def __init__(self, blend_weight: float = 0.5):
        """
        Args:
            blend_weight: Weight given to RL action vs strategy signals.
                0.0 = pure strategy, 1.0 = pure RL.
        """
        self.blend_weight = float(np.clip(blend_weight, 0.0, 1.0))

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def build_env_state(
        self,
        data: pd.DataFrame,
        position: int = 0,
        equity: float = 100_000.0,
        capital: float = 100_000.0,
    ) -> Optional[np.ndarray]:
        """Build a 17-element state vector compatible with TradingEnv._get_state.

        Vector layout:
          [open, high, low, close, volume,          (price/volume — normalised)
           ema9, ema20, atr, rsi, vwap,              (normalised indicators)
           bb_upper, bb_lower,                       (normalised indicators)
           position, equity_ratio,                  (agent state)
           mask_hold, mask_buy, mask_sell]           (action mask)

        Returns ``None`` when data is too short to compute indicators.
        """
        if data.empty or len(data) < 20:
            return None

        df = data.copy()

        # Compute indicators
        df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift()).abs()
        tr3 = (df["low"] - df["close"].shift()).abs()
        df["atr"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = -delta.clip(upper=0).rolling(14).mean()
        rs = gain / loss.replace(0.0, 1.0)
        df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["vwap"] = (typical * df["volume"]).cumsum() / df["volume"].cumsum().replace(0, 1)
        df["bb_mid"] = df["close"].rolling(20).mean()
        df["bb_std"] = df["close"].rolling(20).std().fillna(0.0)
        df["bb_upper"] = df["bb_mid"] + 2.0 * df["bb_std"]
        df["bb_lower"] = df["bb_mid"] - 2.0 * df["bb_std"]
        df = df.bfill().ffill().replace([np.inf, -np.inf], 0.0)

        # Normalisation constants (match TradingEnv)
        price_scale = float(max(df["close"].median(), 1.0))
        vol_scale = float(max(df["volume"].median(), 1.0))
        ind_cols = ["ema9", "ema20", "atr", "rsi", "vwap", "bb_upper", "bb_lower"]
        ind_mean = df[ind_cols].mean().to_numpy(dtype=np.float32)
        ind_std = df[ind_cols].std().replace(0.0, 1.0).to_numpy(dtype=np.float32)

        row = df.iloc[-1]
        features = np.array([
            float(row["open"]) / price_scale,
            float(row["high"]) / price_scale,
            float(row["low"]) / price_scale,
            float(row["close"]) / price_scale,
            float(row["volume"]) / vol_scale,
            float(row["ema9"] - ind_mean[0]) / ind_std[0],
            float(row["ema20"] - ind_mean[1]) / ind_std[1],
            float(row["atr"] - ind_mean[2]) / ind_std[2],
            float(row["rsi"] - ind_mean[3]) / ind_std[3],
            float(row["vwap"] - ind_mean[4]) / ind_std[4],
            float(row["bb_upper"] - ind_mean[5]) / ind_std[5],
            float(row["bb_lower"] - ind_mean[6]) / ind_std[6],
            float(position),
            float(equity / max(capital, 1.0)),
        ], dtype=np.float32)

        mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if position == 0:
            mask[2] = 0.0  # cannot sell when flat
        else:
            mask[1] = 0.0  # cannot buy when already long

        return np.concatenate([features, mask], axis=0)

    # ------------------------------------------------------------------
    # RL signal generation
    # ------------------------------------------------------------------

    def get_rl_signal(
        self,
        agent: Any,
        state: np.ndarray,
        symbol: str,
        current_price: float,
        capital: float = 100_000.0,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Ask the RL agent for an action and convert it to a signal dict.

        Returns ``None`` for a hold action (action 0).
        """
        if agent is None or state is None:
            return None

        action = int(agent.act(state))
        action_name = _RL_ACTIONS.get(action, "hold")
        if action_name == "hold":
            return None

        side = _ACTION_TO_SIDE.get(action, "long")
        # Simple ATR-proxy sizing: 1 % of price as risk unit
        atr_proxy = max(current_price * 0.01, 0.01)
        if side == "long":
            stop = current_price - atr_proxy
            target = current_price + atr_proxy * 2.0
        else:
            stop = current_price + atr_proxy
            target = current_price - atr_proxy * 2.0

        size = max(1, int((capital * 0.01) / atr_proxy))

        return {
            "symbol": symbol,
            "side": side,
            "entry_price": current_price,
            "stop_loss": round(stop, 4),
            "target": round(target, 4),
            "size": size,
            "signal_type": "rl_override",
            "rl_action": action_name,
            "cycle_context": dict(cycle_state) if cycle_state else {},
        }

    # ------------------------------------------------------------------
    # Hybrid blending
    # ------------------------------------------------------------------

    def hybrid_decision(
        self,
        strategy_signals: List[Dict[str, Any]],
        rl_signal: Optional[Dict[str, Any]],
        data: pd.DataFrame,
        cycle_state: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Blend strategy signals with an RL signal using ``blend_weight``.

        - ``blend_weight == 0``: pure strategy signals returned unchanged.
        - ``blend_weight == 1``: only the RL signal is returned (if any).
        - In between: strategy signals are kept; the RL signal is appended
          with its ``ai_score`` scaled by ``blend_weight``.
        """
        if rl_signal is None or self.blend_weight == 0.0:
            return list(strategy_signals)

        if self.blend_weight == 1.0:
            return [rl_signal]

        results = list(strategy_signals)
        rl_boosted = {
            **rl_signal,
            "ai_score": round(
                float(np.clip(float(rl_signal.get("ai_score", 0.5)) * self.blend_weight, 0.0, 1.0)),
                4,
            ),
            "hybrid_blend": self.blend_weight,
        }
        results.append(rl_boosted)
        return results

    # ------------------------------------------------------------------
    # Confidence scaling
    # ------------------------------------------------------------------

    def adjust_size_from_rl_confidence(
        self,
        size: float,
        agent: Any,
        state: Optional[np.ndarray] = None,
    ) -> float:
        """Scale position size by the agent's policy confidence (inverse of epsilon).

        Low epsilon (greedy policy) → scale factor approaches 1.0.
        High epsilon (random policy) → scale factor approaches 0.5.
        """
        if agent is None:
            return size
        epsilon = float(getattr(agent, "epsilon", 1.0))
        confidence = 1.0 - epsilon  # 0 = exploring, 1 = fully greedy
        scale = 0.5 + 0.5 * confidence  # range [0.5, 1.0]
        return float(max(size * scale, 1.0))
