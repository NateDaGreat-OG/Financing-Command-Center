"""Reinforcement learning environment wrapping the trading backtester."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Tuple

ACTIONS = {0: "hold", 1: "buy", 2: "sell"}


class TradingEnv:
    def __init__(
        self,
        data: pd.DataFrame,
        capital: float = 100000.0,
        position_size: float = 1.0,
        transaction_cost: float = 0.001,
    ):
        self.raw_data = data.copy()
        self.capital = float(capital)
        self.position_size = float(position_size)
        self.transaction_cost = float(transaction_cost)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.cash = float(capital)
        self.equity = float(capital)
        self.previous_equity = float(capital)
        self.history: list[float] = []

        self._prepare_indicators()
        self._build_normalization()

    def _prepare_indicators(self) -> None:
        data = self.raw_data.copy()
        data = data.dropna().reset_index(drop=True)
        data["volume"] = data["volume"].replace(0, 1.0)
        data["ema9"] = data["close"].ewm(span=9, adjust=False).mean()
        data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()
        data["atr"] = (data["high"] - data["low"]).rolling(14).mean().fillna(0.0)
        data["rsi"] = self._rsi(data["close"], 14)
        typical = (data["high"] + data["low"] + data["close"]) / 3.0
        data["vwap"] = (typical * data["volume"]).cumsum() / data["volume"].cumsum()
        data["bb_mid"] = data["close"].rolling(20).mean()
        data["bb_std"] = data["close"].rolling(20).std().fillna(0.0)
        data["bb_upper"] = data["bb_mid"] + 2.0 * data["bb_std"]
        data["bb_lower"] = data["bb_mid"] - 2.0 * data["bb_std"]
        data = data.fillna(method="bfill").fillna(method="ffill").replace([np.inf, -np.inf], 0.0)
        self.data = data.reset_index(drop=True)

    def _build_normalization(self) -> None:
        self.price_scale = float(max(self.data["close"].median(), 1.0))
        self.volume_scale = float(max(self.data["volume"].median(), 1.0))
        indicators = self.data[["ema9", "ema20", "atr", "rsi", "vwap", "bb_upper", "bb_lower"]]
        self.indicator_mean = indicators.mean().to_numpy(dtype=np.float32)
        self.indicator_std = indicators.std().replace(0.0, 1.0).to_numpy(dtype=np.float32)

    def reset(self) -> np.ndarray:
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.cash = float(self.capital)
        self.equity = float(self.capital)
        self.previous_equity = float(self.capital)
        self.history = []
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        row = self.data.iloc[self.current_step]
        reward = 0.0
        info: dict[str, Any] = {}

        valid = self._is_valid_action(action)
        if not valid:
            reward -= 0.05
            info["invalid_action"] = True
            action = 0

        prev_equity = self.equity
        current_price = float(row["close"])

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = current_price
            cost = self.transaction_cost * self.position_size * current_price
            self.cash -= self.position_size * current_price + cost
            reward -= min(cost / max(self.capital, 1.0), 0.1)
            info["action"] = "buy"
        elif action == 2 and self.position == 1:
            exit_price = current_price
            cost = self.transaction_cost * self.position_size * exit_price
            self.cash += self.position_size * exit_price - cost
            reward += (exit_price - self.entry_price) * self.position_size - cost
            self.position = 0
            self.entry_price = 0.0
            info["action"] = "sell"
        else:
            info["action"] = "hold"

        self.equity = self.cash + self._position_value(row)
        self.history.append(self.equity)

        reward += self._continuous_reward(prev_equity)
        reward += self._position_reward(row)
        reward -= self._drawdown_penalty()
        reward = float(np.clip(reward, -1.0, 1.0))

        self.previous_equity = self.equity
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        state = self._get_state()
        info["equity"] = float(self.equity)
        info["cash"] = float(self.cash)
        return state, reward, done, info

    def _get_state(self) -> np.ndarray:
        row = self.data.iloc[self.current_step]
        feature_vector = np.array(
            [
                float(row["open"]) / self.price_scale,
                float(row["high"]) / self.price_scale,
                float(row["low"]) / self.price_scale,
                float(row["close"]) / self.price_scale,
                float(row["volume"]) / self.volume_scale,
                float(row["ema9"] - self.indicator_mean[0]) / self.indicator_std[0],
                float(row["ema20"] - self.indicator_mean[1]) / self.indicator_std[1],
                float(row["atr"] - self.indicator_mean[2]) / self.indicator_std[2],
                float(row["rsi"] - self.indicator_mean[3]) / self.indicator_std[3],
                float(row["vwap"] - self.indicator_mean[4]) / self.indicator_std[4],
                float(row["bb_upper"] - self.indicator_mean[5]) / self.indicator_std[5],
                float(row["bb_lower"] - self.indicator_mean[6]) / self.indicator_std[6],
                float(self.position),
                float(self.equity / self.capital),
            ],
            dtype=np.float32,
        )
        mask = self._action_mask().astype(np.float32)
        return np.concatenate([feature_vector, mask], axis=0)

    def _continuous_reward(self, prev_equity: float) -> float:
        return float((self.equity - prev_equity) / max(self.capital, 1.0))

    def _position_reward(self, row: pd.Series) -> float:
        if self.position != 1:
            return 0.0
        unrealized = float(row["close"] - self.entry_price)
        if unrealized > 0:
            return min(unrealized / max(self.entry_price, 1.0), 0.05)
        return -min(abs(unrealized) / max(self.entry_price, 1.0), 0.05)

    def _drawdown_penalty(self) -> float:
        drawdown = max(0.0, (self.capital - self.equity) / max(self.capital, 1.0))
        return float(drawdown * 0.02)

    def _position_value(self, row: pd.Series) -> float:
        return float(self.position_size * float(row["close"])) if self.position == 1 else 0.0

    def _action_mask(self) -> np.ndarray:
        mask = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        if self.position == 0:
            mask[2] = 0.0
        else:
            mask[1] = 0.0
        return mask

    def _is_valid_action(self, action: int) -> bool:
        if action not in ACTIONS:
            return False
        if self.position == 0 and action == 2:
            return False
        if self.position == 1 and action == 1:
            return False
        return True

    def _rsi(self, series: pd.Series, period: int) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()
        rs = gain / loss.replace(0.0, 1.0)
        return 100.0 - (100.0 / (1.0 + rs))
