"""Reinforcement learning environment wrapping the trading backtester."""
import numpy as np
import pandas as pd

ACTIONS = {0: "hold", 1: "buy", 2: "sell"}

class TradingEnv:
    def __init__(self, data: pd.DataFrame, capital: float = 100000):
        self.data = data.copy()
        self.capital = capital
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.equity = capital
        self.history = []
        self._prepare_indicators()

    def _prepare_indicators(self):
        self.data["ema9"] = self.data["close"].ewm(span=9, adjust=False).mean()
        self.data["ema20"] = self.data["close"].ewm(span=20, adjust=False).mean()
        self.data["atr"] = (self.data["high"] - self.data["low"]).rolling(14).mean().fillna(0)
        self.data["rsi"] = self._rsi(self.data["close"], 14)
        self.data["vwap"] = ((self.data["volume"] * (self.data["high"] + self.data["low"] + self.data["close"]) / 3).cumsum() / self.data["volume"].cumsum()).fillna(method="ffill")
        self.data["bb_mid"] = self.data["close"].rolling(20).mean()
        self.data["bb_std"] = self.data["close"].rolling(20).std()
        self.data["bb_upper"] = self.data["bb_mid"] + 2 * self.data["bb_std"]
        self.data["bb_lower"] = self.data["bb_mid"] - 2 * self.data["bb_std"]

    def reset(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.equity = self.capital
        self.history = []
        return self._get_state()

    def step(self, action: int):
        row = self.data.iloc[self.current_step]
        reward = 0.0
        done = False
        info = {}

        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = row["close"]
            info["action"] = "buy"
        elif action == 2 and self.position == 1:
            pnl = row["close"] - self.entry_price
            self.equity += pnl
            reward = pnl - abs(pnl) * 0.001
            self.position = 0
            self.entry_price = 0.0
            info["action"] = "sell"
        else:
            info["action"] = "hold"

        self.current_step += 1
        if self.current_step >= len(self.data) - 1:
            done = True

        state = self._get_state()
        reward -= self._drawdown_penalty()
        info["equity"] = self.equity
        return state, reward, done, info

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        state = np.array([
            row["open"],
            row["high"],
            row["low"],
            row["close"],
            row["volume"],
            row["ema9"],
            row["ema20"],
            row["atr"],
            row["rsi"],
            row["vwap"],
            row["bb_upper"],
            row["bb_lower"],
            self.position,
            self.equity,
        ], dtype=np.float32)
        return state

    def _drawdown_penalty(self):
        if self.equity < self.capital:
            return (self.capital - self.equity) * 0.0001
        return 0.0

    def _rsi(self, series, period):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = -delta.clip(upper=0).rolling(window=period).mean()
        rs = gain / loss.replace(0, 1)
        return 100 - (100 / (1 + rs))
