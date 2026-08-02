import numpy as np
import pandas as pd
from datetime import datetime

class Backtester:
    def __init__(self, strategy_module, capital: float, slippage: float, commission: float, risk_manager, logger):
        self.strategy = strategy_module
        self.capital = capital
        self.slippage = slippage
        self.commission = commission
        self.risk_manager = risk_manager
        self.logger = logger

    def run(self, historical_data: dict) -> dict:
        equity = self.capital
        trade_log = []
        returns = []
        open_trades = []

        for symbol, data in historical_data.items():
            df = self._normalize_data(data)
            if df.empty:
                continue

            candidates = self.strategy.scan_candidates([symbol])
            signals = self.strategy.generate_signals(df)
            for signal in signals:
                trade = self._simulate_trade(symbol, signal, df)
                if trade:
                    trade_log.append(trade)
                    equity += trade["net_pnl"]
                    returns.append(trade["net_pnl"] / self.capital)
                    self.logger.log_trade(trade)

        stats = self._calculate_statistics(equity, returns, trade_log)
        result = {"equity_curve": equity, "metrics": stats, "trade_log": trade_log}
        self.logger.log_backtest(result)
        return result

    def _normalize_data(self, raw_data: dict) -> pd.DataFrame:
        if "bars" in raw_data:
            rows = raw_data["bars"]
        elif "data" in raw_data:
            rows = raw_data["data"]
        else:
            rows = raw_data

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        df["t"] = pd.to_datetime(df["t"])
        df = df.rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        df = df.set_index("timestamp")
        return df

    def _simulate_trade(self, symbol: str, signal: dict, df) -> dict | None:
        if not signal or "entry_price" not in signal or "size" not in signal:
            return None

        entry_price = signal["entry_price"]
        exit_price = signal.get("exit_price", entry_price)
        size = signal["size"]
        side = signal.get("side", "long")
        gross_pnl = (exit_price - entry_price) * size if side == "long" else (entry_price - exit_price) * size
        fees = abs(gross_pnl) * self.commission
        slippage_cost = abs(gross_pnl) * self.slippage
        net_pnl = gross_pnl - fees - slippage_cost

        return {
            "timestamp": datetime.utcnow().isoformat(),
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "size": size,
            "gross_pnl": gross_pnl,
            "fees": fees,
            "slippage": slippage_cost,
            "net_pnl": net_pnl,
            "signal": signal,
        }

    def _calculate_statistics(self, equity: float, returns: list, trade_log: list) -> dict:
        if not returns:
            return {
                "cagr": 0.0,
                "max_drawdown": 0.0,
                "sharpe": 0.0,
                "win_rate": 0.0,
                "trade_count": 0,
            }

        equity_curve = np.cumsum(returns) + self.capital
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (peak - equity_curve).max()
        win_rate = sum(1 for trade in trade_log if trade["net_pnl"] > 0) / len(trade_log)
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

        return {
            "cagr": round(((equity / self.capital) ** (1 / 1) - 1) * 100, 2),
            "max_drawdown": round(drawdown, 2),
            "sharpe": round(sharpe, 2),
            "win_rate": round(win_rate * 100, 2),
            "trade_count": len(trade_log),
        }
