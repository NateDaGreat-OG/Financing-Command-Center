import csv
import json
import os
from datetime import datetime

class TradeLogger:
    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.signal_log = os.path.join(self.log_dir, "signals.json")
        self.trade_log = os.path.join(self.log_dir, "trades.csv")
        self.error_log = os.path.join(self.log_dir, "errors.log")
        self.backtest_log = os.path.join(self.log_dir, "backtests.json")

    def _append_json(self, path: str, payload: dict):
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(payload, default=str) + "\n")

    def log_signals(self, symbol: str, signals: list):
        payload = {"timestamp": datetime.utcnow().isoformat(), "symbol": symbol, "signals": signals}
        self._append_json(self.signal_log, payload)

    def log_trade(self, trade: dict):
        file_exists = os.path.exists(self.trade_log)
        with open(self.trade_log, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=list(trade.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(trade)

    def log_error(self, error: Exception, context: str = ""):
        with open(self.error_log, "a", encoding="utf-8") as file:
            file.write(f"{datetime.utcnow().isoformat()} | {context} | {repr(error)}\n")

    def log_backtest(self, backtest_result: dict):
        payload = {"timestamp": datetime.utcnow().isoformat(), "result": backtest_result}
        self._append_json(self.backtest_log, payload)
