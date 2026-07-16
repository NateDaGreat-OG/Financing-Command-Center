import os
import requests

class AlpacaClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }

    def get_historical(self, symbol: str, timeframe: str = "1D") -> dict:
        url = f"{self.base_url}/v2/stocks/{symbol}/bars"
        params = {"timeframe": timeframe, "limit": 500}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_intraday(self, symbol: str, interval: str = "1Min") -> dict:
        url = f"{self.base_url}/v2/stocks/{symbol}/bars"
        params = {"timeframe": interval, "limit": 200}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_latest_quote(self, symbol: str) -> dict:
        url = f"{self.base_url}/v2/stocks/{symbol}/quotes/latest"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def submit_order(self, symbol: str, qty: float, side: str, type: str = "market", time_in_force: str = "day") -> dict:
        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": type,
            "time_in_force": time_in_force,
        }
        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def get_positions(self) -> dict:
        url = f"{self.base_url}/v2/positions"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_account(self) -> dict:
        url = f"{self.base_url}/v2/account"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
