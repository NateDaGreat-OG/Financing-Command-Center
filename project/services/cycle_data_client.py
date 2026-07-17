"""Optional external data client for cycle analysis.

This client can fetch macro, sector, volatility, or liquidity information to support
cycle-aware signal generation and capital allocation.
"""
from __future__ import annotations

import requests
from typing import Any, Dict, Optional


class CycleDataClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def get_volatility_index(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.base_url}/volatility/{symbol}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_liquidity_snapshot(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.base_url}/liquidity/{symbol}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_macro_indicators(self, symbol: str) -> Dict[str, Any]:
        url = f"{self.base_url}/macro/{symbol}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_sector_data(self, sector: str) -> Dict[str, Any]:
        url = f"{self.base_url}/sector/{sector}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()
