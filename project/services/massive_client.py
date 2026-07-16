import requests

class MassiveClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def get_news(self, symbol: str) -> dict:
        url = f"{self.base_url}/news"
        params = {"symbol": symbol, "limit": 20}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_fundamentals(self, symbol: str) -> dict:
        url = f"{self.base_url}/fundamentals/{symbol}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_sentiment(self, symbol: str) -> dict:
        url = f"{self.base_url}/sentiment"
        params = {"symbol": symbol}
        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()
