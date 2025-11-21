import os
import requests
import json
from typing import Dict, Any, Optional
import time

class KalshiClient:
    """
    Client for interacting with the Kalshi API (v2).
    """
    
    BASE_URL = "https://trading-api.kalshi.com/trade-api/v2"
    
    def __init__(self, api_key: str = None, email: str = None, password: str = None):
        self.api_key = api_key or os.getenv("KALSHI_API_KEY")
        self.email = email or os.getenv("KALSHI_EMAIL")
        self.password = password or os.getenv("KALSHI_PASSWORD")
        self.token = None
        self.member_id = None
        
        if self.email and self.password:
            self.authenticate()

    def authenticate(self):
        """
        Login and retrieve session token.
        """
        url = f"{self.BASE_URL}/login"
        payload = {
            "email": self.email,
            "password": self.password
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            data = response.json()
            self.token = data.get("token")
            self.member_id = data.get("member_id")
            print(f"Successfully authenticated as member {self.member_id}")
        else:
            print(f"Authentication failed: {response.text}")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.token}" if self.token else "",
            "Content-Type": "application/json"
        }

    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get market details and current order book.
        """
        # Get market details
        url = f"{self.BASE_URL}/markets/{ticker}"
        response = requests.get(url, headers=self._get_headers())
        market_data = response.json() if response.status_code == 200 else {}
        
        # Get order book
        book_url = f"{self.BASE_URL}/markets/{ticker}/orderbook"
        book_response = requests.get(book_url, headers=self._get_headers())
        book_data = book_response.json() if book_response.status_code == 200 else {}
        
        return {
            "market": market_data,
            "orderbook": book_data
        }

    def place_order(self, ticker: str, side: str, count: int, price: int) -> Dict[str, Any]:
        """
        Place an order.
        side: 'yes' or 'no' (Kalshi uses 'yes'/'no' for binary options)
        price: in cents (1-99)
        """
        url = f"{self.BASE_URL}/portfolio/orders"
        
        # Map side to Kalshi format if needed
        action = "buy" # Usually we buy 'yes' or buy 'no' contracts
        # Wait, Kalshi structure: You buy 'yes' contracts or 'no' contracts?
        # Actually, you usually buy 'yes' or 'no' side.
        
        payload = {
            "ticker": ticker,
            "action": action,
            "type": "limit",
            "side": side.lower(), # 'yes' or 'no'
            "count": count,
            "yes_price": price if side.lower() == 'yes' else None,
            "no_price": price if side.lower() == 'no' else None,
            # Note: API might require specific price field depending on side
            # Simplified for now based on typical binary API
        }
        
        # Correction for Kalshi v2:
        # "side" is "yes" or "no".
        # "action" is "buy" or "sell".
        # If we are opening a position, it's "buy".
        
        payload = {
            "ticker": ticker,
            "action": "buy",
            "type": "limit",
            "side": side.lower(),
            "count": count,
            "price": price # Price is usually specified for the side you are buying
        }

        response = requests.post(url, json=payload, headers=self._get_headers())
        return response.json()

    def get_balance(self) -> Dict[str, Any]:
        """
        Get portfolio balance.
        """
        url = f"{self.BASE_URL}/portfolio/balance"
        response = requests.get(url, headers=self._get_headers())
        return response.json()
