import os
import requests
import json
from typing import Dict, Any, Optional
import time

class KalshiClient:
    """
    Client for interacting with the Kalshi API (v2).
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
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

    def get_markets(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        event_ticker: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get markets with optional filters.

        Args:
            series_ticker: Filter by series (e.g., 'HIGHFB' for NFL)
            status: Comma-separated list: 'unopened', 'open', 'closed', 'settled'
            event_ticker: Filter by specific event
            limit: Max results per page (default 200)
            cursor: Pagination cursor

        Returns:
            Dict with 'markets' list and 'cursor' for pagination
        """
        url = f"{self.BASE_URL}/markets"
        params = {"limit": limit}

        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if event_ticker:
            params["event_ticker"] = event_ticker
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, params=params, headers=self._get_headers())

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching markets: {response.status_code} - {response.text}")
            return {"markets": [], "cursor": None}

    def get_events(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 200,
        cursor: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get events with optional filters.

        Args:
            series_ticker: Filter by series
            status: Comma-separated list: 'open', 'closed', 'settled'
            limit: Max results per page
            cursor: Pagination cursor

        Returns:
            Dict with 'events' list and 'cursor' for pagination
        """
        url = f"{self.BASE_URL}/events"
        params = {"limit": limit}

        if series_ticker:
            params["series_ticker"] = series_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor

        response = requests.get(url, params=params, headers=self._get_headers())

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching events: {response.status_code} - {response.text}")
            return {"events": [], "cursor": None}

    def get_market_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        period_interval: int = 60,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get historical candlestick data for a market.

        Args:
            series_ticker: Series ticker (e.g., 'HIGHFB')
            market_ticker: Market ticker (e.g., 'HIGHFB-24SEP15-B-KC')
            period_interval: Candlestick interval in minutes (1, 60, or 1440)
            start_ts: Start timestamp (Unix time in seconds)
            end_ts: End timestamp (Unix time in seconds)

        Returns:
            Dict with 'candlesticks' list containing OHLC data
        """
        url = f"{self.BASE_URL}/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        params = {"period_interval": period_interval}

        if start_ts:
            params["start_ts"] = start_ts
        if end_ts:
            params["end_ts"] = end_ts

        response = requests.get(url, params=params, headers=self._get_headers())

        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error fetching candlesticks for {market_ticker}: {response.status_code} - {response.text}")
            return {"candlesticks": []}

    def get_all_markets_paginated(
        self,
        series_ticker: Optional[str] = None,
        status: Optional[str] = None,
        max_results: Optional[int] = None
    ) -> list:
        """
        Get all markets using pagination.

        Args:
            series_ticker: Filter by series
            status: Filter by status
            max_results: Maximum total results (None = unlimited)

        Returns:
            List of all markets
        """
        all_markets = []
        cursor = None
        count = 0

        while True:
            response = self.get_markets(
                series_ticker=series_ticker,
                status=status,
                cursor=cursor
            )

            markets = response.get("markets", [])
            all_markets.extend(markets)
            count += len(markets)

            # Check if we've hit max or no more pages
            if max_results and count >= max_results:
                break

            cursor = response.get("cursor")
            if not cursor:
                break

            # Rate limiting - be nice to the API
            time.sleep(0.1)

        return all_markets
