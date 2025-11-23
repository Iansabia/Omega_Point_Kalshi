import os
import requests
import json
from typing import Dict, Any, Optional
import time
import base64
import hashlib
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.backends import default_backend
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class KalshiClient:
    """
    Client for interacting with the Kalshi API (v2).

    Supports two authentication methods:
    1. API Key (recommended for production): Uses RSA signatures
    2. Email/Password (legacy): Uses session tokens
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, api_key: str = None, private_key_path: str = None,
                 email: str = None, password: str = None):
        """
        Initialize Kalshi client with API key or email/password.

        API Key authentication (preferred):
            api_key: Your Kalshi API key ID
            private_key_path: Path to your RSA private key file (.pem)

        Email/Password authentication (fallback):
            email: Your Kalshi account email
            password: Your Kalshi account password
        """
        # API Key authentication
        self.api_key_id = (api_key or os.getenv("KALSHI_API_KEY_ID") or "").strip('"\'')
        self.private_key_path = (private_key_path or os.getenv("KALSHI_PRIVATE_KEY_PATH") or "").strip('"\'')
        self.private_key = None

        # Email/Password authentication (fallback)
        self.email = (email or os.getenv("KALSHI_EMAIL") or "").strip('"\'')
        self.password = (password or os.getenv("KALSHI_PASSWORD") or "").strip('"\'')
        self.token = None
        self.member_id = None

        # Try API key authentication first
        if self.api_key_id and self.private_key_path and self.private_key_path != "":
            self._load_private_key()
            print(f"Using API key authentication (key ID: {self.api_key_id[:8]}...)")
        # Fallback to email/password
        elif self.email and self.password:
            self.authenticate()
            print("Using email/password authentication (consider upgrading to API keys)")
        else:
            print("Warning: No authentication credentials provided")

    def _load_private_key(self):
        """Load RSA private key from file."""
        try:
            with open(self.private_key_path, 'rb') as key_file:
                self.private_key = serialization.load_pem_private_key(
                    key_file.read(),
                    password=None,
                    backend=default_backend()
                )
        except FileNotFoundError:
            print(f"Error: Private key file not found at {self.private_key_path}")
            self.private_key = None
        except Exception as e:
            print(f"Error loading private key: {e}")
            self.private_key = None

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

    def _sign_request(self, method: str, path: str, body: str = "") -> str:
        """
        Sign a request using RSA private key.

        Args:
            method: HTTP method (GET, POST, etc.)
            path: API path (e.g., /trade-api/v2/markets)
            body: Request body (JSON string, empty for GET requests)

        Returns:
            Base64-encoded signature
        """
        # Create signature message: METHOD + path + body
        message = f"{method}{path}{body}"

        # Sign with private key
        signature = self.private_key.sign(
            message.encode('utf-8'),
            padding.PKCS1v15(),
            hashes.SHA256()
        )

        # Return base64-encoded signature
        return base64.b64encode(signature).decode('utf-8')

    def _get_headers(self, method: str = "GET", path: str = "", body: str = "") -> Dict[str, str]:
        """
        Get request headers with authentication.

        Uses API key authentication if available, otherwise falls back to token auth.
        """
        headers = {
            "Content-Type": "application/json"
        }

        # Use API key authentication if available
        if self.private_key and self.api_key_id:
            timestamp = str(int(time.time() * 1000))  # Milliseconds
            signature = self._sign_request(method, path, body)

            headers["KALSHI-ACCESS-KEY"] = self.api_key_id
            headers["KALSHI-ACCESS-SIGNATURE"] = signature
            headers["KALSHI-ACCESS-TIMESTAMP"] = timestamp
        # Fallback to token authentication
        elif self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        return headers

    def get_market_data(self, ticker: str) -> Dict[str, Any]:
        """
        Get market details and current order book.
        """
        # Get market details
        path = f"/trade-api/v2/markets/{ticker}"
        url = f"{self.BASE_URL}/markets/{ticker}"
        response = requests.get(url, headers=self._get_headers("GET", path))
        market_data = response.json() if response.status_code == 200 else {}

        # Get order book
        book_path = f"/trade-api/v2/markets/{ticker}/orderbook"
        book_url = f"{self.BASE_URL}/markets/{ticker}/orderbook"
        book_response = requests.get(book_url, headers=self._get_headers("GET", book_path))
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
        path = "/trade-api/v2/portfolio/orders"
        url = f"{self.BASE_URL}/portfolio/orders"

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

        body = json.dumps(payload)
        response = requests.post(url, data=body, headers=self._get_headers("POST", path, body))
        return response.json()

    def get_balance(self) -> Dict[str, Any]:
        """
        Get portfolio balance.
        """
        path = "/trade-api/v2/portfolio/balance"
        url = f"{self.BASE_URL}/portfolio/balance"
        response = requests.get(url, headers=self._get_headers("GET", path))
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

        path = "/trade-api/v2/markets"
        response = requests.get(url, params=params, headers=self._get_headers("GET", path))

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

        path = "/trade-api/v2/events"
        response = requests.get(url, params=params, headers=self._get_headers("GET", path))

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

        path = f"/trade-api/v2/series/{series_ticker}/markets/{market_ticker}/candlesticks"
        response = requests.get(url, params=params, headers=self._get_headers("GET", path))

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
