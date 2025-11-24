"""
Kalshi WebSocket Client for Real-Time Market Data.

Provides live orderbook updates and trade notifications via WebSocket connection.
Used for low-latency arbitrage trading (20-100ms price updates).
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional, Set

import websockets
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class KalshiWebSocket:
    """
    WebSocket client for Kalshi real-time market data.

    Features:
    - Real-time orderbook updates (bid/ask prices)
    - Trade notifications
    - Auto-reconnection on disconnect
    - Subscription management
    """

    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize Kalshi WebSocket client.

        Args:
            api_key: Kalshi API key ID (defaults to KALSHI_API_KEY_ID env var)
            base_url: WebSocket base URL (defaults to production)
        """
        self.api_key = api_key or os.getenv("KALSHI_API_KEY_ID")
        self.base_url = base_url or os.getenv("KALSHI_WS_URL", "wss://trading-api.kalshi.com/trade-api/ws/v2")

        self.websocket = None
        self.subscriptions: Set[str] = set()
        self.is_connected = False
        self.reconnect_delay = 1.0  # Start with 1 second
        self.max_reconnect_delay = 60.0  # Max 60 seconds
        self.callback: Optional[Callable[[str, Dict], None]] = None

        if not self.api_key:
            logger.warning("Kalshi API key not found. WebSocket will not connect.")

    async def connect(self):
        """
        Establish WebSocket connection to Kalshi.

        Handles authentication and initial setup.
        """
        if not self.api_key:
            logger.error("Cannot connect: No API key provided")
            return

        try:
            logger.info(f"Connecting to Kalshi WebSocket: {self.base_url}")

            # Connect with authentication header
            headers = {"Authorization": f"Bearer {self.api_key}"}

            self.websocket = await websockets.connect(self.base_url, extra_headers=headers)

            self.is_connected = True
            self.reconnect_delay = 1.0  # Reset backoff on successful connection
            logger.info("✅ Connected to Kalshi WebSocket")

            # Resubscribe to previous channels after reconnection
            if self.subscriptions:
                await self._resubscribe()

        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            self.is_connected = False
            raise

    async def disconnect(self):
        """Close WebSocket connection gracefully."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("Disconnected from Kalshi WebSocket")
            except Exception as e:
                logger.error(f"Error during disconnect: {e}")

        self.is_connected = False
        self.websocket = None

    async def subscribe_orderbook(self, ticker: str):
        """
        Subscribe to real-time orderbook updates for a market.

        Args:
            ticker: Market ticker (e.g., 'KXMVENFLSINGLEGAME-...')

        Orderbook updates include:
        - Best bid/ask prices
        - Orderbook depth
        - Last trade price
        """
        if not self.is_connected:
            logger.warning(f"Cannot subscribe: Not connected (ticker: {ticker})")
            return

        try:
            subscribe_msg = {"cmd": "subscribe", "type": "orderbook_delta", "market_ticker": ticker}

            await self.websocket.send(json.dumps(subscribe_msg))
            self.subscriptions.add(ticker)
            logger.info(f"📊 Subscribed to orderbook: {ticker}")

        except Exception as e:
            logger.error(f"Failed to subscribe to {ticker}: {e}")

    async def unsubscribe_orderbook(self, ticker: str):
        """Unsubscribe from orderbook updates."""
        if not self.is_connected:
            return

        try:
            unsubscribe_msg = {"cmd": "unsubscribe", "type": "orderbook_delta", "market_ticker": ticker}

            await self.websocket.send(json.dumps(unsubscribe_msg))
            self.subscriptions.discard(ticker)
            logger.info(f"Unsubscribed from orderbook: {ticker}")

        except Exception as e:
            logger.error(f"Failed to unsubscribe from {ticker}: {e}")

    async def _resubscribe(self):
        """Re-subscribe to all previous subscriptions after reconnection."""
        logger.info(f"Re-subscribing to {len(self.subscriptions)} markets...")

        for ticker in list(self.subscriptions):
            try:
                subscribe_msg = {"cmd": "subscribe", "type": "orderbook_delta", "market_ticker": ticker}
                await self.websocket.send(json.dumps(subscribe_msg))
            except Exception as e:
                logger.error(f"Failed to resubscribe to {ticker}: {e}")

    async def listen(self, callback: Callable[[str, Dict], None]):
        """
        Listen for incoming WebSocket messages.

        Args:
            callback: Function called with (ticker, message_data) for each update

        The callback receives:
        - ticker: Market ticker string
        - message_data: Dict with orderbook data (bid, ask, last_price, etc.)
        """
        self.callback = callback

        while True:
            try:
                if not self.is_connected:
                    await self.connect()

                async for message in self.websocket:
                    try:
                        data = json.loads(message)
                        await self._handle_message(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse message: {e}")
                    except Exception as e:
                        logger.error(f"Error handling message: {e}")

            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed, reconnecting...")
                self.is_connected = False
                await self._reconnect_with_backoff()

            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.is_connected = False
                await self._reconnect_with_backoff()

    async def _handle_message(self, data: Dict):
        """
        Handle incoming WebSocket message.

        Processes orderbook updates and calls user callback.
        """
        msg_type = data.get("type")

        if msg_type == "orderbook_delta":
            # Orderbook update
            ticker = data.get("market_ticker")
            if ticker and self.callback:
                orderbook_data = self._parse_orderbook(data)
                await self.callback(ticker, orderbook_data)

        elif msg_type == "error":
            logger.error(f"Kalshi WebSocket error: {data.get('message')}")

        elif msg_type == "subscribed":
            logger.debug(f"Subscription confirmed: {data.get('market_ticker')}")

        else:
            logger.debug(f"Unknown message type: {msg_type}")

    def _parse_orderbook(self, data: Dict) -> Dict[str, Any]:
        """
        Parse orderbook delta message into standardized format.

        Returns:
            Dict with bid, ask, last_price, spread, mid_price, timestamp
        """
        # Extract best bid/ask from orderbook delta
        orderbook = data.get("msg", {})

        # Yes side (long position)
        yes_bids = orderbook.get("yes", {}).get("bids", [])
        yes_asks = orderbook.get("yes", {}).get("asks", [])

        # Get best prices (orderbook is sorted by price)
        yes_bid = yes_bids[0]["price"] / 100 if yes_bids else None  # Convert cents to dollars
        yes_ask = yes_asks[0]["price"] / 100 if yes_asks else None

        # Calculate mid price and spread
        if yes_bid and yes_ask:
            mid_price = (yes_bid + yes_ask) / 2
            spread = yes_ask - yes_bid
        else:
            mid_price = yes_bid or yes_ask or None
            spread = None

        return {
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "mid_price": mid_price,
            "spread": spread,
            "timestamp": time.time(),
            "raw": data,  # Keep raw data for debugging
        }

    async def _reconnect_with_backoff(self):
        """Reconnect with exponential backoff."""
        await asyncio.sleep(self.reconnect_delay)
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        logger.info(f"Reconnecting (backoff: {self.reconnect_delay}s)...")

    async def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current mid price for a market (synchronous-style helper).

        Note: This requires an active subscription and recent update.
        For one-off price checks, use REST API instead.

        Args:
            ticker: Market ticker

        Returns:
            Current mid price or None if not available
        """
        # This is a simplified version - in practice, you'd cache latest prices
        logger.warning("get_current_price() requires implementing price caching")
        return None


# Example usage
async def main():
    """Example: Subscribe to NFL market and print price updates."""

    async def on_price_update(ticker: str, orderbook: Dict):
        """Callback for price updates."""
        print(f"[{ticker}] Bid: ${orderbook['yes_bid']:.2f}, Ask: ${orderbook['yes_ask']:.2f}, "
              f"Mid: ${orderbook['mid_price']:.2f}, Spread: ${orderbook['spread']:.4f}")

    # Initialize WebSocket client
    ws_client = KalshiWebSocket()

    # Connect
    await ws_client.connect()

    # Subscribe to a market (replace with actual ticker)
    ticker = "KXMVENFLSINGLEGAME-S2025..."
    await ws_client.subscribe_orderbook(ticker)

    # Listen for updates
    await ws_client.listen(on_price_update)


if __name__ == "__main__":
    asyncio.run(main())
