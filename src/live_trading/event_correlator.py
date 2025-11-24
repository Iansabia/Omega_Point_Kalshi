"""
Event Correlator for NFL Game and Kalshi Market Synchronization.

Correlates live NFL game events with Kalshi market prices to enable
real-time arbitrage detection.

Features:
- Map NFL game IDs to Kalshi market tickers
- Sync timestamps between data streams
- Maintain latest state from both sources
- Detect data freshness and staleness
- Provide unified game state + market price view
"""

import logging
import time
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)


class EventCorrelator:
    """
    Correlates NFL game events with Kalshi market updates.

    Maintains a synchronized view of:
    - NFL game state (score, quarter, possession, etc.)
    - Kalshi market prices (bid, ask, mid, spread)
    - Data freshness/staleness indicators
    """

    def __init__(self, staleness_threshold: float = 10.0):
        """
        Initialize event correlator.

        Args:
            staleness_threshold: Max age (seconds) before data is considered stale
        """
        self.staleness_threshold = staleness_threshold

        # Mapping: game_id -> ticker
        self.game_to_ticker: Dict[str, str] = {}

        # Mapping: ticker -> game_id
        self.ticker_to_game: Dict[str, str] = {}

        # Latest NFL game states: game_id -> state_dict
        self.nfl_states: Dict[str, Dict[str, Any]] = {}

        # Latest Kalshi market prices: ticker -> price_dict
        self.kalshi_prices: Dict[str, Dict[str, Any]] = {}

        # Track active games
        self.active_games: Set[str] = set()

        logger.info(f"EventCorrelator initialized (staleness threshold: {staleness_threshold}s)")

    def register_game_market(self, game_id: str, ticker: str):
        """
        Register a mapping between NFL game ID and Kalshi market ticker.

        Args:
            game_id: Sportradar game ID
            ticker: Kalshi market ticker (e.g., 'KXMVENFLSINGLEGAME-S2025...')

        Example:
            correlator.register_game_market('sr:match:12345', 'KXMVENFLSINGLEGAME-S2025-BAL-KC')
        """
        self.game_to_ticker[game_id] = ticker
        self.ticker_to_game[ticker] = game_id
        self.active_games.add(game_id)

        logger.info(f"📍 Registered: {game_id} <-> {ticker}")

    def update_nfl_state(self, game_id: str, state: Dict[str, Any]):
        """
        Update NFL game state from Sportradar.

        Args:
            game_id: Sportradar game ID
            state: Game state dict (from sportradar_client.parse_game_state())

        Expected state keys:
        - home_score, away_score, score_diff
        - quarter, clock, clock_seconds, time_remaining
        - possession, yardline, down, distance
        - status, timestamp
        """
        if game_id not in self.active_games:
            logger.warning(f"Received update for unregistered game: {game_id}")
            return

        # Add update timestamp if not present
        if "timestamp" not in state:
            state["timestamp"] = time.time()

        self.nfl_states[game_id] = state

        # Check if game is finished
        if state.get("status") in ["closed", "complete"]:
            logger.info(f"🏁 Game finished: {game_id} (Final: {state.get('home_score')}-{state.get('away_score')})")

    def update_kalshi_price(self, ticker: str, price_data: Dict[str, Any]):
        """
        Update Kalshi market price from WebSocket.

        Args:
            ticker: Kalshi market ticker
            price_data: Price dict (from kalshi_websocket._parse_orderbook())

        Expected price_data keys:
        - yes_bid, yes_ask, mid_price, spread
        - timestamp
        """
        if ticker not in self.ticker_to_game:
            logger.warning(f"Received price update for unregistered ticker: {ticker}")
            return

        # Add update timestamp if not present
        if "timestamp" not in price_data:
            price_data["timestamp"] = time.time()

        self.kalshi_prices[ticker] = price_data

    def get_correlated_state(self, game_id: str) -> Optional[Dict[str, Any]]:
        """
        Get unified state combining NFL game state and Kalshi market price.

        Args:
            game_id: Sportradar game ID

        Returns:
            Dict with combined state, or None if data is missing/stale

        Returns structure:
            {
                'game_id': str,
                'ticker': str,
                'nfl': {game state from Sportradar},
                'kalshi': {price data from Kalshi},
                'is_fresh': bool,
                'data_age': {'nfl': float, 'kalshi': float},
                'timestamp': float
            }
        """
        # Check if game is registered
        if game_id not in self.game_to_ticker:
            logger.debug(f"Game not registered: {game_id}")
            return None

        ticker = self.game_to_ticker[game_id]

        # Check if we have data from both sources
        if game_id not in self.nfl_states:
            logger.debug(f"No NFL state for game: {game_id}")
            return None

        if ticker not in self.kalshi_prices:
            logger.debug(f"No Kalshi price for ticker: {ticker}")
            return None

        nfl_state = self.nfl_states[game_id]
        kalshi_price = self.kalshi_prices[ticker]

        # Calculate data age
        current_time = time.time()
        nfl_age = current_time - nfl_state.get("timestamp", 0)
        kalshi_age = current_time - kalshi_price.get("timestamp", 0)

        # Check freshness
        is_fresh = nfl_age < self.staleness_threshold and kalshi_age < self.staleness_threshold

        if not is_fresh:
            logger.warning(
                f"⚠️  Stale data for {game_id}: NFL age={nfl_age:.1f}s, Kalshi age={kalshi_age:.1f}s (threshold={self.staleness_threshold}s)"
            )

        return {
            "game_id": game_id,
            "ticker": ticker,
            "nfl": nfl_state,
            "kalshi": kalshi_price,
            "is_fresh": is_fresh,
            "data_age": {"nfl": nfl_age, "kalshi": kalshi_age},
            "timestamp": current_time,
        }

    def get_all_correlated_states(self, require_fresh: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Get all correlated states for active games.

        Args:
            require_fresh: If True, only return fresh data (age < staleness_threshold)

        Returns:
            Dict mapping game_id -> correlated_state
        """
        states = {}

        for game_id in self.active_games:
            correlated = self.get_correlated_state(game_id)

            if correlated is None:
                continue

            if require_fresh and not correlated["is_fresh"]:
                continue

            states[game_id] = correlated

        return states

    def remove_game(self, game_id: str):
        """
        Remove a game from tracking (e.g., when game is finished).

        Args:
            game_id: Sportradar game ID
        """
        if game_id in self.game_to_ticker:
            ticker = self.game_to_ticker[game_id]

            # Clean up mappings
            del self.game_to_ticker[game_id]
            del self.ticker_to_game[ticker]

            # Clean up states
            self.nfl_states.pop(game_id, None)
            self.kalshi_prices.pop(ticker, None)

            self.active_games.discard(game_id)

            logger.info(f"🗑️  Removed game from tracking: {game_id}")

    def get_stats(self) -> Dict[str, Any]:
        """Get correlation statistics."""
        nfl_with_data = len(self.nfl_states)
        kalshi_with_data = len(self.kalshi_prices)
        fresh_count = len(self.get_all_correlated_states(require_fresh=True))

        return {
            "active_games": len(self.active_games),
            "nfl_states_cached": nfl_with_data,
            "kalshi_prices_cached": kalshi_with_data,
            "fresh_correlated_states": fresh_count,
            "staleness_threshold": self.staleness_threshold,
        }

    def __repr__(self):
        stats = self.get_stats()
        return (
            f"EventCorrelator(active_games={stats['active_games']}, "
            f"fresh_states={stats['fresh_correlated_states']}, "
            f"staleness_threshold={stats['staleness_threshold']}s)"
        )


# Example usage
async def example_usage():
    """
    Example: Wire up Sportradar and Kalshi streams to event correlator.
    """
    import asyncio
    from src.data.sportradar_client import SportradarClient
    from src.execution.kalshi_websocket import KalshiWebSocket

    # Initialize components
    correlator = EventCorrelator(staleness_threshold=10.0)
    sportradar = SportradarClient()
    kalshi_ws = KalshiWebSocket()

    # Register game-to-market mapping
    game_id = "sr:match:12345"  # Sportradar game ID
    ticker = "KXMVENFLSINGLEGAME-S2025-BAL-KC"  # Kalshi market ticker

    correlator.register_game_market(game_id, ticker)

    # Callback for NFL game updates
    def on_nfl_update(game_id: str, game_state: Dict):
        """Handle NFL game state updates."""
        correlator.update_nfl_state(game_id, game_state)

        # Check if we have correlated state
        correlated = correlator.get_correlated_state(game_id)
        if correlated and correlated["is_fresh"]:
            print(f"\n[Correlated State Available]")
            print(f"  Game: {game_id}")
            print(f"  Score: {correlated['nfl']['home_score']}-{correlated['nfl']['away_score']}")
            print(f"  Kalshi Mid Price: ${correlated['kalshi']['mid_price']:.2f}")
            print(f"  Data Age: NFL={correlated['data_age']['nfl']:.1f}s, Kalshi={correlated['data_age']['kalshi']:.1f}s")

    # Callback for Kalshi price updates
    async def on_kalshi_update(ticker: str, price_data: Dict):
        """Handle Kalshi price updates."""
        correlator.update_kalshi_price(ticker, price_data)

    # Connect Kalshi WebSocket
    await kalshi_ws.connect()
    await kalshi_ws.subscribe_orderbook(ticker)

    # Start both streams
    await asyncio.gather(
        sportradar.poll_live_games([game_id], on_nfl_update, interval=2.0),  # Poll NFL every 2s
        kalshi_ws.listen(on_kalshi_update),  # Stream Kalshi prices
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(example_usage())
