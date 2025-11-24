"""
Live Trading Engine for NFL Momentum Arbitrage.

Orchestrates all components for real-time trading:
1. ESPN → NFL game state
2. Kalshi WebSocket → Market prices
3. Event Correlator → Sync streams
4. Win Probability Model → True EV prediction
5. Arbitrage Detector → Trading signals
6. Risk Manager → Position management
7. Kalshi Client → Order execution

Architecture:
    NFL Stream ──┐
                 ├──> Event Correlator ──> Model Inference ──> Arbitrage Detector ──> Trade Execution
    Kalshi WS ───┘

Usage:
    engine = LiveTradingEngine(...)
    await engine.start()
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.data.espn_client import ESPNClient
from src.execution.kalshi_client import KalshiClient
from src.execution.kalshi_websocket import KalshiWebSocket
from src.live_trading.arbitrage_detector import ArbitrageDetector, ArbitrageSignal
from src.live_trading.event_correlator import EventCorrelator
from src.models.win_probability_inference import WinProbabilityInference
from src.risk.momentum_risk_manager import MomentumRiskLimits, MomentumRiskManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

logger = logging.getLogger(__name__)


class LiveTradingEngine:
    """
    Live trading engine for NFL momentum arbitrage.

    Manages the entire trading pipeline from data ingestion to order execution.
    """

    def __init__(
        self,
        model_path: str = "models/win_probability_model.pkl",
        risk_limits: Optional[MomentumRiskLimits] = None,
        paper_trading: bool = True,
        min_edge: float = 0.10,
        min_confidence: float = 0.5,
        max_spread: float = 0.10,
    ):
        """
        Initialize live trading engine.

        Args:
            model_path: Path to trained win probability model
            risk_limits: Momentum risk limits
            paper_trading: If True, simulate trades without real execution
            min_edge: Minimum edge for arbitrage signal (10% default)
            min_confidence: Minimum model confidence (50% default)
            max_spread: Maximum bid-ask spread (10% default)
        """
        self.paper_trading = paper_trading

        logger.info("=" * 60)
        logger.info("Initializing Live Trading Engine")
        logger.info("=" * 60)
        logger.info(f"Mode: {'PAPER TRADING' if paper_trading else '⚠️  LIVE TRADING'}")
        logger.info(f"Model: {model_path}")
        logger.info(f"Min Edge: {min_edge:.1%}")
        logger.info(f"Min Confidence: {min_confidence:.1%}")

        # Initialize components
        logger.info("\n🔧 Initializing components...")

        # 1. Data sources
        self.espn = ESPNClient()
        self.kalshi_ws = KalshiWebSocket()
        self.kalshi_client = KalshiClient()

        # 2. Event correlation
        self.correlator = EventCorrelator(staleness_threshold=10.0)

        # 3. Win probability inference
        self.wp_inference = WinProbabilityInference(model_path=model_path)

        # 4. Arbitrage detection
        self.arbitrage_detector = ArbitrageDetector(
            min_edge=min_edge, min_confidence=min_confidence, max_spread=max_spread, require_fresh_data=True
        )

        # 5. Risk management
        self.risk_manager = MomentumRiskManager(risk_limits or MomentumRiskLimits())

        # Track active games
        self.active_games: Dict[str, Dict[str, Any]] = {}
        # game_id -> {'ticker', 'espn_id', 'home_team', 'away_team'}

        # Statistics
        self.total_signals = 0
        self.total_trades = 0
        self.running = False

        logger.info("✅ All components initialized")

    def register_game(self, sportradar_game_id: str, kalshi_ticker: str, home_team: str, away_team: str):
        """
        Register a game to track.

        Args:
            sportradar_game_id: ESPN game ID (kept for backward compatibility)
            kalshi_ticker: Kalshi market ticker
            home_team: Home team code
            away_team: Away team code
        """
        self.active_games[sportradar_game_id] = {
            "ticker": kalshi_ticker,
            "espn_id": sportradar_game_id,
            "home_team": home_team,
            "away_team": away_team,
        }

        # Register in correlator
        self.correlator.register_game_market(sportradar_game_id, kalshi_ticker)

        logger.info(f"📍 Registered game: {home_team} vs {away_team}")
        logger.info(f"   ESPN ID: {sportradar_game_id}")
        logger.info(f"   Kalshi Ticker: {kalshi_ticker}")

    async def start(self):
        """
        Start the live trading engine.

        Launches all data streams and begins trading.
        """
        if self.running:
            logger.warning("Engine already running")
            return

        self.running = True

        logger.info("\n" + "=" * 60)
        logger.info("🚀 Starting Live Trading Engine")
        logger.info("=" * 60)

        if len(self.active_games) == 0:
            logger.error("❌ No games registered. Call register_game() first.")
            return

        # Connect to Kalshi WebSocket
        await self.kalshi_ws.connect()

        # Subscribe to all registered games
        for game_id, game_info in self.active_games.items():
            ticker = game_info["ticker"]
            await self.kalshi_ws.subscribe_orderbook(ticker)

        # Start data streams
        game_ids = list(self.active_games.keys())

        logger.info(f"\n✅ Tracking {len(game_ids)} games")
        logger.info("🔄 Starting data streams...")

        # Run both streams concurrently
        await asyncio.gather(
            self._run_nfl_stream(game_ids),
            self._run_kalshi_stream(),
            self._run_trading_loop(),
        )

    async def stop(self):
        """Stop the live trading engine."""
        logger.info("\n🛑 Stopping Live Trading Engine...")
        self.running = False

        # Disconnect Kalshi WebSocket
        await self.kalshi_ws.disconnect()

        # Close ESPN client
        await self.espn.close()

        # Close any open positions
        open_positions = self.risk_manager.get_open_positions()
        if open_positions:
            logger.warning(f"⚠️  {len(open_positions)} open positions remaining")
            # In production, you'd want to close these

        logger.info("✅ Engine stopped")

    async def _run_nfl_stream(self, game_ids: List[str]):
        """
        Run ESPN NFL data stream.

        Args:
            game_ids: List of ESPN game IDs to track
        """
        # Poll each game concurrently
        tasks = []
        for game_id in game_ids:

            def make_callback(gid):
                def on_nfl_update(game_state: Dict):
                    """Handle NFL game state update."""
                    self.correlator.update_nfl_state(gid, game_state)

                return on_nfl_update

            task = self.espn.poll_live_game(game_id=game_id, callback=make_callback(game_id), interval=2)
            tasks.append(task)

        # Run all polling tasks concurrently
        await asyncio.gather(*tasks)

    async def _run_kalshi_stream(self):
        """Run Kalshi WebSocket price stream."""

        async def on_kalshi_update(ticker: str, price_data: Dict):
            """Handle Kalshi price update."""
            self.correlator.update_kalshi_price(ticker, price_data)

        # Listen for price updates
        await self.kalshi_ws.listen(on_kalshi_update)

    async def _run_trading_loop(self):
        """
        Main trading loop.

        Checks for arbitrage opportunities every second.
        """
        logger.info("🔄 Trading loop started")

        while self.running:
            try:
                # Get all correlated states (fresh data only)
                correlated_states = self.correlator.get_all_correlated_states(require_fresh=True)

                if not correlated_states:
                    await asyncio.sleep(1.0)
                    continue

                # Run inference for each game
                for game_id, correlated_state in correlated_states.items():
                    await self._process_game(game_id, correlated_state)

            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)

            # Wait before next iteration
            await asyncio.sleep(1.0)

    async def _process_game(self, game_id: str, correlated_state: Dict[str, Any]):
        """
        Process a single game for trading opportunities.

        Args:
            game_id: Sportradar game ID
            correlated_state: Correlated state from EventCorrelator
        """
        # Get win probability prediction
        nfl_state = correlated_state["nfl"]
        wp_result = self.wp_inference.predict_from_sportradar(nfl_state)
        model_wp = wp_result["home_wp"]

        # Detect arbitrage
        signal = self.arbitrage_detector.detect(correlated_state, model_wp)

        if signal:
            self.total_signals += 1
            await self._execute_signal(signal)

    async def _execute_signal(self, signal: ArbitrageSignal):
        """
        Execute a trading signal.

        Args:
            signal: ArbitrageSignal to execute
        """
        ticker = signal.ticker
        direction = signal.direction
        edge = signal.edge

        logger.info("\n" + "=" * 60)
        logger.info(f"🎯 EXECUTING SIGNAL: {signal}")
        logger.info("=" * 60)

        # Check risk management
        game_id = signal.game_id
        data_age = signal.game_state.get("timestamp", 0)
        current_time = signal.timestamp
        age_seconds = current_time - data_age

        # Calculate position size (simplified)
        position_value = 100.0  # $100 per trade (you'd use Kelly here)

        # Check if trade allowed
        allowed, reason = self.risk_manager.can_trade(
            ticker=ticker, edge=abs(edge), data_age=age_seconds, game_id=game_id, position_value=position_value
        )

        if not allowed:
            logger.warning(f"⚠️  Trade blocked: {reason}")
            return

        # Execute trade
        if self.paper_trading:
            self._execute_paper_trade(signal, position_value)
        else:
            await self._execute_live_trade(signal, position_value)

        self.total_trades += 1

    def _execute_paper_trade(self, signal: ArbitrageSignal, position_value: float):
        """
        Execute paper trade (simulation).

        Args:
            signal: Trading signal
            position_value: Position size in dollars
        """
        logger.info("📝 PAPER TRADE:")
        logger.info(f"   Ticker: {signal.ticker}")
        logger.info(f"   Direction: {signal.direction}")
        logger.info(f"   Entry Price: ${signal.market_price:.2f}")
        logger.info(f"   Position Size: ${position_value:.0f}")
        logger.info(f"   Expected Edge: {signal.edge:+.1%}")

        # Record position opening
        self.risk_manager.open_position(
            ticker=signal.ticker,
            side=signal.direction,
            quantity=position_value / signal.market_price,
            entry_price=signal.market_price,
            game_id=signal.game_id,
        )

    async def _execute_live_trade(self, signal: ArbitrageSignal, position_value: float):
        """
        Execute live trade on Kalshi.

        Args:
            signal: Trading signal
            position_value: Position size in dollars
        """
        logger.info("💰 LIVE TRADE:")
        logger.info(f"   Ticker: {signal.ticker}")
        logger.info(f"   Direction: {signal.direction}")

        # Get current orderbook
        orderbook = self.kalshi_client.get_orderbook(signal.ticker)

        # Place market order (simplified - you'd use limit orders in production)
        if signal.direction == "BUY":
            price = orderbook["asks"][0]["price"] if orderbook.get("asks") else signal.market_price
        else:
            price = orderbook["bids"][0]["price"] if orderbook.get("bids") else signal.market_price

        quantity = int(position_value / price)

        try:
            order = self.kalshi_client.place_order(
                ticker=signal.ticker, side=signal.direction.lower(), quantity=quantity, price=price, order_type="market"
            )

            logger.info(f"✅ Order placed: {order}")

            # Record position
            self.risk_manager.open_position(
                ticker=signal.ticker, side=signal.direction, quantity=quantity, entry_price=price, game_id=signal.game_id
            )

        except Exception as e:
            logger.error(f"❌ Order failed: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        risk_stats = self.risk_manager.get_stats()
        correlator_stats = self.correlator.get_stats()
        detector_stats = self.arbitrage_detector.get_stats()

        return {
            "running": self.running,
            "active_games": len(self.active_games),
            "total_signals": self.total_signals,
            "total_trades": self.total_trades,
            "paper_trading": self.paper_trading,
            "risk": risk_stats,
            "correlator": correlator_stats,
            "detector": detector_stats,
        }


# Example usage
async def main():
    """Example: Run live trading engine in paper trading mode."""
    logger.info("\n" + "=" * 60)
    logger.info("Live Trading Engine - Example")
    logger.info("=" * 60)

    # Initialize engine
    engine = LiveTradingEngine(
        model_path="models/win_probability_model.pkl",
        paper_trading=True,  # PAPER TRADING MODE
        min_edge=0.10,
        min_confidence=0.5,
    )

    # Register a game (you'd get this from Kalshi/Sportradar APIs)
    sportradar_game_id = "sr:match:12345"
    kalshi_ticker = "KXMVENFLSINGLEGAME-S2025-BAL-KC"

    engine.register_game(sportradar_game_id=sportradar_game_id, kalshi_ticker=kalshi_ticker, home_team="BAL", away_team="KC")

    # Start engine
    logger.info("\n⚠️  This is a demo - would run indefinitely in production")
    logger.info("   In production, you'd run: await engine.start()")

    # Get stats
    stats = engine.get_stats()
    logger.info(f"\n📊 Engine Stats:")
    logger.info(f"   Active Games: {stats['active_games']}")
    logger.info(f"   Paper Trading: {stats['paper_trading']}")
    logger.info(f"   Signals: {stats['total_signals']}")
    logger.info(f"   Trades: {stats['total_trades']}")

    logger.info("\n" + "=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
