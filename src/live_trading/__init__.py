"""
Live Trading Infrastructure.

Components for real-time NFL arbitrage trading:
- EventCorrelator: Sync NFL game state with Kalshi market prices
- ArbitrageDetector: Identify trading opportunities (Phase 3)
- LiveTradingEngine: Orchestrate all components (Phase 5)
"""

from src.live_trading.arbitrage_detector import ArbitrageDetector, ArbitrageSignal
from src.live_trading.event_correlator import EventCorrelator

__all__ = ["EventCorrelator", "ArbitrageDetector", "ArbitrageSignal"]
