"""
Arbitrage Detector for NFL Momentum Trading.

Detects arbitrage opportunities by comparing:
- Model Win Probability (true EV) - from XGBoost trained on historical data
- Market Price (human EV) - from Kalshi orderbook

Core Strategy:
    "If the ravens score a touchdown the true EV for them winning might be 75%
     but momentum and human error may cause it to go to 90%."

    → Trade when: |Model WP - Market Price| > threshold (e.g., 10%)

Features:
- Edge calculation (model vs market)
- Signal filtering (min edge, confidence, data freshness)
- Trade direction (buy/sell)
- Position sizing recommendations
- Momentum detection (recent price movement)
"""

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ArbitrageSignal:
    """Represents a trading signal with arbitrage opportunity."""

    def __init__(
        self,
        game_id: str,
        ticker: str,
        model_wp: float,
        market_price: float,
        edge: float,
        direction: str,
        confidence: float,
        game_state: Dict[str, Any],
        market_data: Dict[str, Any],
        timestamp: float,
    ):
        """
        Initialize arbitrage signal.

        Args:
            game_id: Sportradar game ID
            ticker: Kalshi market ticker
            model_wp: Model win probability (0-1)
            market_price: Kalshi market mid price (0-1)
            edge: Arbitrage edge (model_wp - market_price)
            direction: Trade direction ('BUY' or 'SELL')
            confidence: Signal confidence (0-1)
            game_state: NFL game state
            market_data: Kalshi market data
            timestamp: Signal generation time
        """
        self.game_id = game_id
        self.ticker = ticker
        self.model_wp = model_wp
        self.market_price = market_price
        self.edge = edge
        self.direction = direction
        self.confidence = confidence
        self.game_state = game_state
        self.market_data = market_data
        self.timestamp = timestamp

    def __repr__(self):
        return (
            f"ArbitrageSignal({self.direction} {self.ticker}, "
            f"edge={self.edge:.1%}, model={self.model_wp:.1%}, "
            f"market={self.market_price:.1%})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dict."""
        return {
            "game_id": self.game_id,
            "ticker": self.ticker,
            "model_wp": self.model_wp,
            "market_price": self.market_price,
            "edge": self.edge,
            "direction": self.direction,
            "confidence": self.confidence,
            "game_state": self.game_state,
            "market_data": self.market_data,
            "timestamp": self.timestamp,
        }


class ArbitrageDetector:
    """
    Detects arbitrage opportunities from model predictions and market prices.

    This is the core of the momentum trading strategy.
    """

    def __init__(
        self,
        min_edge: float = 0.10,  # Minimum 10% edge
        min_confidence: float = 0.5,  # Minimum 50% confidence
        max_spread: float = 0.10,  # Maximum 10% bid-ask spread
        require_fresh_data: bool = True,  # Require fresh data from correlator
    ):
        """
        Initialize arbitrage detector.

        Args:
            min_edge: Minimum edge required for signal (0-1, e.g., 0.10 = 10%)
            min_confidence: Minimum model confidence (0-1)
            max_spread: Maximum allowable bid-ask spread (0-1)
            require_fresh_data: Require fresh data from event correlator
        """
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.max_spread = max_spread
        self.require_fresh_data = require_fresh_data

        self.signals_generated = 0
        self.signals_filtered = 0

        logger.info(f"ArbitrageDetector initialized:")
        logger.info(f"  Min Edge: {min_edge:.1%}")
        logger.info(f"  Min Confidence: {min_confidence:.1%}")
        logger.info(f"  Max Spread: {max_spread:.1%}")

    def detect(self, correlated_state: Dict[str, Any], model_wp: float) -> Optional[ArbitrageSignal]:
        """
        Detect arbitrage opportunity from correlated game state + model prediction.

        Args:
            correlated_state: Correlated state from EventCorrelator
            model_wp: Model win probability from WinProbabilityInference

        Returns:
            ArbitrageSignal if opportunity detected, else None
        """
        # Extract data
        game_id = correlated_state["game_id"]
        ticker = correlated_state["ticker"]
        nfl_state = correlated_state["nfl"]
        kalshi_state = correlated_state["kalshi"]
        is_fresh = correlated_state["is_fresh"]

        # Filter: Require fresh data
        if self.require_fresh_data and not is_fresh:
            logger.debug(f"Filtered: Stale data for {game_id}")
            self.signals_filtered += 1
            return None

        # Get market price (mid price = average of bid/ask)
        market_price = kalshi_state.get("mid_price")
        if market_price is None:
            logger.warning(f"No market price for {ticker}")
            self.signals_filtered += 1
            return None

        # Get bid-ask spread
        spread = kalshi_state.get("spread")
        if spread is None or spread > self.max_spread:
            logger.debug(f"Filtered: Spread too wide ({spread:.1%}) for {ticker}")
            self.signals_filtered += 1
            return None

        # Adjust model WP based on possession
        # The model predicts possession team WP, but market price is for home team
        possession = nfl_state.get("possession", "home")
        if possession == "away":
            # If away team has ball, flip model WP to get home team WP
            home_model_wp = 1.0 - model_wp
        else:
            home_model_wp = model_wp

        # Calculate edge
        edge = home_model_wp - market_price

        # Determine trade direction
        if edge > 0:
            direction = "BUY"  # Model says higher than market → buy
        else:
            direction = "SELL"  # Model says lower than market → sell

        # Filter: Min edge
        if abs(edge) < self.min_edge:
            logger.debug(f"Filtered: Edge too small ({edge:.1%}) for {ticker}")
            self.signals_filtered += 1
            return None

        # Calculate confidence (how confident is the model?)
        # Use model's distance from 50/50
        confidence = abs(home_model_wp - 0.5) * 2

        # Filter: Min confidence
        if confidence < self.min_confidence:
            logger.debug(f"Filtered: Confidence too low ({confidence:.1%}) for {ticker}")
            self.signals_filtered += 1
            return None

        # Generate signal
        signal = ArbitrageSignal(
            game_id=game_id,
            ticker=ticker,
            model_wp=home_model_wp,
            market_price=market_price,
            edge=edge,
            direction=direction,
            confidence=confidence,
            game_state=nfl_state,
            market_data=kalshi_state,
            timestamp=time.time(),
        )

        self.signals_generated += 1

        logger.info(f"🎯 SIGNAL: {signal}")
        logger.info(f"   Score: {nfl_state.get('home_score')}-{nfl_state.get('away_score')}")
        logger.info(f"   Q{nfl_state.get('quarter')}, {nfl_state.get('clock')}")
        logger.info(f"   Model WP: {home_model_wp:.1%}, Market: {market_price:.1%}")
        logger.info(f"   Edge: {edge:+.1%}, Direction: {direction}")

        return signal

    def detect_batch(self, correlated_states: Dict[str, Dict], model_predictions: Dict[str, float]) -> List[ArbitrageSignal]:
        """
        Detect arbitrage opportunities for multiple games.

        Args:
            correlated_states: Dict mapping game_id -> correlated_state
            model_predictions: Dict mapping game_id -> model_wp

        Returns:
            List of ArbitrageSignals
        """
        signals = []

        for game_id, correlated_state in correlated_states.items():
            if game_id not in model_predictions:
                logger.warning(f"No model prediction for {game_id}")
                continue

            model_wp = model_predictions[game_id]

            signal = self.detect(correlated_state, model_wp)
            if signal:
                signals.append(signal)

        return signals

    def get_stats(self) -> Dict[str, Any]:
        """Get detector statistics."""
        total = self.signals_generated + self.signals_filtered
        signal_rate = self.signals_generated / total if total > 0 else 0

        return {
            "signals_generated": self.signals_generated,
            "signals_filtered": self.signals_filtered,
            "total_evaluated": total,
            "signal_rate": signal_rate,
            "min_edge": self.min_edge,
            "min_confidence": self.min_confidence,
            "max_spread": self.max_spread,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.signals_generated = 0
        self.signals_filtered = 0
        logger.info("📊 Statistics reset")


# Example usage
def main():
    """Example: Detect arbitrage from mock data."""
    print("\n" + "=" * 60)
    print("Arbitrage Detector Example")
    print("=" * 60)

    # Initialize detector
    detector = ArbitrageDetector(min_edge=0.10, min_confidence=0.5, max_spread=0.10)

    # Mock correlated state (from EventCorrelator)
    correlated_state = {
        "game_id": "sr:match:12345",
        "ticker": "KXMVENFLSINGLEGAME-S2025-BAL-KC",
        "nfl": {
            "home_score": 21,
            "away_score": 14,
            "score_diff": 7,
            "quarter": 3,
            "clock": "8:45",
            "possession": "home",
            "yardline": 45,
            "down": 2,
            "distance": 7,
        },
        "kalshi": {
            "yes_bid": 0.88,
            "yes_ask": 0.92,
            "mid_price": 0.90,  # Market says 90% chance home wins
            "spread": 0.04,
        },
        "is_fresh": True,
        "data_age": {"nfl": 0.5, "kalshi": 0.3},
    }

    # Model prediction (from WinProbabilityInference)
    model_wp = 0.75  # Model says 75% chance home wins

    print("\n📊 Scenario:")
    print("   Home leading 21-14 in Q3, possession at their 45")
    print(f"   Model WP: {model_wp:.1%}")
    print(f"   Market Price: {correlated_state['kalshi']['mid_price']:.1%}")
    print(f"   Edge: {model_wp - correlated_state['kalshi']['mid_price']:+.1%}")

    # Detect arbitrage
    signal = detector.detect(correlated_state, model_wp)

    if signal:
        print("\n✅ Arbitrage signal generated!")
        print(f"   Direction: {signal.direction}")
        print(f"   Edge: {signal.edge:+.1%}")
        print(f"   Confidence: {signal.confidence:.1%}")
    else:
        print("\n❌ No signal (filtered)")

    # Get stats
    stats = detector.get_stats()
    print(f"\n📊 Detector Stats:")
    print(f"   Signals Generated: {stats['signals_generated']}")
    print(f"   Signals Filtered: {stats['signals_filtered']}")
    print(f"   Signal Rate: {stats['signal_rate']:.1%}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
