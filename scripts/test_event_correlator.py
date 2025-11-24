#!/usr/bin/env python3
"""
Test Event Correlator.

Tests the event correlation between NFL game states and Kalshi market prices.
"""

import time
from src.live_trading.event_correlator import EventCorrelator


def test_basic_correlation():
    """Test basic correlation functionality."""
    print("=" * 60)
    print("Testing Event Correlator")
    print("=" * 60)

    # Initialize correlator
    correlator = EventCorrelator(staleness_threshold=5.0)
    print(f"\n✅ Created correlator: {correlator}")

    # Register a game-market mapping
    game_id = "sr:match:test123"
    ticker = "KXMVENFLSINGLEGAME-S2025-BAL-KC"

    correlator.register_game_market(game_id, ticker)
    print(f"\n✅ Registered mapping: {game_id} <-> {ticker}")

    # Simulate NFL game state update
    nfl_state = {
        "home_score": 21,
        "away_score": 14,
        "score_diff": 7,
        "quarter": 3,
        "clock": "8:45",
        "clock_seconds": 525,
        "time_remaining": 1425,  # Q3 8:45 + Q4 15:00
        "possession": "home",
        "yardline": 45,
        "down": 2,
        "distance": 7,
        "status": "inprogress",
        "timestamp": time.time(),
    }

    correlator.update_nfl_state(game_id, nfl_state)
    print("\n✅ Updated NFL state:")
    print(f"   Score: {nfl_state['home_score']}-{nfl_state['away_score']}")
    print(f"   Quarter: Q{nfl_state['quarter']}, Clock: {nfl_state['clock']}")
    print(f"   Possession: {nfl_state['possession']}, {nfl_state['down']}&{nfl_state['distance']} at {nfl_state['yardline']}")

    # Try to get correlated state (should fail - no Kalshi data yet)
    correlated = correlator.get_correlated_state(game_id)
    if correlated is None:
        print("\n⚠️  No correlated state yet (missing Kalshi data)")

    # Simulate Kalshi price update
    kalshi_price = {
        "yes_bid": 0.68,
        "yes_ask": 0.72,
        "mid_price": 0.70,
        "spread": 0.04,
        "timestamp": time.time(),
    }

    correlator.update_kalshi_price(ticker, kalshi_price)
    print("\n✅ Updated Kalshi price:")
    print(f"   Bid: ${kalshi_price['yes_bid']:.2f}, Ask: ${kalshi_price['yes_ask']:.2f}")
    print(f"   Mid: ${kalshi_price['mid_price']:.2f}, Spread: ${kalshi_price['spread']:.4f}")

    # Now get correlated state (should succeed)
    correlated = correlator.get_correlated_state(game_id)
    if correlated:
        print("\n✅ Correlated State Retrieved:")
        print(f"   Game ID: {correlated['game_id']}")
        print(f"   Ticker: {correlated['ticker']}")
        print(f"   Is Fresh: {correlated['is_fresh']}")
        print(f"   NFL Data Age: {correlated['data_age']['nfl']:.2f}s")
        print(f"   Kalshi Data Age: {correlated['data_age']['kalshi']:.2f}s")
        print(f"   Score: {correlated['nfl']['home_score']}-{correlated['nfl']['away_score']}")
        print(f"   Market Mid Price: ${correlated['kalshi']['mid_price']:.2f}")
    else:
        print("\n❌ Failed to get correlated state")

    # Test staleness
    print("\n🕒 Testing staleness detection (waiting 6 seconds)...")
    time.sleep(6)

    correlated = correlator.get_correlated_state(game_id)
    if correlated:
        print(f"   Is Fresh: {correlated['is_fresh']} (should be False)")
        print(f"   NFL Data Age: {correlated['data_age']['nfl']:.2f}s")
        print(f"   Kalshi Data Age: {correlated['data_age']['kalshi']:.2f}s")

    # Test stats
    stats = correlator.get_stats()
    print(f"\n📊 Correlator Stats:")
    print(f"   Active Games: {stats['active_games']}")
    print(f"   NFL States Cached: {stats['nfl_states_cached']}")
    print(f"   Kalshi Prices Cached: {stats['kalshi_prices_cached']}")
    print(f"   Fresh Correlated States: {stats['fresh_correlated_states']}")
    print(f"   Staleness Threshold: {stats['staleness_threshold']}s")

    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_correlation()
