#!/usr/bin/env python3
"""
Demo Backtest with Synthetic Game Data.

Shows how the momentum arbitrage strategy would work on a simulated NFL game.
Generates realistic price movements with momentum spikes that the strategy can exploit.

Usage:
    python scripts/backtest_demo.py
"""

import logging

import numpy as np
import pandas as pd

from src.live_trading.arbitrage_detector import ArbitrageDetector
from src.models.win_probability_inference import WinProbabilityInference

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def generate_synthetic_game():
    """
    Generate synthetic NFL game with realistic price movements.

    Simulates a close game where:
    - Market prices exhibit momentum (overreact to TDs, big plays)
    - Model predictions are more stable (true win probability)
    - Creates arbitrage opportunities

    Returns:
        DataFrame with columns: minute, score_diff, model_wp, market_price, event
    """
    np.random.seed(42)

    # Game timeline (60 minutes of play)
    minutes = np.arange(0, 61, 1)
    n = len(minutes)

    # Simulate score differential over time (random walk)
    score_changes = np.random.choice([-7, 0, 0, 0, 3, 7], size=n, p=[0.05, 0.70, 0.10, 0.10, 0.025, 0.025])
    score_diff = np.cumsum(score_changes)

    # Calculate true win probability from score diff and time remaining
    time_remaining = 3600 - (minutes * 60)  # Seconds
    model_wp = []

    for i in range(n):
        # Simple model: WP depends on score and time
        # More time = less certain, more score diff = more certain
        base_wp = 0.5 + (score_diff[i] / 28.0)  # +28 points = ~100% win
        time_factor = 1 - (time_remaining[i] / 3600)  # More certain as time runs out
        wp = 0.5 + (base_wp - 0.5) * (0.3 + 0.7 * time_factor)
        wp = np.clip(wp, 0.05, 0.95)
        model_wp.append(wp)

    model_wp = np.array(model_wp)

    # Market price: starts same as model, but overreacts to events
    market_price = model_wp.copy()

    # Add momentum events (TDs, big plays) where market overreacts
    events = []
    for i in range(n):
        if score_changes[i] == 7:  # Touchdown
            # Market overreacts: +15% momentum
            market_price[i : i + 5] += 0.15
            events.append(f"TD @ min {i}")
        elif score_changes[i] == -7:
            market_price[i : i + 5] -= 0.15
            events.append(f"TD allowed @ min {i}")
        elif i in [15, 30, 45]:  # Quarter breaks - price consolidates
            market_price[i : i + 2] = (market_price[i] + model_wp[i]) / 2

    market_price = np.clip(market_price, 0.05, 0.95)

    # Add noise to market price (human irrationality)
    market_price += np.random.normal(0, 0.02, n)
    market_price = np.clip(market_price, 0.05, 0.95)

    # Create DataFrame
    df = pd.DataFrame({"minute": minutes, "score_diff": score_diff, "model_wp": model_wp, "market_price": market_price})

    return df, events


def run_demo_backtest():
    """Run demonstration backtest."""
    logger.info("=" * 80)
    logger.info("MOMENTUM ARBITRAGE DEMO BACKTEST")
    logger.info("=" * 80)
    logger.info("\nSimulating: Ravens vs Chiefs (Close game with momentum swings)")
    logger.info("Strategy: Trade when |Model WP - Market Price| > 10%\n")

    # Generate synthetic game
    game_df, events = generate_synthetic_game()

    logger.info(f"📊 Game Events:")
    for event in events:
        logger.info(f"   - {event}")

    # Initialize detector
    detector = ArbitrageDetector(min_edge=0.10, min_confidence=0.30, max_spread=0.10, require_fresh_data=False)

    # Track trades
    trades = []
    open_position = None
    position_size = 100.0

    logger.info(f"\n🔄 Running backtest (minute-by-minute)...\n")

    # Process each minute
    for idx, row in game_df.iterrows():
        minute = row["minute"]
        model_wp = row["model_wp"]
        market_price = row["market_price"]
        edge = model_wp - market_price

        # Close open position if held for 5 minutes or end of game
        if open_position:
            holding_time = minute - open_position["entry_minute"]

            if holding_time >= 5 or minute == 60:
                # Calculate P&L
                entry_price = open_position["entry_price"]
                exit_price = market_price
                direction = open_position["direction"]

                if direction == "BUY":
                    price_change = (exit_price - entry_price) / entry_price
                else:
                    price_change = (entry_price - exit_price) / entry_price

                contracts = position_size / entry_price
                gross_pnl = contracts * (exit_price - entry_price) if direction == "BUY" else contracts * (entry_price - exit_price)

                # Apply Kalshi fees
                entry_fee = 0.07 * contracts * entry_price * (1 - entry_price)
                exit_fee = 0.07 * contracts * exit_price * (1 - exit_price)
                net_pnl = gross_pnl - entry_fee - exit_fee

                trades.append(
                    {
                        "entry_minute": open_position["entry_minute"],
                        "exit_minute": minute,
                        "direction": direction,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "net_pnl": net_pnl,
                        "holding_time": holding_time,
                    }
                )

                reason = "Max hold time" if holding_time >= 5 else "End of game"
                logger.info(
                    f"📉 Min {minute}: CLOSE {direction} @ ${exit_price:.2f} "
                    f"(Entry ${entry_price:.2f}, P&L: ${net_pnl:+.2f}, {reason})"
                )

                open_position = None

        # Check for new signal if no position
        if not open_position and abs(edge) > 0.10:
            direction = "BUY" if edge > 0 else "SELL"

            logger.info(
                f"📈 Min {minute}: SIGNAL {direction} @ ${market_price:.2f} "
                f"(Model: {model_wp:.1%}, Market: {market_price:.1%}, Edge: {edge:+.1%})"
            )

            open_position = {"entry_minute": minute, "entry_price": market_price, "direction": direction}

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 80)

    if trades:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df["net_pnl"].sum()
        winning_trades = len(trades_df[trades_df["net_pnl"] > 0])
        losing_trades = len(trades_df[trades_df["net_pnl"] < 0])
        win_rate = winning_trades / len(trades) if len(trades) > 0 else 0

        logger.info(f"\n📈 Trading Activity:")
        logger.info(f"   Total Trades: {len(trades)}")
        logger.info(f"   Winning Trades: {winning_trades}")
        logger.info(f"   Losing Trades: {losing_trades}")
        logger.info(f"   Win Rate: {win_rate:.1%}")

        logger.info(f"\n💰 Performance:")
        logger.info(f"   Total P&L: ${total_pnl:+.2f}")
        logger.info(f"   Avg P&L per Trade: ${total_pnl / len(trades):+.2f}")

        logger.info(f"\n📋 Individual Trades:")
        for i, trade in enumerate(trades, 1):
            logger.info(
                f"   {i}. Min {trade['entry_minute']}-{trade['exit_minute']}: "
                f"{trade['direction']} ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} "
                f"= ${trade['net_pnl']:+.2f}"
            )

    else:
        logger.info("\n   No trades executed (no signals met criteria)")

    logger.info("\n" + "=" * 80)
    logger.info("\n💡 This demo shows how your strategy would work:")
    logger.info("   1. Market overreacts to TDs/big plays (momentum)")
    logger.info("   2. Model predicts stable true win probability")
    logger.info("   3. When gap > 10% → Trade the difference")
    logger.info("   4. Close after 5 minutes or when price corrects")
    logger.info("\n   For LIVE trading: Use real Sportradar game state + Kalshi prices")
    logger.info("=" * 80 + "\n")


if __name__ == "__main__":
    run_demo_backtest()
