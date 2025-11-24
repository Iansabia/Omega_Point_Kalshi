#!/usr/bin/env python3
"""
Paper Trading for Monday Night Football.

Runs the full trading system in paper mode (no real money) during tonight's MNF game.

Carolina Panthers @ San Francisco 49ers
November 24, 2025, 8:15 PM ET

Usage:
    # Auto-find game (recommended)
    python scripts/run_paper_trading_mnf.py

    # Or specify manually
    python scripts/run_paper_trading_mnf.py --game-id "sr:match:xxxxx" --ticker "KXMVENFL..."
"""

import argparse
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from src.data.espn_client import ESPNClient
from src.execution.kalshi_client import KalshiClient
from src.live_trading.live_trading_engine import LiveTradingEngine

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"paper_trading_mnf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


async def auto_find_game():
    """Auto-discover tonight's game and market."""
    logger.info("🔍 Auto-discovering tonight's game...")

    espn = ESPNClient()
    kalshi = KalshiClient()

    # Find ESPN game
    try:
        today = datetime.now().strftime("%Y%m%d")
        scoreboard = await espn.get_scoreboard(date=today)
        games = scoreboard.get("events", [])

        game_id = None
        home_team = None
        away_team = None

        for game in games:
            competition = game.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])

            teams = {}
            for competitor in competitors:
                team_abbr = competitor.get("team", {}).get("abbreviation", "").upper()
                home_away = competitor.get("homeAway", "")
                teams[home_away] = team_abbr

            if ("CAR" in teams.values()) and ("SF" in teams.values()):
                game_id = game.get("id")
                home_team = teams.get("home", "SF")
                away_team = teams.get("away", "CAR")
                break

        if not game_id:
            logger.error("❌ Could not find Panthers vs 49ers in today's games")
            await espn.close()
            return None, None, None, None

        logger.info(f"✅ Found game: {game_id}")

    except Exception as e:
        logger.error(f"❌ Error finding ESPN game: {e}")
        await espn.close()
        return None, None, None, None

    # Find Kalshi market
    try:
        response = kalshi.get_markets(series_ticker="KXMVENFLSINGLEGAME", status="open", limit=100)
        markets = response.get("markets", [])

        ticker = None
        search_terms = ["panthers", "carolina", "49ers", "san francisco"]

        for market in markets:
            title = market.get("title", "").lower()
            ticker_str = market.get("ticker", "").lower()

            matches = sum(1 for term in search_terms if term in title or term in ticker_str)
            if matches >= 2:
                ticker = market["ticker"]
                break

        if not ticker:
            logger.error("❌ Could not find Panthers vs 49ers market on Kalshi")
            logger.warning("   Market may not be open yet - try again closer to game time")
            return None, None, None, None

        logger.info(f"✅ Found market: {ticker}")

    except Exception as e:
        logger.error(f"❌ Error finding Kalshi market: {e}")
        await espn.close()
        return None, None, None, None

    await espn.close()
    return game_id, ticker, home_team, away_team


async def run_paper_trading(game_id: str, ticker: str, home_team: str, away_team: str):
    """
    Run paper trading for the game.

    Args:
        game_id: Sportradar game ID
        ticker: Kalshi market ticker
        home_team: Home team code (e.g., 'SF')
        away_team: Away team code (e.g., 'CAR')
    """
    logger.info("\n" + "=" * 80)
    logger.info("MONDAY NIGHT FOOTBALL - PAPER TRADING")
    logger.info("=" * 80)
    logger.info(f"Game: {away_team} @ {home_team}")
    logger.info(f"Time: 8:15 PM ET")
    logger.info(f"Mode: PAPER TRADING (No Real Money)")
    logger.info(f"Log: {log_file}")
    logger.info("=" * 80)

    # Initialize trading engine
    logger.info("\n🔧 Initializing trading engine...")

    engine = LiveTradingEngine(
        model_path="models/win_probability_model.pkl",
        paper_trading=True,  # PAPER MODE - NO REAL MONEY
        min_edge=0.10,  # 10% minimum edge
        min_confidence=0.5,  # 50% minimum confidence
        max_spread=0.10,  # 10% max spread
    )

    # Register game
    logger.info(f"\n📍 Registering game...")
    logger.info(f"   Sportradar ID: {game_id}")
    logger.info(f"   Kalshi Ticker: {ticker}")

    engine.register_game(sportradar_game_id=game_id, kalshi_ticker=ticker, home_team=home_team, away_team=away_team)

    # Start trading
    logger.info("\n🚀 Starting paper trading...")
    logger.info("   System will trade automatically")
    logger.info("   Press Ctrl+C to stop\n")

    try:
        await engine.start()

    except KeyboardInterrupt:
        logger.info("\n\n🛑 Stopping trading engine...")
        await engine.stop()

        # Print final statistics
        stats = engine.get_stats()

        logger.info("\n" + "=" * 80)
        logger.info("FINAL STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total Signals Generated: {stats['total_signals']}")
        logger.info(f"Total Trades Executed: {stats['total_trades']}")
        logger.info(f"Open Positions: {stats['risk']['open_positions']}")
        logger.info(f"Paper Trading Mode: {stats['paper_trading']}")

        if stats["total_trades"] > 0:
            logger.info(f"\n📊 Risk Manager Stats:")
            logger.info(f"   Consecutive Losses: {stats['risk']['consecutive_losses']}")
            logger.info(f"   In Cooldown: {stats['risk']['in_cooldown']}")

        logger.info(f"\n📊 Arbitrage Detector Stats:")
        logger.info(f"   Signals Generated: {stats['detector']['signals_generated']}")
        logger.info(f"   Signals Filtered: {stats['detector']['signals_filtered']}")
        logger.info(f"   Signal Rate: {stats['detector']['signal_rate']:.1%}")

        logger.info("\n" + "=" * 80)
        logger.info(f"Full log saved to: {log_file}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"\n❌ Error during trading: {e}", exc_info=True)
        await engine.stop()


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Paper trading for Monday Night Football")

    parser.add_argument("--game-id", type=str, help="Sportradar game ID")
    parser.add_argument("--ticker", type=str, help="Kalshi market ticker")
    parser.add_argument("--home", type=str, help="Home team code (e.g., 'SF')")
    parser.add_argument("--away", type=str, help="Away team code (e.g., 'CAR')")

    args = parser.parse_args()

    # Check if manual IDs provided
    if args.game_id and args.ticker:
        game_id = args.game_id
        ticker = args.ticker
        home_team = args.home or "SF"
        away_team = args.away or "CAR"

        logger.info("📋 Using provided game ID and ticker")

    else:
        # Auto-find game
        game_id, ticker, home_team, away_team = await auto_find_game()

        if not game_id or not ticker:
            logger.error("\n❌ Could not auto-discover game")
            logger.error("\nPossible solutions:")
            logger.error("1. Check your SPORTRADAR_API_KEY in .env")
            logger.error("2. Wait until closer to game time for Kalshi market to open")
            logger.error("3. Manually specify game ID and ticker:")
            logger.error("   python scripts/run_paper_trading_mnf.py \\")
            logger.error('       --game-id "sr:match:xxxxx" \\')
            logger.error('       --ticker "KXMVENFL..." \\')
            logger.error('       --home "SF" --away "CAR"')
            return

    # Run paper trading
    await run_paper_trading(game_id, ticker, home_team, away_team)


if __name__ == "__main__":
    asyncio.run(main())
