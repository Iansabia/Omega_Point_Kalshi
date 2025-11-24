#!/usr/bin/env python3
"""
Find Tonight's NFL Game and Kalshi Market.

Quick script to discover game ID and market ticker for tonight's MNF game.

Usage:
    python scripts/find_tonights_game.py
"""

import asyncio
import logging
from datetime import datetime

from src.data.espn_client import ESPNClient
from src.execution.kalshi_client import KalshiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def find_game_and_market():
    """Find tonight's MNF game ID and Kalshi market ticker."""
    logger.info("=" * 60)
    logger.info("FINDING TONIGHT'S MONDAY NIGHT FOOTBALL GAME")
    logger.info("=" * 60)

    # Initialize clients
    espn = ESPNClient()
    kalshi = KalshiClient()

    # Step 1: Find ESPN game ID
    logger.info("\n1. Searching for tonight's NFL game...")
    logger.info("   Expected: Carolina Panthers @ San Francisco 49ers")

    try:
        # Get today's scoreboard
        today = datetime.now().strftime("%Y%m%d")
        scoreboard = await espn.get_scoreboard(date=today)
        games = scoreboard.get("events", [])
        logger.info(f"   Found {len(games)} games today")

        # Search for Panthers vs 49ers
        game_id = None
        game_info = None
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

            # Check if this is Panthers (CAR) vs 49ers (SF)
            if ("CAR" in teams.values()) and ("SF" in teams.values()):
                game_id = game.get("id")
                game_info = game
                home_team = teams.get("home", "SF")
                away_team = teams.get("away", "CAR")
                break

        if game_id:
            logger.info(f"\n   ✅ FOUND GAME!")
            logger.info(f"   ESPN Game ID: {game_id}")

            competition = game_info.get("competitions", [{}])[0]
            for competitor in competition.get("competitors", []):
                team = competitor.get("team", {}).get("displayName", "")
                home_away = competitor.get("homeAway", "").upper()
                logger.info(f"   {home_away}: {team}")

            status = competition.get("status", {}).get("type", {}).get("detail", "")
            logger.info(f"   Status: {status}")
        else:
            logger.error("   ❌ Game not found in today's schedule")
            logger.error("   Available games:")
            for g in games[:5]:
                comp = g.get("competitions", [{}])[0]
                comps = comp.get("competitors", [])
                teams = [c.get("team", {}).get("abbreviation", "?") for c in comps]
                logger.error(f"      {' vs '.join(teams)}")

    except Exception as e:
        logger.error(f"   ❌ Error fetching ESPN schedule: {e}")
        game_id = None
        home_team = None
        away_team = None

    # Step 2: Find Kalshi market ticker
    logger.info("\n2. Searching for Kalshi market...")

    try:
        response = kalshi.get_markets(series_ticker="KXMVENFLSINGLEGAME", status="open", limit=100)

        markets = response.get("markets", [])
        logger.info(f"   Found {len(markets)} open NFL markets")

        # Search for Panthers/49ers market
        ticker = None
        market_info = None

        search_terms = ["panthers", "carolina", "49ers", "san francisco", "sf"]

        for market in markets:
            title = market.get("title", "").lower()
            ticker_str = market.get("ticker", "").lower()

            # Check if any search terms match
            matches = sum(1 for term in search_terms if term in title or term in ticker_str)

            if matches >= 2:  # At least 2 terms (e.g., "panthers" + "49ers")
                ticker = market["ticker"]
                market_info = market
                break

        if ticker:
            logger.info(f"\n   ✅ FOUND MARKET!")
            logger.info(f"   Kalshi Ticker: {ticker}")
            logger.info(f"   Title: {market_info.get('title')}")
            logger.info(f"   Event: {market_info.get('event_ticker')}")
        else:
            logger.error("   ❌ Market not found")
            logger.error("   Available markets (first 5):")
            for m in markets[:5]:
                logger.error(f"      {m.get('ticker')}")
                logger.error(f"      {m.get('title')}")

    except Exception as e:
        logger.error(f"   ❌ Error fetching Kalshi markets: {e}")
        ticker = None

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if game_id and ticker:
        logger.info("✅ Ready for paper trading!")
        logger.info(f"\nESPN Game ID:\n   {game_id}")
        logger.info(f"\nKalshi Market Ticker:\n   {ticker}")
        logger.info(f"\nNext step:")
        logger.info(f"   python scripts/run_paper_trading_mnf.py \\")
        logger.info(f'       --game-id "{game_id}" \\')
        logger.info(f'       --ticker "{ticker}"')
        if home_team and away_team:
            logger.info(f'       --home "{home_team}" --away "{away_team}"')
    else:
        logger.error("❌ Missing information:")
        if not game_id:
            logger.error("   - ESPN game ID not found")
            logger.error("   - Check ESPN.com to verify game schedule")
        if not ticker:
            logger.error("   - Kalshi market ticker not found")
            logger.error("   - Market may not be open yet")

    logger.info("=" * 60)

    # Close ESPN client
    await espn.close()

    return game_id, ticker, home_team, away_team


async def main():
    """Main entry point."""
    await find_game_and_market()


if __name__ == "__main__":
    asyncio.run(main())
