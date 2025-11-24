#!/usr/bin/env python3
"""
Test ESPN API Client.

Verify that ESPN API works and can fetch tonight's Panthers vs 49ers game.

Usage:
    python scripts/test_espn_api.py
"""

import asyncio
import json
import logging
from datetime import datetime

from src.data.espn_client import ESPNClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def test_scoreboard():
    """Test fetching today's scoreboard."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Fetch Today's Scoreboard")
    logger.info("=" * 60)

    client = ESPNClient()

    try:
        # Get today's scoreboard
        today = datetime.now().strftime("%Y%m%d")
        logger.info(f"Fetching scoreboard for {today}...")

        scoreboard = await client.get_scoreboard(date=today)
        events = scoreboard.get("events", [])

        logger.info(f"✅ Found {len(events)} games today")

        # List all games
        for idx, event in enumerate(events, 1):
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])

            teams = []
            for competitor in competitors:
                team = competitor.get("team", {}).get("abbreviation", "")
                score = competitor.get("score", 0)
                teams.append(f"{team} ({score})")

            status = competition.get("status", {}).get("type", {}).get("name", "")
            game_id = event.get("id", "")

            logger.info(f"  {idx}. {' vs '.join(teams)} - {status} [ID: {game_id}]")

    except Exception as e:
        logger.error(f"❌ Error fetching scoreboard: {e}", exc_info=True)
        return False

    finally:
        await client.close()

    return True


async def test_find_tonights_game():
    """Test finding tonight's Panthers vs 49ers game."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Find Panthers vs 49ers Game")
    logger.info("=" * 60)

    client = ESPNClient()

    try:
        today = datetime.now().strftime("%Y%m%d")
        logger.info(f"Searching for CAR @ SF on {today}...")

        # Try finding the game
        game = await client.find_game(home_team="SF", away_team="CAR", date=today)

        if game:
            logger.info("✅ FOUND GAME!")

            game_id = game.get("id", "")
            logger.info(f"   ESPN Game ID: {game_id}")

            competition = game.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])

            for competitor in competitors:
                team = competitor.get("team", {}).get("displayName", "")
                score = competitor.get("score", 0)
                home_away = competitor.get("homeAway", "")
                logger.info(f"   {home_away.upper()}: {team} ({score})")

            status = competition.get("status", {}).get("type", {}).get("detail", "")
            logger.info(f"   Status: {status}")

            return game_id
        else:
            logger.warning("❌ Game not found")
            logger.warning("   Try searching for 'SF' or 'CAR' individually:")

            # Try finding SF home game
            sf_game = await client.find_game(home_team="SF", date=today)
            if sf_game:
                competition = sf_game.get("competitions", [{}])[0]
                competitors = competition.get("competitors", [])
                teams = [c.get("team", {}).get("abbreviation", "") for c in competitors]
                logger.warning(f"   Found SF game: {' vs '.join(teams)}")
            else:
                logger.warning("   No SF home games found")

            return None

    except Exception as e:
        logger.error(f"❌ Error finding game: {e}", exc_info=True)
        return None

    finally:
        await client.close()


async def test_game_summary(game_id: str):
    """Test fetching detailed game summary."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Fetch Game Summary")
    logger.info("=" * 60)

    client = ESPNClient()

    try:
        logger.info(f"Fetching summary for game {game_id}...")

        summary = await client.get_game_summary(game_id)

        logger.info("✅ Successfully fetched game summary")

        # Parse to standardized format
        state = client.parse_game_state(summary, is_scoreboard=False)

        logger.info("\n📊 Parsed Game State:")
        logger.info(f"   Teams: {state['away_team']} @ {state['home_team']}")
        logger.info(f"   Score: {state['away_score']} - {state['home_score']}")
        logger.info(f"   Quarter: {state['quarter']}")
        logger.info(f"   Clock: {state['clock']} ({state['clock_seconds']}s)")
        logger.info(f"   Status: {state['status']}")
        logger.info(f"   Possession: {state['possession']}")
        logger.info(f"   Situation: {state['down']} & {state['distance']} at {state['yardline']}")
        logger.info(f"   Time Remaining: {state['time_remaining']}s")

        return state

    except Exception as e:
        logger.error(f"❌ Error fetching game summary: {e}", exc_info=True)
        return None

    finally:
        await client.close()


async def test_live_polling(game_id: str, duration: int = 30):
    """Test live polling of game state."""
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST 4: Live Polling ({duration}s)")
    logger.info("=" * 60)

    client = ESPNClient()
    updates = []

    def on_update(state):
        """Callback for game updates."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.info(
            f"[{timestamp}] {state['away_team']} {state['away_score']} - "
            f"{state['home_score']} {state['home_team']} | "
            f"Q{state['quarter']} {state['clock']} | {state['status']}"
        )
        updates.append(state)

    try:
        logger.info(f"Polling game {game_id} every 2s for {duration}s...")
        logger.info("Press Ctrl+C to stop early\n")

        # Poll for limited time
        max_polls = duration // 2
        await client.poll_live_game(game_id=game_id, callback=on_update, interval=2, max_polls=max_polls)

        logger.info(f"\n✅ Received {len(updates)} updates")

        if len(updates) >= 2:
            # Check if state changed
            first = updates[0]
            last = updates[-1]

            if (
                first["home_score"] != last["home_score"]
                or first["away_score"] != last["away_score"]
                or first["clock_seconds"] != last["clock_seconds"]
            ):
                logger.info("✅ Game state is updating correctly")
            else:
                logger.info("⚠️  Game state unchanged (may be halftime or pre-game)")

    except Exception as e:
        logger.error(f"❌ Error during polling: {e}", exc_info=True)
        return False

    finally:
        await client.close()

    return True


async def main():
    """Run all tests."""
    logger.info("\n" + "=" * 80)
    logger.info("ESPN API CLIENT TEST SUITE")
    logger.info("=" * 80)

    # Test 1: Fetch scoreboard
    success = await test_scoreboard()
    if not success:
        logger.error("\n❌ Scoreboard test failed, stopping")
        return

    # Test 2: Find tonight's game
    game_id = await test_find_tonights_game()
    if not game_id:
        logger.warning("\n⚠️  Could not find tonight's game")
        logger.warning("   Game may not be available yet or wrong date")
        logger.warning("   Check ESPN.com to verify game schedule")

        # Ask user if they want to enter game ID manually
        logger.info("\n💡 You can test with any ESPN game ID")
        logger.info("   Find game ID from ESPN URLs (e.g., espn.com/nfl/game/_/gameId/401671716)")
        return

    # Test 3: Fetch game summary
    state = await test_game_summary(game_id)
    if not state:
        logger.error("\n❌ Game summary test failed")
        return

    # Test 4: Live polling (30 seconds)
    logger.info("\n💡 Testing live polling for 30 seconds...")
    logger.info("   This simulates what will happen during paper trading")
    await test_live_polling(game_id, duration=30)

    # Final summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST SUMMARY")
    logger.info("=" * 80)
    logger.info("✅ ESPN API client is working correctly")
    logger.info(f"✅ Game ID for tonight: {game_id}")
    logger.info("✅ Ready to use for paper trading")
    logger.info("\nNext step: Update paper trading scripts to use ESPN client")
    logger.info("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
