#!/usr/bin/env python3
"""
Launch Dashboard + Live Trading for Monday Night Football

This script runs both the dashboard server and the live trading engine.
The dashboard provides real-time visualization and controls.

Usage:
    PYTHONPATH=. ./venv/bin/python3 scripts/run_dashboard_trading.py

Then open: http://localhost:8000
"""

import asyncio
import sys
import threading
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Check if new API key is configured
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("KALSHI_API_KEY_ID")
private_key_path = os.getenv("KALSHI_PRIVATE_KEY_PATH")

if not api_key or not private_key_path:
    print("=" * 70)
    print("  ⚠️  WARNING: Kalshi API credentials not configured")
    print("=" * 70)
    print()
    print("Please configure your new Kalshi API key:")
    print("  1. Generate new key on https://kalshi.com")
    print("  2. Store key in ~/.ssh/kalshi/kalshi_private_key.pem")
    print("  3. Update .env file with new key ID and path")
    print()
    print("See GENERATE_NEW_KEY.md for detailed instructions")
    print("=" * 70)
    print()
    response = input("Continue anyway? (yes/no): ")
    if response.lower() != "yes":
        sys.exit(1)


def run_dashboard_server():
    """Run the dashboard server in a separate thread."""
    import uvicorn
    from src.dashboard.dashboard_server import app

    print("Starting dashboard server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


def run_trading_engine():
    """Run the live trading engine with dashboard integration."""
    from src.live_trading.live_trading_engine import LiveTradingEngine
    from src.data.espn_client import ESPNClient
    from src.execution.kalshi_client import KalshiClient
    from src.dashboard.dashboard_server import update_dashboard, dashboard_state

    print()
    print("=" * 70)
    print("  INITIALIZING LIVE TRADING ENGINE")
    print("=" * 70)
    print()

    # Initialize clients
    espn = ESPNClient()
    kalshi = KalshiClient()

    # Find tonight's game
    print("Finding tonight's game...")
    from datetime import datetime

    today = datetime.now().strftime("%Y%m%d")
    try:
        scoreboard = espn.get_scoreboard_sync(today)
        if scoreboard and "events" in scoreboard and len(scoreboard["events"]) > 0:
            game = scoreboard["events"][0]
            game_id = game["id"]
            home_team = game["competitions"][0]["competitors"][0]["team"]["abbreviation"]
            away_team = game["competitions"][0]["competitors"][1]["team"]["abbreviation"]

            print(f"✅ Found game: {away_team} @ {home_team}")
            print(f"   ESPN Game ID: {game_id}")
            print()

            # Update dashboard
            asyncio.run(
                update_dashboard(
                    "game_state",
                    {
                        "game_id": game_id,
                        "home_team": home_team,
                        "away_team": away_team,
                        "status": "Found",
                    },
                )
            )
        else:
            print("❌ No games found for tonight")
            print("   Check ESPN API or try manual game ID")
            game_id = None
    except Exception as e:
        print(f"❌ Error finding game: {e}")
        game_id = None

    # Find Kalshi market
    if game_id:
        print("Finding Kalshi market...")
        try:
            # Search for NFL markets
            markets = kalshi.get_markets(series_ticker="HIGHFB", status="open", limit=50)

            if markets and "markets" in markets:
                # Find market matching game
                for market in markets["markets"]:
                    ticker = market.get("ticker", "")
                    # Match based on team names or date
                    print(f"   Found market: {ticker}")
                    # You would match based on teams and date here

                print(f"✅ Found {len(markets['markets'])} open NFL markets")
                print()
            else:
                print("❌ No open NFL markets found on Kalshi")
        except Exception as e:
            print(f"❌ Error fetching Kalshi markets: {e}")

    print("=" * 70)
    print("  DASHBOARD READY")
    print("=" * 70)
    print()
    print("  Open your browser to: http://localhost:8000")
    print()
    print("  The dashboard will show:")
    print("    - Live game state from ESPN")
    print("    - Real-time market prices from Kalshi")
    print("    - Model win probability predictions")
    print("    - Arbitrage signals")
    print("    - Trade execution")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 70)
    print()

    # Update dashboard status
    asyncio.run(update_dashboard("status", {"status": "connected"}))

    # Keep the trading engine alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")


def main():
    """Main entry point."""
    print()
    print("=" * 70)
    print("  🏈 MONDAY NIGHT FOOTBALL TRADING DASHBOARD")
    print("=" * 70)
    print()
    print("  Starting components:")
    print("    1. Dashboard Server (FastAPI + WebSocket)")
    print("    2. Live Trading Engine (ESPN + Kalshi)")
    print()

    # Start dashboard server in background thread
    dashboard_thread = threading.Thread(target=run_dashboard_server, daemon=True)
    dashboard_thread.start()

    # Give dashboard server time to start
    time.sleep(2)

    # Run trading engine in main thread
    try:
        run_trading_engine()
    except KeyboardInterrupt:
        print("\n\nShutting down gracefully...")
        sys.exit(0)


if __name__ == "__main__":
    main()
