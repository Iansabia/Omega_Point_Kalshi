#!/usr/bin/env python3
"""
Download Historical Candlesticks for NFL Games (CORRECT METHOD).

Follows the proper Kalshi API workflow:
1. Search for markets to get series_ticker + market_ticker pair
2. Get market details for open_time/close_time
3. Request candlesticks with correct timestamps
4. Validate data exists

Usage:
    python scripts/download_game_candlesticks.py --search "ravens chiefs"
    python scripts/download_game_candlesticks.py --list-recent
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.execution.kalshi_client import KalshiClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class GameCandlestickDownloader:
    """Download candlestick data for NFL games using correct Kalshi API workflow."""

    def __init__(self, cache_dir: str = "data/game_candlesticks"):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory to cache downloaded candlesticks
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.client = KalshiClient()
        logger.info(f"Cache directory: {self.cache_dir}")

    def search_markets(self, query: str, status: str = "settled", limit: int = 20) -> List[Dict[str, Any]]:
        """
        Step 1: Search for NFL markets.

        Args:
            query: Search query (e.g., "ravens chiefs", "nfl week 12")
            status: Market status ('settled', 'closed', 'open')
            limit: Max results

        Returns:
            List of markets with series_ticker and market_ticker
        """
        logger.info(f"🔍 Searching for: '{query}' (status: {status})")

        # Search using Kalshi API
        response = self.client.get_markets(
            series_ticker="KXMVENFLSINGLEGAME", status=status, limit=limit  # NFL single game series
        )

        markets = response.get("markets", [])

        # Filter by query terms (case-insensitive)
        query_lower = query.lower()
        filtered = []

        for market in markets:
            title = market.get("title", "").lower()
            ticker = market.get("ticker", "").lower()

            # Check if query terms are in title or ticker
            if query_lower in title or query_lower in ticker:
                filtered.append(market)

        logger.info(f"✅ Found {len(filtered)} markets matching '{query}'")

        return filtered

    def get_market_details(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Step 2: Get market details including timestamps.

        Args:
            ticker: Market ticker

        Returns:
            Market details with open_time, close_time, series_ticker
        """
        logger.info(f"📋 Getting details for: {ticker}")

        market = self.client.get_market(ticker)

        if not market:
            logger.error(f"❌ Could not fetch market details for {ticker}")
            return None

        # Extract key fields
        series_ticker = market.get("event_ticker")
        open_time = market.get("open_time")
        close_time = market.get("close_time")
        title = market.get("title")

        logger.info(f"   Title: {title}")
        logger.info(f"   Series: {series_ticker}")
        logger.info(f"   Open: {open_time}")
        logger.info(f"   Close: {close_time}")

        return {
            "ticker": ticker,
            "series_ticker": series_ticker,
            "title": title,
            "open_time": open_time,
            "close_time": close_time,
            "market": market,
        }

    def download_candlesticks(
        self, market_ticker: str, series_ticker: str, open_time: str, close_time: str, interval: int = 1
    ) -> Optional[List[Dict]]:
        """
        Step 3: Download candlesticks with correct parameters.

        Args:
            market_ticker: Market ticker (e.g., 'KXMVENFLSINGLEGAME-S2025...')
            series_ticker: Series ticker (e.g., 'KXMVENFLSINGLEGAME-S2025...')
            open_time: Market open time (ISO format)
            close_time: Market close time (ISO format)
            interval: Candlestick interval (1, 60, or 1440 minutes)

        Returns:
            List of candlestick dicts, or None if no data
        """
        logger.info(f"📥 Downloading candlesticks...")
        logger.info(f"   Interval: {interval} minutes")

        # Convert ISO timestamps to Unix seconds
        try:
            start_ts = int(datetime.fromisoformat(open_time.replace("Z", "+00:00")).timestamp())
            end_ts = int(datetime.fromisoformat(close_time.replace("Z", "+00:00")).timestamp())
        except Exception as e:
            logger.error(f"❌ Failed to parse timestamps: {e}")
            return None

        logger.info(f"   Start: {start_ts} ({open_time})")
        logger.info(f"   End: {end_ts} ({close_time})")

        # Request candlesticks
        response = self.client.get_market_candlesticks(
            series_ticker=series_ticker,
            market_ticker=market_ticker,
            period_interval=interval,
            start_ts=start_ts,
            end_ts=end_ts,
        )

        candlesticks = response.get("candlesticks", [])

        # Step 4: Validate data
        if not candlesticks:
            logger.warning("⚠️  No candlesticks returned (market may have had no trading activity)")
            return None

        logger.info(f"✅ Downloaded {len(candlesticks)} candlesticks")

        # Show sample
        if candlesticks:
            first = candlesticks[0]
            last = candlesticks[-1]
            logger.info(f"   First candle: {first}")
            logger.info(f"   Last candle: {last}")

        return candlesticks

    def download_and_cache(self, ticker: str, interval: int = 1, force: bool = False) -> Optional[str]:
        """
        Full workflow: search, download, cache.

        Args:
            ticker: Market ticker
            interval: Candlestick interval (1, 60, or 1440)
            force: Force re-download even if cached

        Returns:
            Path to cached file, or None if failed
        """
        cache_file = self.cache_dir / f"{ticker}_interval{interval}.json"

        # Check cache
        if cache_file.exists() and not force:
            logger.info(f"📂 Using cached data: {cache_file}")
            return str(cache_file)

        # Step 1: Get market details
        details = self.get_market_details(ticker)
        if not details:
            return None

        # Step 2: Download candlesticks
        candlesticks = self.download_candlesticks(
            market_ticker=details["ticker"],
            series_ticker=details["series_ticker"],
            open_time=details["open_time"],
            close_time=details["close_time"],
            interval=interval,
        )

        if not candlesticks:
            return None

        # Step 3: Cache data
        data = {
            "market": details["market"],
            "candlesticks": candlesticks,
            "interval": interval,
            "downloaded_at": datetime.now().isoformat(),
        }

        with open(cache_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"💾 Cached to: {cache_file}")

        return str(cache_file)

    def list_recent_markets(self, limit: int = 10) -> List[Dict]:
        """List recent NFL markets for easy selection."""
        logger.info(f"📋 Listing {limit} recent NFL markets...")

        response = self.client.get_markets(series_ticker="KXMVENFLSINGLEGAME", status="settled", limit=limit)

        markets = response.get("markets", [])

        logger.info("\n" + "=" * 80)
        logger.info("Recent NFL Markets")
        logger.info("=" * 80)

        for i, market in enumerate(markets, 1):
            ticker = market.get("ticker")
            title = market.get("title")
            result = market.get("result", "unknown")
            open_time = market.get("open_time", "")

            logger.info(f"\n{i}. {ticker}")
            logger.info(f"   Title: {title}")
            logger.info(f"   Result: {result}")
            logger.info(f"   Opened: {open_time}")

        logger.info("\n" + "=" * 80)

        return markets


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download NFL game candlesticks from Kalshi")

    parser.add_argument("--ticker", type=str, help="Market ticker to download")
    parser.add_argument("--search", type=str, help="Search for markets by query (e.g., 'ravens chiefs')")
    parser.add_argument("--list-recent", action="store_true", help="List recent NFL markets")
    parser.add_argument("--interval", type=int, default=1, choices=[1, 60, 1440], help="Interval: 1, 60, or 1440 minutes")
    parser.add_argument("--force", action="store_true", help="Force re-download even if cached")

    args = parser.parse_args()

    downloader = GameCandlestickDownloader()

    if args.list_recent:
        downloader.list_recent_markets(limit=20)
        return

    if args.search:
        markets = downloader.search_markets(args.search)

        if not markets:
            logger.error(f"❌ No markets found for '{args.search}'")
            return

        # Show results
        logger.info("\n" + "=" * 80)
        logger.info("Search Results")
        logger.info("=" * 80)

        for i, market in enumerate(markets[:10], 1):
            logger.info(f"\n{i}. {market['ticker']}")
            logger.info(f"   {market.get('title')}")
            logger.info(f"   Result: {market.get('result')}")

        logger.info("\n" + "=" * 80)
        logger.info(f"Use: python scripts/download_game_candlesticks.py --ticker <TICKER>")
        logger.info("=" * 80)
        return

    if args.ticker:
        logger.info("\n" + "=" * 80)
        logger.info("Downloading Game Candlesticks")
        logger.info("=" * 80)

        cache_file = downloader.download_and_cache(args.ticker, interval=args.interval, force=args.force)

        if cache_file:
            logger.info("\n✅ Success!")
            logger.info(f"   Data saved to: {cache_file}")
            logger.info(f"\n   Use for backtesting:")
            logger.info(f"   python scripts/backtest_single_game.py --data {cache_file}")
        else:
            logger.error("\n❌ Failed to download candlesticks")
            logger.error("   Possible reasons:")
            logger.error("   - Market had no trading activity")
            logger.error("   - Invalid timestamps")
            logger.error("   - API error")

        logger.info("=" * 80)
        return

    # No arguments - show help
    parser.print_help()


if __name__ == "__main__":
    main()
