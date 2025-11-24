"""
Download Historical Kalshi Data.

Standalone script to download and cache historical NFL market data from Kalshi.

Usage:
    # Download 2024 season
    python scripts/download_historical_data.py --season 2024

    # Download all available data
    python scripts/download_historical_data.py --all

    # Test with limited markets
    python scripts/download_historical_data.py --season 2024 --max-markets 10

    # Clear cache and re-download
    python scripts/download_historical_data.py --season 2024 --clear-cache
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.kalshi_historical import KalshiHistoricalDataFetcher


def main():
    """Download historical data from Kalshi."""
    parser = argparse.ArgumentParser(description="Download historical Kalshi NFL data")
    parser.add_argument("--season", type=int, help="NFL season year (e.g., 2024)")
    parser.add_argument("--all", action="store_true", help="Download all seasons")
    parser.add_argument("--max-markets", type=int, help="Limit number of markets (for testing)")
    parser.add_argument("--clear-cache", action="store_true", help="Clear cache before downloading")
    parser.add_argument(
        "--interval", type=int, default=60, choices=[1, 60, 1440], help="Candlestick interval in minutes (1, 60, or 1440)"
    )
    parser.add_argument(
        "--status", type=str, default="settled", choices=["settled", "closed", "open"], help="Market status to fetch"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("KALSHI HISTORICAL DATA DOWNLOADER")
    print("=" * 80)

    # Initialize fetcher
    fetcher = KalshiHistoricalDataFetcher()

    # Clear cache if requested
    if args.clear_cache:
        fetcher.clear_cache()

    # Determine seasons to download
    if args.all:
        seasons = [2023, 2024]  # Adjust based on available data
    elif args.season:
        seasons = [args.season]
    else:
        print("❌ Error: Specify --season YEAR or --all")
        return

    total_markets = 0
    total_candlesticks = 0

    for season in seasons:
        print(f"\n{'='*80}")
        print(f"SEASON {season}")
        print(f"{'='*80}\n")

        # Fetch markets
        markets = fetcher.fetch_nfl_markets(
            season=season, status=args.status, use_cache=not args.clear_cache, max_markets=args.max_markets
        )

        if markets.empty:
            print(f"⚠️  No {args.status} markets found for {season}")
            continue

        print(f"\n📊 Market Summary:")
        print(f"   Total markets: {len(markets)}")

        if "status" in markets.columns:
            print(f"   By status:")
            for status, count in markets["status"].value_counts().items():
                print(f"      {status}: {count}")

        # Download candlesticks
        candlesticks = fetcher.download_candlesticks(
            markets, period_interval=args.interval, use_cache=not args.clear_cache, max_markets=args.max_markets
        )

        total_markets += len(markets)
        total_candlesticks += len(candlesticks)

        # Show sample
        if candlesticks:
            print(f"\n📈 Sample candlestick data:")
            sample_ticker = list(candlesticks.keys())[0]
            sample_df = candlesticks[sample_ticker]
            print(f"   Ticker: {sample_ticker}")
            print(f"   Candlesticks: {len(sample_df)}")
            if not sample_df.empty:
                print(f"   Columns: {', '.join(sample_df.columns)}")
                print(f"\n   First 3 rows:")
                print(sample_df.head(3).to_string(index=False))

    # Final summary
    print(f"\n{'='*80}")
    print("DOWNLOAD COMPLETE")
    print("=" * 80)
    print(f"\n📊 Summary:")
    print(f"   Seasons: {len(seasons)}")
    print(f"   Total markets: {total_markets}")
    print(f"   Candlestick datasets: {total_candlesticks}")

    # Cache stats
    stats = fetcher.get_cache_stats()
    print(f"\n💾 Cache Statistics:")
    print(f"   Market files: {stats['market_files']}")
    print(f"   Candlestick files: {stats['candlestick_files']}")
    print(f"   Total size: {stats['total_size_mb']:.2f} MB")
    print(f"   Location: {stats['cache_dir']}")

    print(f"\n✅ Data ready for backtesting!")
    print(f"   Run: python run_backtest.py --use-real-data --season {seasons[0]}")


if __name__ == "__main__":
    main()
