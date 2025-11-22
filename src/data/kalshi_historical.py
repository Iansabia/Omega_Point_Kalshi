"""
Kalshi Historical Data Fetcher.

Downloads and caches historical market data from Kalshi API for backtesting.

Features:
- Fetch settled NFL markets
- Download candlestick price data
- Cache data locally to avoid re-fetching
- Handle rate limits and pagination
- Data validation and quality checks

Usage:
    from src.data.kalshi_historical import KalshiHistoricalDataFetcher

    fetcher = KalshiHistoricalDataFetcher()
    markets = fetcher.fetch_nfl_markets(season=2024)
    fetcher.download_candlesticks(markets)
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time

from src.execution.kalshi_client import KalshiClient


class KalshiHistoricalDataFetcher:
    """Fetch and cache historical Kalshi market data."""

    def __init__(
        self,
        cache_dir: str = "data/historical",
        kalshi_client: Optional[KalshiClient] = None
    ):
        """
        Initialize historical data fetcher.

        Args:
            cache_dir: Directory to cache downloaded data
            kalshi_client: Optional pre-configured KalshiClient
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.cache_dir / "candlesticks").mkdir(exist_ok=True)
        (self.cache_dir / "markets").mkdir(exist_ok=True)

        # Initialize Kalshi client
        self.client = kalshi_client or KalshiClient()

    def fetch_nfl_markets(
        self,
        season: Optional[int] = None,
        status: str = "settled",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch NFL markets from Kalshi.

        Args:
            season: NFL season year (e.g., 2024), None = all seasons
            status: Market status ('settled', 'closed', 'open')
            use_cache: Load from cache if available

        Returns:
            DataFrame with market metadata
        """
        cache_file = self.cache_dir / f"markets/nfl_{status}_{season or 'all'}.csv"

        # Check cache
        if use_cache and cache_file.exists():
            print(f"📂 Loading cached markets from {cache_file}")
            return pd.read_csv(cache_file)

        print(f"🌐 Fetching {status} NFL markets from Kalshi API...")

        # Fetch markets using pagination
        # Kalshi uses series tickers like 'HIGHFB' for NFL
        all_markets = self.client.get_all_markets_paginated(
            series_ticker="HIGHFB",  # NFL high scorer markets
            status=status
        )

        if not all_markets:
            print("⚠️  No markets found")
            return pd.DataFrame()

        print(f"✅ Found {len(all_markets)} markets")

        # Convert to DataFrame
        df = pd.DataFrame(all_markets)

        # Filter by season if specified
        if season and 'ticker' in df.columns:
            # Kalshi tickers often include date: HIGHFB-24SEP15-B-KC
            # Extract year from ticker
            df['year'] = df['ticker'].str.extract(r'-(\d{2})', expand=False)
            df['year'] = '20' + df['year']  # Convert 24 -> 2024
            df = df[df['year'] == str(season)]

        # Save to cache
        df.to_csv(cache_file, index=False)
        print(f"💾 Cached markets to {cache_file}")

        return df

    def download_candlesticks(
        self,
        markets: pd.DataFrame,
        period_interval: int = 60,
        use_cache: bool = True,
        max_markets: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Download candlestick data for multiple markets.

        Args:
            markets: DataFrame with market metadata
            period_interval: Candlestick interval (1, 60, or 1440 minutes)
            use_cache: Skip download if cached
            max_markets: Limit number of markets (for testing)

        Returns:
            Dict mapping ticker -> candlestick DataFrame
        """
        candlesticks = {}
        total = min(len(markets), max_markets) if max_markets else len(markets)

        print(f"\n📥 Downloading candlesticks for {total} markets...")
        print(f"   Interval: {period_interval} minutes")

        for i, row in markets.head(total).iterrows():
            ticker = row.get('ticker')
            series_ticker = row.get('series_ticker', 'HIGHFB')

            if not ticker:
                continue

            cache_file = self.cache_dir / f"candlesticks/{ticker}.csv"

            # Check cache
            if use_cache and cache_file.exists():
                candlesticks[ticker] = pd.read_csv(cache_file)
                if (i + 1) % 10 == 0:
                    print(f"   📂 Loaded {i+1}/{total} from cache")
                continue

            # Download from API
            try:
                response = self.client.get_market_candlesticks(
                    series_ticker=series_ticker,
                    market_ticker=ticker,
                    period_interval=period_interval
                )

                candles = response.get('candlesticks', [])

                if candles:
                    df = pd.DataFrame(candles)
                    df.to_csv(cache_file, index=False)
                    candlesticks[ticker] = df

                    if (i + 1) % 10 == 0:
                        print(f"   🌐 Downloaded {i+1}/{total}")

                # Rate limiting
                time.sleep(0.2)

            except Exception as e:
                print(f"   ⚠️  Error downloading {ticker}: {e}")
                continue

        print(f"✅ Downloaded {len(candlesticks)} candlestick datasets")
        return candlesticks

    def load_historical_backtest_data(
        self,
        season: int = 2024,
        max_games: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Load historical data formatted for backtesting.

        Args:
            season: NFL season year
            max_games: Limit number of games

        Returns:
            List of game scenarios with price paths and outcomes
        """
        # Fetch markets
        markets = self.fetch_nfl_markets(season=season, status="settled")

        if markets.empty:
            print(f"❌ No settled markets found for {season}")
            return []

        # Download candlesticks
        candlesticks = self.download_candlesticks(
            markets,
            period_interval=60,  # 1-hour intervals
            max_markets=max_games
        )

        # Format for backtest
        scenarios = []

        for ticker, candles_df in candlesticks.items():
            if candles_df.empty:
                continue

            market_info = markets[markets['ticker'] == ticker].iloc[0]

            # Extract price path
            if 'close' in candles_df.columns:
                price_path = candles_df['close'].tolist()
            else:
                continue

            # Get outcome (yes=1, no=0)
            result = market_info.get('result', None)
            outcome = 1 if result == 'yes' else 0

            # Get initial and final prices
            initial_price = price_path[0] if price_path else 0.5
            final_price = price_path[-1] if price_path else 0.5

            # Get event metadata
            event_ticker = market_info.get('event_ticker', '')
            market_title = market_info.get('title', ticker)

            scenarios.append({
                'game_id': len(scenarios),
                'ticker': ticker,
                'event_ticker': event_ticker,
                'title': market_title,
                'true_prob': final_price,  # Market's final assessment
                'initial_price': initial_price,
                'price_path': price_path,
                'outcome': outcome,
                'start_time': datetime.fromtimestamp(candles_df['start_ts'].iloc[0]) if 'start_ts' in candles_df.columns else None,
                'end_time': datetime.fromtimestamp(candles_df['end_ts'].iloc[-1]) if 'end_ts' in candles_df.columns else None,
                'num_candlesticks': len(candles_df),
                'is_real_data': True  # Flag to indicate real vs synthetic
            })

        print(f"\n✅ Loaded {len(scenarios)} historical game scenarios")
        print(f"   Average price path length: {np.mean([len(s['price_path']) for s in scenarios]):.1f} candlesticks")

        return scenarios

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about cached data."""
        market_files = list((self.cache_dir / "markets").glob("*.csv"))
        candle_files = list((self.cache_dir / "candlesticks").glob("*.csv"))

        total_size = sum(f.stat().st_size for f in market_files + candle_files)

        return {
            'market_files': len(market_files),
            'candlestick_files': len(candle_files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

    def clear_cache(self):
        """Clear all cached data."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            (self.cache_dir / "candlesticks").mkdir(exist_ok=True)
            (self.cache_dir / "markets").mkdir(exist_ok=True)
            print(f"🗑️  Cleared cache: {self.cache_dir}")
