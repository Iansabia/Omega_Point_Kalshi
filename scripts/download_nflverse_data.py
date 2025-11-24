#!/usr/bin/env python3
"""
Download nflverse Play-by-Play Data for Win Probability Model Training.

Downloads historical NFL play-by-play data from nflverse (formerly nflfastR).
This data is used to train the XGBoost win probability model.

Data source: https://github.com/nflverse/nflverse-data
Format: Parquet files hosted on GitHub

Features:
- Play-by-play data with 372+ columns per play
- Pre-calculated win probability (EPA model)
- Game state: score, time, field position, down & distance
- Play outcomes: yards gained, TDs, turnovers, etc.

Usage:
    python scripts/download_nflverse_data.py --seasons 2020 2021 2022 2023
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class NFLverseDataDownloader:
    """Download and cache nflverse play-by-play data."""

    BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{season}.parquet"

    def __init__(self, cache_dir: str = "data/nflverse"):
        """
        Initialize downloader.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Cache directory: {self.cache_dir}")

    def download_season(self, season: int, force_redownload: bool = False) -> pd.DataFrame:
        """
        Download play-by-play data for a single season.

        Args:
            season: NFL season year (e.g., 2023)
            force_redownload: Re-download even if cached

        Returns:
            DataFrame with play-by-play data
        """
        cache_file = self.cache_dir / f"pbp_{season}.parquet"

        # Check cache
        if cache_file.exists() and not force_redownload:
            logger.info(f"📂 Loading cached data for {season} from {cache_file}")
            return pd.read_parquet(cache_file)

        # Download from nflverse
        url = self.BASE_URL.format(season=season)
        logger.info(f"🌐 Downloading {season} season data from nflverse...")
        logger.info(f"   URL: {url}")

        try:
            df = pd.read_parquet(url)

            # Save to cache
            df.to_parquet(cache_file, index=False)
            logger.info(f"✅ Downloaded {len(df):,} plays for {season} season")
            logger.info(f"💾 Cached to {cache_file}")

            return df

        except Exception as e:
            logger.error(f"❌ Failed to download {season} data: {e}")
            raise

    def download_multiple_seasons(self, seasons: List[int], force_redownload: bool = False) -> pd.DataFrame:
        """
        Download and combine data for multiple seasons.

        Args:
            seasons: List of season years
            force_redownload: Re-download even if cached

        Returns:
            Combined DataFrame with all seasons
        """
        logger.info(f"📥 Downloading data for {len(seasons)} seasons: {seasons}")

        all_data = []

        for season in seasons:
            try:
                df = self.download_season(season, force_redownload=force_redownload)
                all_data.append(df)
            except Exception as e:
                logger.error(f"⚠️  Skipping {season} due to error: {e}")
                continue

        if not all_data:
            raise ValueError("No data downloaded successfully")

        # Combine all seasons
        combined = pd.concat(all_data, ignore_index=True)
        logger.info(f"\n✅ Combined {len(combined):,} total plays from {len(all_data)} seasons")

        return combined

    def get_win_probability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features relevant for win probability modeling.

        Args:
            df: Raw play-by-play data

        Returns:
            DataFrame with WP model features

        Key features for win probability:
        - score_differential: Point differential (positive = team winning)
        - half_seconds_remaining: Time remaining (seconds)
        - posteam_timeouts_remaining: Timeouts left for possession team
        - defteam_timeouts_remaining: Timeouts left for defense
        - yardline_100: Field position (0-100, 0=own endzone, 100=opponent endzone)
        - down: Current down (1-4)
        - ydstogo: Yards to go for first down
        - posteam_score: Possession team score
        - defteam_score: Defense team score
        - qtr: Quarter (1-4, 5=OT)
        - wp: Pre-calculated win probability (target variable)
        """
        logger.info("🔧 Extracting win probability features...")

        # Select relevant columns
        feature_cols = [
            # Target variable
            "wp",  # Win probability (our target)
            # Game state
            "game_id",
            "season",
            "week",
            "posteam",  # Possession team
            "defteam",  # Defense team
            "game_date",
            "home_team",
            "away_team",
            # Score
            "posteam_score",
            "defteam_score",
            "score_differential",
            # Time
            "qtr",
            "quarter_seconds_remaining",
            "half_seconds_remaining",
            "game_seconds_remaining",
            # Field position
            "yardline_100",  # 0-100 (distance to opponent endzone)
            "down",
            "ydstogo",  # Yards to go
            # Timeouts
            "posteam_timeouts_remaining",
            "defteam_timeouts_remaining",
            # Play type (for filtering)
            "play_type",
            # Stadium (home field advantage)
            "home_score",
            "away_score",
        ]

        # Filter to only columns that exist
        available_cols = [col for col in feature_cols if col in df.columns]
        missing_cols = set(feature_cols) - set(available_cols)

        if missing_cols:
            logger.warning(f"⚠️  Missing columns: {missing_cols}")

        wp_df = df[available_cols].copy()

        # Filter to regular plays (exclude kickoffs, timeouts, etc.)
        if "play_type" in wp_df.columns:
            regular_plays = wp_df["play_type"].isin(["pass", "run", "punt", "field_goal"])
            wp_df = wp_df[regular_plays]
            logger.info(f"   Filtered to {len(wp_df):,} regular plays")

        # Remove rows with missing WP
        if "wp" in wp_df.columns:
            wp_df = wp_df.dropna(subset=["wp"])
            logger.info(f"   {len(wp_df):,} plays with valid win probability")

        logger.info(f"✅ Extracted {len(wp_df):,} plays with {len(wp_df.columns)} features")

        return wp_df

    def get_cache_stats(self) -> dict:
        """Get statistics about cached data."""
        cache_files = list(self.cache_dir.glob("pbp_*.parquet"))

        total_size = sum(f.stat().st_size for f in cache_files) / (1024**2)  # MB

        seasons = sorted([int(f.stem.split("_")[1]) for f in cache_files])

        return {
            "cached_seasons": seasons,
            "num_files": len(cache_files),
            "total_size_mb": total_size,
            "cache_dir": str(self.cache_dir),
        }

    def clear_cache(self):
        """Clear all cached data."""
        import shutil

        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"🗑️  Cleared cache: {self.cache_dir}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Download nflverse play-by-play data")

    parser.add_argument("--seasons", type=int, nargs="+", default=[2020, 2021, 2022, 2023], help="Seasons to download")

    parser.add_argument("--cache-dir", type=str, default="data/nflverse", help="Cache directory")

    parser.add_argument("--force-redownload", action="store_true", help="Force re-download even if cached")

    parser.add_argument("--extract-features", action="store_true", help="Extract win probability features")

    parser.add_argument("--stats", action="store_true", help="Show cache statistics")

    args = parser.parse_args()

    downloader = NFLverseDataDownloader(cache_dir=args.cache_dir)

    if args.stats:
        stats = downloader.get_cache_stats()
        print("\n" + "=" * 60)
        print("NFLverse Cache Statistics")
        print("=" * 60)
        print(f"Cache Directory: {stats['cache_dir']}")
        print(f"Cached Seasons: {stats['cached_seasons']}")
        print(f"Number of Files: {stats['num_files']}")
        print(f"Total Size: {stats['total_size_mb']:.1f} MB")
        print("=" * 60 + "\n")
        return

    # Download data
    print("\n" + "=" * 60)
    print("NFLverse Data Downloader")
    print("=" * 60)

    combined_df = downloader.download_multiple_seasons(args.seasons, force_redownload=args.force_redownload)

    print(f"\n📊 Dataset Summary:")
    print(f"   Total Plays: {len(combined_df):,}")
    print(f"   Columns: {len(combined_df.columns)}")
    print(f"   Memory Usage: {combined_df.memory_usage(deep=True).sum() / (1024**2):.1f} MB")

    # Extract features if requested
    if args.extract_features:
        wp_df = downloader.get_win_probability_features(combined_df)

        # Save feature dataset
        features_file = Path(args.cache_dir) / "wp_features.parquet"
        wp_df.to_parquet(features_file, index=False)
        print(f"\n💾 Saved win probability features to {features_file}")
        print(f"   Features: {len(wp_df.columns)}")
        print(f"   Samples: {len(wp_df):,}")

    print("\n" + "=" * 60)
    print("✅ Download complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
