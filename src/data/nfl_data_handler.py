import pandas as pd
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class NFLDataHandler:
    """
    Handler for fetching and processing NFL data from nflverse.
    """
    
    BASE_URL = "https://github.com/nflverse/nflverse-data/releases/download"
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = cache_dir
        # Ensure cache dir exists if we were to use local caching
        # os.makedirs(cache_dir, exist_ok=True)

    def load_pbp_data(self, seasons: List[int]) -> pd.DataFrame:
        """
        Load play-by-play data for specified seasons.
        """
        dfs = []
        for season in seasons:
            try:
                url = f"{self.BASE_URL}/pbp/play_by_play_{season}.parquet"
                logger.info(f"Fetching PBP data for {season} from {url}...")
                df = pd.read_parquet(url)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load PBP data for {season}: {e}")
        
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)

    def load_player_stats(self, seasons: List[int]) -> pd.DataFrame:
        """
        Load player stats.
        """
        # nflverse stores player stats in a slightly different structure, 
        # often aggregated. For now, let's pull weekly stats.
        dfs = []
        for season in seasons:
            try:
                url = f"{self.BASE_URL}/player_stats/player_stats_{season}.parquet"
                logger.info(f"Fetching player stats for {season}...")
                df = pd.read_parquet(url)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load player stats for {season}: {e}")
                
        if not dfs:
            return pd.DataFrame()
            
        return pd.concat(dfs, ignore_index=True)

    def get_team_stats(self, season: int) -> pd.DataFrame:
        """
        Get team level stats/schedules.
        """
        try:
            url = f"{self.BASE_URL}/schedules/schedule_{season}.parquet"
            return pd.read_parquet(url)
        except Exception as e:
            logger.error(f"Failed to load schedule for {season}: {e}")
            return pd.DataFrame()
