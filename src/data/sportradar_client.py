import os
import json
import asyncio
import aiohttp
import time
from typing import Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)

class SportradarClient:
    """
    Client for interacting with Sportradar NFL API (v7).
    """
    
    BASE_URL = "https://api.sportradar.us/nfl/official/trial/v7/en"
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("SPORTRADAR_API_KEY")
        self.last_request_time = 0
        self.rate_limit_delay = 1.1  # Trial key limit: 1 req/sec
        
        if not self.api_key:
            logger.warning("Sportradar API key not found. Live data will be unavailable.")

    async def _make_request(self, endpoint: str) -> Dict[str, Any]:
        """
        Make rate-limited HTTP request.
        """
        if not self.api_key:
            return {}
            
        # Rate limiting
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - elapsed)
            
        url = f"{self.BASE_URL}/{endpoint}.json?api_key={self.api_key}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url) as response:
                    self.last_request_time = time.time()
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.error(f"Sportradar API Error {response.status}: {await response.text()}")
                        return {}
            except Exception as e:
                logger.error(f"Request failed: {e}")
                return {}

    async def get_weekly_schedule(self, year: int, season_type: str, week: int) -> Dict[str, Any]:
        """
        Get schedule for a specific week.
        season_type: 'PRE', 'REG', 'PST'
        """
        endpoint = f"games/{year}/{season_type}/{week}/schedule"
        return await self._make_request(endpoint)

    async def get_game_boxscore(self, game_id: str) -> Dict[str, Any]:
        """
        Get boxscore for a specific game.
        """
        endpoint = f"games/{game_id}/boxscore"
        return await self._make_request(endpoint)

    async def stream_live_data(self, callback: Callable[[Dict], None]):
        """
        Simulate WebSocket stream using polling for Trial API (WebSockets usually require paid tier).
        """
        logger.info("Starting live data stream (polling mode)...")
        while True:
            # In a real scenario, we would poll specific live games
            # For now, just sleep to simulate the loop
            await asyncio.sleep(5)
            # data = await self.get_game_boxscore(game_id)
            # callback(data)
            pass

    def parse_play_event(self, event: Dict) -> Dict[str, Any]:
        """
        Parse a raw play event into a standardized format.
        """
        return {
            "type": event.get("type"),
            "description": event.get("description"),
            "clock": event.get("clock"),
            "quarter": event.get("period"),
            "home_score": event.get("score", {}).get("home"),
            "away_score": event.get("score", {}).get("away")
        }
