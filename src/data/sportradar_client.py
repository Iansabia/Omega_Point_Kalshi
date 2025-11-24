import asyncio
import json
import logging
import os
import time
from typing import Any, Callable, Dict, Optional

import aiohttp

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

    async def get_live_game_summary(self, game_id: str) -> Dict[str, Any]:
        """
        Get live game summary with current score, quarter, and game state.
        This is faster than boxscore and sufficient for win probability calculations.
        """
        endpoint = f"games/{game_id}/summary"
        return await self._make_request(endpoint)

    async def poll_live_games(self, game_ids: list[str], callback: Callable[[str, Dict], None], interval: float = 2.0):
        """
        Poll multiple live games continuously.

        Args:
            game_ids: List of Sportradar game IDs to track
            callback: Function called with (game_id, game_state) on each update
            interval: Polling interval in seconds (default 2s for live games)
        """
        logger.info(f"Starting live game polling for {len(game_ids)} games (interval: {interval}s)")

        while True:
            for game_id in game_ids:
                try:
                    game_data = await self.get_live_game_summary(game_id)
                    if game_data:
                        game_state = self.parse_game_state(game_data)
                        callback(game_id, game_state)
                except Exception as e:
                    logger.error(f"Error polling game {game_id}: {e}")

            await asyncio.sleep(interval)

    def parse_game_state(self, game_data: Dict) -> Dict[str, Any]:
        """
        Parse Sportradar game summary into standardized game state for win probability model.

        Returns dict with keys needed for WP calculation:
        - home_score, away_score: Current scores
        - quarter: Current quarter (1-4, 5=OT)
        - clock: Time remaining in quarter (seconds)
        - possession: Team with ball ('home' or 'away')
        - yardline: Field position (0-100, 50=midfield)
        - down: Current down (1-4)
        - distance: Yards to go for first down
        - status: Game status ('inprogress', 'closed', etc.)
        """
        try:
            summary = game_data.get("summary", {}) if "summary" in game_data else game_data

            # Score
            home_score = summary.get("home", {}).get("points", 0)
            away_score = summary.get("away", {}).get("points", 0)

            # Quarter and clock
            quarter = summary.get("quarter", 1)
            clock = summary.get("clock", "15:00")  # MM:SS format

            # Convert clock to seconds
            if ":" in str(clock):
                mins, secs = map(int, str(clock).split(":"))
                clock_seconds = mins * 60 + secs
            else:
                clock_seconds = 900  # Default to full quarter (15 min)

            # Possession and situation
            situation = summary.get("situation", {})
            possession = situation.get("possession", {}).get("alias", "home")  # 'home' or 'away'

            # Field position (yardline_100 format: 0-100, 50=midfield)
            yardline = situation.get("location", {}).get("yardline", 50)

            # Down and distance
            down = situation.get("down", 1)
            distance = situation.get("yfd", 10)  # Yards for first down

            # Game status
            status = summary.get("status", "inprogress")

            return {
                "home_score": home_score,
                "away_score": away_score,
                "score_diff": home_score - away_score,  # Positive = home winning
                "quarter": quarter,
                "clock": clock,
                "clock_seconds": clock_seconds,
                "time_remaining": self._calculate_time_remaining(quarter, clock_seconds),
                "possession": possession,
                "yardline": yardline,
                "down": down,
                "distance": distance,
                "status": status,
                "timestamp": time.time(),
            }

        except Exception as e:
            logger.error(f"Error parsing game state: {e}")
            return {}

    def _calculate_time_remaining(self, quarter: int, clock_seconds: int) -> int:
        """
        Calculate total time remaining in game (in seconds).

        Args:
            quarter: Current quarter (1-4, 5=OT)
            clock_seconds: Seconds remaining in current quarter

        Returns:
            Total seconds remaining in regulation
        """
        if quarter >= 5:  # Overtime
            return 0  # OT is sudden death, use 0 for model

        seconds_per_quarter = 900  # 15 minutes
        quarters_remaining = 4 - quarter

        return (quarters_remaining * seconds_per_quarter) + clock_seconds

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
            "away_score": event.get("score", {}).get("away"),
        }
