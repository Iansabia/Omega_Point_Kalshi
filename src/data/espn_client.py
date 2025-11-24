"""
ESPN API Client for Live NFL Game Data.

Free alternative to Sportradar - no API key required.
Uses the same unofficial API that powers ESPN.com's scoreboard.

Author: Claude
Date: 2025-11-24
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import requests

logger = logging.getLogger(__name__)


class ESPNClient:
    """
    ESPN API client for live NFL game data.

    Free, no API key required. Uses the same unofficial API that powers
    ESPN.com's scoreboard and game pages.

    Example:
        client = ESPNClient()
        games = await client.get_scoreboard(date="20251124")
        game_data = await client.get_game_summary(game_id="401671716")
    """

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

    def __init__(self):
        """Initialize ESPN client."""
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info("Initialized ESPN client (no API key required)")

    async def _ensure_session(self):
        """Ensure aiohttp session is created."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()

    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def get_scoreboard(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get NFL scoreboard for a specific date.

        Args:
            date: Date in YYYYMMDD format (e.g., "20251124").
                  If None, gets current date.

        Returns:
            Scoreboard data containing all games for the date.

        Example:
            scoreboard = await client.get_scoreboard(date="20251124")
            games = scoreboard.get("events", [])
        """
        await self._ensure_session()

        url = f"{self.BASE_URL}/scoreboard"
        params = {}
        if date:
            params["dates"] = date

        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                logger.debug(f"Fetched scoreboard for {date or 'today'}: {len(data.get('events', []))} games")
                return data

        except Exception as e:
            logger.error(f"Error fetching scoreboard: {e}")
            raise

    async def get_game_summary(self, game_id: str) -> Dict[str, Any]:
        """
        Get detailed summary for a specific game.

        Args:
            game_id: ESPN game ID (e.g., "401671716")

        Returns:
            Game summary with live data including:
            - Score
            - Quarter/clock
            - Possession
            - Field position
            - Down/distance

        Example:
            summary = await client.get_game_summary(game_id="401671716")
            status = summary.get("header", {}).get("competitions", [{}])[0].get("status", {})
        """
        await self._ensure_session()

        url = f"{self.BASE_URL}/summary"
        params = {"event": game_id}

        try:
            async with self.session.get(url, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                logger.debug(f"Fetched game summary for {game_id}")
                return data

        except Exception as e:
            logger.error(f"Error fetching game summary for {game_id}: {e}")
            raise

    def parse_game_state(self, game_data: Dict[str, Any], is_scoreboard: bool = True) -> Dict[str, Any]:
        """
        Parse ESPN API response into standardized game state format.

        Args:
            game_data: Raw ESPN API response (from scoreboard or summary)
            is_scoreboard: If True, expects scoreboard format; if False, expects summary format

        Returns:
            Standardized game state dict matching SportradarClient format:
            {
                "home_score": int,
                "away_score": int,
                "score_diff": int,
                "quarter": int,
                "clock": str,
                "clock_seconds": int,
                "time_remaining": int,
                "possession": str,
                "yardline": int,
                "down": int,
                "distance": int,
                "status": str,
                "timestamp": float,
                "home_team": str,
                "away_team": str,
                "game_id": str
            }
        """
        try:
            if is_scoreboard:
                # Scoreboard format: game_data is an "event" object
                competition = game_data.get("competitions", [{}])[0]
                competitors = competition.get("competitors", [])
                situation = competition.get("situation", {})
                status = competition.get("status", {})
            else:
                # Summary format: game_data is the full summary response
                competition = game_data.get("header", {}).get("competitions", [{}])[0]
                competitors = competition.get("competitors", [])
                situation = competition.get("situation", {})
                status = competition.get("status", {})

            # Extract teams
            home_team = None
            away_team = None
            home_score = 0
            away_score = 0

            for competitor in competitors:
                team_abbr = competitor.get("team", {}).get("abbreviation", "")
                score = int(competitor.get("score", 0))

                if competitor.get("homeAway") == "home":
                    home_team = team_abbr
                    home_score = score
                else:
                    away_team = team_abbr
                    away_score = score

            # Extract status
            status_type = status.get("type", {}).get("name", "").lower()
            period = int(status.get("period", 0))
            display_clock = status.get("displayClock", "0:00")

            # Parse clock
            clock_seconds = self._parse_clock_to_seconds(display_clock)

            # Calculate time remaining (including all future quarters)
            time_remaining = clock_seconds + (4 - period) * 900  # 900 seconds per quarter

            # Extract situation (possession, field position, down/distance)
            possession = situation.get("possession", {})
            down_distance_text = situation.get("downDistanceText", "")

            # Determine possession team
            possession_team = None
            if possession:
                # ESPN provides team ID in possession, need to match to home/away
                poss_id = str(possession.get("id", ""))
                for competitor in competitors:
                    if str(competitor.get("id", "")) == poss_id:
                        possession_team = "home" if competitor.get("homeAway") == "home" else "away"
                        break

            # Parse down and distance
            down = 0
            distance = 0
            if down_distance_text:
                # Example: "2nd & 7", "1st & 10", "3rd & Goal"
                parts = down_distance_text.split("&")
                if len(parts) == 2:
                    # Parse down
                    down_str = parts[0].strip().lower()
                    if "1st" in down_str:
                        down = 1
                    elif "2nd" in down_str:
                        down = 2
                    elif "3rd" in down_str:
                        down = 3
                    elif "4th" in down_str:
                        down = 4

                    # Parse distance
                    dist_str = parts[1].strip().lower()
                    if "goal" in dist_str:
                        distance = situation.get("yardLine", 10)  # Use yardline as proxy
                    else:
                        try:
                            distance = int("".join(filter(str.isdigit, dist_str)))
                        except ValueError:
                            distance = 0

            # Parse yardline
            yardline = situation.get("yardLine", 50)

            # Determine game status
            if "pre" in status_type or "scheduled" in status_type:
                game_status = "scheduled"
            elif "in" in status_type or "play" in status_type:
                game_status = "inprogress"
            elif "final" in status_type or "complete" in status_type:
                game_status = "closed"
            elif "half" in status_type:
                game_status = "halftime"
            else:
                game_status = "unknown"

            # Build standardized state
            state = {
                "home_score": home_score,
                "away_score": away_score,
                "score_diff": home_score - away_score,
                "quarter": period,
                "clock": display_clock,
                "clock_seconds": clock_seconds,
                "time_remaining": time_remaining,
                "possession": possession_team or "none",
                "yardline": yardline,
                "down": down,
                "distance": distance,
                "status": game_status,
                "timestamp": datetime.utcnow().timestamp(),
                "home_team": home_team or "HOME",
                "away_team": away_team or "AWAY",
            }

            # Add game ID if available
            if is_scoreboard:
                state["game_id"] = game_data.get("id", "")
            else:
                state["game_id"] = game_data.get("header", {}).get("id", "")

            return state

        except Exception as e:
            logger.error(f"Error parsing game state: {e}", exc_info=True)
            # Return minimal state
            return {
                "home_score": 0,
                "away_score": 0,
                "score_diff": 0,
                "quarter": 0,
                "clock": "0:00",
                "clock_seconds": 0,
                "time_remaining": 0,
                "possession": "none",
                "yardline": 50,
                "down": 0,
                "distance": 0,
                "status": "unknown",
                "timestamp": datetime.utcnow().timestamp(),
                "home_team": "HOME",
                "away_team": "AWAY",
                "game_id": "",
            }

    def _parse_clock_to_seconds(self, clock_str: str) -> int:
        """
        Parse clock string to seconds.

        Args:
            clock_str: Clock string (e.g., "8:45", "0:12")

        Returns:
            Seconds remaining in current quarter
        """
        try:
            parts = clock_str.split(":")
            if len(parts) == 2:
                minutes = int(parts[0])
                seconds = int(parts[1])
                return minutes * 60 + seconds
            return 0
        except Exception:
            return 0

    async def find_game(
        self, home_team: Optional[str] = None, away_team: Optional[str] = None, date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Find a specific game by teams and date.

        Args:
            home_team: Home team abbreviation (e.g., "SF")
            away_team: Away team abbreviation (e.g., "CAR")
            date: Date in YYYYMMDD format (default: today)

        Returns:
            Game data if found, None otherwise
        """
        scoreboard = await self.get_scoreboard(date=date)
        events = scoreboard.get("events", [])

        for event in events:
            competition = event.get("competitions", [{}])[0]
            competitors = competition.get("competitors", [])

            teams = {}
            for competitor in competitors:
                team_abbr = competitor.get("team", {}).get("abbreviation", "")
                home_away = competitor.get("homeAway", "")
                teams[home_away] = team_abbr

            # Check if this is the game
            if home_team and away_team:
                if teams.get("home") == home_team and teams.get("away") == away_team:
                    return event
            elif home_team:
                if teams.get("home") == home_team:
                    return event
            elif away_team:
                if teams.get("away") == away_team:
                    return event

        return None

    async def poll_live_game(
        self, game_id: str, callback: Callable[[Dict[str, Any]], None], interval: int = 2, max_polls: int = 0
    ):
        """
        Poll a live game and call callback with updated state.

        This method matches the interface of SportradarClient.poll_live_game()
        for drop-in replacement.

        Args:
            game_id: ESPN game ID
            callback: Function to call with game state updates
            interval: Polling interval in seconds (default: 2)
            max_polls: Maximum number of polls (0 = unlimited)

        Example:
            def on_update(state):
                print(f"Score: {state['home_score']}-{state['away_score']}")

            await client.poll_live_game(game_id="401671716", callback=on_update)
        """
        polls = 0
        logger.info(f"Starting to poll game {game_id} every {interval}s")

        try:
            while True:
                # Check max polls
                if max_polls > 0 and polls >= max_polls:
                    logger.info(f"Reached max polls ({max_polls})")
                    break

                try:
                    # Fetch game summary
                    summary = await self.get_game_summary(game_id)

                    # Parse to standardized format
                    state = self.parse_game_state(summary, is_scoreboard=False)

                    # Call callback
                    callback(state)

                    # Check if game is over
                    if state["status"] == "closed":
                        logger.info(f"Game {game_id} is closed, stopping polling")
                        break

                except Exception as e:
                    logger.error(f"Error polling game {game_id}: {e}")

                # Increment and wait
                polls += 1
                await asyncio.sleep(interval)

        except asyncio.CancelledError:
            logger.info(f"Polling cancelled for game {game_id}")

    # Synchronous helpers for convenience
    def get_scoreboard_sync(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Synchronous version of get_scoreboard()."""
        url = f"{self.BASE_URL}/scoreboard"
        params = {}
        if date:
            params["dates"] = date

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_game_summary_sync(self, game_id: str) -> Dict[str, Any]:
        """Synchronous version of get_game_summary()."""
        url = f"{self.BASE_URL}/summary"
        params = {"event": game_id}

        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
