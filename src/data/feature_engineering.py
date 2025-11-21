import pandas as pd
import numpy as np
from typing import Dict, Any, List

class FeatureEngineer:
    """
    Calculates derived features from raw NFL data for agent consumption.
    """
    
    def __init__(self):
        self.team_elos = {} # team -> elo
        self.base_elo = 1500
        self.k_factor = 20

    def calculate_elo(self, home_team: str, away_team: str, home_score: int, away_score: int) -> Dict[str, float]:
        """
        Update ELO ratings based on game result.
        """
        home_elo = self.team_elos.get(home_team, self.base_elo)
        away_elo = self.team_elos.get(away_team, self.base_elo)
        
        # Expected scores
        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 / (1 + 10 ** ((home_elo - away_elo) / 400))
        
        # Actual scores (1 = win, 0.5 = tie, 0 = loss)
        if home_score > away_score:
            actual_home = 1.0
            actual_away = 0.0
        elif home_score < away_score:
            actual_home = 0.0
            actual_away = 1.0
        else:
            actual_home = 0.5
            actual_away = 0.5
            
        # Update
        new_home_elo = home_elo + self.k_factor * (actual_home - expected_home)
        new_away_elo = away_elo + self.k_factor * (actual_away - expected_away)
        
        self.team_elos[home_team] = new_home_elo
        self.team_elos[away_team] = new_away_elo
        
        return {home_team: new_home_elo, away_team: new_away_elo}

    def calculate_momentum(self, team_scores: List[int], window: int = 3) -> float:
        """
        Calculate momentum based on recent scoring margin or wins.
        Simple implementation: Average score over last N games.
        """
        if not team_scores:
            return 0.0
        
        recent = team_scores[-window:]
        return np.mean(recent)

    def calculate_volatility(self, price_history: List[float], window: int = 10) -> float:
        """
        Calculate realized volatility of an asset.
        """
        if len(price_history) < 2:
            return 0.0
            
        returns = np.diff(np.log(price_history))
        if len(returns) < window:
            return np.std(returns)
            
        return np.std(returns[-window:])

    def process_game_features(self, game_data: Dict) -> Dict[str, Any]:
        """
        Extract all relevant features for a game.
        """
        home = game_data.get('home_team')
        away = game_data.get('away_team')
        
        return {
            "home_elo": self.team_elos.get(home, self.base_elo),
            "away_elo": self.team_elos.get(away, self.base_elo),
            "elo_diff": self.team_elos.get(home, self.base_elo) - self.team_elos.get(away, self.base_elo),
            # Add more features here
        }
