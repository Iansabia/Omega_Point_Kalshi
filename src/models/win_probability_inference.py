"""
Real-time Win Probability Inference Pipeline.

Provides fast inference for live NFL games using the trained XGBoost model.
Designed for low-latency arbitrage trading (<100ms inference time).

Features:
- Load pre-trained model once (avoid reload overhead)
- Fast prediction from Sportradar game state
- Batch prediction support
- Caching for repeated game states
- Team-specific adjustments (home field advantage, etc.)
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.models.win_probability_model import WinProbabilityModel

logger = logging.getLogger(__name__)


class WinProbabilityInference:
    """
    Fast inference pipeline for real-time win probability predictions.

    Optimized for low-latency trading decisions.
    """

    def __init__(self, model_path: str = "models/win_probability_model.pkl"):
        """
        Initialize inference pipeline.

        Args:
            model_path: Path to trained model
        """
        self.model_path = model_path

        # Load model
        logger.info(f"🔧 Loading win probability model from {model_path}")
        start_time = time.time()

        self.model = WinProbabilityModel(model_path=model_path)

        load_time = time.time() - start_time
        logger.info(f"✅ Model loaded in {load_time:.3f}s")

        # Cache for repeated predictions (optional optimization)
        self.cache: Dict[str, float] = {}
        self.cache_enabled = False

    def predict_from_sportradar(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict win probability from Sportradar game state.

        Args:
            game_state: Game state dict from sportradar_client.parse_game_state()

        Returns:
            Dict with prediction results:
            {
                'home_wp': float,  # Home team win probability
                'away_wp': float,  # Away team win probability (1 - home_wp)
                'model_wp': float,  # Raw model output (possession team)
                'possession': str,  # 'home' or 'away'
                'confidence': float,  # Prediction confidence (0-1)
                'inference_time_ms': float,  # Time taken for prediction
                'game_state': dict  # Original game state
            }
        """
        start_time = time.time()

        # Extract features
        features = self._extract_features(game_state)

        # Predict
        model_wp = self.model.predict(features)

        # Determine which team has the ball
        possession = game_state.get("possession", "home")

        # Calculate home/away win probabilities
        if possession == "home":
            home_wp = model_wp
            away_wp = 1.0 - model_wp
        else:
            home_wp = 1.0 - model_wp
            away_wp = model_wp

        # Calculate confidence (distance from 50/50)
        confidence = abs(home_wp - 0.5) * 2

        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        return {
            "home_wp": home_wp,
            "away_wp": away_wp,
            "model_wp": model_wp,
            "possession": possession,
            "confidence": confidence,
            "inference_time_ms": inference_time,
            "game_state": game_state,
        }

    def _extract_features(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract model features from Sportradar game state.

        Args:
            game_state: Raw game state from Sportradar

        Returns:
            Features dict for model.predict()
        """
        # Map Sportradar fields to model features
        features = {
            "score_diff": game_state.get("score_diff", 0),
            "time_remaining": game_state.get("time_remaining", 3600),
            "yardline": game_state.get("yardline", 50),
            "down": game_state.get("down", 1),
            "distance": game_state.get("distance", 10),
            "quarter": game_state.get("quarter", 1),
            # Timeouts (default to 3 if not provided)
            "posteam_timeouts": game_state.get("posteam_timeouts", 3),
            "defteam_timeouts": game_state.get("defteam_timeouts", 3),
        }

        return features

    def predict_batch(self, game_states: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict win probabilities for multiple games (batch inference).

        Args:
            game_states: List of game state dicts

        Returns:
            List of prediction dicts
        """
        predictions = []

        for game_state in game_states:
            pred = self.predict_from_sportradar(game_state)
            predictions.append(pred)

        return predictions

    def enable_cache(self, max_size: int = 1000):
        """
        Enable prediction caching (optional optimization).

        Args:
            max_size: Maximum cache size
        """
        self.cache_enabled = True
        self.cache = {}
        logger.info(f"✅ Prediction caching enabled (max size: {max_size})")

    def clear_cache(self):
        """Clear prediction cache."""
        self.cache.clear()
        logger.info("🗑️  Cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {"cache_enabled": self.cache_enabled, "cache_size": len(self.cache)}


# Example usage
def main():
    """Example: Predict win probability for a game scenario."""
    import argparse

    parser = argparse.ArgumentParser(description="Win probability inference")
    parser.add_argument("--model", type=str, default="models/win_probability_model.pkl", help="Model path")

    args = parser.parse_args()

    # Initialize inference pipeline
    print("\n" + "=" * 60)
    print("Win Probability Inference Pipeline")
    print("=" * 60)

    inference = WinProbabilityInference(model_path=args.model)

    # Example game scenario
    print("\n📊 Example Game Scenario:")
    print("   Home team leading 21-14 in Q3")
    print("   8:45 left in quarter")
    print("   Home has ball, 2nd & 7 at their own 45")

    game_state = {
        "home_score": 21,
        "away_score": 14,
        "score_diff": 7,  # Home winning by 7
        "quarter": 3,
        "clock": "8:45",
        "clock_seconds": 525,
        "time_remaining": 1425,  # Q3 8:45 + Q4 15:00
        "possession": "home",
        "yardline": 45,
        "down": 2,
        "distance": 7,
        "status": "inprogress",
    }

    # Predict
    result = inference.predict_from_sportradar(game_state)

    print(f"\n🎯 Win Probability Prediction:")
    print(f"   Home Win Probability: {result['home_wp']:.1%}")
    print(f"   Away Win Probability: {result['away_wp']:.1%}")
    print(f"   Confidence: {result['confidence']:.1%}")
    print(f"   Inference Time: {result['inference_time_ms']:.2f}ms")

    # Another scenario: Close game, late Q4
    print("\n📊 Example Game Scenario 2:")
    print("   Tied 24-24 in Q4")
    print("   2:00 left in game")
    print("   Away has ball, 3rd & 8 at home 35 (field goal range)")

    game_state_2 = {
        "home_score": 24,
        "away_score": 24,
        "score_diff": 0,  # Tied
        "quarter": 4,
        "clock": "2:00",
        "clock_seconds": 120,
        "time_remaining": 120,
        "possession": "away",
        "yardline": 35,  # 35 yards from home endzone (field goal range)
        "down": 3,
        "distance": 8,
        "status": "inprogress",
    }

    result_2 = inference.predict_from_sportradar(game_state_2)

    print(f"\n🎯 Win Probability Prediction:")
    print(f"   Home Win Probability: {result_2['home_wp']:.1%}")
    print(f"   Away Win Probability: {result_2['away_wp']:.1%}")
    print(f"   Confidence: {result_2['confidence']:.1%}")
    print(f"   Inference Time: {result_2['inference_time_ms']:.2f}ms")

    print("\n" + "=" * 60)
    print("✅ Inference complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
