"""
Win Probability Model using XGBoost.

Trains an XGBoost model to predict NFL win probability based on game state.
Uses nflverse play-by-play data (nflfastR format).

Model features:
- Score differential
- Time remaining
- Field position (yardline)
- Down and distance
- Timeouts remaining
- Quarter

This model predicts the probability that the possession team will win the game.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss, mean_absolute_error, roc_auc_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WinProbabilityModel:
    """
    XGBoost model for NFL win probability prediction.

    Based on nflfastR's approach but simplified for real-time inference.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize win probability model.

        Args:
            model_path: Path to saved model (optional, for loading pre-trained model)
        """
        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []
        self.model_path = model_path

        if model_path and Path(model_path).exists():
            self.load_model(model_path)

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for model training/inference.

        Args:
            df: DataFrame with nflverse play-by-play data

        Returns:
            DataFrame with engineered features
        """
        logger.info("🔧 Engineering features...")

        features = df.copy()

        # Core features (already in dataset)
        # - score_differential: Point difference (positive = posteam winning)
        # - half_seconds_remaining: Time remaining (seconds)
        # - yardline_100: Distance to opponent endzone (0-100)
        # - down: Current down (1-4)
        # - ydstogo: Yards to go for first down
        # - qtr: Quarter (1-4, 5=OT)
        # - posteam_timeouts_remaining
        # - defteam_timeouts_remaining

        # Engineered features

        # 1. Timeout advantage
        if "posteam_timeouts_remaining" in features.columns and "defteam_timeouts_remaining" in features.columns:
            features["timeout_advantage"] = features["posteam_timeouts_remaining"] - features["defteam_timeouts_remaining"]

        # 2. Field position zones (easier to interpret than raw yardline)
        if "yardline_100" in features.columns:
            features["in_red_zone"] = (features["yardline_100"] <= 20).astype(int)
            features["in_fg_range"] = (features["yardline_100"] <= 35).astype(int)
            features["backed_up"] = (features["yardline_100"] >= 95).astype(int)

        # 3. Down situation
        if "down" in features.columns and "ydstogo" in features.columns:
            # 3rd/4th down conversion difficulty
            features["is_3rd_4th_down"] = (features["down"] >= 3).astype(int)
            features["conversion_difficulty"] = features["ydstogo"] / (5 - features["down"].clip(1, 4))

        # 4. Time pressure (late game situations)
        if "half_seconds_remaining" in features.columns:
            features["is_two_minute_drill"] = (features["half_seconds_remaining"] <= 120).astype(int)
            features["is_final_minute"] = (features["half_seconds_remaining"] <= 60).astype(int)

        # 5. Score situation
        if "score_differential" in features.columns:
            features["is_close_game"] = (features["score_differential"].abs() <= 7).astype(int)
            features["is_blowout"] = (features["score_differential"].abs() >= 21).astype(int)

        # Select final feature set
        self.feature_names = [
            # Core features
            "score_differential",
            "half_seconds_remaining",
            "yardline_100",
            "down",
            "ydstogo",
            "qtr",
            "posteam_timeouts_remaining",
            "defteam_timeouts_remaining",
            # Engineered features
            "timeout_advantage",
            "in_red_zone",
            "in_fg_range",
            "backed_up",
            "is_3rd_4th_down",
            "conversion_difficulty",
            "is_two_minute_drill",
            "is_final_minute",
            "is_close_game",
            "is_blowout",
        ]

        # Filter to available features
        available_features = [f for f in self.feature_names if f in features.columns]
        missing_features = set(self.feature_names) - set(available_features)

        if missing_features:
            logger.warning(f"⚠️  Missing features: {missing_features}")
            self.feature_names = available_features

        logger.info(f"✅ Engineered {len(self.feature_names)} features")

        return features[self.feature_names]

    def train(
        self,
        data_path: str = "data/nflverse/wp_features.parquet",
        test_size: float = 0.2,
        random_state: int = 42,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ) -> Dict[str, float]:
        """
        Train XGBoost win probability model.

        Args:
            data_path: Path to nflverse features parquet file
            test_size: Fraction of data for testing
            random_state: Random seed for reproducibility
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)

        Returns:
            Dict with training metrics
        """
        logger.info("=" * 60)
        logger.info("Training Win Probability Model")
        logger.info("=" * 60)

        # Load data
        logger.info(f"📂 Loading training data from {data_path}")
        df = pd.read_parquet(data_path)
        logger.info(f"   Loaded {len(df):,} plays")

        # Prepare features
        X = self.prepare_features(df)

        # Target: Win probability (0-1)
        y = df["wp"]

        # Remove NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]

        logger.info(f"📊 Training dataset: {len(X):,} samples, {len(self.feature_names)} features")

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        logger.info(f"   Train: {len(X_train):,} samples")
        logger.info(f"   Test:  {len(X_test):,} samples")

        # Initialize XGBoost model
        logger.info(f"\n🚀 Training XGBoost model...")
        logger.info(f"   n_estimators: {n_estimators}")
        logger.info(f"   max_depth: {max_depth}")
        logger.info(f"   learning_rate: {learning_rate}")

        self.model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            objective="reg:squarederror",  # Regression for probability
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
        )

        # Train model
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        logger.info("✅ Training complete!")

        # Evaluate model
        metrics = self.evaluate(X_test, y_test)

        # Feature importance
        self._log_feature_importance()

        return metrics

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dict with evaluation metrics
        """
        logger.info("\n📈 Evaluating model...")

        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred = np.clip(y_pred, 0, 1)  # Clip to [0, 1] range

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)

        # Convert to binary for AUC (threshold at 0.5)
        y_binary = (y_test > 0.5).astype(int)
        y_pred_binary = (y_pred > 0.5).astype(int)
        auc = roc_auc_score(y_binary, y_pred)

        # Log loss (calibration metric)
        logloss = log_loss(y_binary, y_pred)

        metrics = {
            "mae": mae,
            "auc": auc,
            "log_loss": logloss,
        }

        logger.info(f"   Mean Absolute Error: {mae:.4f}")
        logger.info(f"   AUC-ROC: {auc:.4f}")
        logger.info(f"   Log Loss: {logloss:.4f}")

        return metrics

    def _log_feature_importance(self):
        """Log feature importance."""
        if self.model is None:
            return

        importance = self.model.feature_importances_
        feature_importance = sorted(zip(self.feature_names, importance), key=lambda x: x[1], reverse=True)

        logger.info("\n🔝 Top 10 Most Important Features:")
        for i, (feature, score) in enumerate(feature_importance[:10], 1):
            logger.info(f"   {i}. {feature}: {score:.4f}")

    def predict(self, game_state: Dict[str, Any]) -> float:
        """
        Predict win probability for a single game state.

        Args:
            game_state: Dict with game state features (from Sportradar)

        Expected keys:
        - score_differential (or home_score/away_score)
        - time_remaining (seconds)
        - yardline (0-100)
        - down (1-4)
        - distance (yards to go)
        - quarter (1-4, 5=OT)
        - posteam_timeouts (optional)
        - defteam_timeouts (optional)

        Returns:
            Win probability (0-1) for possession team
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Convert game state to feature DataFrame
        features_df = self._game_state_to_features(game_state)

        # Predict
        wp = self.model.predict(features_df)[0]
        wp = float(np.clip(wp, 0, 1))

        return wp

    def _game_state_to_features(self, game_state: Dict[str, Any]) -> pd.DataFrame:
        """
        Convert raw game state to model features.

        Args:
            game_state: Dict with game state from Sportradar

        Returns:
            DataFrame with model features
        """
        # Map Sportradar fields to nflverse format
        features = {
            "score_differential": game_state.get("score_diff", 0),
            "half_seconds_remaining": game_state.get("time_remaining", 1800),
            "yardline_100": game_state.get("yardline", 50),
            "down": game_state.get("down", 1),
            "ydstogo": game_state.get("distance", 10),
            "qtr": game_state.get("quarter", 1),
            "posteam_timeouts_remaining": game_state.get("posteam_timeouts", 3),
            "defteam_timeouts_remaining": game_state.get("defteam_timeouts", 3),
        }

        # Create DataFrame
        df = pd.DataFrame([features])

        # Engineer features
        df = self.prepare_features(df)

        return df

    def save_model(self, path: str):
        """Save trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save")

        model_data = {"model": self.model, "feature_names": self.feature_names}

        Path(path).parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"💾 Saved model to {path}")

    def load_model(self, path: str):
        """Load trained model from disk."""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]

        logger.info(f"📂 Loaded model from {path}")
        logger.info(f"   Features: {len(self.feature_names)}")


# CLI for training
def main():
    """Train and save win probability model."""
    import argparse

    parser = argparse.ArgumentParser(description="Train win probability model")
    parser.add_argument("--data", type=str, default="data/nflverse/wp_features.parquet", help="Training data path")
    parser.add_argument("--output", type=str, default="models/win_probability_model.pkl", help="Output model path")
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.1, help="Learning rate")

    args = parser.parse_args()

    # Train model
    model = WinProbabilityModel()
    metrics = model.train(
        data_path=args.data,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
    )

    # Save model
    model.save_model(args.output)

    print("\n" + "=" * 60)
    print("✅ Model training complete!")
    print("=" * 60)
    print(f"Model saved to: {args.output}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Log Loss: {metrics['log_loss']:.4f}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
