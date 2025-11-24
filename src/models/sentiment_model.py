import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class SentimentModel:
    """
    Sentiment Quantification System integrating FinBERT, VADER, and market microstructure signals.
    """

    # Parameters for Panic Coefficient
    ALPHA = 3.5
    BETA = 1.2
    GAMMA = -2.0

    def __init__(self):
        self._finbert = None
        self._vader = None
        self.csad_history = []
        self.market_return_history = []

    def analyze_sentiment_finbert(self, text: str) -> float:
        """
        Integrate FinBERT for sentiment classification.
        Returns a score between -1 (Negative) and 1 (Positive).
        """
        try:
            from transformers import pipeline

            # Lazy loading to avoid overhead if not used
            if not hasattr(self, "_finbert"):
                # Use a smaller distilled model if possible, but standard FinBERT is requested
                self._finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

            result = self._finbert(text)[0]
            # FinBERT returns labels: 'positive', 'negative', 'neutral'
            label = result["label"]
            score = result["score"]

            if label == "positive":
                return score
            elif label == "negative":
                return -score
            else:  # neutral
                return 0.0

        except ImportError:
            print("Transformers library not found. Using VADER fallback.")
            return self.analyze_sentiment_vader(text)
        except Exception as e:
            print(f"FinBERT error: {e}")
            return 0.0

    def analyze_sentiment_vader(self, text: str) -> float:
        """
        Implement VADER lexicon-based backup.
        Returns compound score between -1 and 1.
        """
        try:
            if self._vader is None:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

                self._vader = SentimentIntensityAnalyzer()

            scores = self._vader.polarity_scores(text)
            return scores["compound"]  # Returns value between -1 and 1

        except ImportError:
            logger.warning("vaderSentiment library not found. Using fallback.")
            # Simple fallback dictionary
            positive_words = {"bullish", "up", "growth", "profit", "win", "good", "high", "victory", "strong", "beat"}
            negative_words = {"bearish", "down", "loss", "crash", "lose", "bad", "low", "defeat", "weak", "fail"}

            words = text.lower().split()
            score = 0
            for word in words:
                if word in positive_words:
                    score += 1
                elif word in negative_words:
                    score -= 1

            # Normalize
            if len(words) > 0:
                return max(-1.0, min(1.0, score / len(words) * 5))
            return 0.0

    def aggregate_sentiment(self, n_pos: int, n_neg: int) -> float:
        """
        Create sentiment score aggregation: S = (N_pos - N_neg) / (N_pos + N_neg)
        """
        if n_pos + n_neg == 0:
            return 0.0
        return (n_pos - n_neg) / (n_pos + n_neg)

    def calculate_panic_coefficient(self, csad_t: float, volatility_t: float, sentiment_t: float) -> float:
        """
        Build panic coefficient: Panic_t = exp(α×CSAD_t + β×Volatility_t + γ×Sentiment_t)
        """
        return np.exp(self.ALPHA * csad_t + self.BETA * volatility_t + self.GAMMA * sentiment_t)

    def calculate_csad(self, returns: np.ndarray, market_return: float) -> float:
        """
        Calculate Cross-Sectional Absolute Deviation (CSAD) for a single time step.

        CSAD_t = (1/N) * Σ|R_i,t - R_m,t|

        Args:
            returns: Array of individual agent/asset returns at time t
            market_return: Market return at time t

        Returns:
            CSAD value
        """
        if len(returns) == 0:
            return 0.0

        abs_devs = np.abs(returns - market_return)
        return np.mean(abs_devs)

    def update_herding_history(self, returns: np.ndarray, market_return: float):
        """
        Update history for herding detection regression.

        Args:
            returns: Array of returns for current time step
            market_return: Market return for current time step
        """
        csad = self.calculate_csad(returns, market_return)
        self.csad_history.append(csad)
        self.market_return_history.append(market_return)

        # Keep history manageable (last 100 periods)
        if len(self.csad_history) > 100:
            self.csad_history.pop(0)
            self.market_return_history.pop(0)

    def detect_herding(self, min_periods: int = 30) -> Dict[str, Any]:
        """
        Detect herding behavior using Chang et al. (2000) regression:

        CSAD_t = α + γ₁|R_m,t| + γ₂(R_m,t)² + ε_t

        Herding is indicated when γ₂ < 0 (and statistically significant).

        Args:
            min_periods: Minimum number of periods required for regression

        Returns:
            Dictionary with regression results and herding indicator
        """
        if len(self.csad_history) < min_periods:
            return {
                "gamma1": 0.0,
                "gamma2": 0.0,
                "is_herding": False,
                "message": f"Insufficient data: {len(self.csad_history)}/{min_periods}",
            }

        try:
            from scipy import stats

            # Prepare regression data
            y = np.array(self.csad_history)
            rm = np.array(self.market_return_history)

            # Independent variables: |R_m| and (R_m)²
            x1 = np.abs(rm)
            x2 = rm**2

            # Stack into design matrix [intercept, |R_m|, (R_m)²]
            X = np.column_stack([np.ones(len(y)), x1, x2])

            # OLS regression
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha, gamma1, gamma2 = beta

            # Calculate residuals and statistics
            y_pred = X @ beta
            residuals = y - y_pred
            mse = np.mean(residuals**2)

            # Standard errors (simplified)
            se_gamma2 = np.sqrt(mse / len(y))  # Simplified standard error

            # T-statistic for gamma2
            t_stat = gamma2 / se_gamma2 if se_gamma2 > 0 else 0

            # Herding detected if gamma2 < 0 and statistically significant (t < -1.96 for 5%)
            is_herding = gamma2 < 0 and t_stat < -1.96

            return {
                "alpha": alpha,
                "gamma1": gamma1,
                "gamma2": gamma2,
                "t_statistic": t_stat,
                "is_herding": is_herding,
                "r_squared": 1 - (np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)),
                "n_periods": len(y),
            }

        except Exception as e:
            logger.error(f"Herding detection error: {e}")
            return {"gamma1": 0.0, "gamma2": 0.0, "is_herding": False, "error": str(e)}
