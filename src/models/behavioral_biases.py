import random
from typing import Any, Dict

import numpy as np


class BehavioralBiases:
    """
    Implementation of behavioral biases for agents: Recency, Homer (Loyalty), Gambler's Fallacy, Herding.
    """

    def __init__(self):
        self.recency_weight = 0.7
        self.herding_coefficient = 0.2

    def apply_recency_bias(self, historical_returns: list) -> float:
        """
        Implement recency bias: w = 0.7 (overweight recent vs optimal 0.3)
        Returns a weighted average return.
        """
        if not historical_returns:
            return 0.0

        # Simple exponential decay or just overweighting the last item?
        # Checklist says "w = 0.7". Let's assume it means 70% weight on most recent, 30% on rest.
        if len(historical_returns) == 1:
            return historical_returns[0]

        recent = historical_returns[-1]
        past_avg = np.mean(historical_returns[:-1])

        return self.recency_weight * recent + (1 - self.recency_weight) * past_avg

    def calculate_loyalty_adjustment(
        self, fundamental_value: float, loyalty_strength: float, is_preferred_outcome: bool
    ) -> float:
        """
        Add homer bias: loyalty_strength ∈ [0.5, 0.9]
        Adjusts perceived value based on loyalty.
        """
        if is_preferred_outcome:
            # Overvalue the preferred outcome
            return fundamental_value * (1 + (loyalty_strength - 0.5))  # Scale 0.5-0.9 to 0.0-0.4 boost?
            # Or just direct multiplier? Let's use a simple multiplier logic.
            # If loyalty is high (0.9), they perceive it as much more valuable.
            return fundamental_value * (1 + loyalty_strength * 0.2)
        else:
            # Undervalue the opponent? Or just standard?
            # Usually homer bias is about overestimating your team.
            return fundamental_value

    def detect_gamblers_fallacy(self, recent_outcomes: list, target_outcome: str) -> bool:
        """
        Build gambler's fallacy detector.
        If a streak of one outcome has occurred, they expect the other.
        """
        if len(recent_outcomes) < 3:
            return False

        # Check for streak
        streak = True
        last = recent_outcomes[-1]
        for outcome in recent_outcomes[-3:]:
            if outcome != last:
                streak = False
                break

        # If streak of 'HEADS', gambler expects 'TAILS'
        if streak and last != target_outcome:
            return True  # They expect a reversal
        return False

    def calculate_perceived_value(self, fundamental_value: float, sentiment_effect: float, herding_effect: float) -> float:
        """
        Create sentiment-driven adjustment: V_perceived = V_fundamental + sentiment_effect + herding
        """
        return fundamental_value + sentiment_effect + (herding_effect * self.herding_coefficient)
