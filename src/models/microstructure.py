import numpy as np
from typing import Dict, Any

class MicrostructureModel:
    """
    Market Microstructure Models for price impact, spread calculation, and liquidity.
    """
    
    # Parameters from research
    KYLE_LAMBDA = 1.5  # Prediction markets tend to be less liquid
    ETA = 0.314
    GAMMA = 0.142

    def __init__(self):
        pass

    def calculate_kyle_lambda(self, sigma_v: float, sigma_u: float) -> float:
        """
        Implement Kyle's Lambda: λ = 0.5 × √(Σ_v/Σ_u)
        sigma_v: Volatility of fundamental value
        sigma_u: Volatility of noise trading
        """
        if sigma_u == 0:
            return self.KYLE_LAMBDA # Fallback
        return 0.5 * np.sqrt(sigma_v / sigma_u)

    def calculate_spread(self, order_processing_cost: float, inventory_cost: float, adverse_selection_cost: float) -> float:
        """
        Calculate bid-ask spreads: Spread = Order_Processing + Inventory + Adverse_Selection
        """
        return order_processing_cost + inventory_cost + adverse_selection_cost

    def calculate_price_impact_sqrt(self, quantity: float, lambda_param: float = None) -> float:
        """
        Implement price impact model: ΔP = λ × √Q (square-root law)
        """
        lam = lambda_param if lambda_param else self.KYLE_LAMBDA
        return lam * np.sqrt(abs(quantity))

    def calculate_almgren_chriss_impact(self, quantity: float, daily_volume: float, volatility: float) -> float:
        """
        Add Almgren-Chriss market impact: Impact = η × σ × (Q/V)^γ
        """
        if daily_volume == 0:
            return 0.0
        return self.ETA * volatility * (abs(quantity) / daily_volume) ** self.GAMMA
