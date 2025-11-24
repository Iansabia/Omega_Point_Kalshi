from typing import Any, Dict

import numpy as np


class TransactionCostModel:
    """
    Estimates transaction costs including fees, slippage, and market impact.
    """

    def __init__(self, fee_rate: float = 0.0):
        # Kalshi often has low/no fees for makers, but takers might pay.
        # Using 0.0 as default for now, configurable.
        self.fee_rate = fee_rate

    def estimate_cost(self, order_size: int, price: float, is_maker: bool = False) -> float:
        """
        Calculate total estimated cost (fees).
        """
        notional = order_size * (price / 100.0)
        fees = notional * self.fee_rate
        return fees

    def estimate_market_impact(self, order_size: int, daily_volume: float, volatility: float) -> float:
        """
        Estimate price impact of a trade.
        Impact = eta * sigma * (size / volume)^gamma
        """
        if daily_volume <= 0:
            return 0.0

        eta = 0.314
        gamma = 0.142
        psi = order_size / daily_volume
        impact = eta * volatility * (psi**gamma)

        # Returns expected price move in percentage terms
        return impact

    def simulate_slippage(self, order_type: str, price: float, market_data: Dict[str, Any], is_buy: bool) -> float:
        """
        Simulate execution price with slippage.
        """
        if order_type == "MARKET":
            # Simple slippage model: 5bps
            slippage = 0.0005
            if is_buy:
                return price * (1 + slippage)
            else:
                return price * (1 - slippage)

        # LIMIT orders don't slip on price, but might not fill (not modeled here)
        return price
