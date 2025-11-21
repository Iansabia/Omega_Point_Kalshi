import random
import numpy as np
from typing import List
from src.agents.base_agent import BaseTrader
from src.orderbook.order import Order, OrderType
import time
import uuid

class HomerAgent(BaseTrader):
    """
    Homer Agent (Loyalty Bias).
    Overweights specific outcomes due to loyalty.
    """
    
    def __init__(self, unique_id: int, model, initial_wealth: float = 2000.0, loyalty_asset: str = "YES", loyalty_strength: float = 0.7):
        super().__init__(unique_id, model, initial_wealth=initial_wealth)
        self.loyalty_strength = loyalty_strength # [0.5, 0.9]
        self.loyalty_asset = loyalty_asset

    def observe_market(self):
        pass

    def update_loyalty(self, positive_outcome: bool):
        """
        Update loyalty strength based on outcomes.
        """
        # Decay
        self.loyalty_strength *= 0.99
        
        # Reinforcement
        if positive_outcome:
            self.loyalty_strength *= 1.05
            
        self.loyalty_strength = min(0.99, max(0.1, self.loyalty_strength))

    def make_decision(self):
        """
        Buy the loyal asset regardless of price (within reason).
        """
        market_price = self.model.current_price
        
        # Perceived value is higher
        perceived_value = market_price * (1 + (self.loyalty_strength - 0.5))
        
        if perceived_value > market_price:
            # Buy!
            size = 20 * self.loyalty_strength
            self._place_limit_order('BUY', market_price * 1.05, size) # Aggressive limit

    def _place_limit_order(self, side: str, price: float, quantity: float):
        order = Order(
            order_id=str(uuid.uuid4()),
            side=side,
            price=price,
            quantity=quantity,
            timestamp=time.time(),
            trader_id=self.trader_id,
            order_type=OrderType.LIMIT
        )
        self.submit_orders([order])
