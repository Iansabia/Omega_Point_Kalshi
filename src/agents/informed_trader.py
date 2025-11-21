import random
import numpy as np
from typing import List
from src.agents.base_agent import BaseTrader
from src.orderbook.order import Order, OrderType
import time
import uuid

class InformedTrader(BaseTrader):
    """
    Informed trader who receives a noisy signal of the true value.
    """
    
    def __init__(self, unique_id: int, model, initial_wealth: float = 10000.0, information_quality: float = 0.8):
        super().__init__(unique_id, model, initial_wealth=initial_wealth)
        self.information_quality = information_quality # [0.5, 1.0]
        self.true_value = 0.5 # This should come from model or external source

    def observe_market(self):
        pass

    def acquire_information(self) -> float:
        """
        Generate signal: signal = true_value + N(0, 1-quality)
        """
        noise = np.random.normal(0, 1 - self.information_quality)
        signal = self.true_value + noise
        return max(0.0, min(1.0, signal)) # Clamp to [0, 1]

    def make_decision(self):
        """
        Trading logic based on signal vs market price.
        """
        signal = self.acquire_information()
        market_price = self.model.current_price
        
        # Strategic trading: spread orders over time?
        # For now, simple threshold logic
        
        size = 50 # Base size
        
        if signal > market_price * 1.02:
            self._place_limit_order('BUY', market_price * 1.01, size)
        elif signal < market_price * 0.98:
            self._place_limit_order('SELL', market_price * 0.99, size)

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
