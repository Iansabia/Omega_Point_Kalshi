import random
import time
import uuid
from typing import List

import numpy as np

from src.agents.base_agent import BaseTrader
from src.orderbook.order import Order, OrderType


class Arbitrageur(BaseTrader):
    """
    Arbitrageur who exploits price discrepancies between market price and fundamental value.
    """

    def __init__(self, model, initial_wealth: float = 50000.0, detection_speed: float = 0.8):
        super().__init__(model, initial_wealth=initial_wealth)
        self.detection_speed = detection_speed  # [0.7, 1.0]
        self.min_spread = 0.02
        self.true_value = 0.5  # Should be shared/global

    def observe_market(self):
        """Observe market for arbitrage opportunities."""
        market_price = self.model.current_price
        spread = abs(market_price - self.true_value)

        return {
            "price": market_price,
            "fundamental_value": self.true_value,
            "spread": spread,
            "arbitrage_opportunity": spread > self.min_spread,
        }

    def detect_arbitrage(self) -> float:
        """
        Detect if spread > min_spread and random() < detection_speed
        Returns the spread size if detected, else 0.
        """
        market_price = self.model.current_price
        spread = abs(market_price - self.true_value)

        if spread > self.min_spread and random.random() < self.detection_speed:
            return spread
        return 0.0

    def make_decision(self):
        """
        Execute strategy to close mispricing.
        """
        spread = self.detect_arbitrage()
        if spread > 0:
            market_price = self.model.current_price
            # Capital constraints and leverage limits would go here

            size = 100  # Aggressive size

            if market_price < self.true_value:
                # Buy undervalued
                self._place_limit_order("BUY", market_price * 1.01, size)
            else:
                # Sell overvalued
                self._place_limit_order("SELL", market_price * 0.99, size)

    def _place_limit_order(self, side: str, price: float, quantity: float):
        order = Order(
            order_id=str(uuid.uuid4()),
            side=side,
            price=price,
            quantity=quantity,
            timestamp=time.time(),
            trader_id=self.trader_id,
            order_type=OrderType.LIMIT,
        )
        self.submit_orders([order])
