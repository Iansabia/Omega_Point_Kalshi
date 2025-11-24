import random
import numpy as np
from typing import List
from src.agents.base_agent import BaseTrader
from src.orderbook.order import Order, OrderType
import time
import uuid

class NoiseTrader(BaseTrader):
    """
    Noise trader with multiple strategies: Random, Contrarian, TrendFollower.
    """

    def __init__(self, unique_id: int, model, strategy: str = "random", initial_wealth: float = 1000.0):
        super().__init__(unique_id, model, initial_wealth=initial_wealth)
        self.strategy = strategy
        self.recency_weight = 0.7
        self.trade_probability = 0.1

        # Strategy specific params
        self.contrarian_threshold = 0.02
        self.trend_windows = (10, 30)
        self.price_history = []

    def observe_market(self):
        """Read current market price."""
        current_price = self.model.current_price
        self.price_history.append(current_price)
        # Keep history manageable
        if len(self.price_history) > 100:
            self.price_history.pop(0)

    def make_decision(self):
        """Generate trading signal based on strategy."""
        if self.strategy == "random":
            self._random_walk_strategy()
        elif self.strategy == "contrarian":
            self._contrarian_strategy()
        elif self.strategy == "trend":
            self._trend_following_strategy()

    def _random_walk_strategy(self):
        """Random walk with 10% trade probability."""
        if random.random() < self.trade_probability:
            side = 'BUY' if random.random() < 0.5 else 'SELL'
            self._place_market_order(side)

    def _contrarian_strategy(self):
        """Trade against recent returns."""
        if len(self.price_history) < 2:
            return
            
        ret = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2] if self.price_history[-2] > 0 else 0
        
        if ret > self.contrarian_threshold:
            self._place_market_order('SELL')
        elif ret < -self.contrarian_threshold:
            self._place_market_order('BUY')

    def _trend_following_strategy(self):
        """Moving average crossover."""
        short_window, long_window = self.trend_windows
        if len(self.price_history) < long_window:
            return
            
        short_ma = np.mean(self.price_history[-short_window:])
        long_ma = np.mean(self.price_history[-long_window:])
        
        if short_ma > long_ma:
            self._place_market_order('BUY')
        elif short_ma < long_ma:
            self._place_market_order('SELL')

    def _place_market_order(self, side: str):
        """Helper to place a market order."""
        # Overconfidence: Trade size multiplier 1.2-1.5
        size_multiplier = random.uniform(1.2, 1.5)
        quantity = 10 * size_multiplier # Base size 10

        order = Order(
            order_id=str(uuid.uuid4()),
            side=side,
            price=1.0 if side == 'BUY' else 0.0, # Market order: willing to pay any price
            quantity=quantity,
            timestamp=time.time(),
            trader_id=self.trader_id,  # Use trader_id property from base class
            order_type=OrderType.MARKET
        )
        self.submit_orders([order])
        order = Order(
            order_id=str(uuid.uuid4()),
            side=side,
            price=1.0 if side == "BUY" else 0.0,  # Market order: willing to pay any price
            quantity=quantity,
            timestamp=time.time(),
            trader_id=self.trader_id,  # Use trader_id property from base class
            order_type=OrderType.MARKET,
        )
        self.submit_orders([order])
