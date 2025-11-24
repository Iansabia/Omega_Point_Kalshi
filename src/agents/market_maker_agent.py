import random
import time
import uuid
from typing import List, Tuple

import numpy as np

from src.agents.base_agent import BaseTrader
from src.orderbook.order import Order, OrderType


class MarketMakerAgent(BaseTrader):
    """
    Market Maker using Avellaneda-Stoikov framework.
    """

    def __init__(
        self, model, initial_wealth: float = 100000.0, target_inventory: float = 0.0, risk_param: float = 0.1, risk_limits=None
    ):
        super().__init__(model, initial_wealth=initial_wealth, risk_limits=risk_limits)
        self.risk_param = risk_param
        self.target_inventory = target_inventory
        self.inventory = 0.0  # Tracks current inventory
        self.half_spread = 0.02  # Base spread

    def observe_market(self):
        """Observe market and update inventory."""
        self.inventory = self.position

        return {
            "price": self.model.current_price,
            "inventory": self.inventory,
            "target_inventory": self.target_inventory,
            "deviation": self.inventory - self.target_inventory,
            "bid": self.model.order_book.get_best_bid() if hasattr(self.model, "order_book") else None,
            "ask": self.model.order_book.get_best_ask() if hasattr(self.model, "order_book") else None,
        }

    def estimate_mid_price(self) -> float:
        # Use model's current price or last trade
        return self.model.current_price

    def quote_prices(self) -> Tuple[float, float]:
        """
        Calculate bid and ask prices based on inventory skew.

        Avellaneda-Stoikov framework:
        - If we're long (inventory > target), we want to:
          * Lower ask to attract buyers (sell our excess)
          * Lower bid to discourage more buying
        - If we're short (inventory < target), we want to:
          * Raise bid to attract sellers (buy to cover)
          * Raise ask to discourage selling

        The skew is SUBTRACTED from quotes when we're long (positive inventory),
        which lowers both bid and ask, encouraging us to sell.
        """
        mid = self.estimate_mid_price()

        # Calculate inventory deviation from target
        inventory_deviation = self.inventory - self.target_inventory

        # Inventory skew: positive when long, negative when short
        # We subtract this from quotes, so when long, quotes drop (encouraging sells)
        skew = self.risk_param * inventory_deviation

        # Apply Avellaneda-Stoikov pricing
        # When long (skew > 0): both prices drop to encourage selling
        # When short (skew < 0): both prices rise to encourage buying
        bid = mid - self.half_spread - skew
        ask = mid + self.half_spread - skew

        # Ensure prices stay within valid range [0, 1] for prediction markets
        bid = max(0.01, min(0.99, bid))
        ask = max(0.01, min(0.99, ask))

        # Ensure bid < ask
        if bid >= ask:
            spread = self.half_spread * 2
            bid = mid - spread / 2
            ask = mid + spread / 2

        return bid, ask

    def make_decision(self):
        """
        Place two-sided quotes.
        """
        bid_price, ask_price = self.quote_prices()
        size = 10  # Standard quote size

        # Cancel previous orders?
        # In this simple model, we just place new ones.
        # Real MM would cancel/replace.

        self._place_limit_order("BUY", bid_price, size)
        self._place_limit_order("SELL", ask_price, size)

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
