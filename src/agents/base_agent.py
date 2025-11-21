import mesa
from abc import abstractmethod
from typing import List, Optional
from src.orderbook.order import Order, OrderType
import logging

logger = logging.getLogger(__name__)

class BaseTrader(mesa.Agent):
    """
    Abstract base class for all trader agents.
    """

    def __init__(self, unique_id: int, model: mesa.Model, initial_wealth: float = 1000.0):
        super().__init__(unique_id, model)
        self.initial_wealth = initial_wealth
        self.wealth = initial_wealth
        self.position = 0.0  # Net position in contracts
        self.orders: List[Order] = []
        self.trade_history: List[dict] = []

    @property
    def trader_id(self) -> str:
        """Generate trader ID for orders."""
        return f"agent_{self.unique_id}"

    @abstractmethod
    def observe_market(self):
        """Read current market state."""
        pass

    @abstractmethod
    def make_decision(self):
        """Generate trading signal."""
        pass

    def submit_orders(self, orders: List[Order]):
        """Place orders in order book."""
        for order in orders:
            # Basic validation against wealth/risk
            cost = order.price * order.quantity
            if order.side == 'BUY' and cost > self.wealth:
                continue # Insufficient funds
            
            # In a real implementation, we'd lock funds here
            
            self.model.matching_engine.match_order(order)
            self.orders.append(order)

    def execute_trade(self, quantity: float, price: float, side: str):
        """Update portfolio after a trade execution."""
        cost = quantity * price

        if side == 'BUY':
            self.wealth -= cost
            self.position += quantity
        else:
            self.wealth += cost
            self.position -= quantity

        trade_record = {
            "side": side,
            "quantity": quantity,
            "price": price,
            "step": getattr(self.model, 'step_count', 0),
            "timestamp": getattr(self.model, 'schedule', None) and getattr(self.model.schedule, 'steps', 0)
        }
        self.trade_history.append(trade_record)

        logger.debug(f"Agent {self.unique_id} executed {side} {quantity:.2f} @ {price:.4f}, "
                    f"wealth={self.wealth:.2f}, position={self.position:.2f}")

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        return self.wealth + (self.position * current_price)

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate profit and loss."""
        return self.get_portfolio_value(current_price) - self.initial_wealth

    def step(self):
        """
        Mesa step function.
        """
        self.observe_market()
        self.make_decision()
