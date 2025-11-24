import logging
from abc import abstractmethod
from typing import List, Optional

import mesa
import numpy as np

from src.orderbook.order import Order, OrderType
from src.risk.risk_manager import RiskLimits, RiskManager

logger = logging.getLogger(__name__)


class BaseTrader(mesa.Agent):
    """
    Abstract base class for all trader agents.

    Includes integrated risk management with:
    - Position limits
    - Trade frequency controls
    - Stop losses
    - Kelly Criterion position sizing
    """

    def __init__(self, model: mesa.Model, initial_wealth: float = 1000.0, risk_limits: RiskLimits = None):
        # Mesa 2.x requires unique_id, Mesa 3.3+ auto-assigns it
        # Generate unique_id based on the next agent ID
        unique_id = model.next_id() if hasattr(model, 'next_id') else len(model.schedule.agents)
        super().__init__(unique_id, model)

        self.initial_wealth = initial_wealth
        self.wealth = initial_wealth
        self.position = 0.0  # Net position in contracts
        self.orders: List[Order] = []
        self.trade_history: List[dict] = []

        # Risk management
        self.risk_manager = RiskManager(risk_limits or RiskLimits())
        self.risk_manager.update_capital(initial_wealth)

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

    def can_trade(self, ticker: str, edge: float) -> bool:
        """
        Check if trade is allowed by risk manager.

        Args:
            ticker: Market ticker
            edge: Estimated edge (expected value %)

        Returns:
            True if trade allowed
        """
        allowed, reason = self.risk_manager.can_trade(ticker, edge)
        if not allowed:
            logger.debug(f"Agent {self.unique_id} trade blocked: {reason}")
        return allowed

    def calculate_position_size(self, edge: float, win_prob: float, avg_win: float = 50.0, avg_loss: float = 50.0) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            edge: Expected value as fraction
            win_prob: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount

        Returns:
            Position size in dollars
        """
        return self.risk_manager.calculate_position_size(
            edge=edge, win_prob=win_prob, avg_win=avg_win, avg_loss=avg_loss, available_capital=self.wealth
        )

    def submit_orders(self, orders: List[Order]):
        """Place orders in order book with risk checks."""
        for order in orders:
            # Basic validation against wealth/risk
            cost = order.price * order.quantity if order.price else self.wealth * 0.1
            if order.side == "BUY" and cost > self.wealth:
                logger.debug(f"Agent {self.unique_id} order rejected: insufficient funds")
                continue  # Insufficient funds

            # Risk management check
            # Estimate edge (simplified - agents should override this)
            edge = 0.05  # Default 5% edge assumption
            ticker = getattr(self.model, "current_ticker", "MARKET")  # Use model's current market
            if not self.can_trade(ticker, edge):
                continue

            # In a real implementation, we'd lock funds here

            self.model.matching_engine.match_order(order)
            self.orders.append(order)

            # Record trade with risk manager
            self.risk_manager.record_trade(
                ticker=ticker,
                side=order.side,
                quantity=order.quantity,
                price=order.price if order.price else 0.5,  # Default price for market orders
                is_entry=True,
            )

    def execute_trade(self, side: str, quantity: float, price: float):
        """
        Update portfolio after a trade execution.

        Args:
            side: 'BUY' or 'SELL'
            quantity: Number of units
            price: Price per unit

        Raises:
            ValueError: If insufficient wealth or position
        """
        cost = quantity * price

        if side == "BUY":
            # Check if enough wealth to buy
            if cost > self.wealth:
                raise ValueError(f"Insufficient wealth: need {cost:.2f}, have {self.wealth:.2f}")
            self.wealth -= cost
            self.position += quantity
        else:
            # Check if enough position to sell
            if quantity > self.position:
                raise ValueError(f"Insufficient position: need {quantity:.2f}, have {self.position:.2f}")
            self.wealth += cost
            self.position -= quantity

        # Update risk manager capital
        self.risk_manager.update_capital(self.wealth)

        trade_record = {
            "side": side,
            "quantity": quantity,
            "price": price,
            "step": getattr(self.model, "step_count", 0),
            "timestamp": getattr(self.model, "step_count", 0),  # Mesa 3.3+ uses step_count directly
        }
        self.trade_history.append(trade_record)

        logger.debug(
            f"Agent {self.unique_id} executed {side} {quantity:.2f} @ {price:.4f}, "
            f"wealth={self.wealth:.2f}, position={self.position:.2f}"
        )

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
