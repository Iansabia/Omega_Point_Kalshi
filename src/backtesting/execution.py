"""
Execution handlers for backtesting - simulates order execution with realistic fill models.
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from .backtest_engine import Event, FillEvent, OrderEvent

logger = logging.getLogger(__name__)


class ExecutionHandler:
    """
    Base class for execution handlers.

    Simulates order execution with slippage and commission.
    """

    def __init__(self, events_queue, data_handler, commission_model="fixed"):
        self.events = events_queue
        self.data_handler = data_handler
        self.commission_model = commission_model

    def execute_order(self, event: OrderEvent):
        """
        Execute an order and generate a FillEvent.
        """
        if isinstance(event, OrderEvent):
            # Get current market price
            fill_price = self._calculate_fill_price(event)

            if fill_price is None:
                logger.warning(f"Could not execute order for {event.symbol} - no price data")
                return

            # Calculate commission
            commission = self._calculate_commission(event.quantity, fill_price)

            # Calculate total cost
            if event.direction == "BUY":
                cost = fill_price * event.quantity + commission
            else:  # SELL
                cost = fill_price * event.quantity - commission

            # Generate FillEvent
            fill = FillEvent(
                timestamp=datetime.now(),
                symbol=event.symbol,
                exchange="SIMULATED",
                quantity=event.quantity,
                direction=event.direction,
                fill_price=fill_price,
                commission=commission,
                cost=cost,
            )

            self.events.put(fill)

    def _calculate_fill_price(self, event: OrderEvent) -> Optional[float]:
        """
        Calculate fill price with slippage model.
        """
        # Get latest bar for the symbol
        bar = self.data_handler.get_latest_bar_value(event.symbol, "close")

        if bar is None:
            return None

        # Apply slippage model
        if event.order_type == "MKT":
            # Market orders: add slippage
            slippage_bps = 5  # 5 basis points
            slippage_factor = 1 + (slippage_bps / 10000)

            if event.direction == "BUY":
                fill_price = bar * slippage_factor
            else:  # SELL
                fill_price = bar / slippage_factor

            return fill_price

        elif event.order_type == "LMT":
            # Limit orders: use limit price if available
            if event.price is not None:
                # Simple model: execute if market price better than limit
                if event.direction == "BUY" and bar <= event.price:
                    return event.price
                elif event.direction == "SELL" and bar >= event.price:
                    return event.price
                else:
                    # Limit not hit, no fill
                    return None
            else:
                return bar

        return bar

    def _calculate_commission(self, quantity: int, price: float) -> float:
        """
        Calculate commission based on model.
        """
        if self.commission_model == "fixed":
            # Fixed commission per trade
            return 1.0

        elif self.commission_model == "percentage":
            # Percentage of trade value
            commission_rate = 0.001  # 10 bps
            return quantity * price * commission_rate

        elif self.commission_model == "tiered":
            # Tiered based on trade size
            trade_value = quantity * price
            if trade_value < 1000:
                return 5.0
            elif trade_value < 10000:
                return 10.0
            else:
                return 20.0

        return 0.0


class SimulatedExecutionHandler(ExecutionHandler):
    """
    Simulated execution with realistic slippage and partial fills.
    """

    def __init__(self, events_queue, data_handler, slippage_model="constant"):
        super().__init__(events_queue, data_handler, commission_model="percentage")
        self.slippage_model = slippage_model

    def _calculate_fill_price(self, event: OrderEvent) -> Optional[float]:
        """
        Calculate fill price with more realistic slippage model.
        """
        bar = self.data_handler.get_latest_bar_value(event.symbol, "close")

        if bar is None:
            return None

        # Apply slippage based on model
        if self.slippage_model == "constant":
            # Fixed slippage
            slippage_bps = 5
            slippage_factor = 1 + (slippage_bps / 10000)

            if event.direction == "BUY":
                return bar * slippage_factor
            else:
                return bar / slippage_factor

        elif self.slippage_model == "volume_dependent":
            # Slippage increases with order size
            volume = self.data_handler.get_latest_bar_value(event.symbol, "volume")

            if volume is None or volume == 0:
                volume = 10000  # Default

            participation_rate = event.quantity / volume
            slippage_bps = min(50, 5 + participation_rate * 100)  # Cap at 50 bps

            slippage_factor = 1 + (slippage_bps / 10000)

            if event.direction == "BUY":
                return bar * slippage_factor
            else:
                return bar / slippage_factor

        elif self.slippage_model == "random":
            # Random slippage with normal distribution
            slippage_bps = np.random.normal(5, 2)  # Mean 5 bps, std 2 bps
            slippage_bps = max(0, slippage_bps)  # No negative slippage

            slippage_factor = 1 + (slippage_bps / 10000)

            if event.direction == "BUY":
                return bar * slippage_factor
            else:
                return bar / slippage_factor

        return bar


class PredictionMarketExecutionHandler(ExecutionHandler):
    """
    Execution handler specific to prediction markets.

    Accounts for binary outcome structure and typical bid-ask spreads.
    """

    def __init__(self, events_queue, data_handler, spread_bps: float = 50):
        super().__init__(events_queue, data_handler, commission_model="percentage")
        self.spread_bps = spread_bps

    def _calculate_fill_price(self, event: OrderEvent) -> Optional[float]:
        """
        Calculate fill price accounting for bid-ask spread.
        """
        mid_price = self.data_handler.get_latest_bar_value(event.symbol, "close")

        if mid_price is None:
            return None

        # Calculate half-spread
        half_spread = (self.spread_bps / 10000) / 2

        # Market taker pays the spread
        if event.direction == "BUY":
            fill_price = mid_price * (1 + half_spread)
        else:  # SELL
            fill_price = mid_price * (1 - half_spread)

        # Ensure price stays in [0, 1] range for prediction markets
        fill_price = max(0.01, min(0.99, fill_price))

        return fill_price

    def _calculate_commission(self, quantity: int, price: float) -> float:
        """
        Commission model for prediction markets.

        Typical: maker rebate, taker fee model.
        """
        # Assume all orders are market orders (takers)
        taker_fee_bps = 5  # 5 bps taker fee
        return quantity * price * (taker_fee_bps / 10000)
