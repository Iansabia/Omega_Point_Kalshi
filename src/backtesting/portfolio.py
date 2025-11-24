"""
Portfolio management for backtesting.

Tracks positions, cash, and generates orders from signals.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .backtest_engine import Event, FillEvent, OrderEvent, SignalEvent

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Manages portfolio state, positions, and P&L during backtesting.
    """

    def __init__(self, events_queue, data_handler, start_date: datetime, initial_capital: float = 100000.0):
        self.events = events_queue
        self.data_handler = data_handler
        self.symbol_list = data_handler.symbol_list
        self.start_date = start_date
        self.initial_capital = initial_capital

        # Track positions
        self.positions = {s: 0 for s in self.symbol_list}
        self.holdings = {s: 0.0 for s in self.symbol_list}

        # Portfolio history
        self.all_positions = []
        self.all_holdings = []
        self.current_positions = {s: 0 for s in self.symbol_list}
        self.current_holdings = self._construct_current_holdings()

        # Performance tracking
        self.equity_curve = []

    def _construct_current_holdings(self) -> Dict:
        """
        Initialize holdings dictionary with cash and total.
        """
        holdings = {s: 0.0 for s in self.symbol_list}
        holdings["datetime"] = self.start_date
        holdings["cash"] = self.initial_capital
        holdings["commission"] = 0.0
        holdings["total"] = self.initial_capital
        return holdings

    def update_timeindex(self, event: Event):
        """
        Update portfolio based on market data (mark-to-market).
        """
        bars = {}
        for symbol in self.symbol_list:
            bars[symbol] = self.data_handler.get_latest_bars(symbol, N=1)

        # Update positions
        self.current_positions["datetime"] = event.timestamp

        # Update holdings
        self.current_holdings["datetime"] = event.timestamp

        # Mark-to-market
        for symbol in self.symbol_list:
            if bars[symbol] is not None and len(bars[symbol]) > 0:
                try:
                    market_price = bars[symbol].iloc[-1]["close"]
                    self.current_holdings[symbol] = self.current_positions[symbol] * market_price
                except (KeyError, IndexError):
                    self.current_holdings[symbol] = 0

        # Update total equity
        self.current_holdings["total"] = self.current_holdings["cash"] + sum(
            [self.current_holdings[s] for s in self.symbol_list]
        )

        # Append to history
        self.all_positions.append(dict(self.current_positions))
        self.all_holdings.append(dict(self.current_holdings))

    def update_signal(self, event: SignalEvent):
        """
        Convert signal to order and place in queue.
        """
        # Simple position sizing: fixed size or percentage of equity
        order_quantity = self._calculate_order_quantity(event)

        if order_quantity != 0:
            order_type = "MKT"  # Use market orders for simplicity
            direction = "BUY" if event.signal_type == "LONG" else "SELL"

            order = OrderEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                order_type=order_type,
                quantity=abs(order_quantity),
                direction=direction,
                price=event.target_price,
            )

            self.events.put(order)

    def _calculate_order_quantity(self, event: SignalEvent) -> int:
        """
        Calculate order size based on signal and current position.

        Simple implementation: fixed size or percentage of equity.
        """
        if event.signal_type == "LONG":
            # Enter or add to long position
            target_quantity = 100  # Fixed size for now
            current_quantity = self.current_positions[event.symbol]
            return target_quantity - current_quantity

        elif event.signal_type == "SHORT":
            # Enter or add to short position
            target_quantity = -100
            current_quantity = self.current_positions[event.symbol]
            return target_quantity - current_quantity

        elif event.signal_type == "EXIT":
            # Close position
            return -self.current_positions[event.symbol]

        return 0

    def update_fill(self, event: FillEvent):
        """
        Update portfolio based on filled order.
        """
        # Update positions
        if event.direction == "BUY":
            self.current_positions[event.symbol] += event.quantity
        elif event.direction == "SELL":
            self.current_positions[event.symbol] -= event.quantity

        # Update holdings
        fill_cost = event.cost
        self.current_holdings[event.symbol] = self.current_positions[event.symbol] * event.fill_price
        self.current_holdings["cash"] -= fill_cost
        self.current_holdings["commission"] += event.commission
        self.current_holdings["total"] = self.current_holdings["cash"] + sum(
            [self.current_holdings[s] for s in self.symbol_list]
        )

    def create_equity_curve(self) -> pd.DataFrame:
        """
        Create equity curve DataFrame from holdings history.
        """
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index("datetime", inplace=True)
        curve["returns"] = curve["total"].pct_change()
        curve["equity_curve"] = (1.0 + curve["returns"]).cumprod()
        return curve

    def get_current_positions(self) -> Dict:
        """Return current positions."""
        return self.current_positions

    def get_current_holdings(self) -> Dict:
        """Return current holdings."""
        return self.current_holdings
