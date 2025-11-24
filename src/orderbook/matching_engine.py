import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .order import Order, OrderType
from .orderbook import OrderBook


@dataclass
class Trade:
    buyer_id: str
    seller_id: str
    price: float
    quantity: float
    timestamp: float
    aggressor_side: str  # 'BUY' or 'SELL'


class MatchingEngine:
    def __init__(self, orderbook: OrderBook):
        self.orderbook = orderbook
        self.model = None  # Will be set by the model during initialization
        self.trades: List[Trade] = []  # Track all trades

    def match_order(self, incoming: Order) -> List[Trade]:
        """
        Match an incoming order against the order book.
        Returns a list of Trades executed.
        """
        if incoming.order_type == OrderType.FOK:
            return self._handle_fok(incoming)

        fills = []

        # Select the opposite book
        # If buying, look at asks (min-heap). If selling, look at bids (max-heap).
        book = self.orderbook.asks if incoming.side == "BUY" else self.orderbook.bids

        while book and incoming.remaining > 0:
            # Peek at best price
            # Bids are stored as (-price, ts, order), Asks as (price, ts, order)
            best_price_tuple = book[0]
            best_price = abs(best_price_tuple[0])
            resting_order = best_price_tuple[2]

            if not self._can_match(incoming, best_price):
                break

            # Calculate fill quantity
            fill_qty = min(incoming.remaining, resting_order.remaining)

            # Execute trade
            trade = self._execute_trade(incoming, resting_order, fill_qty, best_price)
            fills.append(trade)

            # Update resting order
            resting_order.remaining -= fill_qty
            if resting_order.remaining <= 1e-9:  # Float tolerance
                heapq.heappop(book)
                # Also remove from order map if needed, though OrderBook.orders might need cleanup
                if resting_order.order_id in self.orderbook.orders:
                    del self.orderbook.orders[resting_order.order_id]

            # Update incoming order
            incoming.remaining -= fill_qty

        # If order is not filled and not IOC/FOK/MARKET, add to book
        if incoming.remaining > 1e-9:
            if incoming.order_type == OrderType.LIMIT:
                self.orderbook.add_order(incoming)
            # MARKET and IOC orders are cancelled if not fully filled (remainder)

        return fills

    def _can_match(self, incoming: Order, resting_price: float) -> bool:
        if incoming.order_type == OrderType.MARKET:
            return True

        if incoming.side == "BUY":
            return incoming.price >= resting_price
        else:
            return incoming.price <= resting_price

    def _execute_trade(self, incoming: Order, resting: Order, quantity: float, price: float) -> Trade:
        """
        Execute a trade and notify both parties.
        """
        trade = Trade(
            buyer_id=incoming.trader_id if incoming.side == "BUY" else resting.trader_id,
            seller_id=incoming.trader_id if incoming.side == "SELL" else resting.trader_id,
            price=price,
            quantity=quantity,
            timestamp=time.time(),
            aggressor_side=incoming.side,
        )

        # Track trade
        self.trades.append(trade)

        # Notify agents of execution if model is available
        if self.model:
            self._notify_agents(trade, incoming, resting)

        return trade

    def _notify_agents(self, trade: Trade, incoming: Order, resting: Order):
        """
        Notify both trading parties of the execution.
        """
        # Find agents by their unique IDs
        # trader_id format is "agent_{unique_id}"
        try:
            # Extract agent IDs from trader_ids
            incoming_agent_id = int(incoming.trader_id.split("_")[1])
            resting_agent_id = int(resting.trader_id.split("_")[1])

            # Get agents from model (Mesa 3.3+ uses model.agents directly)
            incoming_agent = None
            resting_agent = None

            for agent in self.model.agents:
                if agent.unique_id == incoming_agent_id:
                    incoming_agent = agent
                elif agent.unique_id == resting_agent_id:
                    resting_agent = agent

            # Notify agents
            if incoming_agent:
                incoming_agent.execute_trade(side=incoming.side, quantity=trade.quantity, price=trade.price)

            if resting_agent:
                # Resting order is on opposite side
                resting_side = "SELL" if incoming.side == "BUY" else "BUY"
                resting_agent.execute_trade(side=resting_side, quantity=trade.quantity, price=trade.price)

        except (ValueError, IndexError, AttributeError) as e:
            # If we can't parse trader IDs or find agents, log but don't crash
            import logging

            logging.warning(f"Could not notify agents of trade: {e}")

    def _handle_fok(self, incoming: Order) -> List[Trade]:
        """
        Fill-or-Kill: Must be fully filled immediately or cancelled.
        """
        book = self.orderbook.asks if incoming.side == "BUY" else self.orderbook.bids

        # Simulate matching to see if full quantity is available at valid prices
        temp_fills = []
        remaining_qty = incoming.quantity

        # We can't modify the heap while simulating, so we might need a copy or just iterate
        # Since heaps aren't sorted lists, iterating in order is destructive (heappop).
        # For FOK, we can check depth.

        # Optimization: Check if enough liquidity exists at valid prices
        # This is O(N log N) if we pop everything, which is bad.
        # But for FOK, we usually just peek.

        # Simplified implementation: Copy book (expensive) or just try to match and rollback?
        # Rollback is hard.
        # Let's use a list of potential matches.

        potential_matches = []

        # Make a copy of the heap to simulate
        book_copy = list(book)
        # Note: list(heap) does NOT sort it. We must heappop from the copy.
        # But heapq.nsmallest might be faster if we need top N.
        # Actually, we can just pop from the copy.

        # We need to preserve the heap property in the copy if we pop?
        # No, we can just use heappop on the copy.
        heapq.heapify(book_copy)  # Ensure it's a heap (it should be, but list() is just shallow copy of underlying list)

        while book_copy and remaining_qty > 0:
            best_price_tuple = heapq.heappop(book_copy)
            best_price = abs(best_price_tuple[0])
            resting_order = best_price_tuple[2]

            if not self._can_match(incoming, best_price):
                break

            qty = min(remaining_qty, resting_order.remaining)
            potential_matches.append((resting_order, qty, best_price))
            remaining_qty -= qty

        if remaining_qty > 1e-9:
            # Cannot fill fully - reject the FOK order
            return []

        # If we get here, we verified enough liquidity exists.
        # Now execute the actual fills by matching against the real order book.
        incoming.remaining = incoming.quantity
        fills = []

        while book and incoming.remaining > 0:
            # We know this will succeed because we verified liquidity above
            best_price_tuple = book[0]
            resting_order = best_price_tuple[2]
            best_price = abs(best_price_tuple[0])

            fill_qty = min(incoming.remaining, resting_order.remaining)
            trade = self._execute_trade(incoming, resting_order, fill_qty, best_price)
            fills.append(trade)

            # Update quantities
            resting_order.remaining -= fill_qty
            incoming.remaining -= fill_qty

            # Remove filled orders from book
            if resting_order.remaining <= 1e-9:
                heapq.heappop(book)
                if resting_order.order_id in self.orderbook.orders:
                    del self.orderbook.orders[resting_order.order_id]

        return fills
