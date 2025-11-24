import heapq
from typing import Dict, List, Tuple

from .order import Order


class OrderBook:
    def __init__(self):
        self.bids: List[Tuple[float, float, Order]] = []  # max-heap (negated prices)
        self.asks: List[Tuple[float, float, Order]] = []  # min-heap
        self.orders: Dict[str, Order] = {}  # order_id -> Order

    def add_order(self, order: Order):
        """
        Add an order to the order book.
        Bids are stored in a max-heap (using negated prices).
        Asks are stored in a min-heap.
        """
        if order.side == "BUY":
            # Negate price for max-heap behavior using Python's min-heap
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))
        else:
            heapq.heappush(self.asks, (order.price, order.timestamp, order))

        self.orders[order.order_id] = order

    def get_best_bid(self) -> float:
        """Get best bid price."""
        if not self.bids:
            return 0.0
        return -self.bids[0][0]

    def get_best_ask(self) -> float:
        """Get best ask price."""
        if not self.asks:
            return float("inf")
        return self.asks[0][0]

    def get_best_bid_order(self) -> Order:
        """Get best bid Order object."""
        if not self.bids:
            return None
        return self.bids[0][2]

    def get_best_ask_order(self) -> Order:
        """Get best ask Order object."""
        if not self.asks:
            return None
        return self.asks[0][2]

    def get_mid_price(self) -> float:
        """Get mid-price between best bid and ask."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid == 0.0 or ask == float("inf"):
            return None  # Changed to None for consistency
        return (bid + ask) / 2.0

    def get_spread(self) -> float:
        """Get bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid == 0.0 or ask == float("inf"):
            return None  # Changed to None for consistency
        return ask - bid

    def remove_order(self, order_id: str):
        """Remove an order from the book."""
        if order_id not in self.orders:
            return

        order = self.orders.pop(order_id)

        # Remove from appropriate heap
        # Note: This is O(n) operation. For production, consider using a better data structure.
        if order.side == "BUY":
            self.bids = [(p, ts, o) for p, ts, o in self.bids if o.order_id != order_id]
            heapq.heapify(self.bids)
        else:
            self.asks = [(p, ts, o) for p, ts, o in self.asks if o.order_id != order_id]
            heapq.heapify(self.asks)

    def get_imbalance(self) -> float:
        """
        Calculate volume imbalance: (Bid Vol - Ask Vol) / (Bid Vol + Ask Vol)
        Note: This iterates heaps, which is O(N). For high freq, maintain running sums.
        """
        bid_vol = sum(o.remaining for _, _, o in self.bids)
        ask_vol = sum(o.remaining for _, _, o in self.asks)

        if bid_vol + ask_vol == 0:
            return 0.0
        return (bid_vol - ask_vol) / (bid_vol + ask_vol)

    def get_depth(self, levels: int = 5) -> Dict[str, List[Tuple[float, float]]]:
        """
        Return top N levels of depth.
        Note: This is expensive on a heap (requires sorting/copying).
        """
        # Efficient approach for top N: nsmallest
        # Bids are (-price, ts, order)
        top_bids = heapq.nsmallest(levels, self.bids)
        bid_levels = [(-p, o.remaining) for p, _, o in top_bids]

        top_asks = heapq.nsmallest(levels, self.asks)
        ask_levels = [(p, o.remaining) for p, _, o in top_asks]

        return {"bids": bid_levels, "asks": ask_levels}
