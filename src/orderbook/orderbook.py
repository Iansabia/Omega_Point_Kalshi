import heapq
from typing import List, Dict, Tuple
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
        if order.side == 'BUY':
            # Negate price for max-heap behavior using Python's min-heap
            heapq.heappush(self.bids, (-order.price, order.timestamp, order))
        else:
            heapq.heappush(self.asks, (order.price, order.timestamp, order))
        
        self.orders[order.order_id] = order
