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

    def get_best_bid(self) -> float:
        if not self.bids:
            return 0.0
        return -self.bids[0][0]

    def get_best_ask(self) -> float:
        if not self.asks:
            return float('inf')
        return self.asks[0][0]

    def get_mid_price(self) -> float:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid == 0.0 or ask == float('inf'):
            return 0.0
        return (bid + ask) / 2.0

    def get_spread(self) -> float:
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid == 0.0 or ask == float('inf'):
            return 0.0
        return ask - bid

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
        
        return {
            "bids": bid_levels,
            "asks": ask_levels
        }
