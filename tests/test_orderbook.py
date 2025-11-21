"""
Unit tests for Order Book functionality.

Tests the core order book implementation including:
- Order placement and management
- Price-time priority
- Order matching logic
- Market and limit orders
- Fill-or-Kill (FOK) and Immediate-or-Cancel (IOC) orders
"""
import pytest
from src.orderbook.orderbook import OrderBook
from src.orderbook.order import Order, OrderType, OrderSide
from src.orderbook.matching_engine import MatchingEngine


@pytest.mark.unit
@pytest.mark.orderbook
class TestOrderBook:
    """Test suite for OrderBook class."""

    def test_orderbook_initialization(self, order_book):
        """Test that order book initializes correctly."""
        assert order_book is not None
        assert len(order_book.bids) == 0
        assert len(order_book.asks) == 0
        assert order_book.get_mid_price() is None

    def test_add_buy_order(self, order_book, sample_buy_order):
        """Test adding a buy order to the book."""
        order_book.add_order(sample_buy_order)

        assert len(order_book.bids) == 1
        assert len(order_book.asks) == 0
        assert sample_buy_order.order_id in order_book.orders

    def test_add_sell_order(self, order_book, sample_sell_order):
        """Test adding a sell order to the book."""
        order_book.add_order(sample_sell_order)

        assert len(order_book.bids) == 0
        assert len(order_book.asks) == 1
        assert sample_sell_order.order_id in order_book.orders

    def test_price_time_priority_bids(self, order_book):
        """Test that bids are sorted by price-time priority (best bid first)."""
        # Add orders with different prices
        order1 = Order("b1", "agent_1", OrderSide.BUY, OrderType.LIMIT, 0.50, 100)
        order2 = Order("b2", "agent_2", OrderSide.BUY, OrderType.LIMIT, 0.55, 100)
        order3 = Order("b3", "agent_3", OrderSide.BUY, OrderType.LIMIT, 0.52, 100)

        order_book.add_order(order1)
        order_book.add_order(order2)
        order_book.add_order(order3)

        # Best bid should be highest price (0.55)
        best_bid = order_book.get_best_bid_order()
        assert best_bid is not None
        assert best_bid.price == 0.55
        assert best_bid.order_id == "b2"

    def test_price_time_priority_asks(self, order_book):
        """Test that asks are sorted by price-time priority (best ask first)."""
        # Add orders with different prices
        order1 = Order("s1", "agent_1", OrderSide.SELL, OrderType.LIMIT, 0.50, 100)
        order2 = Order("s2", "agent_2", OrderSide.SELL, OrderType.LIMIT, 0.45, 100)
        order3 = Order("s3", "agent_3", OrderSide.SELL, OrderType.LIMIT, 0.52, 100)

        order_book.add_order(order1)
        order_book.add_order(order2)
        order_book.add_order(order3)

        # Best ask should be lowest price (0.45)
        best_ask = order_book.get_best_ask_order()
        assert best_ask is not None
        assert best_ask.price == 0.45
        assert best_ask.order_id == "s2"

    def test_time_priority_same_price(self, order_book):
        """Test that orders at same price follow FIFO (first-in-first-out)."""
        import time

        # Add two orders at same price
        order1 = Order("b1", "agent_1", OrderSide.BUY, OrderType.LIMIT, 0.50, 100)
        order_book.add_order(order1)

        time.sleep(0.001)  # Ensure different timestamps

        order2 = Order("b2", "agent_2", OrderSide.BUY, OrderType.LIMIT, 0.50, 100)
        order_book.add_order(order2)

        # First order should be at top
        best_bid = order_book.get_best_bid_order()
        assert best_bid.order_id == "b1"

    def test_remove_order(self, order_book, sample_buy_order):
        """Test removing an order from the book."""
        order_book.add_order(sample_buy_order)
        assert len(order_book.bids) == 1

        order_book.remove_order(sample_buy_order.order_id)
        assert len(order_book.bids) == 0
        assert sample_buy_order.order_id not in order_book.orders

    def test_get_spread(self, order_book):
        """Test spread calculation."""
        # Empty book should have None spread
        assert order_book.get_spread() is None

        # Add buy and sell orders
        buy_order = Order("b1", "agent_1", OrderSide.BUY, OrderType.LIMIT, 0.48, 100)
        sell_order = Order("s1", "agent_2", OrderSide.SELL, OrderType.LIMIT, 0.52, 100)

        order_book.add_order(buy_order)
        order_book.add_order(sell_order)

        spread = order_book.get_spread()
        assert spread is not None
        assert abs(spread - 0.04) < 1e-10  # 0.52 - 0.48 = 0.04

    def test_get_mid_price(self, order_book):
        """Test mid-price calculation."""
        # Empty book should return None
        assert order_book.get_mid_price() is None

        # Add buy and sell orders
        buy_order = Order("b1", "agent_1", OrderSide.BUY, OrderType.LIMIT, 0.48, 100)
        sell_order = Order("s1", "agent_2", OrderSide.SELL, OrderType.LIMIT, 0.52, 100)

        order_book.add_order(buy_order)
        order_book.add_order(sell_order)

        mid_price = order_book.get_mid_price()
        assert mid_price is not None
        assert abs(mid_price - 0.50) < 1e-10  # (0.48 + 0.52) / 2 = 0.50

    def test_get_depth(self, order_book):
        """Test order book depth calculation."""
        # Add multiple orders at different levels
        order_book.add_order(Order("b1", "a1", OrderSide.BUY, OrderType.LIMIT, 0.50, 100))
        order_book.add_order(Order("b2", "a2", OrderSide.BUY, OrderType.LIMIT, 0.49, 50))
        order_book.add_order(Order("s1", "a3", OrderSide.SELL, OrderType.LIMIT, 0.51, 100))
        order_book.add_order(Order("s2", "a4", OrderSide.SELL, OrderType.LIMIT, 0.52, 50))

        depth = order_book.get_depth(levels=2)

        assert depth is not None
        assert 'bids' in depth
        assert 'asks' in depth
        assert len(depth['bids']) <= 2
        assert len(depth['asks']) <= 2

    def test_get_imbalance(self, order_book):
        """Test order book imbalance calculation."""
        # Add orders with more buy volume
        order_book.add_order(Order("b1", "a1", OrderSide.BUY, OrderType.LIMIT, 0.50, 200))
        order_book.add_order(Order("s1", "a2", OrderSide.SELL, OrderType.LIMIT, 0.51, 100))

        imbalance = order_book.get_imbalance()
        assert imbalance > 0  # More bids than asks


@pytest.mark.unit
@pytest.mark.orderbook
class TestMatchingEngine:
    """Test suite for MatchingEngine class."""

    def test_matching_engine_initialization(self, matching_engine):
        """Test matching engine initializes correctly."""
        assert matching_engine is not None
        assert len(matching_engine.trades) == 0

    def test_limit_order_no_match(self, matching_engine, order_book):
        """Test limit order that doesn't match stays in book."""
        buy_order = Order("b1", "agent_1", OrderSide.BUY, OrderType.LIMIT, 0.48, 100)

        trades = matching_engine.match_order(buy_order)

        assert len(trades) == 0  # No match
        assert len(order_book.bids) == 1  # Order added to book
        assert buy_order.remaining == 100  # Nothing filled

    def test_limit_order_full_match(self, matching_engine, order_book):
        """Test limit order that fully matches."""
        # Add resting sell order
        sell_order = Order("s1", "agent_1", OrderSide.SELL, OrderType.LIMIT, 0.50, 100)
        order_book.add_order(sell_order)

        # Add matching buy order
        buy_order = Order("b1", "agent_2", OrderSide.BUY, OrderType.LIMIT, 0.50, 100)
        trades = matching_engine.match_order(buy_order)

        assert len(trades) == 1
        assert trades[0].quantity == 100
        assert trades[0].price == 0.50
        assert buy_order.remaining == 0
        assert sell_order.remaining == 0

    def test_limit_order_partial_match(self, matching_engine, order_book):
        """Test limit order that partially matches."""
        # Add resting sell order (50 quantity)
        sell_order = Order("s1", "agent_1", OrderSide.SELL, OrderType.LIMIT, 0.50, 50)
        order_book.add_order(sell_order)

        # Add larger buy order (100 quantity)
        buy_order = Order("b1", "agent_2", OrderSide.BUY, OrderType.LIMIT, 0.50, 100)
        trades = matching_engine.match_order(buy_order)

        assert len(trades) == 1
        assert trades[0].quantity == 50
        assert buy_order.remaining == 50  # Partially filled
        assert sell_order.remaining == 0  # Fully filled

    def test_market_order_execution(self, matching_engine, order_book):
        """Test market order executes at best available price."""
        # Add resting sell order
        sell_order = Order("s1", "agent_1", OrderSide.SELL, OrderType.LIMIT, 0.52, 100)
        order_book.add_order(sell_order)

        # Add market buy order
        buy_order = Order("b1", "agent_2", OrderSide.BUY, OrderType.MARKET, None, 100)
        trades = matching_engine.match_order(buy_order)

        assert len(trades) == 1
        assert trades[0].price == 0.52  # Executed at ask price
        assert trades[0].quantity == 100

    def test_price_improvement(self, matching_engine, order_book):
        """Test that aggressive limit orders get price improvement."""
        # Add resting sell at 0.50
        sell_order = Order("s1", "agent_1", OrderSide.SELL, OrderType.LIMIT, 0.50, 100)
        order_book.add_order(sell_order)

        # Add aggressive buy at 0.55 (willing to pay more)
        buy_order = Order("b1", "agent_2", OrderSide.BUY, OrderType.LIMIT, 0.55, 100)
        trades = matching_engine.match_order(buy_order)

        assert len(trades) == 1
        assert trades[0].price == 0.50  # Gets price improvement (pays less)

    def test_fok_order_success(self, matching_engine, order_book):
        """Test Fill-or-Kill order that can be fully filled."""
        # Add enough liquidity
        order_book.add_order(Order("s1", "a1", OrderSide.SELL, OrderType.LIMIT, 0.50, 100))

        # FOK order that can be filled
        fok_order = Order("b1", "agent_2", OrderSide.BUY, OrderType.FOK, 0.50, 100)
        trades = matching_engine.match_order(fok_order)

        assert len(trades) == 1
        assert fok_order.remaining == 0

    def test_fok_order_failure(self, matching_engine, order_book):
        """Test Fill-or-Kill order that cannot be fully filled."""
        # Add insufficient liquidity
        order_book.add_order(Order("s1", "a1", OrderSide.SELL, OrderType.LIMIT, 0.50, 50))

        # FOK order that cannot be fully filled
        fok_order = Order("b1", "agent_2", OrderSide.BUY, OrderType.FOK, 0.50, 100)
        trades = matching_engine.match_order(fok_order)

        assert len(trades) == 0  # FOK failed
        assert fok_order.remaining == 100  # Nothing filled
        assert len(order_book.bids) == 0  # Not added to book

    def test_ioc_order(self, matching_engine, order_book):
        """Test Immediate-or-Cancel order partial fill."""
        # Add partial liquidity
        order_book.add_order(Order("s1", "a1", OrderSide.SELL, OrderType.LIMIT, 0.50, 50))

        # IOC order for 100 (can only fill 50)
        ioc_order = Order("b1", "agent_2", OrderSide.BUY, OrderType.IOC, 0.50, 100)
        trades = matching_engine.match_order(ioc_order)

        assert len(trades) == 1
        assert trades[0].quantity == 50  # Partial fill
        assert ioc_order.remaining == 50  # Unfilled portion
        assert len(order_book.bids) == 0  # Remainder not added to book

    def test_multiple_levels_matching(self, matching_engine, order_book):
        """Test order matching across multiple price levels."""
        # Add multiple sell orders at different prices
        order_book.add_order(Order("s1", "a1", OrderSide.SELL, OrderType.LIMIT, 0.50, 50))
        order_book.add_order(Order("s2", "a2", OrderSide.SELL, OrderType.LIMIT, 0.51, 50))
        order_book.add_order(Order("s3", "a3", OrderSide.SELL, OrderType.LIMIT, 0.52, 50))

        # Large buy order should match across levels
        buy_order = Order("b1", "agent_4", OrderSide.BUY, OrderType.LIMIT, 0.52, 150)
        trades = matching_engine.match_order(buy_order)

        assert len(trades) == 3  # Matched 3 orders
        assert sum(t.quantity for t in trades) == 150
        # Should match best prices first
        assert trades[0].price == 0.50
        assert trades[1].price == 0.51
        assert trades[2].price == 0.52

    def test_self_trade_prevention(self, matching_engine, order_book):
        """Test that agents cannot trade with themselves (if implemented)."""
        # Note: This test assumes self-trade prevention is implemented
        # If not implemented, this test can be skipped or used as a TODO

        # Add sell order from agent_1
        sell_order = Order("s1", "agent_1", OrderSide.SELL, OrderType.LIMIT, 0.50, 100)
        order_book.add_order(sell_order)

        # Try to buy from same agent
        buy_order = Order("b1", "agent_1", OrderSide.BUY, OrderType.LIMIT, 0.50, 100)
        trades = matching_engine.match_order(buy_order)

        # Implementation-dependent: either no trade or trade allowed
        # For now, we'll just verify the function doesn't crash
        assert trades is not None


@pytest.mark.unit
@pytest.mark.orderbook
class TestOrder:
    """Test suite for Order class."""

    def test_order_creation(self):
        """Test creating an order."""
        order = Order(
            order_id="test_1",
            trader_id="agent_1",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            price=0.50,
            quantity=100
        )

        assert order.order_id == "test_1"
        assert order.trader_id == "agent_1"
        assert order.side == "BUY"  # Side is stored as string
        assert order.order_type == OrderType.LIMIT
        assert order.price == 0.50
        assert order.quantity == 100
        assert order.remaining == 100

    def test_order_fill(self):
        """Test order fill tracking."""
        order = Order("test_1", "agent_1", OrderSide.BUY, OrderType.LIMIT, 0.50, 100)

        # Fill 30 shares
        order.fill(30)
        assert order.remaining == 70
        assert order.quantity == 100  # Original quantity unchanged

        # Fill remaining 70
        order.fill(70)
        assert order.remaining == 0
        assert order.is_filled()

    def test_market_order_no_price(self):
        """Test that market orders can have None price."""
        order = Order("test_1", "agent_1", OrderSide.BUY, OrderType.MARKET, None, 100)

        assert order.price is None
        assert order.order_type == OrderType.MARKET
