"""
Pytest fixtures and configuration for Omega Point tests.
"""

import sys
from pathlib import Path

import pytest

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.market_model import PredictionMarketModel
from src.orderbook.matching_engine import MatchingEngine
from src.orderbook.order import Order, OrderSide, OrderType
from src.orderbook.orderbook import OrderBook


@pytest.fixture
def simple_market_config():
    """Basic market configuration for testing."""
    return {
        "market": {"initial_price": 0.5, "trading_hours": {"start": "09:00", "end": "17:00"}},
        "simulation": {"seed": 42, "steps": 10},
    }


@pytest.fixture
def agent_config():
    """Basic agent configuration for testing."""
    return {
        "noise_trader": {"count": 5, "wealth_distribution": {"type": "uniform", "mean": 1000}},
        "informed_trader": {"count": 2, "wealth": 10000, "information_quality": 0.8},
        "market_maker": {"count": 1, "wealth": 100000, "target_inventory": 0},
    }


@pytest.fixture
def order_book():
    """Create a fresh order book."""
    return OrderBook()


@pytest.fixture
def matching_engine(order_book):
    """Create a matching engine with order book."""
    return MatchingEngine(order_book)


@pytest.fixture
def sample_buy_order():
    """Create a sample buy limit order."""
    return Order(
        order_id="test_buy_1", trader_id="agent_1", side=OrderSide.BUY, order_type=OrderType.LIMIT, price=0.55, quantity=100
    )


@pytest.fixture
def sample_sell_order():
    """Create a sample sell limit order."""
    return Order(
        order_id="test_sell_1", trader_id="agent_2", side=OrderSide.SELL, order_type=OrderType.LIMIT, price=0.45, quantity=100
    )


@pytest.fixture
def market_model(simple_market_config, agent_config):
    """Create a market model for testing."""
    model = PredictionMarketModel(config=simple_market_config, agent_config=agent_config, seed=42)
    return model


@pytest.fixture
def market_model_no_agents(simple_market_config):
    """Create a market model without agents."""
    model = PredictionMarketModel(config=simple_market_config, agent_config={}, seed=42)
    return model
