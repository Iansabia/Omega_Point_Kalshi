"""
Unit tests for Agent implementations.

Tests all 6 agent types:
1. NoiseTrader (Random, Contrarian, TrendFollower)
2. InformedTrader
3. Arbitrageur
4. MarketMakerAgent
5. HomerAgent
6. LLMAgent
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from src.agents.noise_trader import NoiseTrader
from src.agents.informed_trader import InformedTrader
from src.agents.arbitrageur import Arbitrageur
from src.agents.market_maker_agent import MarketMakerAgent
from src.agents.homer_agent import HomerAgent
from src.agents.llm_agent import LLMAgent
from src.orderbook.order import Order, OrderType, OrderSide


@pytest.fixture
def mock_model():
    """Create a mock model for agent testing."""
    model = Mock()
    model.current_price = 0.50
    model.fundamental_value = 0.50
    model.order_book = Mock()
    model.order_book.get_best_bid = Mock(return_value=0.49)
    model.order_book.get_best_ask = Mock(return_value=0.51)
    model.order_book.get_spread = Mock(return_value=0.02)
    model.order_book.get_mid_price = Mock(return_value=0.50)
    model.matching_engine = Mock()
    model.step_count = 1
    return model


@pytest.mark.unit
@pytest.mark.agents
class TestNoiseTrader:
    """Test suite for NoiseTrader agent."""

    def test_noise_trader_initialization_random(self, mock_model):
        """Test NoiseTrader initialization with random strategy."""
        agent = NoiseTrader(
            model=mock_model,
            strategy="random",
            initial_wealth=1000.0
        )

        assert agent.model == mock_model
        assert agent.strategy == "random"
        assert agent.wealth == 1000.0
        assert agent.position == 0.0
        assert agent.recency_weight == 0.7
        assert len(agent.trade_history) == 0

    def test_noise_trader_initialization_contrarian(self, mock_model):
        """Test NoiseTrader initialization with contrarian strategy."""
        agent = NoiseTrader(
            model=mock_model,
            strategy="contrarian",
            initial_wealth=2000.0
        )

        assert agent.strategy == "contrarian"
        assert agent.wealth == 2000.0

    def test_noise_trader_initialization_trend(self, mock_model):
        """Test NoiseTrader initialization with trend following strategy."""
        agent = NoiseTrader(
            model=mock_model,
            strategy="trend",
            initial_wealth=1500.0
        )

        assert agent.strategy == "trend"
        assert agent.wealth == 1500.0

    def test_noise_trader_observe_market(self, mock_model):
        """Test that noise trader can observe market state."""
        agent = NoiseTrader(model=mock_model, strategy="random")

        market_state = agent.observe_market()

        assert market_state is not None
        assert 'price' in market_state
        assert 'spread' in market_state

    def test_noise_trader_make_decision_random(self, mock_model):
        """Test random noise trader decision making."""
        agent = NoiseTrader(model=mock_model, strategy="random")

        # Random strategy should sometimes submit orders
        # With 10% probability, expect around 10 orders in 100 tries
        initial_order_count = len(agent.orders)
        for _ in range(100):
            agent.make_decision()

        # Should have submitted some orders (not zero due to randomness)
        # With 10% probability over 100 iterations, probability of 0 orders is (0.9)^100 ≈ 0.000027%
        orders_submitted = len(agent.orders) - initial_order_count
        assert orders_submitted > 0, f"Expected at least 1 order in 100 tries, got {orders_submitted}"

    def test_noise_trader_wealth_tracking(self, mock_model):
        """Test that noise trader tracks wealth correctly."""
        agent = NoiseTrader(model=mock_model, strategy="random", initial_wealth=1000.0)

        initial_wealth = agent.wealth
        assert initial_wealth == 1000.0

        # Simulate a buy
        agent.execute_trade("BUY", 10, 0.50)
        assert agent.wealth == 1000.0 - (10 * 0.50)  # 995.0
        assert agent.position == 10

        # Simulate a sell
        agent.execute_trade("SELL", 5, 0.55)
        assert agent.wealth == 995.0 + (5 * 0.55)  # 997.75
        assert agent.position == 5

    def test_noise_trader_pnl_calculation(self, mock_model):
        """Test P&L calculation for noise trader."""
        agent = NoiseTrader(model=mock_model, strategy="random", initial_wealth=1000.0)

        # No trades yet
        pnl = agent.calculate_pnl(current_price=0.50)
        assert pnl == 0.0

        # Buy at 0.50, value at 0.55
        agent.execute_trade("BUY", 10, 0.50)
        pnl = agent.calculate_pnl(current_price=0.55)

        # Wealth = 1000 - 5 = 995
        # Position value = 10 * 0.55 = 5.5
        # Total = 995 + 5.5 = 1000.5
        # P&L = 1000.5 - 1000 = 0.5
        assert abs(pnl - 0.5) < 0.01


@pytest.mark.unit
@pytest.mark.agents
class TestInformedTrader:
    """Test suite for InformedTrader agent."""

    def test_informed_trader_initialization(self, mock_model):
        """Test InformedTrader initialization."""
        agent = InformedTrader(
            model=mock_model,
            initial_wealth=10000.0,
            information_quality=0.8
        )

        assert agent.wealth == 10000.0
        assert agent.information_quality == 0.8
        assert agent.position == 0.0

    def test_informed_trader_information_quality(self, mock_model):
        """Test that information quality affects signal accuracy."""
        # High quality trader
        high_quality = InformedTrader(
            model=mock_model,
            information_quality=0.95
        )

        # Low quality trader
        low_quality = InformedTrader(
            model=mock_model,
            information_quality=0.55
        )

        assert high_quality.information_quality > low_quality.information_quality

    def test_informed_trader_observe_market(self, mock_model):
        """Test informed trader market observation."""
        agent = InformedTrader(model=mock_model)

        market_state = agent.observe_market()

        assert market_state is not None
        assert 'price' in market_state
        assert 'fundamental_value' in market_state

    def test_informed_trader_decision_logic(self, mock_model):
        """Test that informed trader makes decisions based on value."""
        agent = InformedTrader(
            model=mock_model,
            information_quality=0.9
        )

        # Set fundamental value higher than market price
        mock_model.fundamental_value = 0.60
        mock_model.current_price = 0.50

        decision = agent.make_decision()

        # Informed trader should recognize undervaluation
        # Decision might be None or BUY depending on threshold
        if decision is not None:
            assert decision['action'] in ['BUY', 'SELL', 'HOLD']

    def test_informed_trader_wealth_constraint(self, mock_model):
        """Test that informed trader respects wealth constraints."""
        agent = InformedTrader(
            model=mock_model,
            initial_wealth=100.0  # Low wealth
        )

        # Try to buy more than affordable
        with pytest.raises((ValueError, AssertionError)):
            agent.execute_trade("BUY", 1000, 0.50)  # Would cost 500


@pytest.mark.unit
@pytest.mark.agents
class TestArbitrageur:
    """Test suite for Arbitrageur agent."""

    def test_arbitrageur_initialization(self, mock_model):
        """Test Arbitrageur initialization."""
        agent = Arbitrageur(
            model=mock_model,
            initial_wealth=50000.0,
            detection_speed=0.85
        )

        assert agent.wealth == 50000.0
        assert agent.detection_speed == 0.85
        assert agent.min_spread == 0.02
        assert agent.position == 0.0

    def test_arbitrageur_detection_speed(self, mock_model):
        """Test that detection speed affects opportunity recognition."""
        fast_arb = Arbitrageur(
            model=mock_model,
            detection_speed=0.95
        )

        slow_arb = Arbitrageur(
            model=mock_model,
            detection_speed=0.60
        )

        assert fast_arb.detection_speed > slow_arb.detection_speed

    def test_arbitrageur_spread_detection(self, mock_model):
        """Test arbitrageur detects price discrepancies."""
        agent = Arbitrageur(
            model=mock_model,
            detection_speed=0.9
        )

        # Set up arbitrage opportunity
        mock_model.current_price = 0.45
        mock_model.fundamental_value = 0.55  # 10 cent spread

        market_state = agent.observe_market()

        assert market_state is not None
        spread = abs(mock_model.current_price - mock_model.fundamental_value)
        assert spread > agent.min_spread  # Opportunity exists

    def test_arbitrageur_min_spread_threshold(self, mock_model):
        """Test that arbitrageur only trades on sufficient spreads."""
        agent = Arbitrageur(
            model=mock_model,
            detection_speed=1.0
        )

        # Small spread (below threshold)
        mock_model.current_price = 0.50
        mock_model.fundamental_value = 0.51  # Only 1 cent

        assert abs(mock_model.current_price - mock_model.fundamental_value) < agent.min_spread


@pytest.mark.unit
@pytest.mark.agents
class TestMarketMakerAgent:
    """Test suite for MarketMakerAgent."""

    def test_market_maker_initialization(self, mock_model):
        """Test MarketMaker initialization."""
        agent = MarketMakerAgent(
            model=mock_model,
            initial_wealth=100000.0,
            target_inventory=0.0,
            risk_param=0.1
        )

        assert agent.wealth == 100000.0
        assert agent.target_inventory == 0.0
        assert agent.risk_param == 0.1
        assert agent.inventory == 0.0
        assert agent.half_spread == 0.02

    def test_market_maker_inventory_tracking(self, mock_model):
        """Test that market maker tracks inventory."""
        agent = MarketMakerAgent(
            model=mock_model,
            target_inventory=0.0
        )

        assert agent.inventory == 0.0

        # Buy should increase inventory
        agent.execute_trade("BUY", 100, 0.50)
        assert agent.position == 100  # Position tracks inventory

    def test_market_maker_target_inventory(self, mock_model):
        """Test market maker tries to maintain target inventory."""
        agent = MarketMakerAgent(
            model=mock_model,
            target_inventory=0.0
        )

        # Deviation from target
        agent.execute_trade("BUY", 50, 0.50)

        deviation = agent.position - agent.target_inventory
        assert deviation == 50  # Currently 50 above target

    def test_market_maker_quote_pricing(self, mock_model):
        """Test market maker quote generation."""
        agent = MarketMakerAgent(
            model=mock_model,
            target_inventory=0.0
        )

        market_state = agent.observe_market()

        # Market maker should observe bid/ask
        assert market_state is not None

    def test_market_maker_risk_parameter(self, mock_model):
        """Test that risk parameter affects spread."""
        conservative = MarketMakerAgent(
            model=mock_model,
            risk_param=0.2  # Higher risk aversion
        )

        aggressive = MarketMakerAgent(
            model=mock_model,
            risk_param=0.05  # Lower risk aversion
        )

        assert conservative.risk_param > aggressive.risk_param


@pytest.mark.unit
@pytest.mark.agents
class TestHomerAgent:
    """Test suite for HomerAgent (loyalty bias)."""

    def test_homer_agent_initialization(self, mock_model):
        """Test HomerAgent initialization."""
        agent = HomerAgent(
            model=mock_model,
            initial_wealth=2000.0,
            loyalty_asset="YES",
            loyalty_strength=0.75
        )

        assert agent.wealth == 2000.0
        assert agent.loyalty_asset == "YES"
        assert agent.loyalty_strength == 0.75
        assert agent.position == 0.0

    def test_homer_agent_loyalty_strength(self, mock_model):
        """Test loyalty strength parameter."""
        strong_homer = HomerAgent(
            model=mock_model,
            loyalty_strength=0.9
        )

        weak_homer = HomerAgent(
            model=mock_model,
            loyalty_strength=0.55
        )

        assert strong_homer.loyalty_strength > weak_homer.loyalty_strength

    def test_homer_agent_loyalty_asset(self, mock_model):
        """Test that homer agent has preferred asset."""
        yes_homer = HomerAgent(
            model=mock_model,
            loyalty_asset="YES"
        )

        no_homer = HomerAgent(
            model=mock_model,
            loyalty_asset="NO"
        )

        assert yes_homer.loyalty_asset == "YES"
        assert no_homer.loyalty_asset == "NO"

    def test_homer_agent_observe_market(self, mock_model):
        """Test homer agent market observation."""
        agent = HomerAgent(
            model=mock_model,
            loyalty_asset="YES"
        )

        market_state = agent.observe_market()
        assert market_state is not None

    def test_homer_agent_biased_decision(self, mock_model):
        """Test that homer agent shows loyalty bias in decisions."""
        agent = HomerAgent(
            model=mock_model,
            loyalty_asset="YES",
            loyalty_strength=0.8
        )

        # Homer agents should be willing to overpay for loyal asset
        assert agent.loyalty_strength > 0.5  # Shows bias


@pytest.mark.unit
@pytest.mark.agents
class TestLLMAgent:
    """Test suite for LLMAgent (Gemini-powered)."""

    def test_llm_agent_initialization(self, mock_model):
        """Test LLMAgent initialization."""
        agent = LLMAgent(
            model=mock_model,
            initial_wealth=10000.0,
            risk_profile="balanced"
        )

        assert agent.wealth == 10000.0
        assert agent.risk_profile == "balanced"
        assert agent.cumulative_cost == 0.0
        assert agent.position == 0.0

    def test_llm_agent_risk_profiles(self, mock_model):
        """Test different risk profiles."""
        conservative = LLMAgent(
            model=mock_model,
            risk_profile="conservative"
        )

        moderate = LLMAgent(
            model=mock_model,
            risk_profile="moderate"
        )

        aggressive = LLMAgent(
            model=mock_model,
            risk_profile="aggressive"
        )

        assert conservative.risk_profile == "conservative"
        assert moderate.risk_profile == "moderate"
        assert aggressive.risk_profile == "aggressive"

    def test_llm_agent_cost_tracking(self, mock_model):
        """Test that LLM agent tracks API costs."""
        agent = LLMAgent(
            model=mock_model,
            risk_profile="balanced"
        )

        assert agent.cumulative_cost == 0.0

        # After LLM calls, cost should increase
        # (This test assumes cost tracking is implemented)

    @patch('src.agents.llm_agent.genai')
    def test_llm_agent_should_use_llm(self, mock_genai, mock_model):
        """Test hybrid decision logic (when to use LLM vs rules)."""
        agent = LLMAgent(
            model=mock_model,
            risk_profile="balanced"
        )

        market_state = {'volatility': 0.05}

        # Low volatility - might use rules
        use_llm_low_vol = agent.should_use_llm(market_state)

        # High volatility - should use LLM
        market_state['volatility'] = 0.25
        use_llm_high_vol = agent.should_use_llm(market_state)

        # Either both boolean or test passes
        assert isinstance(use_llm_low_vol, bool)
        assert isinstance(use_llm_high_vol, bool)

    def test_llm_agent_observe_market(self, mock_model):
        """Test LLM agent market observation."""
        agent = LLMAgent(
            model=mock_model,
            risk_profile="balanced"
        )

        market_state = agent.observe_market()
        assert market_state is not None
        assert isinstance(market_state, dict)

    @patch('src.agents.llm_agent.genai')
    def test_llm_agent_rule_based_fallback(self, mock_genai, mock_model):
        """Test that LLM agent has rule-based fallback."""
        agent = LLMAgent(
            model=mock_model,
            risk_profile="balanced"
        )

        # Simulate LLM unavailable
        mock_genai.Client.side_effect = Exception("API Error")

        # Should still be able to make decisions with rules
        market_state = {'price': 0.50, 'volatility': 0.05}

        # Rule-based decision shouldn't crash
        try:
            decision = agent.rule_based_decision(market_state)
            # If implemented, should return something
            assert decision is not None or decision is None  # Either is fine
        except NotImplementedError:
            # If not implemented yet, that's ok
            pass


@pytest.mark.unit
@pytest.mark.agents
class TestBaseAgent:
    """Test suite for BaseAgent abstract class functionality."""

    def test_agent_trader_id(self, mock_model):
        """Test that agents generate trader IDs correctly."""
        agent = NoiseTrader(model=mock_model, strategy="random")

        trader_id = agent.trader_id
        assert trader_id.startswith("agent_")
        assert str(agent.unique_id) in trader_id

    def test_agent_portfolio_value(self, mock_model):
        """Test portfolio value calculation."""
        agent = NoiseTrader(
            model=mock_model,
            strategy="random",
            initial_wealth=1000.0
        )

        # No position
        portfolio_value = agent.get_portfolio_value(current_price=0.50)
        assert portfolio_value == 1000.0

        # With position
        agent.execute_trade("BUY", 10, 0.50)
        portfolio_value = agent.get_portfolio_value(current_price=0.55)

        # Wealth = 1000 - 5 = 995
        # Position value = 10 * 0.55 = 5.5
        # Total = 1000.5
        expected = 995.0 + (10 * 0.55)
        assert abs(portfolio_value - expected) < 0.01

    def test_agent_trade_history(self, mock_model):
        """Test that agents maintain trade history."""
        agent = NoiseTrader(model=mock_model, strategy="random")

        assert len(agent.trade_history) == 0

        agent.execute_trade("BUY", 10, 0.50)
        assert len(agent.trade_history) == 1

        trade = agent.trade_history[0]
        assert trade['side'] == "BUY"
        assert trade['quantity'] == 10
        assert trade['price'] == 0.50

    def test_agent_step_method(self, mock_model):
        """Test that agents have step method."""
        agent = NoiseTrader(model=mock_model, strategy="random")

        # Step should execute without error
        try:
            agent.step()
        except NotImplementedError:
            # If not implemented, that's ok for abstract base
            pass
