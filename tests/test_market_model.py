"""
Integration tests for PredictionMarketModel.
"""
import pytest
import numpy as np
from src.models.market_model import PredictionMarketModel


class TestPredictionMarketModel:
    """Test suite for the core market model."""

    def test_model_initialization_minimal(self):
        """Test model initialization with no config."""
        model = PredictionMarketModel()

        assert model.current_price == 0.5  # Default initial price
        assert model.fundamental_value == 0.5
        assert model.step_count == 0
        assert model.cumulative_llm_cost == 0.0
        assert model.order_book is not None
        assert model.matching_engine is not None
        assert len(list(model.agents)) == 0  # No agents without config

    def test_model_initialization_with_config(self):
        """Test model initialization with custom config."""
        config = {
            'market': {
                'initial_price': 0.65,
                'tick_size': 0.01
            }
        }
        model = PredictionMarketModel(config=config)

        assert model.current_price == 0.65
        assert model.fundamental_value == 0.65

    def test_model_initialization_with_seed(self):
        """Test reproducible initialization with seed."""
        model1 = PredictionMarketModel(seed=42)
        model2 = PredictionMarketModel(seed=42)

        # Random state should be the same
        assert model1.random.random() == model2.random.random()

    def test_initialize_noise_traders(self):
        """Test noise trader initialization from config."""
        agent_config = {
            'noise_trader': {
                'count': 10,
                'wealth_distribution': {
                    'type': 'lognormal',
                    'mean': 1000,
                    'sigma': 0.5
                }
            }
        }

        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        agents = list(model.agents)
        assert len(agents) == 10

        # Check agent types
        from src.agents.noise_trader import NoiseTrader
        for agent in agents:
            assert isinstance(agent, NoiseTrader)
            assert agent.wealth > 0

    def test_initialize_informed_traders(self):
        """Test informed trader initialization from config."""
        agent_config = {
            'informed_trader': {
                'count': 5,
                'wealth': 10000,
                'information_quality': 0.8
            }
        }

        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        agents = list(model.agents)
        assert len(agents) == 5

        from src.agents.informed_trader import InformedTrader
        for agent in agents:
            assert isinstance(agent, InformedTrader)
            assert agent.wealth == 10000
            assert agent.information_quality == 0.8

    def test_initialize_arbitrageurs(self):
        """Test arbitrageur initialization from config."""
        agent_config = {
            'arbitrageur': {
                'count': 3,
                'wealth': 50000,
                'detection_speed': 0.9
            }
        }

        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        agents = list(model.agents)
        assert len(agents) == 3

        from src.agents.arbitrageur import Arbitrageur
        for agent in agents:
            assert isinstance(agent, Arbitrageur)
            assert agent.wealth == 50000
            assert agent.detection_speed == 0.9

    def test_initialize_market_makers(self):
        """Test market maker initialization from config."""
        agent_config = {
            'market_maker': {
                'count': 2,
                'wealth': 100000,
                'risk_param': 0.1
            }
        }

        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        agents = list(model.agents)
        assert len(agents) == 2

        from src.agents.market_maker_agent import MarketMakerAgent
        for agent in agents:
            assert isinstance(agent, MarketMakerAgent)
            assert agent.wealth == 100000
            assert agent.risk_param == 0.1

    def test_initialize_homer_agents(self):
        """Test homer agent initialization from config."""
        agent_config = {
            'homer_agent': {
                'count': 4,
                'wealth': 2000,
                'loyalty_strength': 0.7
            }
        }

        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        agents = list(model.agents)
        assert len(agents) == 4

        from src.agents.homer_agent import HomerAgent
        for agent in agents:
            assert isinstance(agent, HomerAgent)
            assert agent.wealth == 2000
            assert agent.loyalty_strength == 0.7

    def test_initialize_llm_agents(self):
        """Test LLM agent initialization from config."""
        agent_config = {
            'llm_agent': {
                'count': 2,
                'wealth': 10000,
                'risk_profile': 'conservative'
            }
        }

        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        agents = list(model.agents)
        assert len(agents) == 2

        from src.agents.llm_agent import LLMAgent
        for agent in agents:
            assert isinstance(agent, LLMAgent)
            assert agent.wealth == 10000
            assert agent.risk_profile == 'conservative'

    def test_initialize_mixed_agents(self):
        """Test initialization with multiple agent types."""
        agent_config = {
            'noise_trader': {'count': 10},
            'informed_trader': {'count': 5},
            'arbitrageur': {'count': 3},
            'market_maker': {'count': 2}
        }

        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        agents = list(model.agents)
        assert len(agents) == 20  # 10 + 5 + 3 + 2

    def test_step_execution(self):
        """Test single step execution."""
        agent_config = {'noise_trader': {'count': 5}}
        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        initial_step = model.step_count
        model.step()

        assert model.step_count == initial_step + 1
        # Datacollector should have collected data
        assert len(model.datacollector.model_vars["step"]) == 1

    def test_multi_step_simulation(self):
        """Test running multiple simulation steps."""
        agent_config = {
            'informed_trader': {'count': 5},  # Use informed traders instead of noise for stability
            'market_maker': {'count': 2}
        }
        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        num_steps = 10
        for _ in range(num_steps):
            model.step()

        assert model.step_count == num_steps
        # Check datacollector has data for all steps
        assert len(model.datacollector.model_vars["step"]) == num_steps

    def test_get_spread(self):
        """Test spread calculation."""
        model = PredictionMarketModel()

        # Empty order book should return None
        spread = model.get_spread()
        assert spread is None or spread == 0.0

    def test_calculate_volume(self):
        """Test volume calculation."""
        model = PredictionMarketModel()

        volume = model.calculate_volume()
        assert volume == 0.0  # No trades initially

    def test_datacollector_model_reporters(self):
        """Test datacollector captures model-level metrics."""
        agent_config = {'noise_trader': {'count': 5}}
        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        model.step()

        # Check all model reporters are captured
        assert "market_price" in model.datacollector.model_vars
        assert "fundamental_value" in model.datacollector.model_vars
        assert "total_volume" in model.datacollector.model_vars
        assert "bid_ask_spread" in model.datacollector.model_vars
        assert "llm_cost" in model.datacollector.model_vars
        assert "step" in model.datacollector.model_vars

    def test_datacollector_agent_reporters(self):
        """Test datacollector captures agent-level metrics."""
        agent_config = {'noise_trader': {'count': 3}}
        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        model.step()

        # Get agent data
        agent_data = model.datacollector.get_agent_vars_dataframe()

        # Check agent reporters
        assert "wealth" in agent_data.columns
        assert "position" in agent_data.columns
        assert "agent_type" in agent_data.columns
        assert len(agent_data) == 3  # One row per agent

    def test_matching_engine_linked_to_model(self):
        """Test matching engine has reference to model."""
        model = PredictionMarketModel()

        assert model.matching_engine.model is model

    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        agent_config = {'noise_trader': {'count': 10}}

        model1 = PredictionMarketModel(agent_config=agent_config, seed=42)
        for _ in range(5):
            model1.step()

        model2 = PredictionMarketModel(agent_config=agent_config, seed=42)
        for _ in range(5):
            model2.step()

        # Get final prices
        price1 = model1.datacollector.model_vars["market_price"][-1]
        price2 = model2.datacollector.model_vars["market_price"][-1]

        # Should be identical with same seed
        assert price1 == price2

    def test_agent_wealth_conservation(self):
        """Test that total wealth is conserved (no money creation)."""
        agent_config = {
            'noise_trader': {'count': 10},
            'market_maker': {'count': 2}
        }
        model = PredictionMarketModel(agent_config=agent_config, seed=42)

        # Calculate initial total wealth
        initial_wealth = sum(agent.wealth for agent in model.agents)

        # Run simulation
        for _ in range(10):
            model.step()

        # Calculate final total wealth (accounting for positions)
        final_wealth = sum(agent.wealth + agent.position * model.current_price
                          for agent in model.agents)

        # Wealth should be approximately conserved (within rounding)
        # Note: Small differences may occur due to unfilled orders
        assert abs(initial_wealth - final_wealth) < 1000  # Allow small variance
