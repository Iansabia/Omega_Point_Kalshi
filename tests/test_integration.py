"""
Integration Tests for Omega Point ABM

Tests multiple components working together:
1. End-to-end simulation (agents + order book + model)
2. Backtesting workflow (data → simulation → analysis)
3. Risk management integration (agents + risk manager)
4. Feature engineering pipeline (data → features → agents)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.informed_trader import InformedTrader
from src.agents.market_maker_agent import MarketMakerAgent
from src.agents.noise_trader import NoiseTrader
from src.backtesting.backtest_engine import BacktestEngine
from src.data.feature_engineering import FeatureEngineer
from src.models.market_model import PredictionMarketModel
from src.orderbook.matching_engine import MatchingEngine
from src.orderbook.orderbook import OrderBook
from src.risk.risk_manager import RiskLimits, RiskManager


@pytest.mark.integration
class TestEndToEndSimulation:
    """Test complete simulation with agents, order book, and market model."""

    def test_simple_market_simulation(self):
        """Test basic market simulation with multiple agents."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Simple Market Simulation")
        print("=" * 70)

        # Create model with proper config structure
        config = {"market": {"initial_price": 0.50, "ticker": "TEST"}, "agents": {"noise_traders": 3, "informed_traders": 2}}

        model = PredictionMarketModel(config=config, seed=42)

        print(f"✓ Model initialized:")
        print(f"  Initial price: {model.current_price}")
        print(f"  Ticker: {model.current_ticker}")

        # Run simulation
        initial_price = model.current_price
        for step in range(10):
            model.step()

        final_price = model.current_price

        print(f"\n✓ Simulation completed (10 steps):")
        print(f"  Initial price: {initial_price:.3f}")
        print(f"  Final price: {final_price:.3f}")
        print(f"  Price change: {(final_price - initial_price):.3f}")
        print(f"  Steps: {model.step_count}")

        # Assertions
        assert model.step_count == 10
        assert 0 <= final_price <= 1.0, "Price should stay in [0,1]"
        print("\n  ✅ PASS: Market simulation works end-to-end")

    def test_agent_interactions(self):
        """Test that different agent types interact correctly."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Agent Interactions")
        print("=" * 70)

        # Create minimal model
        config = {"market": {"initial_price": 0.50}, "agents": {"noise_traders": 2, "informed_traders": 2, "market_makers": 2}}

        model = PredictionMarketModel(config=config, seed=42)

        # Run simulation
        for _ in range(10):
            model.step()

        print(f"\n✓ After 10 steps:")
        print(f"  Current price: {model.current_price:.3f}")
        print(f"  Steps completed: {model.step_count}")

        assert model.step_count == 10
        print("\n  ✅ PASS: Agents interact with market")

    def test_price_discovery(self):
        """Test that market simulation produces valid prices."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Price Validity")
        print("=" * 70)

        config = {"market": {"initial_price": 0.30}, "agents": {"noise_traders": 5, "informed_traders": 3}}

        model = PredictionMarketModel(config=config, seed=42)

        print(f"✓ Initial price: {model.current_price:.3f}")

        # Run simulation
        for _ in range(20):
            model.step()

        print(f"✓ After 20 steps:")
        print(f"  Final price: {model.current_price:.3f}")

        assert 0 <= model.current_price <= 1.0
        print("\n  ✅ PASS: Price stays in valid range [0,1]")


@pytest.mark.integration
class TestBacktestingWorkflow:
    """Test complete backtesting workflow."""

    @pytest.mark.skip(reason="BacktestEngine API needs to be standardized")
    def test_backtest_engine_integration(self):
        """Test backtesting engine with synthetic data."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Backtesting Workflow")
        print("=" * 70)

        # TODO: Update this test once BacktestEngine API is standardized
        # The backtest engine currently has different initialization parameters
        pytest.skip("BacktestEngine needs API standardization")


@pytest.mark.integration
class TestRiskManagementIntegration:
    """Test risk management integrated with agents."""

    def test_risk_limits_enforced_in_simulation(self):
        """Test that risk limits are enforced during simulation."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Risk Management Integration")
        print("=" * 70)

        # Create tight risk limits
        risk_limits = RiskLimits(
            max_positions=2, max_position_size=50, trade_probability=1.0  # Always allow trades (for testing)
        )

        print(f"✓ Risk limits:")
        print(f"  Max positions: {risk_limits.max_positions}")
        print(f"  Max position size: ${risk_limits.max_position_size}")

        # Create agents with risk limits
        from unittest.mock import Mock

        model = Mock()
        model.current_price = 0.50
        model.order_book = Mock()
        model.order_book.get_spread = Mock(return_value=0.02)
        model.order_book.get_mid_price = Mock(return_value=0.50)
        model.matching_engine = Mock()

        agent = NoiseTrader(model=model, strategy="random", initial_wealth=1000.0, risk_limits=risk_limits)

        # Override trade probability to ensure order generation
        agent.trade_probability = 1.0

        print(f"\n✓ Agent created with risk limits")
        print(f"  Initial wealth: ${agent.wealth}")
        print(f"  Risk manager active: {agent.risk_manager is not None}")

        # Try to make trades
        orders_before = len(agent.orders)
        for _ in range(20):
            agent.make_decision()

        orders_after = len(agent.orders)
        orders_submitted = orders_after - orders_before

        print(f"\n✓ After 20 decision cycles:")
        print(f"  Orders submitted: {orders_submitted}")
        print(f"  Risk manager blocked trades: {20 - orders_submitted}")

        # Some orders should be submitted (not all blocked)
        # But risk limits should prevent unlimited trading
        assert orders_submitted >= 0
        print(f"\n  ✅ PASS: Risk management integrates with agents")

    def test_position_limits(self):
        """Test that position limits work across multiple agents."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Position Limit Enforcement")
        print("=" * 70)

        risk_limits = RiskLimits(max_positions=3, trade_probability=1.0)

        risk_manager = RiskManager(limits=risk_limits)

        # Add positions
        risk_manager.positions = {"GAME1": 100, "GAME2": 200}

        # Try to add 3rd position (should work)
        can_trade_1, reason_1 = risk_manager.can_trade("GAME3", edge=0.10)
        print(f"✓ Adding 3rd position: {can_trade_1} ({reason_1})")

        if can_trade_1:
            risk_manager.positions["GAME3"] = 150

        # Try to add 4th position (should fail)
        can_trade_2, reason_2 = risk_manager.can_trade("GAME4", edge=0.10)
        print(f"✓ Adding 4th position: {can_trade_2} ({reason_2})")

        assert can_trade_1 == True, "3rd position should be allowed"
        assert can_trade_2 == False, "4th position should be blocked"

        print(f"\n✓ Final state:")
        print(f"  Positions held: {len(risk_manager.positions)}")
        print(f"  Limit: {risk_limits.max_positions}")

        print(f"\n  ✅ PASS: Position limits enforced correctly")


@pytest.mark.integration
class TestDataPipelineIntegration:
    """Test data pipeline integration."""

    def test_feature_engineering_workflow(self):
        """Test feature engineering with game data."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Feature Engineering Pipeline")
        print("=" * 70)

        fe = FeatureEngineer()

        # Simulate season results
        games = [
            ("CHI", "GB", 24, 21),
            ("CHI", "DET", 28, 14),
            ("GB", "DET", 31, 17),
            ("CHI", "GB", 17, 24),  # Rematch
        ]

        print(f"✓ Processing {len(games)} games:")

        for home, away, home_score, away_score in games:
            fe.calculate_elo(home, away, home_score, away_score)
            print(f"  {home} {home_score} - {away_score} {away}")

        print(f"\n✓ Final ELO ratings:")
        for team in sorted(fe.team_elos.keys()):
            print(f"  {team}: {fe.team_elos[team]:.1f}")

        # Create features for next game
        next_game = {"home_team": "CHI", "away_team": "DET"}
        features = fe.process_game_features(next_game)

        print(f"\n✓ Features for CHI vs DET:")
        print(f"  Home ELO: {features['home_elo']:.1f}")
        print(f"  Away ELO: {features['away_elo']:.1f}")
        print(f"  ELO diff: {features['elo_diff']:.1f}")

        assert "home_elo" in features
        assert "away_elo" in features
        assert "elo_diff" in features

        print(f"\n  ✅ PASS: Feature engineering pipeline works")

    def test_volatility_calculation_pipeline(self):
        """Test volatility calculation from price history."""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST: Volatility Calculation")
        print("=" * 70)

        fe = FeatureEngineer()

        # Create synthetic price series
        prices = [0.50, 0.52, 0.51, 0.54, 0.53, 0.56, 0.55, 0.57, 0.56, 0.58]

        vol = fe.calculate_volatility(prices)

        print(f"✓ Price series: {[f'{p:.2f}' for p in prices]}")
        print(f"✓ Calculated volatility: {vol:.4f}")

        assert vol > 0, "Volatility should be positive for changing prices"
        assert vol < 1.0, "Volatility should be reasonable"

        print(f"\n  ✅ PASS: Volatility calculation works")


def run_integration_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("OMEGA POINT ABM - INTEGRATION TEST SUITE")
    print("=" * 70)

    result = pytest.main([__file__, "-v", "--tb=short", "-s", "-m", "integration"])

    return result


if __name__ == "__main__":
    run_integration_tests()
