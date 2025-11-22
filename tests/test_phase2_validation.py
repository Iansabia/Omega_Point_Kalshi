"""
Phase 2 Validation Tests: Mathematical Foundations
Tests for jump-diffusion models, sentiment quantification, and microstructure models
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.jump_diffusion import JumpDiffusionModel
from src.models.sentiment_model import SentimentModel, detect_herding
from src.models.microstructure import MarketMicrostructure
from src.models.behavioral_biases import BehavioralBiases


class TestJumpDiffusionModel:
    """Test 2.1: Jump-Diffusion Model Validation"""

    def test_simulate_price_paths(self):
        """Validate: Simulate 1000 price paths, verify statistical properties"""
        model = JumpDiffusionModel()
        n_paths = 1000
        n_steps = 100

        paths = []
        for _ in range(n_paths):
            path = model.simulate_path(
                initial_price=0.5,
                n_steps=n_steps,
                dt=1/252  # Daily steps
            )
            paths.append(path)

        paths = np.array(paths)

        # Calculate returns
        returns = np.diff(paths, axis=1) / paths[:, :-1]
        returns_flat = returns.flatten()

        # Remove any NaN or inf values
        returns_flat = returns_flat[np.isfinite(returns_flat)]

        # Check for fat tails (kurtosis > 3 indicates heavy tails)
        from scipy.stats import kurtosis
        kurt = kurtosis(returns_flat, fisher=False)  # Pearson kurtosis

        print(f"\n=== Jump-Diffusion Model Statistics ===")
        print(f"Number of paths: {n_paths}")
        print(f"Steps per path: {n_steps}")
        print(f"Mean return: {np.mean(returns_flat):.6f}")
        print(f"Std return: {np.std(returns_flat):.6f}")
        print(f"Kurtosis: {kurt:.4f} (target: > 3.0 for fat tails)")
        print(f"Min price: {np.min(paths):.4f}")
        print(f"Max price: {np.max(paths):.4f}")

        # Assertion: Kurtosis should be > 3 for jump-diffusion
        assert kurt > 3.0, f"Kurtosis {kurt:.4f} should be > 3.0 for fat tails"

        # Prices should stay in reasonable bounds [0, 1] for prediction markets
        assert np.all(paths >= 0), "Prices should be non-negative"
        # Allow some flexibility for extreme jumps
        assert np.percentile(paths, 99) <= 1.2, "99th percentile should be reasonable"

    def test_calibration_methods(self):
        """Test calibration methods work without errors"""
        model = JumpDiffusionModel()

        # Generate synthetic data
        true_params = {
            'sigma': 0.35,
            'lambda_base': 5,
            'eta_up': 20,
            'eta_down': 12,
            'p_up': 0.4
        }

        # Simulate price path
        prices = model.simulate_path(initial_price=0.5, n_steps=500, dt=1/252)

        # Test that calibration runs (don't require exact recovery)
        try:
            calibrated_params = model.calibrate(prices, method='mle')
            print(f"\n=== Calibration Test ===")
            print(f"Calibrated sigma: {calibrated_params.get('sigma', 'N/A'):.4f}")
            print(f"True sigma: {true_params['sigma']:.4f}")
            assert calibrated_params is not None
            assert 'sigma' in calibrated_params
        except Exception as e:
            pytest.fail(f"Calibration failed: {e}")


class TestSentimentModel:
    """Test 2.2: Sentiment Quantification System"""

    def test_sentiment_scoring(self):
        """Test sentiment analysis on sample texts"""
        model = SentimentModel()

        # Test cases
        positive_text = "The team looks amazing! They're going to dominate this game. Easy win!"
        negative_text = "This team is terrible. They have no chance. Complete disaster."
        neutral_text = "The game starts at 8pm. Weather is clear."

        pos_score = model.analyze_sentiment(positive_text)
        neg_score = model.analyze_sentiment(negative_text)
        neu_score = model.analyze_sentiment(neutral_text)

        print(f"\n=== Sentiment Analysis ===")
        print(f"Positive text score: {pos_score:.4f}")
        print(f"Negative text score: {neg_score:.4f}")
        print(f"Neutral text score: {neu_score:.4f}")

        # Positive should be > 0, negative should be < 0
        assert pos_score > 0, "Positive text should have positive score"
        assert neg_score < 0, "Negative text should have negative score"
        assert abs(neu_score) < abs(pos_score), "Neutral should be less extreme"

    def test_panic_coefficient(self):
        """Test panic coefficient calculation"""
        model = SentimentModel()

        # High volatility, negative sentiment should increase panic
        panic_high = model.calculate_panic_coefficient(
            csad=0.05,
            volatility=0.4,
            sentiment=-0.8
        )

        # Low volatility, positive sentiment should decrease panic
        panic_low = model.calculate_panic_coefficient(
            csad=0.01,
            volatility=0.1,
            sentiment=0.5
        )

        print(f"\n=== Panic Coefficient ===")
        print(f"High panic scenario: {panic_high:.4f}")
        print(f"Low panic scenario: {panic_low:.4f}")

        assert panic_high > panic_low, "High volatility/negative sentiment should increase panic"
        assert panic_high > 1.0, "Panic coefficient should be elevated in crisis"

    def test_herding_detection(self):
        """Test CSAD calculation for herding detection"""
        # Create returns with herding behavior (low dispersion during market moves)
        n_assets = 100
        n_periods = 50

        # Generate returns with some herding
        market_returns = np.random.randn(n_periods) * 0.02
        individual_returns = np.outer(market_returns, np.ones(n_assets))
        individual_returns += np.random.randn(n_periods, n_assets) * 0.005  # Small idiosyncratic

        csad_values = []
        for t in range(n_periods):
            csad = detect_herding(individual_returns[t, :], market_returns[t])
            csad_values.append(csad)

        mean_csad = np.mean(csad_values)
        print(f"\n=== Herding Detection ===")
        print(f"Mean CSAD: {mean_csad:.6f}")
        print(f"CSAD range: [{np.min(csad_values):.6f}, {np.max(csad_values):.6f}]")

        # CSAD should be relatively small indicating herding
        assert mean_csad < 0.02, "CSAD should indicate herding behavior"


class TestMarketMicrostructure:
    """Test 2.3: Market Microstructure Models"""

    def test_kyles_lambda(self):
        """Test Kyle's lambda calculation"""
        ms = MarketMicrostructure()

        # Calculate lambda for different market conditions
        lambda_illiquid = ms.calculate_kyles_lambda(
            sigma_v=100,  # High fundamental volatility
            sigma_u=10    # Low noise trader volatility
        )

        lambda_liquid = ms.calculate_kyles_lambda(
            sigma_v=10,   # Low fundamental volatility
            sigma_u=100   # High noise trader volatility
        )

        print(f"\n=== Kyle's Lambda ===")
        print(f"Illiquid market lambda: {lambda_illiquid:.4f}")
        print(f"Liquid market lambda: {lambda_liquid:.4f}")

        # Illiquid markets should have higher price impact
        assert lambda_illiquid > lambda_liquid, "Illiquid markets should have higher lambda"
        assert lambda_illiquid > 0, "Lambda should be positive"

    def test_bid_ask_spread(self):
        """Test bid-ask spread decomposition"""
        ms = MarketMicrostructure()

        spread = ms.calculate_spread(
            order_processing=0.0005,
            inventory=0.001,
            adverse_selection=0.002
        )

        print(f"\n=== Bid-Ask Spread ===")
        print(f"Total spread: {spread:.4f}")
        print(f"Components: Processing=0.0005, Inventory=0.001, Adverse=0.002")

        assert spread == 0.0035, "Spread should sum components"
        assert spread > 0, "Spread should be positive"

    def test_price_impact(self):
        """Test price impact models"""
        ms = MarketMicrostructure()

        # Square root law
        impact_small = ms.calculate_price_impact(quantity=100, kyle_lambda=1.5)
        impact_large = ms.calculate_price_impact(quantity=400, kyle_lambda=1.5)

        print(f"\n=== Price Impact ===")
        print(f"Impact for Q=100: {impact_small:.4f}")
        print(f"Impact for Q=400: {impact_large:.4f}")
        print(f"Ratio: {impact_large/impact_small:.4f} (expected: 2.0 for square root)")

        # Should follow square root law: 4x quantity = 2x impact
        assert abs(impact_large / impact_small - 2.0) < 0.1, "Should follow square root law"

    def test_almgren_chriss_impact(self):
        """Test Almgren-Chriss market impact"""
        ms = MarketMicrostructure()

        impact = ms.calculate_almgren_chriss_impact(
            quantity=1000,
            daily_volume=50000,
            volatility=0.3,
            eta=0.314,
            gamma=0.142
        )

        print(f"\n=== Almgren-Chriss Impact ===")
        print(f"Impact: {impact:.6f}")
        print(f"Parameters: Q=1000, V=50000, σ=0.3, η=0.314, γ=0.142")

        assert impact > 0, "Impact should be positive"
        assert impact < 1.0, "Impact should be reasonable"


class TestBehavioralBiases:
    """Test 2.4: Behavioral Bias Implementation"""

    def test_recency_bias(self):
        """Test recency bias weighting"""
        biases = BehavioralBiases()

        # Recent data should be weighted more
        recent_weight = biases.recency_weight

        print(f"\n=== Recency Bias ===")
        print(f"Recent data weight: {recent_weight:.2f}")
        print(f"Optimal weight should be ~0.3, biased weight is {recent_weight:.2f}")

        assert recent_weight > 0.5, "Should overweight recent data"
        assert recent_weight < 1.0, "Should not completely ignore history"

    def test_homer_bias(self):
        """Test loyalty/homer bias strength"""
        biases = BehavioralBiases()

        # Generate homer bias strength
        loyalty_strength = biases.generate_homer_bias()

        print(f"\n=== Homer Bias ===")
        print(f"Loyalty strength: {loyalty_strength:.2f}")

        assert 0.5 <= loyalty_strength <= 0.9, "Loyalty should be in expected range"

    def test_herding_coefficient(self):
        """Test herding behavior coefficient"""
        biases = BehavioralBiases()

        herding_coef = biases.herding_coefficient

        print(f"\n=== Herding Coefficient ===")
        print(f"Herding coefficient: {herding_coef:.2f}")

        assert 0.1 <= herding_coef <= 0.3, "Herding coefficient in expected range"

    def test_sentiment_driven_adjustment(self):
        """Test sentiment-driven value adjustment"""
        biases = BehavioralBiases()

        fundamental_value = 0.6
        sentiment_effect = 0.1
        herding_effect = 0.05

        perceived_value = biases.calculate_perceived_value(
            fundamental_value,
            sentiment_effect,
            herding_effect
        )

        print(f"\n=== Sentiment-Driven Adjustment ===")
        print(f"Fundamental value: {fundamental_value:.2f}")
        print(f"Sentiment effect: {sentiment_effect:.2f}")
        print(f"Herding effect: {herding_effect:.2f}")
        print(f"Perceived value: {perceived_value:.2f}")

        assert perceived_value != fundamental_value, "Biases should adjust value"
        assert perceived_value > fundamental_value, "Positive sentiment should increase value"


def run_all_phase2_validations():
    """Run all Phase 2 validation tests and generate report"""

    print("\n" + "="*70)
    print("PHASE 2 VALIDATION: MATHEMATICAL FOUNDATIONS")
    print("="*70)

    # Run pytest programmatically
    pytest_args = [
        __file__,
        '-v',
        '--tb=short',
        '-s'  # Show print statements
    ]

    result = pytest.main(pytest_args)

    return result


if __name__ == "__main__":
    run_all_phase2_validations()
