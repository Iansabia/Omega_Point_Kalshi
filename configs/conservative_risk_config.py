"""
Conservative Risk Configuration for Maximum Risk-Adjusted Returns.

Designed to achieve:
- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Profit Factor > 2.0
- Win Rate > 55%
- Minimal over-trading
"""

from src.risk.risk_manager import RiskLimits

# Conservative risk limits for all agents
CONSERVATIVE_RISK_LIMITS = RiskLimits(
    # Position limits
    max_position_size=800.0,  # Max $800 per position
    max_portfolio_exposure=4000.0,  # Max $4,000 total exposure
    max_positions=5,  # Max 5 concurrent positions
    # Trade frequency limits (CRITICAL for preventing over-trading)
    max_trades_per_game=100,  # Max 100 trades per game (reduced from unlimited)
    min_edge_threshold=0.03,  # Require 3% edge minimum (loosened from 8%)
    trade_probability=0.25,  # Trade 25% of opportunities (increased from 10%)
    # Risk management
    max_loss_per_trade=200.0,  # Stop loss at $200 per trade
    max_daily_drawdown=0.10,  # Max 10% drawdown per game
    max_total_drawdown=0.20,  # Max 20% total drawdown
    # Kelly Criterion for optimal sizing
    use_kelly=True,
    kelly_fraction=0.25,  # Use 1/4 Kelly
    min_position_size=50.0,  # Minimum $50 trade
    # Circuit breakers
    max_consecutive_losses=5,  # Pause after 5 losses
    circuit_breaker_cooldown=3,  # Skip 3 games after circuit breaker
)

# Aggressive risk limits (for testing higher risk tolerance)
AGGRESSIVE_RISK_LIMITS = RiskLimits(
    max_position_size=1000.0,
    max_portfolio_exposure=5000.0,
    max_positions=5,
    max_trades_per_game=50,
    min_edge_threshold=0.05,
    trade_probability=0.15,
    max_loss_per_trade=200.0,
    max_daily_drawdown=0.10,
    max_total_drawdown=0.20,
    use_kelly=True,
    kelly_fraction=0.25,
    min_position_size=10.0,
    max_consecutive_losses=5,
    circuit_breaker_cooldown=10,
)

# Ultra-conservative limits (for real money trading)
ULTRA_CONSERVATIVE_RISK_LIMITS = RiskLimits(
    max_position_size=250.0,  # Max $250 per position
    max_portfolio_exposure=1000.0,  # Max $1,000 total exposure
    max_positions=2,  # Max 2 concurrent positions
    max_trades_per_game=10,  # Max 10 trades per game
    min_edge_threshold=0.12,  # Require 12% edge minimum
    trade_probability=0.05,  # Only trade 5% of opportunities
    max_loss_per_trade=100.0,  # Stop loss at $100
    max_daily_drawdown=0.05,  # Max 5% drawdown per game
    max_total_drawdown=0.10,  # Max 10% total drawdown
    use_kelly=True,
    kelly_fraction=0.15,  # Use 1/6 Kelly (ultra conservative)
    min_position_size=50.0,  # Minimum $50 trade
    max_consecutive_losses=3,  # Pause after 3 losses
    circuit_breaker_cooldown=10,  # Skip 10 games after circuit breaker
)

# Optimized agent configuration (fewer agents = less over-trading)
CONSERVATIVE_AGENT_CONFIG = {
    "noise_trader": {"count": 3, "wealth": 1000, "risk_limits": CONSERVATIVE_RISK_LIMITS},  # Reduced from 30
    "informed_trader": {
        "count": 3,  # Reduced from 10
        "wealth": 10000,
        "information_quality": 0.75,  # Reduced from 0.8 (more conservative)
        "risk_limits": CONSERVATIVE_RISK_LIMITS,
    },
    "market_maker": {
        "count": 1,  # Reduced from 2
        "wealth": 100000,
        "risk_param": 0.15,  # Increased from 0.1 (wider spreads = more conservative)
        "risk_limits": CONSERVATIVE_RISK_LIMITS,
    },
    "momentum_trader": {"count": 1, "wealth": 5000, "lookback": 10, "risk_limits": CONSERVATIVE_RISK_LIMITS},
    "contrarian_trader": {"count": 1, "wealth": 5000, "mean_reversion": 0.3, "risk_limits": CONSERVATIVE_RISK_LIMITS},
    "value_trader": {
        "count": 1,
        "wealth": 8000,
        "edge_threshold": 0.10,  # Higher threshold
        "risk_limits": CONSERVATIVE_RISK_LIMITS,
    },
}

# Aggressive agent configuration (for comparison)
AGGRESSIVE_AGENT_CONFIG = {
    "noise_trader": {"count": 5, "wealth": 1000, "risk_limits": AGGRESSIVE_RISK_LIMITS},
    "informed_trader": {"count": 5, "wealth": 10000, "information_quality": 0.8, "risk_limits": AGGRESSIVE_RISK_LIMITS},
    "market_maker": {"count": 1, "wealth": 100000, "risk_param": 0.1, "risk_limits": AGGRESSIVE_RISK_LIMITS},
    "momentum_trader": {"count": 2, "wealth": 5000, "lookback": 5, "risk_limits": AGGRESSIVE_RISK_LIMITS},
    "contrarian_trader": {"count": 2, "wealth": 5000, "mean_reversion": 0.5, "risk_limits": AGGRESSIVE_RISK_LIMITS},
    "value_trader": {"count": 2, "wealth": 8000, "edge_threshold": 0.05, "risk_limits": AGGRESSIVE_RISK_LIMITS},
}

# Ultra-conservative for real money
ULTRA_CONSERVATIVE_AGENT_CONFIG = {
    "informed_trader": {
        "count": 2,  # Only use informed traders
        "wealth": 10000,
        "information_quality": 0.70,  # Conservative information quality
        "risk_limits": ULTRA_CONSERVATIVE_RISK_LIMITS,
    },
    "value_trader": {
        "count": 1,
        "wealth": 8000,
        "edge_threshold": 0.15,  # Very high threshold
        "risk_limits": ULTRA_CONSERVATIVE_RISK_LIMITS,
    },
}
