"""
Risk Management Framework for Kalshi Trading.

Implements comprehensive risk controls to maximize risk-adjusted returns:
- Position limits
- Stop losses
- Kelly Criterion position sizing
- Drawdown controls
- Circuit breakers
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RiskLimits:
    """Risk limit configuration."""

    # Position limits
    max_position_size: float = 1000.0  # Max $ per position
    max_portfolio_exposure: float = 5000.0  # Max total $ exposed
    max_positions: int = 5  # Max concurrent positions

    # Trade frequency limits
    max_trades_per_game: int = 50  # Prevent over-trading
    min_edge_threshold: float = 0.05  # Minimum 5% edge to trade
    trade_probability: float = 0.15  # Only trade 15% of opportunities

    # Risk management
    max_loss_per_trade: float = 200.0  # Stop loss per trade
    max_daily_drawdown: float = 0.10  # Max 10% daily drawdown
    max_total_drawdown: float = 0.20  # Max 20% total drawdown

    # Kelly Criterion
    use_kelly: bool = True
    kelly_fraction: float = 0.25  # Use 1/4 Kelly for safety
    min_position_size: float = 10.0  # Minimum trade size

    # Circuit breakers
    max_consecutive_losses: int = 5
    circuit_breaker_cooldown: int = 10  # Games to pause after breaker


class RiskManager:
    """
    Manages risk for trading agents.

    Enforces position limits, stop losses, and trade frequency controls.
    """

    def __init__(self, limits: RiskLimits = None):
        """Initialize risk manager."""
        self.limits = limits or RiskLimits()

        # Track state
        self.positions: Dict[str, float] = {}  # ticker -> position size
        self.position_entry_prices: Dict[str, float] = {}  # ticker -> entry price
        self.trades_this_game: int = 0
        self.consecutive_losses: int = 0
        self.circuit_breaker_active: bool = False
        self.circuit_breaker_countdown: int = 0

        # Performance tracking
        self.peak_capital: float = 10000.0
        self.current_capital: float = 10000.0
        self.game_start_capital: float = 10000.0

    def reset_game(self, starting_capital: float):
        """Reset for new game."""
        self.trades_this_game = 0
        self.game_start_capital = starting_capital
        self.current_capital = starting_capital

        # Update peak if new high
        if starting_capital > self.peak_capital:
            self.peak_capital = starting_capital

        # Check circuit breaker
        if self.circuit_breaker_active:
            self.circuit_breaker_countdown -= 1
            if self.circuit_breaker_countdown <= 0:
                self.circuit_breaker_active = False
                self.consecutive_losses = 0

    def update_capital(self, capital: float):
        """Update current capital."""
        self.current_capital = capital
        if capital > self.peak_capital:
            self.peak_capital = capital

    def check_drawdown_limits(self) -> Tuple[bool, str]:
        """
        Check if drawdown limits exceeded.

        Returns:
            (allowed, reason)
        """
        # Check daily drawdown
        daily_dd = (self.game_start_capital - self.current_capital) / self.game_start_capital
        if daily_dd > self.limits.max_daily_drawdown:
            return False, f"Daily drawdown {daily_dd:.2%} exceeds limit {self.limits.max_daily_drawdown:.2%}"

        # Check total drawdown
        total_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        if total_dd > self.limits.max_total_drawdown:
            return False, f"Total drawdown {total_dd:.2%} exceeds limit {self.limits.max_total_drawdown:.2%}"

        return True, "OK"

    def can_trade(self, ticker: str, edge: float) -> Tuple[bool, str]:
        """
        Check if trade is allowed.

        Args:
            ticker: Market ticker
            edge: Estimated edge (expected value %)

        Returns:
            (allowed, reason)
        """
        # Check circuit breaker
        if self.circuit_breaker_active:
            return False, "Circuit breaker active"

        # Check drawdown limits
        allowed, reason = self.check_drawdown_limits()
        if not allowed:
            return False, reason

        # Check trade frequency
        if self.trades_this_game >= self.limits.max_trades_per_game:
            return False, f"Max trades per game ({self.limits.max_trades_per_game}) reached"

        # Check minimum edge
        if edge < self.limits.min_edge_threshold:
            return False, f"Edge {edge:.2%} below threshold {self.limits.min_edge_threshold:.2%}"

        # Probabilistic trading (prevent over-trading)
        if np.random.random() > self.limits.trade_probability:
            return False, "Trade probability filter"

        # Check max positions
        if len(self.positions) >= self.limits.max_positions and ticker not in self.positions:
            return False, f"Max positions ({self.limits.max_positions}) reached"

        return True, "OK"

    def calculate_position_size(
        self,
        edge: float,
        win_prob: float,
        avg_win: float,
        avg_loss: float,
        available_capital: float
    ) -> float:
        """
        Calculate optimal position size using Kelly Criterion.

        Args:
            edge: Expected value as fraction (e.g., 0.05 for 5%)
            win_prob: Probability of winning
            avg_win: Average win amount
            avg_loss: Average loss amount
            available_capital: Capital available for trading

        Returns:
            Position size in dollars
        """
        if not self.limits.use_kelly:
            # Simple fixed fraction
            return min(self.limits.max_position_size, available_capital * 0.02)

        # Kelly formula: f* = (p * b - q) / b
        # where p = win prob, q = loss prob, b = win/loss ratio
        if avg_loss == 0:
            avg_loss = 1.0  # Avoid division by zero

        b = avg_win / avg_loss  # Win/loss ratio
        q = 1 - win_prob

        kelly_fraction = (win_prob * b - q) / b

        # Apply safety fraction (typically 1/4 or 1/2 Kelly)
        kelly_fraction *= self.limits.kelly_fraction

        # Ensure positive and reasonable
        kelly_fraction = max(0, min(kelly_fraction, 0.20))  # Cap at 20%

        # Calculate position size
        position_size = available_capital * kelly_fraction

        # Apply limits
        position_size = max(self.limits.min_position_size, position_size)
        position_size = min(self.limits.max_position_size, position_size)

        # Check portfolio exposure
        total_exposure = sum(abs(p) for p in self.positions.values())
        available_exposure = self.limits.max_portfolio_exposure - total_exposure

        if available_exposure <= 0:
            return 0

        position_size = min(position_size, available_exposure)

        return position_size

    def record_trade(
        self,
        ticker: str,
        side: str,
        quantity: float,
        price: float,
        is_entry: bool = True
    ):
        """
        Record a trade.

        Args:
            ticker: Market ticker
            side: 'buy' or 'sell'
            quantity: Position size in $
            price: Entry/exit price
            is_entry: True if opening position, False if closing
        """
        self.trades_this_game += 1

        if is_entry:
            self.positions[ticker] = quantity
            self.position_entry_prices[ticker] = price
        else:
            # Closing position - check if loss
            if ticker in self.position_entry_prices:
                entry_price = self.position_entry_prices[ticker]
                pnl = (price - entry_price) * quantity if side == 'buy' else (entry_price - price) * quantity

                if pnl < 0:
                    self.consecutive_losses += 1

                    # Check circuit breaker
                    if self.consecutive_losses >= self.limits.max_consecutive_losses:
                        self.circuit_breaker_active = True
                        self.circuit_breaker_countdown = self.limits.circuit_breaker_cooldown
                else:
                    self.consecutive_losses = 0

            # Remove position
            if ticker in self.positions:
                del self.positions[ticker]
            if ticker in self.position_entry_prices:
                del self.position_entry_prices[ticker]

    def should_stop_loss(self, ticker: str, current_price: float) -> bool:
        """
        Check if stop loss should be triggered.

        Args:
            ticker: Market ticker
            current_price: Current market price

        Returns:
            True if stop loss should trigger
        """
        if ticker not in self.positions or ticker not in self.position_entry_prices:
            return False

        entry_price = self.position_entry_prices[ticker]
        position_size = self.positions[ticker]

        # Calculate current P&L
        pnl = (current_price - entry_price) * position_size

        # Check if loss exceeds limit
        if pnl < -self.limits.max_loss_per_trade:
            return True

        return False

    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics."""
        total_exposure = sum(abs(p) for p in self.positions.values())
        total_dd = (self.peak_capital - self.current_capital) / self.peak_capital if self.peak_capital > 0 else 0
        daily_dd = (self.game_start_capital - self.current_capital) / self.game_start_capital if self.game_start_capital > 0 else 0

        return {
            'total_exposure': total_exposure,
            'num_positions': len(self.positions),
            'trades_this_game': self.trades_this_game,
            'total_drawdown': total_dd,
            'daily_drawdown': daily_dd,
            'consecutive_losses': self.consecutive_losses,
            'circuit_breaker_active': self.circuit_breaker_active,
            'exposure_utilization': total_exposure / self.limits.max_portfolio_exposure if self.limits.max_portfolio_exposure > 0 else 0
        }


class PortfolioRiskManager:
    """
    Portfolio-level risk management.

    Coordinates risk across all agents and ensures portfolio-wide limits.
    """

    def __init__(self, limits: RiskLimits = None):
        """Initialize portfolio risk manager."""
        self.limits = limits or RiskLimits()
        self.agent_managers: Dict[str, RiskManager] = {}

    def get_agent_manager(self, agent_id: str) -> RiskManager:
        """Get or create risk manager for agent."""
        if agent_id not in self.agent_managers:
            self.agent_managers[agent_id] = RiskManager(self.limits)
        return self.agent_managers[agent_id]

    def reset_game(self, starting_capital: float):
        """Reset all agents for new game."""
        for manager in self.agent_managers.values():
            manager.reset_game(starting_capital)

    def get_portfolio_metrics(self) -> Dict[str, any]:
        """Get portfolio-wide risk metrics."""
        total_positions = sum(len(m.positions) for m in self.agent_managers.values())
        total_trades = sum(m.trades_this_game for m in self.agent_managers.values())
        total_exposure = sum(
            sum(abs(p) for p in m.positions.values())
            for m in self.agent_managers.values()
        )

        active_breakers = sum(1 for m in self.agent_managers.values() if m.circuit_breaker_active)

        return {
            'total_positions': total_positions,
            'total_trades': total_trades,
            'total_exposure': total_exposure,
            'active_circuit_breakers': active_breakers,
            'num_agents': len(self.agent_managers),
            'avg_trades_per_agent': total_trades / len(self.agent_managers) if self.agent_managers else 0
        }
