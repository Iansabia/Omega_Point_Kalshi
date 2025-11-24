"""
Momentum-Specific Risk Manager.

Enhanced risk controls for momentum arbitrage trading where:
- Edges can disappear quickly (20-100ms market reaction)
- Positions need quick exit strategies
- Momentum reversals are common
- Data staleness is critical

Additional controls beyond base RiskManager:
- Maximum holding time for positions
- Momentum reversal detection
- Data freshness requirements
- Rapid position unwinding
- Correlated exposure limits (same game)
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.risk.risk_manager import RiskLimits, RiskManager

logger = logging.getLogger(__name__)


@dataclass
class MomentumRiskLimits(RiskLimits):
    """
    Extended risk limits for momentum trading.

    Inherits base limits and adds momentum-specific controls.
    """

    # Momentum-specific limits
    max_holding_time_seconds: float = 300.0  # Max 5 minutes per position
    max_data_age_seconds: float = 10.0  # Reject if data >10s old
    max_correlated_exposure: float = 10000.0  # Max exposure to same game
    momentum_reversal_threshold: float = 0.05  # 5% price move against us
    max_consecutive_losses: int = 3  # Stop after 3 losses in a row
    cooldown_period_seconds: float = 60.0  # Wait 60s after max losses


class MomentumRiskManager(RiskManager):
    """
    Enhanced risk manager for momentum arbitrage trading.

    Adds momentum-specific controls on top of base risk management.
    """

    def __init__(self, limits: MomentumRiskLimits = None):
        """
        Initialize momentum risk manager.

        Args:
            limits: Momentum-specific risk limits
        """
        super().__init__(limits or MomentumRiskLimits())
        self.limits: MomentumRiskLimits = self.limits  # Type hint

        # Track open positions with entry time
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        # ticker -> {'entry_time', 'entry_price', 'quantity', 'side', 'game_id'}

        # Track consecutive losses
        self.consecutive_losses = 0
        self.last_loss_time: Optional[float] = None

        # Track per-game exposure
        self.game_exposure: Dict[str, float] = {}  # game_id -> total_exposure

        logger.info(f"MomentumRiskManager initialized:")
        logger.info(f"  Max Holding Time: {self.limits.max_holding_time_seconds}s")
        logger.info(f"  Max Data Age: {self.limits.max_data_age_seconds}s")
        logger.info(f"  Momentum Reversal Threshold: {self.limits.momentum_reversal_threshold:.1%}")
        logger.info(f"  Max Consecutive Losses: {self.limits.max_consecutive_losses}")

    def can_trade(
        self,
        ticker: str,
        edge: float,
        data_age: Optional[float] = None,
        game_id: Optional[str] = None,
        position_value: float = 0,
    ) -> Tuple[bool, str]:
        """
        Check if trade is allowed with momentum-specific checks.

        Args:
            ticker: Market ticker
            edge: Expected edge
            data_age: Age of data in seconds (optional)
            game_id: Game ID for correlation tracking (optional)
            position_value: Value of proposed position (optional)

        Returns:
            (allowed, reason) tuple
        """
        # Base risk checks
        allowed, reason = super().can_trade(ticker, edge)
        if not allowed:
            return False, reason

        # Check data freshness
        if data_age is not None and data_age > self.limits.max_data_age_seconds:
            return False, f"Data too stale ({data_age:.1f}s > {self.limits.max_data_age_seconds}s)"

        # Check consecutive losses cooldown
        if self.consecutive_losses >= self.limits.max_consecutive_losses:
            if self.last_loss_time is None:
                return False, f"Max consecutive losses reached ({self.consecutive_losses})"

            cooldown_elapsed = time.time() - self.last_loss_time
            if cooldown_elapsed < self.limits.cooldown_period_seconds:
                remaining = self.limits.cooldown_period_seconds - cooldown_elapsed
                return False, f"In cooldown period ({remaining:.1f}s remaining)"

            # Cooldown expired, reset counter
            logger.info("✅ Cooldown period expired, resetting consecutive losses")
            self.consecutive_losses = 0
            self.last_loss_time = None

        # Check correlated exposure (same game)
        if game_id is not None:
            current_exposure = self.game_exposure.get(game_id, 0)
            if current_exposure + position_value > self.limits.max_correlated_exposure:
                return (
                    False,
                    f"Correlated exposure limit (${current_exposure:.0f} + ${position_value:.0f} > ${self.limits.max_correlated_exposure:.0f})",
                )

        return True, "Trade allowed"

    def check_position_exit(self, ticker: str, current_price: float) -> Tuple[bool, str]:
        """
        Check if position should be exited due to momentum controls.

        Args:
            ticker: Market ticker
            current_price: Current market price

        Returns:
            (should_exit, reason) tuple
        """
        if ticker not in self.open_positions:
            return False, "No position"

        position = self.open_positions[ticker]
        entry_time = position["entry_time"]
        entry_price = position["entry_price"]
        side = position["side"]

        # Check holding time limit
        holding_time = time.time() - entry_time
        if holding_time > self.limits.max_holding_time_seconds:
            return True, f"Max holding time exceeded ({holding_time:.1f}s)"

        # Check momentum reversal
        if side == "BUY":
            price_change = (current_price - entry_price) / entry_price
        else:  # SELL
            price_change = (entry_price - current_price) / entry_price

        if price_change < -self.limits.momentum_reversal_threshold:
            return True, f"Momentum reversal detected ({price_change:+.1%})"

        return False, "Position ok"

    def open_position(self, ticker: str, side: str, quantity: float, entry_price: float, game_id: Optional[str] = None):
        """
        Record position opening.

        Args:
            ticker: Market ticker
            side: 'BUY' or 'SELL'
            quantity: Position size
            entry_price: Entry price
            game_id: Game ID (optional)
        """
        self.open_positions[ticker] = {
            "entry_time": time.time(),
            "entry_price": entry_price,
            "quantity": quantity,
            "side": side,
            "game_id": game_id,
        }

        # Track game exposure
        if game_id:
            position_value = quantity * entry_price
            self.game_exposure[game_id] = self.game_exposure.get(game_id, 0) + position_value

        logger.info(f"📈 Position opened: {ticker} {side} {quantity} @ ${entry_price:.2f}")

    def close_position(self, ticker: str, exit_price: float, realized_pnl: float):
        """
        Record position closing.

        Args:
            ticker: Market ticker
            exit_price: Exit price
            realized_pnl: Realized P&L
        """
        if ticker not in self.open_positions:
            logger.warning(f"Attempted to close non-existent position: {ticker}")
            return

        position = self.open_positions[ticker]
        entry_price = position["entry_price"]
        holding_time = time.time() - position["entry_time"]
        game_id = position.get("game_id")

        # Update game exposure
        if game_id:
            position_value = position["quantity"] * entry_price
            self.game_exposure[game_id] = max(0, self.game_exposure.get(game_id, 0) - position_value)

        # Track consecutive losses
        if realized_pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = time.time()
            logger.warning(f"❌ Loss on {ticker}: ${realized_pnl:.2f} (consecutive: {self.consecutive_losses})")
        else:
            self.consecutive_losses = 0  # Reset on win
            self.last_loss_time = None
            logger.info(f"✅ Win on {ticker}: ${realized_pnl:.2f} (held {holding_time:.1f}s)")

        # Remove position
        del self.open_positions[ticker]

        logger.info(f"📉 Position closed: {ticker} @ ${exit_price:.2f}")

    def get_open_positions(self) -> Dict[str, Dict[str, Any]]:
        """Get all open positions."""
        return self.open_positions.copy()

    def get_game_exposure(self, game_id: str) -> float:
        """Get total exposure to a specific game."""
        return self.game_exposure.get(game_id, 0)

    def get_stats(self) -> Dict[str, Any]:
        """Get risk manager statistics."""
        base_stats = super().get_risk_metrics()

        momentum_stats = {
            "open_positions": len(self.open_positions),
            "consecutive_losses": self.consecutive_losses,
            "in_cooldown": (
                self.consecutive_losses >= self.limits.max_consecutive_losses
                and self.last_loss_time is not None
                and (time.time() - self.last_loss_time) < self.limits.cooldown_period_seconds
            ),
            "tracked_games": len(self.game_exposure),
            "total_game_exposure": sum(self.game_exposure.values()),
        }

        return {**base_stats, **momentum_stats}

    def reset_stats(self):
        """Reset statistics."""
        super().reset_game()
        self.consecutive_losses = 0
        self.last_loss_time = None
        logger.info("📊 Momentum risk stats reset")


# Example usage
def main():
    """Example: Momentum risk management."""
    print("\n" + "=" * 60)
    print("Momentum Risk Manager Example")
    print("=" * 60)

    # Initialize with strict limits
    limits = MomentumRiskLimits(
        max_holding_time_seconds=300,  # 5 minutes max
        max_data_age_seconds=10,  # 10 seconds max data age
        max_correlated_exposure=5000,  # $5k max per game
        momentum_reversal_threshold=0.05,  # 5% adverse move
        max_consecutive_losses=3,
        cooldown_period_seconds=60,
    )

    risk_mgr = MomentumRiskManager(limits)

    game_id = "sr:match:12345"
    ticker = "KXMVENFLSINGLEGAME-S2025-BAL-KC"

    # Update capital first
    risk_mgr.update_capital(10000.0)

    print("\n📊 Demonstrating momentum-specific controls...")

    # Open position (simulated trade)
    print("\n1. Opening position...")
    risk_mgr.open_position(ticker=ticker, side="BUY", quantity=100, entry_price=0.75, game_id=game_id)

    # Check if should exit (after some time)
    time.sleep(1)
    print("\n2. Checking position after 1 second...")
    should_exit, exit_reason = risk_mgr.check_position_exit(ticker=ticker, current_price=0.72)

    if should_exit:
        print(f"   ⚠️  Should exit: {exit_reason}")
    else:
        print(f"   ✅ Position ok: {exit_reason}")

    # Close position (simulated loss)
    print("\n3. Closing position with loss...")
    risk_mgr.close_position(ticker=ticker, exit_price=0.72, realized_pnl=-3.0)

    # Get stats
    stats = risk_mgr.get_stats()
    print(f"\n📊 Risk Manager Stats:")
    print(f"   Open Positions: {stats['open_positions']}")
    print(f"   Consecutive Losses: {stats['consecutive_losses']}")
    print(f"   In Cooldown: {stats['in_cooldown']}")
    print(f"   Game Exposure: ${stats['total_game_exposure']:.0f}")

    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    main()
