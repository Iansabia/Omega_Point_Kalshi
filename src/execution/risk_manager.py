import logging
from typing import Any, Dict, Optional

from src.execution.audit_log import audit_logger

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Enforces pre-trade risk checks.
    """

    def __init__(self, max_position_size: int = 100, max_daily_loss: float = 1000.0, user_id: Optional[str] = None):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.current_positions = {}  # ticker -> quantity
        self.daily_pnl = 0.0
        self.user_id = user_id or "unknown"

    def check_risk(self, signal: Dict[str, Any]) -> bool:
        """
        Approve or reject a signal based on risk limits.
        """
        ticker = signal.get("ticker")
        count = signal.get("count", 0)
        price = signal.get("price", 0)  # In cents

        # 1. Position Limit Check
        current_pos = self.current_positions.get(ticker, 0)
        new_position = current_pos + count
        if new_position > self.max_position_size:
            logger.warning(f"Risk Reject: Position limit exceeded for {ticker}")
            # Log to audit trail
            audit_logger.log_risk_violation(
                user_id=self.user_id,
                violation_type="position_limit_exceeded",
                current_value=new_position,
                limit=self.max_position_size,
                details={
                    "ticker": ticker,
                    "current_position": current_pos,
                    "requested_count": count,
                    "new_position": new_position,
                    "price": price
                }
            )
            return False

        # 2. Notional Value Check (e.g., max order value)
        notional = count * (price / 100.0)
        max_order_value = 500.0
        if notional > max_order_value:
            logger.warning(f"Risk Reject: Order value ${notional} exceeds limit")
            # Log to audit trail
            audit_logger.log_risk_violation(
                user_id=self.user_id,
                violation_type="order_value_exceeded",
                current_value=notional,
                limit=max_order_value,
                details={
                    "ticker": ticker,
                    "count": count,
                    "price": price,
                    "notional_value": notional
                }
            )
            return False

        # 3. Daily Loss Check (simplified)
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning("Risk Reject: Max daily loss exceeded")
            # Log to audit trail
            audit_logger.log_risk_violation(
                user_id=self.user_id,
                violation_type="daily_loss_exceeded",
                current_value=abs(self.daily_pnl),
                limit=self.max_daily_loss,
                details={
                    "ticker": ticker,
                    "daily_pnl": self.daily_pnl,
                    "max_daily_loss": self.max_daily_loss
                }
            )
            return False

        return True

    def update_position(self, ticker: str, quantity: int, pnl_change: float = 0.0):
        """
        Update state after execution.
        """
        self.current_positions[ticker] = self.current_positions.get(ticker, 0) + quantity
        self.daily_pnl += pnl_change
