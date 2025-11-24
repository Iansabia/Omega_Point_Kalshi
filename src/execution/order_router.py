import logging
from typing import Any, Dict, Optional

from src.execution.kalshi_client import KalshiClient
from src.execution.risk_manager import RiskManager
from src.execution.signal_generator import SignalGenerator
from src.execution.audit_log import audit_logger
from src.execution.circuit_breaker import CircuitBreakerOpenError
from src.orderbook.order import Order

logger = logging.getLogger(__name__)


class OrderRouter:
    """
    Routes orders to the appropriate exchange (Kalshi) after risk checks.
    """

    def __init__(self, kalshi_client: KalshiClient):
        self.client = kalshi_client
        self.signal_gen = SignalGenerator()
        self.risk_manager = RiskManager()

    def route_order(self, order: Order) -> Dict[str, Any]:
        """
        Process an internal order and execute it on Kalshi.
        """
        # 1. Generate Signal
        # We need market data for signal gen, fetching simplified version
        # In real system, this comes from a MarketDataService
        signal = self.signal_gen.generate_signal(order, {})

        if not signal:
            logger.error("Failed to generate signal from order")
            return {"status": "failed", "reason": "signal_generation_error"}

        # 2. Risk Check
        if not self.risk_manager.check_risk(signal):
            # Log risk rejection to audit log
            audit_logger.log_risk_violation(
                user_id=getattr(self.client, 'member_id', None) or getattr(self.client, 'api_key_id', None) or "unknown",
                violation_type="risk_check_failed",
                current_value=signal.get("count", 0),
                limit=0,  # Would need to get from risk manager
                details={
                    "ticker": signal.get("ticker", "unknown"),
                    "side": signal.get("side", "unknown"),
                    "count": signal.get("count", 0),
                    "price": signal.get("price", 0),
                    "reason": "Risk check failed"
                }
            )
            logger.warning(f"Risk check failed for signal: {signal}")
            return {"status": "rejected", "reason": "risk_check_failed"}

        # 3. Execute
        try:
            logger.info(f"Routing order to Kalshi: {signal}")

            # Place order (already logs via KalshiClient)
            response = self.client.place_order(
                ticker=signal["ticker"], side=signal["side"], count=signal["count"], price=signal["price"]
            )

            # 4. Update Risk State (Optimistic)
            self.risk_manager.update_position(signal["ticker"], signal["count"])

            # 5. Log successful trade execution to audit log
            order_data = response.get("order", {})
            if order_data:
                audit_logger.log_trade(
                    user_id=getattr(self.client, 'member_id', None) or getattr(self.client, 'api_key_id', None) or "unknown",
                    order_id=order_data.get("order_id", "unknown"),
                    side=signal["side"],
                    quantity=signal["count"],
                    price=signal["price"],
                    market=signal["ticker"]
                )

            logger.info(f"Order executed successfully: {response}")
            return {"status": "submitted", "response": response}

        except CircuitBreakerOpenError as e:
            # Circuit breaker is open - API is down
            logger.error(f"Circuit breaker OPEN - cannot execute order: {e}")
            audit_logger.log_risk_violation(
                user_id=getattr(self.client, 'member_id', None) or getattr(self.client, 'api_key_id', None) or "unknown",
                violation_type="circuit_breaker_open",
                current_value=0,
                limit=0,
                details={
                    "ticker": signal.get("ticker", "unknown"),
                    "side": signal.get("side", "unknown"),
                    "count": signal.get("count", 0),
                    "price": signal.get("price", 0),
                    "reason": str(e)
                }
            )
            return {"status": "error", "reason": "circuit_breaker_open", "message": str(e)}

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {"status": "error", "reason": str(e)}
