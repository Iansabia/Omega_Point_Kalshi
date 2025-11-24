import logging
from typing import Any, Dict, Optional

from src.execution.kalshi_client import KalshiClient
from src.execution.risk_manager import RiskManager
from src.execution.signal_generator import SignalGenerator
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
            return {"status": "rejected", "reason": "risk_check_failed"}

        # 3. Execute
        try:
            logger.info(f"Routing order to Kalshi: {signal}")
            response = self.client.place_order(
                ticker=signal["ticker"], side=signal["side"], count=signal["count"], price=signal["price"]
            )

            # 4. Update Risk State (Optimistic)
            self.risk_manager.update_position(signal["ticker"], signal["count"])

            return {"status": "submitted", "response": response}

        except Exception as e:
            logger.error(f"Execution failed: {e}")
            return {"status": "error", "reason": str(e)}
