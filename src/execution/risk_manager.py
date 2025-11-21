from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Enforces pre-trade risk checks.
    """
    
    def __init__(self, max_position_size: int = 100, max_daily_loss: float = 1000.0):
        self.max_position_size = max_position_size
        self.max_daily_loss = max_daily_loss
        self.current_positions = {} # ticker -> quantity
        self.daily_pnl = 0.0

    def check_risk(self, signal: Dict[str, Any]) -> bool:
        """
        Approve or reject a signal based on risk limits.
        """
        ticker = signal.get("ticker")
        count = signal.get("count", 0)
        price = signal.get("price", 0) # In cents
        
        # 1. Position Limit Check
        current_pos = self.current_positions.get(ticker, 0)
        if current_pos + count > self.max_position_size:
            logger.warning(f"Risk Reject: Position limit exceeded for {ticker}")
            return False
            
        # 2. Notional Value Check (e.g., max order value)
        notional = count * (price / 100.0)
        if notional > 500.0: # Hardcoded max order value $500
             logger.warning(f"Risk Reject: Order value ${notional} exceeds limit")
             return False
             
        # 3. Daily Loss Check (simplified)
        if self.daily_pnl < -self.max_daily_loss:
            logger.warning("Risk Reject: Max daily loss exceeded")
            return False
            
        return True

    def update_position(self, ticker: str, quantity: int, pnl_change: float = 0.0):
        """
        Update state after execution.
        """
        self.current_positions[ticker] = self.current_positions.get(ticker, 0) + quantity
        self.daily_pnl += pnl_change
