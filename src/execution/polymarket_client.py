import os
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import the official client
try:
    from py_clob_client.client import ClobClient
    from py_clob_client.clob_types import OrderArgs, OrderType
    from py_clob_client.constants import POLYGON
    PY_CLOB_AVAILABLE = True
except ImportError:
    PY_CLOB_AVAILABLE = False
    ClobClient = None
    OrderArgs = None
    OrderType = None
    POLYGON = None
    logger.warning("py_clob_client not found. Polymarket execution will be disabled.")

class PolymarketClient:
    """
    Client for interacting with Polymarket's CLOB API.
    Wraps py_clob_client to provide a unified interface.
    """
    
    def __init__(self, private_key: str = None):
        self.private_key = private_key or os.getenv("POLYMARKET_PRIVATE_KEY")
        self.client: Optional[ClobClient] = None
        
        if PY_CLOB_AVAILABLE and self.private_key:
            try:
                self.client = ClobClient(
                    "https://clob.polymarket.com",
                    key=self.private_key,
                    chain_id=137, # Polygon Mainnet
                    signature_type=1 # EOA (Externally Owned Account)
                )
                # Derive API creds from private key
                self.client.set_api_creds(self.client.create_or_derive_api_creds())
                logger.info("Polymarket CLOB client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Polymarket client: {e}")
        else:
            logger.warning("Polymarket client not initialized (missing key or library).")

    def get_market_data(self, token_id: str) -> Tuple[Optional[float], Dict]:
        """
        Get midpoint price and orderbook for a specific token.
        """
        if not self.client:
            return None, {}
            
        try:
            mid = self.client.get_midpoint(token_id)
            book = self.client.get_order_book(token_id)
            return float(mid) if mid else None, book
        except Exception as e:
            logger.error(f"Error fetching market data for {token_id}: {e}")
            return None, {}

    def place_order(self, token_id: str, price: float, size: float, side: str) -> Dict[str, Any]:
        """
        Place a Limit order on the CLOB.
        side: 'BUY' or 'SELL'
        """
        if not self.client:
            return {"status": "failed", "reason": "client_not_initialized"}
            
        try:
            # Convert side string to enum
            # Assuming py_clob_client uses specific enums, usually imported from clob_types
            # For safety, we'll assume the library handles the enum conversion or we pass the specific object
            # If OrderArgs expects specific constants, we'd need to import them.
            # Here we use the imported OrderArgs and check availability.
            
            from py_clob_client.clob_types import OrderArgs, OrderType, Buy, Sell
            
            order_side = Buy if side.upper() == 'BUY' else Sell
            
            order_args = OrderArgs(
                token_id=token_id,
                price=price,
                size=size,
                side=order_side
            )
            
            # Create and sign
            signed_order = self.client.create_order(order_args)
            
            # Post
            resp = self.client.post_order(signed_order, OrderType.GTC)
            return {"status": "submitted", "response": resp}
            
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            return {"status": "error", "reason": str(e)}

    def cancel_all(self):
        """
        Cancel all open orders.
        """
        if self.client:
            try:
                self.client.cancel_all()
            except Exception as e:
                logger.error(f"Failed to cancel orders: {e}")
