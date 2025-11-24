import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    FOK = auto()  # Fill-or-Kill
    IOC = auto()  # Immediate-or-Cancel


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Order:
    order_id: str
    trader_id: str
    side: str  # 'BUY' or 'SELL' or OrderSide enum
    order_type: OrderType
    price: Optional[float]  # Can be None for market orders
    quantity: float
    timestamp: Optional[float] = None
    remaining: Optional[float] = None

    def __post_init__(self):
        # Convert OrderSide enum to string if needed
        if isinstance(self.side, OrderSide):
            self.side = self.side.value

        # Set timestamp if not provided
        if self.timestamp is None:
            self.timestamp = time.time()

        # Set remaining quantity
        if self.remaining is None:
            self.remaining = self.quantity

        self.validate()

    def validate(self):
        if self.price is not None and self.price < 0:
            raise ValueError("Price must be non-negative")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.side not in ["BUY", "SELL"]:
            raise ValueError("Side must be 'BUY' or 'SELL'")

    def fill(self, quantity: float):
        """Mark quantity as filled."""
        if quantity > self.remaining:
            raise ValueError(f"Cannot fill {quantity}, only {self.remaining} remaining")
        self.remaining -= quantity

    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.remaining == 0
