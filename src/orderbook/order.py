from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

class OrderType(Enum):
    MARKET = auto()
    LIMIT = auto()
    FOK = auto()  # Fill-or-Kill
    IOC = auto()  # Immediate-or-Cancel

@dataclass
class Order:
    order_id: str
    side: str  # 'BUY' or 'SELL'
    price: float
    quantity: float
    timestamp: float
    trader_id: str
    order_type: OrderType = OrderType.LIMIT
    remaining: Optional[float] = None

    def __post_init__(self):
        if self.remaining is None:
            self.remaining = self.quantity
        self.validate()

    def validate(self):
        if self.price < 0:
            raise ValueError("Price must be non-negative")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.side not in ['BUY', 'SELL']:
            raise ValueError("Side must be 'BUY' or 'SELL'")
