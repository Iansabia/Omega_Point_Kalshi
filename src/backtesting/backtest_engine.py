import queue
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# --- Events ---

class Event(ABC):
    pass

@dataclass
class MarketEvent(Event):
    """
    Triggered when new market data is available.
    """
    timestamp: datetime
    symbol: str
    data: Dict[str, Any] # OHLCV, etc.

@dataclass
class SignalEvent(Event):
    """
    Triggered by a Strategy/Agent generating a trading signal.
    """
    timestamp: datetime
    symbol: str
    signal_type: str # 'LONG', 'SHORT', 'EXIT'
    strength: float
    target_price: Optional[float] = None

@dataclass
class OrderEvent(Event):
    """
    Triggered when a Signal is converted to an Order.
    """
    timestamp: datetime
    symbol: str
    order_type: str # 'MKT', 'LMT'
    quantity: int
    direction: str # 'BUY', 'SELL'
    price: Optional[float] = None

@dataclass
class FillEvent(Event):
    """
    Triggered when an Order is filled by the ExecutionHandler.
    """
    timestamp: datetime
    symbol: str
    exchange: str
    quantity: int
    direction: str
    fill_price: float
    commission: float
    cost: float # Total cost (price * qty + comm)

# --- Engine ---

class BacktestEngine:
    """
    Event-driven backtesting engine.
    """
    
    def __init__(self, start_date: datetime, end_date: datetime):
        self.events = queue.Queue()
        self.start_date = start_date
        self.end_date = end_date
        self.continue_backtest = True
        
        # Components (to be injected)
        self.data_handler = None
        self.strategy = None
        self.portfolio = None
        self.execution_handler = None

    def run(self):
        """
        Main event loop.
        """
        print(f"Starting backtest from {self.start_date} to {self.end_date}...")
        
        while self.continue_backtest:
            try:
                event = self.events.get(False)
            except queue.Empty:
                # If queue is empty, trigger data handler to load next tick/bar
                if self.data_handler and self.data_handler.continue_backtest:
                    self.data_handler.update_bars()
                else:
                    self.continue_backtest = False
            else:
                if event is not None:
                    if isinstance(event, MarketEvent):
                        self.strategy.calculate_signals(event)
                        self.portfolio.update_timeindex(event)

                    elif isinstance(event, SignalEvent):
                        self.portfolio.update_signal(event)

                    elif isinstance(event, OrderEvent):
                        self.execution_handler.execute_order(event)

                    elif isinstance(event, FillEvent):
                        self.portfolio.update_fill(event)

        print("Backtest complete.")

    def set_components(self, data_handler, strategy, portfolio, execution_handler):
        """
        Inject dependencies.
        """
        self.data_handler = data_handler
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution_handler = execution_handler
