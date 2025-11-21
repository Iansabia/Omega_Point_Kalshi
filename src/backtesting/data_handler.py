"""
Data handlers for backtesting - provides historical market data in drip-feed fashion.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
import logging

from .backtest_engine import Event, MarketEvent

logger = logging.getLogger(__name__)


class DataHandler(ABC):
    """
    Abstract base class for data handlers.

    Provides interface for drip-feeding historical data to avoid look-ahead bias.
    """

    def __init__(self, events_queue):
        self.events = events_queue
        self.continue_backtest = True
        self.symbol_list = []
        self.latest_symbol_data = {}
        self.symbol_data = {}

    @abstractmethod
    def get_latest_bars(self, symbol: str, N: int = 1) -> Optional[pd.DataFrame]:
        """
        Returns last N bars from latest_symbol_data, or fewer if less available.
        """
        raise NotImplementedError("Must implement get_latest_bars()")

    @abstractmethod
    def update_bars(self):
        """
        Pushes latest bar to latest_symbol_data and generates MarketEvent.
        """
        raise NotImplementedError("Must implement update_bars()")


class CSVDataHandler(DataHandler):
    """
    Reads CSV files containing market data.
    """

    def __init__(self, events_queue, csv_dir: str, symbol_list: List[str]):
        super().__init__(events_queue)
        self.csv_dir = csv_dir
        self.symbol_list = symbol_list
        self.symbol_data = {}
        self.latest_symbol_data = {}
        self.continue_backtest = True
        self._current_bars_index = {}

        self._load_data()

    def _load_data(self):
        """
        Load CSV files into memory.
        """
        for symbol in self.symbol_list:
            try:
                self.symbol_data[symbol] = pd.read_csv(
                    f"{self.csv_dir}/{symbol}.csv",
                    index_col='datetime',
                    parse_dates=True
                )
                self.latest_symbol_data[symbol] = []
                self._current_bars_index[symbol] = 0

                logger.info(f"Loaded {len(self.symbol_data[symbol])} bars for {symbol}")
            except Exception as e:
                logger.error(f"Could not load data for {symbol}: {e}")
                self.continue_backtest = False

    def get_latest_bars(self, symbol: str, N: int = 1) -> Optional[pd.DataFrame]:
        """
        Returns last N bars for symbol.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.warning(f"Symbol {symbol} not available")
            return None
        else:
            if len(bars_list) == 0:
                return None
            return pd.DataFrame(bars_list[-N:])

    def get_latest_bar_value(self, symbol: str, val_type: str) -> Optional[float]:
        """
        Returns specific value (e.g., 'close', 'volume') from latest bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            logger.warning(f"Symbol {symbol} not available")
            return None
        else:
            if len(bars_list) == 0:
                return None
            return getattr(bars_list[-1], val_type, None)

    def update_bars(self):
        """
        Drip-feed next bar for each symbol.
        """
        for symbol in self.symbol_list:
            try:
                bar = self.symbol_data[symbol].iloc[self._current_bars_index[symbol]]

                # Convert to dict for easier access
                bar_dict = {
                    'datetime': bar.name,
                    'open': bar.get('open', bar.get('Open')),
                    'high': bar.get('high', bar.get('High')),
                    'low': bar.get('low', bar.get('Low')),
                    'close': bar.get('close', bar.get('Close')),
                    'volume': bar.get('volume', bar.get('Volume', 0))
                }

                self.latest_symbol_data[symbol].append(bar_dict)
                self._current_bars_index[symbol] += 1

            except IndexError:
                # No more bars for this symbol
                self.continue_backtest = False

        # Generate MarketEvent
        self.events.put(MarketEvent(
            timestamp=datetime.now(),
            symbol=self.symbol_list[0],  # Use first symbol as reference
            data=self.latest_symbol_data
        ))


class PandasDataHandler(DataHandler):
    """
    Uses pandas DataFrame directly (for in-memory backtesting).
    """

    def __init__(self, events_queue, data: Dict[str, pd.DataFrame]):
        super().__init__(events_queue)
        self.symbol_list = list(data.keys())
        self.symbol_data = data
        self.latest_symbol_data = {s: [] for s in self.symbol_list}
        self._current_bars_index = {s: 0 for s in self.symbol_list}
        self.continue_backtest = True

        logger.info(f"Loaded data for symbols: {self.symbol_list}")

    def get_latest_bars(self, symbol: str, N: int = 1) -> Optional[pd.DataFrame]:
        """
        Returns last N bars for symbol.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            return None
        else:
            if len(bars_list) == 0:
                return None
            return pd.DataFrame(bars_list[-N:])

    def get_latest_bar_value(self, symbol: str, val_type: str) -> Optional[float]:
        """
        Returns specific value from latest bar.
        """
        try:
            bars_list = self.latest_symbol_data[symbol]
        except KeyError:
            return None
        else:
            if len(bars_list) == 0:
                return None
            return bars_list[-1].get(val_type)

    def update_bars(self):
        """
        Drip-feed next bar for each symbol.
        """
        for symbol in self.symbol_list:
            try:
                idx = self._current_bars_index[symbol]
                bar = self.symbol_data[symbol].iloc[idx]

                bar_dict = {
                    'datetime': bar.name if hasattr(bar, 'name') else idx,
                    'open': bar.get('open', bar.get('Open', 0)),
                    'high': bar.get('high', bar.get('High', 0)),
                    'low': bar.get('low', bar.get('Low', 0)),
                    'close': bar.get('close', bar.get('Close', 0)),
                    'volume': bar.get('volume', bar.get('Volume', 0)),
                    'price': bar.get('price', bar.get('Close', 0))  # For single price series
                }

                self.latest_symbol_data[symbol].append(bar_dict)
                self._current_bars_index[symbol] += 1

            except (IndexError, KeyError):
                self.continue_backtest = False

        if self.continue_backtest:
            self.events.put(MarketEvent(
                timestamp=datetime.now(),
                symbol=self.symbol_list[0],
                data=self.latest_symbol_data
            ))
