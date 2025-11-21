"""
Trading strategies for backtesting.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional
import logging

from .backtest_engine import Event, MarketEvent, SignalEvent

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    """

    def __init__(self, events_queue, data_handler):
        self.events = events_queue
        self.data_handler = data_handler
        self.symbol_list = data_handler.symbol_list

    @abstractmethod
    def calculate_signals(self, event: Event):
        """
        Generate trading signals from market events.
        """
        raise NotImplementedError("Must implement calculate_signals()")


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy-and-hold strategy for testing.
    """

    def __init__(self, events_queue, data_handler):
        super().__init__(events_queue, data_handler)
        self.bought = {symbol: False for symbol in self.symbol_list}

    def calculate_signals(self, event: MarketEvent):
        """
        Generate a single BUY signal for each symbol (buy-and-hold).
        """
        if isinstance(event, MarketEvent):
            for symbol in self.symbol_list:
                if not self.bought[symbol]:
                    signal = SignalEvent(
                        timestamp=event.timestamp,
                        symbol=symbol,
                        signal_type='LONG',
                        strength=1.0
                    )
                    self.events.put(signal)
                    self.bought[symbol] = True


class MovingAverageCrossStrategy(Strategy):
    """
    Moving average crossover strategy.
    """

    def __init__(self, events_queue, data_handler, short_window: int = 10, long_window: int = 30):
        super().__init__(events_queue, data_handler)
        self.short_window = short_window
        self.long_window = long_window
        self.positions = {symbol: 0 for symbol in self.symbol_list}

    def calculate_signals(self, event: MarketEvent):
        """
        Generate signals when short MA crosses long MA.
        """
        if isinstance(event, MarketEvent):
            for symbol in self.symbol_list:
                bars = self.data_handler.get_latest_bars(symbol, N=self.long_window)

                if bars is not None and len(bars) >= self.long_window:
                    closes = bars['close'].values
                    short_ma = closes[-self.short_window:].mean()
                    long_ma = closes.mean()

                    current_position = self.positions[symbol]

                    # Bullish crossover
                    if short_ma > long_ma and current_position == 0:
                        signal = SignalEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            signal_type='LONG',
                            strength=1.0
                        )
                        self.events.put(signal)
                        self.positions[symbol] = 1

                    # Bearish crossover
                    elif short_ma < long_ma and current_position == 1:
                        signal = SignalEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            signal_type='EXIT',
                            strength=1.0
                        )
                        self.events.put(signal)
                        self.positions[symbol] = 0


class MeanReversionStrategy(Strategy):
    """
    Simple mean reversion strategy for prediction markets.
    """

    def __init__(self, events_queue, data_handler, lookback: int = 20, entry_threshold: float = 0.1):
        super().__init__(events_queue, data_handler)
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.positions = {symbol: 0 for symbol in self.symbol_list}

    def calculate_signals(self, event: MarketEvent):
        """
        Generate signals when price deviates from moving average.
        """
        if isinstance(event, MarketEvent):
            for symbol in self.symbol_list:
                bars = self.data_handler.get_latest_bars(symbol, N=self.lookback)

                if bars is not None and len(bars) >= self.lookback:
                    closes = bars['close'].values if 'close' in bars.columns else bars['price'].values
                    mean_price = closes.mean()
                    current_price = closes[-1]

                    deviation = (current_price - mean_price) / mean_price
                    current_position = self.positions[symbol]

                    # Price below mean -> BUY (expecting reversion up)
                    if deviation < -self.entry_threshold and current_position == 0:
                        signal = SignalEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            signal_type='LONG',
                            strength=abs(deviation),
                            target_price=current_price
                        )
                        self.events.put(signal)
                        self.positions[symbol] = 1

                    # Price above mean -> SELL/EXIT (expecting reversion down)
                    elif deviation > self.entry_threshold and current_position == 1:
                        signal = SignalEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            signal_type='EXIT',
                            strength=abs(deviation)
                        )
                        self.events.put(signal)
                        self.positions[symbol] = 0


class ABMDrivenStrategy(Strategy):
    """
    Strategy driven by ABM simulation forecasts.

    Integrates with the agent-based model to use simulated prices as signals.
    """

    def __init__(self, events_queue, data_handler, abm_model=None, confidence_threshold: float = 0.6):
        super().__init__(events_queue, data_handler)
        self.abm_model = abm_model
        self.confidence_threshold = confidence_threshold
        self.positions = {symbol: 0 for symbol in self.symbol_list}

    def calculate_signals(self, event: MarketEvent):
        """
        Generate signals based on ABM simulation output.
        """
        if isinstance(event, MarketEvent) and self.abm_model:
            for symbol in self.symbol_list:
                # Get ABM forecast (this would interface with your simulation)
                forecast = self._get_abm_forecast(symbol)

                if forecast is not None:
                    current_price = self.data_handler.get_latest_bar_value(symbol, 'close')

                    if current_price is None:
                        continue

                    # Calculate expected return
                    expected_return = (forecast['price'] - current_price) / current_price
                    confidence = forecast.get('confidence', 0.5)

                    current_position = self.positions[symbol]

                    # Long signal if forecast is above current price with high confidence
                    if expected_return > 0.02 and confidence > self.confidence_threshold and current_position == 0:
                        signal = SignalEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            signal_type='LONG',
                            strength=confidence,
                            target_price=forecast['price']
                        )
                        self.events.put(signal)
                        self.positions[symbol] = 1

                    # Exit if forecast drops or confidence wanes
                    elif (expected_return < -0.01 or confidence < 0.4) and current_position == 1:
                        signal = SignalEvent(
                            timestamp=event.timestamp,
                            symbol=symbol,
                            signal_type='EXIT',
                            strength=1.0 - confidence
                        )
                        self.events.put(signal)
                        self.positions[symbol] = 0

    def _get_abm_forecast(self, symbol: str) -> Optional[dict]:
        """
        Get forecast from ABM simulation.

        This is a placeholder - would integrate with actual ABM.
        """
        if self.abm_model:
            return {
                'price': self.abm_model.current_price,
                'confidence': 0.7
            }
        return None
