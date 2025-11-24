#!/usr/bin/env python3
"""
Single-Game Backtest with Detailed Statistics.

Simulates momentum arbitrage trading on one historical Kalshi NFL market.
Shows:
- All signals generated during the game
- Entry/exit prices and timing
- Profit/loss for each trade
- Overall performance metrics

Usage:
    python scripts/backtest_single_game.py --ticker KXMVENFLSINGLEGAME-S2025-...
    python scripts/backtest_single_game.py --list-games  # Show available games
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.live_trading.arbitrage_detector import ArbitrageDetector, ArbitrageSignal
from src.models.win_probability_inference import WinProbabilityInference

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class SingleGameBacktest:
    """
    Backtest momentum arbitrage strategy on a single historical game.
    """

    def __init__(
        self,
        model_path: str = "models/win_probability_model.pkl",
        min_edge: float = 0.10,
        min_confidence: float = 0.5,
        max_spread: float = 0.10,
        position_size: float = 100.0,
        max_holding_candles: int = 5,  # Max 5 candlesticks (~5 hours for 1h candles)
    ):
        """
        Initialize single-game backtest.

        Args:
            model_path: Path to trained model
            min_edge: Minimum edge for signal (10% default)
            min_confidence: Minimum model confidence (50% default)
            max_spread: Maximum bid-ask spread (10% default)
            position_size: Fixed position size in dollars
            max_holding_candles: Maximum candles to hold position
        """
        self.model_path = model_path
        self.min_edge = min_edge
        self.min_confidence = min_confidence
        self.max_spread = max_spread
        self.position_size = position_size
        self.max_holding_candles = max_holding_candles

        # Initialize model (for simplified backtesting, we'll use market price as proxy)
        # In production, you'd predict WP from game state
        self.wp_inference = WinProbabilityInference(model_path=model_path)

        # Initialize detector
        self.detector = ArbitrageDetector(
            min_edge=min_edge, min_confidence=min_confidence, max_spread=max_spread, require_fresh_data=False  # Historical data
        )

        # Track trades
        self.trades: List[Dict[str, Any]] = []
        self.signals: List[ArbitrageSignal] = []

    def load_game_data(self, ticker: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load historical candlestick data for a game.

        Args:
            ticker: Kalshi market ticker

        Returns:
            (candlesticks_df, market_info)
        """
        # Load candlesticks
        candles_path = Path(f"data/historical/candlesticks/{ticker}.csv")
        if not candles_path.exists():
            raise FileNotFoundError(f"Candlestick data not found: {candles_path}")

        candles_df = pd.read_csv(candles_path)

        # Load market metadata
        markets_path = Path("data/historical/markets/nfl_settled_2025.csv")
        if not markets_path.exists():
            raise FileNotFoundError(f"Market metadata not found: {markets_path}")

        markets_df = pd.read_csv(markets_path)
        market_info = markets_df[markets_df["ticker"] == ticker].iloc[0].to_dict()

        logger.info(f"📂 Loaded data for {ticker}")
        logger.info(f"   Title: {market_info.get('title', 'N/A')}")
        logger.info(f"   Outcome: {market_info.get('result', 'N/A')}")
        logger.info(f"   Candlesticks: {len(candles_df)}")

        return candles_df, market_info

    def simulate_game_state(self, candle_idx: int, total_candles: int, market_info: Dict) -> Dict[str, Any]:
        """
        Simulate game state from candlestick position.

        In a real backtest, you'd have actual game events. Here we approximate:
        - Early game: Q1-Q2
        - Mid game: Q3
        - Late game: Q4

        Args:
            candle_idx: Current candlestick index
            total_candles: Total number of candlesticks
            market_info: Market metadata

        Returns:
            Simulated game state dict
        """
        progress = candle_idx / total_candles

        # Approximate quarter from progress
        if progress < 0.25:
            quarter = 1
        elif progress < 0.50:
            quarter = 2
        elif progress < 0.75:
            quarter = 3
        else:
            quarter = 4

        # Approximate time remaining
        time_remaining = int(3600 * (1 - progress))  # Total game ~3600s

        # Simulate score (based on title if available)
        # This is a simplified approximation
        title = market_info.get("title", "")

        # For demonstration, use neutral game state
        game_state = {
            "home_score": 14,
            "away_score": 14,
            "score_diff": 0,
            "quarter": quarter,
            "time_remaining": time_remaining,
            "yardline": 50,
            "down": 1,
            "distance": 10,
            "possession": "home",
            "timestamp": candle_idx,
        }

        return game_state

    def run_backtest(self, ticker: str) -> Dict[str, Any]:
        """
        Run backtest on a single game.

        Args:
            ticker: Kalshi market ticker

        Returns:
            Dict with backtest results
        """
        logger.info("\n" + "=" * 80)
        logger.info(f"Single-Game Backtest: {ticker}")
        logger.info("=" * 80)

        # Load data
        candles_df, market_info = self.load_game_data(ticker)

        # Ensure we have required columns
        if "close" not in candles_df.columns:
            logger.error("❌ Candlestick data missing 'close' column")
            return {}

        # Track positions
        open_position = None
        total_pnl = 0.0

        # Iterate through candlesticks (each represents a time period)
        logger.info(f"\n🔄 Processing {len(candles_df)} candlesticks...")

        for idx, row in candles_df.iterrows():
            market_price = row["close"]

            # Simulate game state
            game_state = self.simulate_game_state(idx, len(candles_df), market_info)

            # Get model prediction (simplified - using trend)
            # In reality, you'd predict from game_state
            model_wp = self._estimate_model_wp(idx, candles_df, market_info)

            # Create mock correlated state
            correlated_state = {
                "game_id": ticker,
                "ticker": ticker,
                "nfl": game_state,
                "kalshi": {"mid_price": market_price, "spread": 0.02, "yes_bid": market_price - 0.01, "yes_ask": market_price + 0.01},
                "is_fresh": True,
                "data_age": {"nfl": 0, "kalshi": 0},
            }

            # Check if we should close existing position
            if open_position:
                holding_time = idx - open_position["entry_idx"]

                # Close if: max holding time OR price moved favorably OR end of game
                should_close = False
                close_reason = ""

                if holding_time >= self.max_holding_candles:
                    should_close = True
                    close_reason = "Max holding time"
                elif idx == len(candles_df) - 1:
                    should_close = True
                    close_reason = "End of game"
                elif open_position["direction"] == "BUY" and market_price > open_position["entry_price"] * 1.05:
                    should_close = True
                    close_reason = "5% profit target"
                elif open_position["direction"] == "SELL" and market_price < open_position["entry_price"] * 0.95:
                    should_close = True
                    close_reason = "5% profit target"

                if should_close:
                    pnl = self._close_position(open_position, market_price, idx, close_reason)
                    total_pnl += pnl
                    open_position = None

            # If no position, check for new signal
            if not open_position:
                signal = self.detector.detect(correlated_state, model_wp)

                if signal:
                    self.signals.append(signal)
                    open_position = self._open_position(signal, market_price, idx)

        # Close any remaining position
        if open_position:
            final_price = candles_df.iloc[-1]["close"]
            pnl = self._close_position(open_position, final_price, len(candles_df) - 1, "Forced close")
            total_pnl += pnl

        # Generate results
        results = self._generate_results(ticker, market_info, total_pnl, candles_df)

        return results

    def _estimate_model_wp(self, idx: int, candles_df: pd.DataFrame, market_info: Dict) -> float:
        """
        Estimate what the model's win probability would have been.

        For backtesting, we use a simple heuristic:
        - If market is trending up → model predicts slightly lower (momentum overreaction)
        - If market is trending down → model predicts slightly higher

        Args:
            idx: Current candlestick index
            candles_df: All candlesticks
            market_info: Market metadata

        Returns:
            Estimated model win probability
        """
        current_price = candles_df.iloc[idx]["close"]

        # Look at trend (last 3 candles)
        if idx >= 3:
            prev_prices = candles_df.iloc[idx - 3 : idx]["close"].values
            trend = current_price - prev_prices.mean()

            # If price rising fast → model predicts lower (humans overreacting)
            # If price falling fast → model predicts higher
            if trend > 0.05:  # 5% upward trend
                model_wp = max(0.1, current_price - 0.10)  # Model 10% lower
            elif trend < -0.05:  # 5% downward trend
                model_wp = min(0.9, current_price + 0.10)  # Model 10% higher
            else:
                model_wp = current_price  # No strong trend, model agrees
        else:
            # Early game, no trend yet
            model_wp = current_price

        return model_wp

    def _open_position(self, signal: ArbitrageSignal, entry_price: float, candle_idx: int) -> Dict[str, Any]:
        """Record position opening."""
        position = {
            "entry_idx": candle_idx,
            "entry_price": entry_price,
            "direction": signal.direction,
            "edge": signal.edge,
            "ticker": signal.ticker,
        }

        logger.info(f"\n📈 SIGNAL {len(self.signals)}: {signal.direction} @ ${entry_price:.2f} (Edge: {signal.edge:+.1%})")

        return position

    def _close_position(self, position: Dict, exit_price: float, candle_idx: int, reason: str) -> float:
        """Record position closing and calculate P&L."""
        entry_price = position["entry_price"]
        direction = position["direction"]
        holding_time = candle_idx - position["entry_idx"]

        # Calculate P&L
        if direction == "BUY":
            price_change = (exit_price - entry_price) / entry_price
        else:  # SELL
            price_change = (entry_price - exit_price) / entry_price

        # Calculate P&L in dollars (simplified - no fees yet)
        contracts = self.position_size / entry_price
        gross_pnl = contracts * (exit_price - entry_price) if direction == "BUY" else contracts * (entry_price - exit_price)

        # Apply Kalshi fees: 0.07 × C × P × (1-P)
        entry_fee = 0.07 * contracts * entry_price * (1 - entry_price)
        exit_fee = 0.07 * contracts * exit_price * (1 - exit_price)
        net_pnl = gross_pnl - entry_fee - exit_fee

        # Record trade
        trade = {
            "entry_idx": position["entry_idx"],
            "exit_idx": candle_idx,
            "holding_time": holding_time,
            "direction": direction,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "price_change": price_change,
            "gross_pnl": gross_pnl,
            "fees": entry_fee + exit_fee,
            "net_pnl": net_pnl,
            "reason": reason,
        }

        self.trades.append(trade)

        logger.info(f"📉 CLOSE {len(self.trades)}: {direction} @ ${exit_price:.2f} ({reason})")
        logger.info(f"   Held: {holding_time} candles, P&L: ${net_pnl:+.2f} (Fees: ${entry_fee + exit_fee:.2f})")

        return net_pnl

    def _generate_results(self, ticker: str, market_info: Dict, total_pnl: float, candles_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate backtest results summary."""
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        # Calculate metrics
        num_trades = len(self.trades)
        num_signals = len(self.signals)

        if num_trades > 0:
            winning_trades = len(trades_df[trades_df["net_pnl"] > 0])
            losing_trades = len(trades_df[trades_df["net_pnl"] < 0])
            win_rate = winning_trades / num_trades

            avg_win = trades_df[trades_df["net_pnl"] > 0]["net_pnl"].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df["net_pnl"] < 0]["net_pnl"].mean() if losing_trades > 0 else 0

            total_fees = trades_df["fees"].sum()
            gross_pnl = trades_df["gross_pnl"].sum()
        else:
            winning_trades = losing_trades = 0
            win_rate = avg_win = avg_loss = total_fees = gross_pnl = 0

        # Market outcome
        final_price = candles_df.iloc[-1]["close"]
        result = market_info.get("result", "unknown")

        results = {
            "ticker": ticker,
            "title": market_info.get("title", "N/A"),
            "result": result,
            "final_price": final_price,
            "num_signals": num_signals,
            "num_trades": num_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pnl": total_pnl,
            "gross_pnl": gross_pnl,
            "total_fees": total_fees,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "trades": self.trades,
        }

        # Print summary
        self._print_results(results)

        return results

    def _print_results(self, results: Dict[str, Any]):
        """Print formatted results."""
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 80)

        logger.info(f"\n📊 Market Info:")
        logger.info(f"   Ticker: {results['ticker']}")
        logger.info(f"   Title: {results['title']}")
        logger.info(f"   Outcome: {results['result']}")
        logger.info(f"   Final Price: ${results['final_price']:.2f}")

        logger.info(f"\n📈 Trading Activity:")
        logger.info(f"   Signals Generated: {results['num_signals']}")
        logger.info(f"   Trades Executed: {results['num_trades']}")

        if results["num_trades"] > 0:
            logger.info(f"   Winning Trades: {results['winning_trades']}")
            logger.info(f"   Losing Trades: {results['losing_trades']}")
            logger.info(f"   Win Rate: {results['win_rate']:.1%}")

            logger.info(f"\n💰 Performance:")
            logger.info(f"   Gross P&L: ${results['gross_pnl']:+.2f}")
            logger.info(f"   Total Fees: ${results['total_fees']:.2f}")
            logger.info(f"   Net P&L: ${results['total_pnl']:+.2f}")
            logger.info(f"   Avg Win: ${results['avg_win']:+.2f}")
            logger.info(f"   Avg Loss: ${results['avg_loss']:+.2f}")

            # Print individual trades
            logger.info(f"\n📋 Trade History:")
            for i, trade in enumerate(results["trades"], 1):
                logger.info(
                    f"   {i}. {trade['direction']} ${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} "
                    f"= ${trade['net_pnl']:+.2f} ({trade['reason']})"
                )
        else:
            logger.info(f"   No trades executed (signals didn't meet criteria)")

        logger.info("\n" + "=" * 80)


def list_available_games():
    """List all available games for backtesting."""
    markets_path = Path("data/historical/markets/nfl_settled_2025.csv")

    if not markets_path.exists():
        logger.error(f"❌ Market data not found: {markets_path}")
        logger.error("   Run: python scripts/download_kalshi_historical.py first")
        return

    markets_df = pd.read_csv(markets_path)

    logger.info("\n" + "=" * 80)
    logger.info("Available Games for Backtesting")
    logger.info("=" * 80)

    for idx, row in markets_df.head(20).iterrows():
        ticker = row["ticker"]
        title = row.get("title", "N/A")
        result = row.get("result", "unknown")

        # Check if candlestick data exists
        candles_path = Path(f"data/historical/candlesticks/{ticker}.csv")
        has_data = "✅" if candles_path.exists() else "❌"

        logger.info(f"{idx + 1}. {has_data} {ticker}")
        logger.info(f"   {title}")
        logger.info(f"   Result: {result}\n")

    logger.info("=" * 80)
    logger.info(f"Total markets: {len(markets_df)}")
    logger.info("Use: python scripts/backtest_single_game.py --ticker <TICKER>")
    logger.info("=" * 80)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Backtest momentum strategy on single game")

    parser.add_argument("--ticker", type=str, help="Kalshi market ticker")
    parser.add_argument("--list-games", action="store_true", help="List available games")
    parser.add_argument("--min-edge", type=float, default=0.10, help="Minimum edge (default 10%%)")
    parser.add_argument("--position-size", type=float, default=100.0, help="Position size in dollars")
    parser.add_argument("--max-holding", type=int, default=5, help="Max holding time in candlesticks")

    args = parser.parse_args()

    if args.list_games:
        list_available_games()
        return

    if not args.ticker:
        logger.error("❌ Please specify --ticker or use --list-games")
        return

    # Run backtest
    backtest = SingleGameBacktest(
        min_edge=args.min_edge, position_size=args.position_size, max_holding_candles=args.max_holding
    )

    results = backtest.run_backtest(args.ticker)


if __name__ == "__main__":
    main()
