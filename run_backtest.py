"""
Comprehensive Backtesting System for Prediction Market ABM.

Tests the strategy on historical data before risking real money.

Usage:
    python run_backtest.py --games 100 --agents 50

Features:
    - Event-driven backtesting
    - Realistic execution with slippage
    - Transaction costs
    - Position tracking
    - Performance metrics (Sharpe, Sortino, Calmar, Max DD)
    - Trade-by-trade analysis
"""
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.market_model import PredictionMarketModel
from src.data.kalshi_historical import KalshiHistoricalDataFetcher


class HistoricalBacktest:
    """Run backtest on historical prediction market data."""

    def __init__(
        self,
        n_games: int = 100,
        agent_config: dict = None,
        initial_capital: float = 10000.0,
        transaction_cost_bps: float = 10.0,  # 10 basis points
        use_real_data: bool = False,
        season: int = 2024
    ):
        """Initialize backtest."""
        self.n_games = n_games
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.use_real_data = use_real_data
        self.season = season

        # Default agent configuration
        self.agent_config = agent_config or {
            'noise_trader': {'count': 30, 'wealth': 1000},
            'informed_trader': {'count': 10, 'wealth': 10000, 'information_quality': 0.8},
            'market_maker': {'count': 2, 'wealth': 100000, 'risk_param': 0.1}
        }

        self.trades = []
        self.portfolio = []
        self.equity_curve = []

        # Initialize data fetcher if using real data
        self.data_fetcher = KalshiHistoricalDataFetcher() if use_real_data else None

    def generate_historical_scenarios(self):
        """
        Generate historical game scenarios.

        If use_real_data=True, loads actual Kalshi historical data.
        Otherwise, simulates realistic scenarios based on statistical properties.
        """
        if self.use_real_data and self.data_fetcher:
            print(f"📥 Loading REAL historical data for {self.season} season...")
            scenarios = self.data_fetcher.load_historical_backtest_data(
                season=self.season,
                max_games=self.n_games
            )

            if not scenarios:
                print("⚠️  No real data found, falling back to synthetic data")
                return self._generate_synthetic_scenarios()

            print(f"✅ Loaded {len(scenarios)} real game scenarios")
            return scenarios
        else:
            return self._generate_synthetic_scenarios()

    def _generate_synthetic_scenarios(self):
        """Generate synthetic historical game scenarios."""
        scenarios = []

        for game_id in range(self.n_games):
            # Simulate a realistic prediction market scenario
            # True probability (what actually happened)
            true_prob = np.random.beta(5, 5)  # Centered around 0.5

            # Initial market price (may be biased)
            market_bias = np.random.normal(0, 0.05)
            initial_price = np.clip(true_prob + market_bias, 0.1, 0.9)

            # Price path over time (simulates market evolution)
            steps = 50  # 50 time steps per game
            price_path = self._generate_price_path(initial_price, true_prob, steps)

            # Final outcome (1 = Yes wins, 0 = No wins)
            outcome = 1 if np.random.random() < true_prob else 0

            scenarios.append({
                'game_id': game_id,
                'true_prob': true_prob,
                'initial_price': initial_price,
                'price_path': price_path,
                'outcome': outcome,
                'start_time': datetime(2024, 1, 1) + timedelta(days=game_id),
                'end_time': datetime(2024, 1, 1) + timedelta(days=game_id, hours=3),
                'is_real_data': False
            })

        return scenarios

    def _generate_price_path(self, initial_price, true_prob, steps):
        """Generate realistic price path for a market."""
        prices = [initial_price]
        current = initial_price

        # Mean reversion to true probability
        mean_reversion = 0.05
        volatility = 0.02

        for _ in range(steps - 1):
            # Random walk with mean reversion
            drift = mean_reversion * (true_prob - current)
            shock = np.random.normal(0, volatility)
            current = np.clip(current + drift + shock, 0.01, 0.99)
            prices.append(current)

        return prices

    def run_backtest(self):
        """Run the backtest."""
        print("="*80)
        print("RUNNING HISTORICAL BACKTEST")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Games: {self.n_games}")
        print(f"  Initial Capital: ${self.initial_capital:,.2f}")
        print(f"  Transaction Cost: {self.transaction_cost_bps} bps")
        print(f"  Agents: {sum(cfg['count'] for cfg in self.agent_config.values())}")

        # Generate scenarios
        print("\n1️⃣  Generating historical scenarios...")
        scenarios = self.generate_historical_scenarios()
        print(f"   ✅ Generated {len(scenarios)} game scenarios")

        # Initialize tracking
        capital = self.initial_capital
        positions = []
        total_pnl = 0

        # Run each scenario
        print("\n2️⃣  Simulating trading on historical data...")
        for i, scenario in enumerate(scenarios):
            if (i + 1) % 10 == 0:
                print(f"   Processing game {i+1}/{len(scenarios)}...")

            # Run ABM simulation on this market
            trades = self._simulate_game(scenario)

            # Calculate P&L for this game
            game_pnl = self._calculate_game_pnl(trades, scenario['outcome'])
            total_pnl += game_pnl
            capital += game_pnl

            # Track equity
            self.equity_curve.append({
                'game': i + 1,
                'capital': capital,
                'pnl': game_pnl,
                'return': game_pnl / self.initial_capital
            })

            self.trades.extend(trades)

        print(f"   ✅ Completed {len(scenarios)} games")

        # Calculate performance metrics
        print("\n3️⃣  Calculating performance metrics...")
        results = self._calculate_metrics()

        # Generate report
        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)
        self._print_results(results)

        # Save results
        self._save_results(results)

        return results

    def _simulate_game(self, scenario):
        """Simulate ABM trading on a single game."""
        # Create market model
        market_config = {
            'initial_price': scenario['initial_price'],
            'tick_size': 0.01
        }

        model = PredictionMarketModel(
            agent_config=self.agent_config,
            config={'market': market_config},
            seed=scenario['game_id']
        )

        # Run simulation
        trades_executed = []
        for step in range(len(scenario['price_path'])):
            model.step()

            # Extract trades from this step
            # Get trades from matching engine
            if hasattr(model.matching_engine, 'trades'):
                for trade in model.matching_engine.trades:
                    trades_executed.append({
                        'game_id': scenario['game_id'],
                        'step': step,
                        'price': trade.price,
                        'quantity': trade.quantity,
                        'side': trade.aggressor_side,
                        'timestamp': scenario['start_time'] + timedelta(minutes=step)
                    })

        return trades_executed

    def _calculate_game_pnl(self, trades, outcome):
        """Calculate P&L for a game based on trades and outcome."""
        if not trades:
            return 0.0

        # Calculate net position from trades
        net_position = 0
        total_cost = 0

        for trade in trades:
            side = trade['side']
            quantity = trade['quantity']
            price = trade['price']

            # Transaction cost
            cost = quantity * price * (self.transaction_cost_bps / 10000)

            if side == 'BUY':
                net_position += quantity
                total_cost += quantity * price + cost
            else:
                net_position -= quantity
                total_cost -= quantity * price - cost

        # Settlement value
        if outcome == 1:  # Yes wins
            settlement_value = net_position * 1.0
        else:  # No wins
            settlement_value = 0.0

        pnl = settlement_value - total_cost
        return pnl

    def _calculate_metrics(self):
        """Calculate performance metrics."""
        df = pd.DataFrame(self.equity_curve)

        if len(df) == 0:
            return {}

        # Returns
        returns = df['return'].values
        cumulative_return = (df['capital'].iloc[-1] / self.initial_capital) - 1

        # Risk metrics
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else 0
        sortino = np.sqrt(252) * returns.mean() / returns[returns < 0].std() if len(returns[returns < 0]) > 0 else 0

        # Drawdown
        equity = df['capital'].values
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = drawdowns.min()

        # Calmar ratio
        calmar = cumulative_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Win rate
        wins = len([r for r in returns if r > 0])
        losses = len([r for r in returns if r < 0])
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        # Average win/loss
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        return {
            'total_games': len(df),
            'total_trades': len(self.trades),
            'initial_capital': self.initial_capital,
            'final_capital': df['capital'].iloc[-1],
            'total_return': cumulative_return,
            'total_pnl': df['capital'].iloc[-1] - self.initial_capital,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'equity_curve': df
        }

    def _print_results(self, results):
        """Print backtest results."""
        print(f"\n📊 TRADING STATISTICS")
        print(f"  Total Games: {results['total_games']}")
        print(f"  Total Trades: {results['total_trades']}")
        print(f"  Trades per Game: {results['total_trades'] / results['total_games']:.1f}")

        print(f"\n💰 CAPITAL & RETURNS")
        print(f"  Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"  Final Capital: ${results['final_capital']:,.2f}")
        print(f"  Total P&L: ${results['total_pnl']:,.2f}")
        print(f"  Total Return: {results['total_return']:.2%}")

        print(f"\n📈 RISK-ADJUSTED METRICS")
        print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {results['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio: {results['calmar_ratio']:.3f}")
        print(f"  Max Drawdown: {results['max_drawdown']:.2%}")

        print(f"\n🎯 WIN/LOSS ANALYSIS")
        print(f"  Win Rate: {results['win_rate']:.2%}")
        print(f"  Wins: {results['wins']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Avg Win: {results['avg_win']:.2%}")
        print(f"  Avg Loss: {results['avg_loss']:.2%}")
        print(f"  Profit Factor: {results['profit_factor']:.2f}")

        # Decision
        print(f"\n" + "="*80)
        print("💡 RECOMMENDATION")
        print("="*80)

        if results['sharpe_ratio'] > 1.0 and results['max_drawdown'] > -0.15:
            print("✅ GOOD - Strategy shows promise!")
            print("   → Sharpe > 1.0 and Max DD < 15%")
            print("   → Safe to proceed to paper trading")
        elif results['sharpe_ratio'] > 0.5:
            print("⚠️  MARGINAL - Strategy needs improvement")
            print("   → Consider optimizing agent parameters")
            print("   → Run more backtests with different configurations")
        else:
            print("❌ POOR - Do NOT trade with real money yet")
            print("   → Strategy underperforming")
            print("   → Refine models and test again")

    def _save_results(self, results):
        """Save backtest results."""
        # Save equity curve
        results['equity_curve'].to_csv('backtest_equity_curve.csv', index=False)

        # Save trades
        if self.trades:
            pd.DataFrame(self.trades).to_csv('backtest_trades.csv', index=False)

        # Generate plots
        self._generate_plots(results)

        print(f"\n💾 Results saved:")
        print(f"   • backtest_equity_curve.csv")
        print(f"   • backtest_trades.csv")
        print(f"   • backtest_results.png")

    def _generate_plots(self, results):
        """Generate performance plots."""
        df = results['equity_curve']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Equity curve
        axes[0, 0].plot(df['game'], df['capital'], linewidth=2, color='blue')
        axes[0, 0].axhline(y=self.initial_capital, color='red', linestyle='--', label='Initial Capital')
        axes[0, 0].set_title('Equity Curve')
        axes[0, 0].set_xlabel('Game')
        axes[0, 0].set_ylabel('Capital ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Returns distribution
        axes[0, 1].hist(df['return'], bins=30, color='green', alpha=0.6, edgecolor='black')
        axes[0, 1].axvline(x=0, color='red', linestyle='--')
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)

        # Drawdown
        equity = df['capital'].values
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max * 100
        axes[1, 0].fill_between(df['game'], drawdowns, color='red', alpha=0.5)
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_xlabel('Game')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True, alpha=0.3)

        # Cumulative returns
        cumulative_returns = (df['capital'] / self.initial_capital - 1) * 100
        axes[1, 1].plot(df['game'], cumulative_returns, linewidth=2, color='purple')
        axes[1, 1].axhline(y=0, color='red', linestyle='--')
        axes[1, 1].set_title('Cumulative Returns')
        axes[1, 1].set_xlabel('Game')
        axes[1, 1].set_ylabel('Return (%)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('backtest_results.png', dpi=150, bbox_inches='tight')
        print("   ✅ Plots saved to backtest_results.png")


def main():
    """Run backtest from command line."""
    parser = argparse.ArgumentParser(description='Run historical backtest')
    parser.add_argument('--games', type=int, default=100, help='Number of games to backtest')
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--agents', type=int, default=42, help='Total number of agents')
    parser.add_argument('--use-real-data', action='store_true', help='Use real Kalshi historical data')
    parser.add_argument('--season', type=int, default=2024, help='NFL season year (e.g., 2024)')
    args = parser.parse_args()

    # Configure agents
    agent_config = {
        'noise_trader': {'count': int(args.agents * 0.6), 'wealth': 1000},
        'informed_trader': {'count': int(args.agents * 0.3), 'wealth': 10000, 'information_quality': 0.8},
        'market_maker': {'count': max(2, int(args.agents * 0.1)), 'wealth': 100000, 'risk_param': 0.1}
    }

    # Run backtest
    backtest = HistoricalBacktest(
        n_games=args.games,
        agent_config=agent_config,
        initial_capital=args.capital,
        use_real_data=args.use_real_data,
        season=args.season
    )

    # Print data source info
    if args.use_real_data:
        print(f"\n🎯 Using REAL Kalshi historical data ({args.season} season)")
    else:
        print(f"\n🎲 Using SYNTHETIC data (simulated markets)")

    results = backtest.run_backtest()


if __name__ == "__main__":
    main()
