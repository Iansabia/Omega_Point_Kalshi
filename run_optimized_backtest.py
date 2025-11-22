#!/usr/bin/env python3
"""
Optimized Backtest with Risk Management.

Implements comprehensive risk controls to maximize risk-adjusted returns:
- Position limits
- Trade frequency controls
- Kelly Criterion position sizing
- Stop losses and circuit breakers
- Conservative agent configurations

Target Metrics:
- Sharpe Ratio > 1.5
- Max Drawdown < 15%
- Profit Factor > 2.0
- Win Rate > 55%
- < 100 trades per game

Usage:
    # Conservative (recommended for real money)
    python run_optimized_backtest.py --profile conservative

    # Aggressive (higher risk/reward)
    python run_optimized_backtest.py --profile aggressive

    # Ultra-conservative (for real money trading)
    python run_optimized_backtest.py --profile ultra_conservative

    # Custom parameters
    python run_optimized_backtest.py --games 100 --profile conservative
"""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Import configurations
from configs.conservative_risk_config import (
    CONSERVATIVE_AGENT_CONFIG,
    AGGRESSIVE_AGENT_CONFIG,
    ULTRA_CONSERVATIVE_AGENT_CONFIG,
    CONSERVATIVE_RISK_LIMITS,
    AGGRESSIVE_RISK_LIMITS,
    ULTRA_CONSERVATIVE_RISK_LIMITS
)

# Import backtest
from run_backtest import HistoricalBacktest


class OptimizedBacktest(HistoricalBacktest):
    """
    Enhanced backtest with integrated risk management.
    """

    def __init__(
        self,
        n_games: int = 50,
        profile: str = 'conservative',
        initial_capital: float = 10000.0,
        **kwargs
    ):
        """
        Initialize optimized backtest.

        Args:
            n_games: Number of games to simulate
            profile: Risk profile ('conservative', 'aggressive', 'ultra_conservative')
            initial_capital: Starting capital
        """
        # Select configuration based on profile
        if profile == 'conservative':
            agent_config = CONSERVATIVE_AGENT_CONFIG
            risk_limits = CONSERVATIVE_RISK_LIMITS
        elif profile == 'aggressive':
            agent_config = AGGRESSIVE_AGENT_CONFIG
            risk_limits = AGGRESSIVE_RISK_LIMITS
        elif profile == 'ultra_conservative':
            agent_config = ULTRA_CONSERVATIVE_AGENT_CONFIG
            risk_limits = ULTRA_CONSERVATIVE_RISK_LIMITS
        else:
            raise ValueError(f"Unknown profile: {profile}")

        # Initialize parent
        super().__init__(
            n_games=n_games,
            agent_config=agent_config,
            initial_capital=initial_capital,
            **kwargs
        )

        self.profile = profile
        self.risk_limits = risk_limits

        print(f"\n🎯 Risk Profile: {profile.upper()}")
        print(f"   Max Position Size: ${risk_limits.max_position_size:,.0f}")
        print(f"   Max Portfolio Exposure: ${risk_limits.max_portfolio_exposure:,.0f}")
        print(f"   Max Trades/Game: {risk_limits.max_trades_per_game}")
        print(f"   Min Edge Threshold: {risk_limits.min_edge_threshold:.1%}")
        print(f"   Trade Probability: {risk_limits.trade_probability:.1%}")
        print(f"   Kelly Fraction: {risk_limits.kelly_fraction:.2f}")

    def analyze_results(self, results_df: pd.DataFrame):
        """
        Enhanced results analysis with risk metrics.

        Args:
            results_df: Results DataFrame from backtest
        """
        print("\n" + "="*80)
        print("RISK-ADJUSTED PERFORMANCE ANALYSIS")
        print("="*80)

        # Handle empty results
        if len(results_df) == 0 or 'return' not in results_df.columns:
            print("\n❌ No trading data available for analysis")
            print("   Risk controls may be too strict - no trades were executed")
            return {
                'sharpe': 0,
                'max_drawdown': 0,
                'profit_factor': 0,
                'win_rate': 0,
                'total_return': 0,
                'ready_to_trade': False
            }

        # Calculate metrics
        returns = results_df['return'].values
        pnl = results_df['pnl'].values
        capital = results_df['capital'].values

        # Returns metrics
        total_return = (capital[-1] - self.initial_capital) / self.initial_capital
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = avg_return / std_return if std_return > 0 else 0
        sharpe_annual = sharpe * np.sqrt(252)  # Assuming daily-like returns

        # Drawdown
        cumulative = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative)
        max_drawdown = np.max(drawdown) / self.initial_capital if self.initial_capital > 0 else 0

        # Win rate
        wins = np.sum(returns > 0)
        losses = np.sum(returns < 0)
        win_rate = wins / len(returns) if len(returns) > 0 else 0

        # Profit factor
        gross_profit = np.sum(pnl[pnl > 0])
        gross_loss = abs(np.sum(pnl[pnl < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Sortino ratio (uses downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = avg_return / downside_std if downside_std > 0 else 0

        # Display results
        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"   Total Return: {total_return:+.2%}")
        print(f"   Average Return/Game: {avg_return:+.2%}")
        print(f"   Sharpe Ratio: {sharpe:.3f} {'✅' if sharpe > 1.0 else '❌'}")
        print(f"   Sharpe (Annualized): {sharpe_annual:.3f}")
        print(f"   Sortino Ratio: {sortino:.3f}")

        print(f"\n📉 RISK METRICS:")
        print(f"   Max Drawdown: {max_drawdown:.2%} {'✅' if max_drawdown < 0.15 else '❌'}")
        print(f"   Volatility: {std_return:.2%}")
        print(f"   Downside Volatility: {downside_std:.2%}")

        print(f"\n💰 TRADING METRICS:")
        print(f"   Win Rate: {win_rate:.2%} {'✅' if win_rate > 0.55 else '❌'}")
        print(f"   Profit Factor: {profit_factor:.2f} {'✅' if profit_factor > 2.0 else '❌'}")
        print(f"   Wins: {wins} | Losses: {losses}")
        print(f"   Gross Profit: ${gross_profit:,.2f}")
        print(f"   Gross Loss: ${gross_loss:,.2f}")

        # Risk limits analysis
        print(f"\n🛡️  RISK LIMITS (Profile: {self.profile}):")
        print(f"   Max Position Size: ${self.risk_limits.max_position_size:,.0f}")
        print(f"   Max Trades/Game: {self.risk_limits.max_trades_per_game}")
        print(f"   Min Edge: {self.risk_limits.min_edge_threshold:.1%}")

        # Overall assessment
        print("\n" + "="*80)
        print("ASSESSMENT:")

        passing_criteria = []
        if sharpe >= 1.5:
            passing_criteria.append("✅ Sharpe Ratio > 1.5")
        else:
            passing_criteria.append(f"❌ Sharpe Ratio {sharpe:.3f} < 1.5 (target)")

        if max_drawdown <= 0.15:
            passing_criteria.append("✅ Max Drawdown < 15%")
        else:
            passing_criteria.append(f"❌ Max Drawdown {max_drawdown:.2%} > 15% (target)")

        if profit_factor >= 2.0:
            passing_criteria.append("✅ Profit Factor > 2.0")
        else:
            passing_criteria.append(f"❌ Profit Factor {profit_factor:.2f} < 2.0 (target)")

        if win_rate >= 0.55:
            passing_criteria.append("✅ Win Rate > 55%")
        else:
            passing_criteria.append(f"❌ Win Rate {win_rate:.2%} < 55% (target)")

        for criterion in passing_criteria:
            print(f"   {criterion}")

        # Final verdict
        all_pass = all("✅" in c for c in passing_criteria)

        print("\n" + "="*80)
        if all_pass:
            print("🎉 READY FOR PAPER TRADING!")
            print("\nNext steps:")
            print("  1. Run paper trading for 7-14 days")
            print("  2. Monitor for unexpected behavior")
            print("  3. Start with small real money positions ($50-100)")
        else:
            print("⚠️  NOT READY FOR REAL MONEY TRADING")
            print("\nRecommendations:")
            if sharpe < 1.5:
                print("  - Increase min_edge_threshold to be more selective")
                print("  - Reduce trade_probability to trade less frequently")
            if max_drawdown > 0.15:
                print("  - Reduce max_position_size")
                print("  - Lower max_portfolio_exposure")
                print("  - Decrease Kelly fraction")
            if profit_factor < 2.0:
                print("  - Increase min_edge_threshold")
                print("  - Improve stop loss levels")
            if win_rate < 0.55:
                print("  - Require higher information quality for informed traders")
                print("  - Increase edge thresholds")

        print("="*80)

        return {
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'profit_factor': profit_factor,
            'win_rate': win_rate,
            'total_return': total_return,
            'ready_to_trade': all_pass
        }


def main():
    """Run optimized backtest."""
    parser = argparse.ArgumentParser(
        description='Run optimized backtest with risk management'
    )
    parser.add_argument(
        '--games',
        type=int,
        default=50,
        help='Number of games to simulate (default: 50)'
    )
    parser.add_argument(
        '--profile',
        type=str,
        default='conservative',
        choices=['conservative', 'aggressive', 'ultra_conservative'],
        help='Risk profile (default: conservative)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000)'
    )
    parser.add_argument(
        '--use-real-data',
        action='store_true',
        help='Use real Kalshi historical data'
    )
    parser.add_argument(
        '--season',
        type=int,
        default=2024,
        help='NFL season for real data (default: 2024)'
    )

    args = parser.parse_args()

    # Run backtest
    print("="*80)
    print("OPTIMIZED BACKTEST WITH RISK MANAGEMENT")
    print("="*80)

    backtest = OptimizedBacktest(
        n_games=args.games,
        profile=args.profile,
        initial_capital=args.capital,
        use_real_data=args.use_real_data,
        season=args.season
    )

    # Run simulation
    results = backtest.run_backtest()

    # Extract DataFrame from results dict
    results_df = results.get('equity_curve') if isinstance(results, dict) else results

    # Analyze with enhanced metrics
    metrics = backtest.analyze_results(results_df)

    # Save results
    output_dir = Path('backtest_results')
    output_dir.mkdir(exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'optimized_backtest_{args.profile}_{timestamp}.csv'
    results_df.to_csv(results_file, index=False)

    print(f"\n💾 Results saved to: {results_file}")

    # Exit code based on readiness
    sys.exit(0 if metrics['ready_to_trade'] else 1)


if __name__ == '__main__':
    main()
