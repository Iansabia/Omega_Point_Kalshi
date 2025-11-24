"""
Monte Carlo simulation for risk assessment and strategy validation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo simulation for backtesting results analysis.

    Uses trade resampling and bootstrap methods to estimate strategy robustness.
    """

    def __init__(self, random_seed: Optional[int] = None):
        if random_seed is not None:
            np.random.seed(random_seed)

    def resample_trades(self, trades: List[float], n_simulations: int = 1000) -> List[Dict]:
        """
        Resample trades with replacement to simulate alternate histories.

        Args:
            trades: List of trade returns (can be P&L or percentages)
            n_simulations: Number of Monte Carlo simulations

        Returns:
            List of simulation results with stats
        """
        if len(trades) == 0:
            logger.warning("No trades to resample")
            return []

        results = []

        for i in range(n_simulations):
            # Resample trades with replacement
            resampled = np.random.choice(trades, size=len(trades), replace=True)

            # Calculate metrics
            cumulative_returns = np.cumsum(resampled)
            final_return = cumulative_returns[-1]
            max_dd = self._calculate_drawdown(cumulative_returns)
            sharpe = self._calculate_sharpe(resampled)

            results.append(
                {
                    "simulation": i,
                    "final_return": final_return,
                    "max_drawdown": max_dd,
                    "sharpe_ratio": sharpe,
                    "win_rate": (resampled > 0).sum() / len(resampled),
                }
            )

        return results

    def estimate_confidence_intervals(
        self, results: List[Dict], metric: str = "final_return", confidence_levels: List[float] = [0.05, 0.5, 0.95]
    ) -> Dict:
        """
        Calculate confidence intervals from Monte Carlo results.

        Args:
            results: List of simulation results
            metric: Metric to analyze
            confidence_levels: Percentiles to calculate (e.g., [0.05, 0.5, 0.95])

        Returns:
            Dictionary with percentile values
        """
        if not results:
            return {}

        values = [r[metric] for r in results]
        percentiles = np.percentile(values, [level * 100 for level in confidence_levels])

        intervals = {}
        for level, value in zip(confidence_levels, percentiles):
            intervals[f"p{int(level*100)}"] = value

        return intervals

    def calculate_probability_of_ruin(
        self, trades: List[float], initial_capital: float, ruin_threshold: float = 0.5, n_simulations: int = 1000
    ) -> float:
        """
        Estimate probability of losing more than ruin_threshold of capital.

        Args:
            trades: List of trade P&L values (absolute amounts)
            initial_capital: Starting capital
            ruin_threshold: Fraction of capital loss that constitutes ruin (e.g., 0.5 = 50%)
            n_simulations: Number of simulations

        Returns:
            Probability of ruin (0 to 1)
        """
        if len(trades) == 0:
            return 1.0

        ruin_count = 0
        ruin_capital = initial_capital * (1 - ruin_threshold)

        for _ in range(n_simulations):
            # Resample trades
            resampled = np.random.choice(trades, size=len(trades), replace=True)

            # Simulate capital path
            capital = initial_capital
            for trade in resampled:
                capital += trade
                if capital <= ruin_capital:
                    ruin_count += 1
                    break

        return ruin_count / n_simulations

    def simulate_future_paths(self, historical_returns: pd.Series, n_days: int = 252, n_simulations: int = 1000) -> np.ndarray:
        """
        Simulate future price/return paths based on historical distribution.

        Args:
            historical_returns: Historical return series
            n_days: Number of days to simulate forward
            n_simulations: Number of simulated paths

        Returns:
            Array of shape (n_simulations, n_days) with simulated returns
        """
        if len(historical_returns) == 0:
            return np.zeros((n_simulations, n_days))

        # Estimate parameters from historical returns
        mean_return = historical_returns.mean()
        std_return = historical_returns.std()

        # Generate simulated paths
        simulated_paths = np.random.normal(loc=mean_return, scale=std_return, size=(n_simulations, n_days))

        return simulated_paths

    def analyze_drawdown_distribution(self, trades: List[float], n_simulations: int = 1000) -> Dict:
        """
        Analyze distribution of maximum drawdown from Monte Carlo simulations.

        Args:
            trades: List of trade returns
            n_simulations: Number of simulations

        Returns:
            Dictionary with drawdown statistics
        """
        if len(trades) == 0:
            return {}

        max_drawdowns = []

        for _ in range(n_simulations):
            resampled = np.random.choice(trades, size=len(trades), replace=True)
            cumulative = np.cumsum(resampled)
            max_dd = self._calculate_drawdown(cumulative)
            max_drawdowns.append(max_dd)

        return {
            "mean_max_dd": np.mean(max_drawdowns),
            "median_max_dd": np.median(max_drawdowns),
            "p5_max_dd": np.percentile(max_drawdowns, 5),
            "p95_max_dd": np.percentile(max_drawdowns, 95),
            "worst_max_dd": np.max(max_drawdowns),
        }

    def generate_report(self, trades: List[float], initial_capital: float = 100000, n_simulations: int = 1000) -> Dict:
        """
        Generate comprehensive Monte Carlo risk report.

        Args:
            trades: List of trade returns
            initial_capital: Starting capital
            n_simulations: Number of simulations

        Returns:
            Dictionary with comprehensive risk metrics
        """
        logger.info(f"Running {n_simulations} Monte Carlo simulations...")

        # Run simulations
        results = self.resample_trades(trades, n_simulations=n_simulations)

        # Calculate confidence intervals
        final_return_ci = self.estimate_confidence_intervals(results, "final_return")
        max_dd_ci = self.estimate_confidence_intervals(results, "max_drawdown")
        sharpe_ci = self.estimate_confidence_intervals(results, "sharpe_ratio")

        # Calculate probability of ruin
        prob_ruin_50 = self.calculate_probability_of_ruin(trades, initial_capital, 0.5, n_simulations)
        prob_ruin_25 = self.calculate_probability_of_ruin(trades, initial_capital, 0.25, n_simulations)

        # Drawdown analysis
        dd_stats = self.analyze_drawdown_distribution(trades, n_simulations)

        report = {
            "n_simulations": n_simulations,
            "n_trades": len(trades),
            "final_return": {"mean": np.mean([r["final_return"] for r in results]), **final_return_ci},
            "max_drawdown": {"mean": np.mean([r["max_drawdown"] for r in results]), **max_dd_ci, **dd_stats},
            "sharpe_ratio": {"mean": np.mean([r["sharpe_ratio"] for r in results]), **sharpe_ci},
            "win_rate": {
                "mean": np.mean([r["win_rate"] for r in results]),
                "median": np.median([r["win_rate"] for r in results]),
            },
            "probability_of_ruin": {"50pct_loss": prob_ruin_50, "25pct_loss": prob_ruin_25},
        }

        return report

    @staticmethod
    def _calculate_drawdown(cumulative_returns: np.ndarray) -> float:
        """Calculate maximum drawdown from cumulative returns."""
        if len(cumulative_returns) == 0:
            return 0.0

        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = cumulative_returns - running_max
        return abs(drawdown.min()) if len(drawdown) > 0 else 0.0

    @staticmethod
    def _calculate_sharpe(returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio from returns."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())


def print_monte_carlo_report(report: Dict):
    """
    Print formatted Monte Carlo report.

    Args:
        report: Monte Carlo report dictionary
    """
    logger.info("=" * 60)
    logger.info("MONTE CARLO RISK REPORT")
    logger.info("=" * 60)

    logger.info(f"\nSimulations: {report['n_simulations']}")
    logger.info(f"Trades: {report['n_trades']}")

    logger.info("\nFinal Return Distribution:")
    logger.info(f"  Mean: {report['final_return']['mean']:.2%}")
    logger.info(f"  5th Percentile: {report['final_return']['p5']:.2%}")
    logger.info(f"  Median: {report['final_return']['p50']:.2%}")
    logger.info(f"  95th Percentile: {report['final_return']['p95']:.2%}")

    logger.info("\nMaximum Drawdown Distribution:")
    logger.info(f"  Mean: {report['max_drawdown']['mean']:.2%}")
    logger.info(f"  Median: {report['max_drawdown']['median_max_dd']:.2%}")
    logger.info(f"  95th Percentile (Worst): {report['max_drawdown']['p95_max_dd']:.2%}")

    logger.info("\nSharpe Ratio Distribution:")
    logger.info(f"  Mean: {report['sharpe_ratio']['mean']:.2f}")
    logger.info(f"  5th Percentile: {report['sharpe_ratio']['p5']:.2f}")
    logger.info(f"  95th Percentile: {report['sharpe_ratio']['p95']:.2f}")

    logger.info("\nProbability of Ruin:")
    logger.info(f"  50% Capital Loss: {report['probability_of_ruin']['50pct_loss']:.2%}")
    logger.info(f"  25% Capital Loss: {report['probability_of_ruin']['25pct_loss']:.2%}")

    logger.info("=" * 60)
