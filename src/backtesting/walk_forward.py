"""
Walk-forward optimization for backtesting.

Implements time-series cross-validation with rolling windows to prevent overfitting.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable, Any, Optional
from dataclasses import dataclass
import logging
from scipy.optimize import minimize, differential_evolution
from concurrent.futures import ProcessPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardWindow:
    """
    Represents a single walk-forward optimization window.
    """
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    fold_number: int


class WalkForwardOptimizer:
    """
    Walk-forward optimization framework for time-series backtesting.

    Implements rolling window optimization to validate strategy robustness
    and prevent overfitting.
    """

    def __init__(
        self,
        train_ratio: float = 0.7,
        n_folds: int = 5,
        anchored: bool = False,
        gap: int = 0,
        parallel: bool = False,
        n_jobs: int = 4
    ):
        """
        Initialize walk-forward optimizer.

        Args:
            train_ratio: Ratio of training data (0-1), e.g., 0.7 = 70% train, 30% test
            n_folds: Number of walk-forward folds
            anchored: If True, training window grows; if False, window slides
            gap: Number of periods to skip between train and test to avoid lookahead
            parallel: Whether to run folds in parallel
            n_jobs: Number of parallel jobs
        """
        self.train_ratio = train_ratio
        self.n_folds = n_folds
        self.anchored = anchored
        self.gap = gap
        self.parallel = parallel
        self.n_jobs = n_jobs

    def create_folds(self, data_length: int) -> List[WalkForwardWindow]:
        """
        Create walk-forward windows for optimization.

        Args:
            data_length: Total number of data points

        Returns:
            List of WalkForwardWindow objects
        """
        if data_length < self.n_folds * 2:
            raise ValueError(f"Data length {data_length} too short for {self.n_folds} folds")

        windows = []

        # Calculate initial window sizes
        initial_train_size = int(data_length * self.train_ratio / self.n_folds)
        test_size = int(data_length * (1 - self.train_ratio) / self.n_folds)

        for fold in range(self.n_folds):
            if self.anchored:
                # Anchored: Training window grows with each fold
                train_start = 0
                train_end = initial_train_size + fold * test_size
            else:
                # Rolling: Training window slides
                train_start = fold * test_size
                train_end = train_start + initial_train_size

            # Add gap to avoid lookahead bias
            test_start = train_end + self.gap
            test_end = test_start + test_size

            # Ensure we don't exceed data bounds
            if test_end > data_length:
                break

            window = WalkForwardWindow(
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                fold_number=fold
            )

            windows.append(window)

        logger.info(f"Created {len(windows)} walk-forward windows")
        return windows

    def optimize_fold(
        self,
        window: WalkForwardWindow,
        objective_function: Callable,
        param_bounds: Dict[str, Tuple[float, float]],
        data: pd.DataFrame,
        method: str = 'differential_evolution',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimize parameters for a single fold.

        Args:
            window: Walk-forward window
            objective_function: Function to minimize (should return scalar)
            param_bounds: Dictionary of parameter names to (min, max) tuples
            data: Full dataset
            method: Optimization method ('differential_evolution', 'scipy')
            **kwargs: Additional arguments for optimizer

        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Optimizing fold {window.fold_number}...")

        # Extract training data
        train_data = data.iloc[window.train_start:window.train_end]
        test_data = data.iloc[window.test_start:window.test_end]

        # Prepare bounds for optimizer
        param_names = list(param_bounds.keys())
        bounds = [param_bounds[name] for name in param_names]

        # Wrapper function that unpacks parameters
        def objective_wrapper(params):
            param_dict = {name: val for name, val in zip(param_names, params)}
            return objective_function(train_data, param_dict)

        # Run optimization
        if method == 'differential_evolution':
            result = differential_evolution(
                objective_wrapper,
                bounds,
                seed=window.fold_number,
                workers=1,
                **kwargs
            )
        elif method == 'scipy':
            # Start from middle of bounds
            x0 = [(b[0] + b[1]) / 2 for b in bounds]
            result = minimize(
                objective_wrapper,
                x0,
                bounds=bounds,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")

        # Extract optimal parameters
        optimal_params = {name: val for name, val in zip(param_names, result.x)}

        # Evaluate on test set
        test_score = objective_function(test_data, optimal_params)

        logger.info(f"Fold {window.fold_number}: Train score={result.fun:.4f}, Test score={test_score:.4f}")

        return {
            'fold': window.fold_number,
            'window': window,
            'optimal_params': optimal_params,
            'train_score': result.fun,
            'test_score': test_score,
            'optimization_result': result,
            'train_data': train_data,
            'test_data': test_data
        }

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        objective_function: Callable,
        param_bounds: Dict[str, Tuple[float, float]],
        method: str = 'differential_evolution',
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run complete walk-forward optimization.

        Args:
            data: Full dataset (must be time-ordered)
            objective_function: Function to minimize
            param_bounds: Parameter bounds for optimization
            method: Optimization method
            **kwargs: Additional optimizer arguments

        Returns:
            Dictionary with comprehensive results
        """
        logger.info("=" * 70)
        logger.info("WALK-FORWARD OPTIMIZATION")
        logger.info("=" * 70)
        logger.info(f"Data length: {len(data)}")
        logger.info(f"Train ratio: {self.train_ratio}")
        logger.info(f"Number of folds: {self.n_folds}")
        logger.info(f"Anchored: {self.anchored}")
        logger.info(f"Gap: {self.gap}")
        logger.info("=" * 70)

        # Create folds
        windows = self.create_folds(len(data))

        # Optimize each fold
        fold_results = []

        if self.parallel and self.n_jobs > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
                futures = {
                    executor.submit(
                        self.optimize_fold,
                        window,
                        objective_function,
                        param_bounds,
                        data,
                        method,
                        **kwargs
                    ): window
                    for window in windows
                }

                for future in as_completed(futures):
                    fold_results.append(future.result())
        else:
            # Sequential execution
            for window in windows:
                result = self.optimize_fold(
                    window,
                    objective_function,
                    param_bounds,
                    data,
                    method,
                    **kwargs
                )
                fold_results.append(result)

        # Sort by fold number
        fold_results.sort(key=lambda x: x['fold'])

        # Calculate aggregate statistics
        train_scores = [r['train_score'] for r in fold_results]
        test_scores = [r['test_score'] for r in fold_results]

        # Check for overfitting
        avg_train = np.mean(train_scores)
        avg_test = np.mean(test_scores)
        overfit_ratio = avg_test / avg_train if avg_train != 0 else np.inf

        logger.info("\n" + "=" * 70)
        logger.info("WALK-FORWARD RESULTS")
        logger.info("=" * 70)
        logger.info(f"Average In-Sample Score: {avg_train:.4f}")
        logger.info(f"Average Out-of-Sample Score: {avg_test:.4f}")
        logger.info(f"Overfit Ratio (OOS/IS): {overfit_ratio:.2f}")
        logger.info(f"OOS Std Dev: {np.std(test_scores):.4f}")
        logger.info("=" * 70)

        # Determine if overfitting
        is_overfitting = overfit_ratio > 1.5  # OOS score 50% worse than IS

        if is_overfitting:
            logger.warning("⚠️  Potential overfitting detected!")
        else:
            logger.info("✓ Strategy appears robust")

        return {
            'fold_results': fold_results,
            'train_scores': train_scores,
            'test_scores': test_scores,
            'avg_train_score': avg_train,
            'avg_test_score': avg_test,
            'overfit_ratio': overfit_ratio,
            'is_overfitting': is_overfitting,
            'n_folds': len(fold_results),
            'windows': windows
        }

    def get_optimal_params_ensemble(
        self,
        results: Dict[str, Any],
        method: str = 'median'
    ) -> Dict[str, float]:
        """
        Get ensemble optimal parameters from all folds.

        Args:
            results: Results from run_walk_forward
            method: How to ensemble ('median', 'mean', 'best_test')

        Returns:
            Dictionary of optimal parameters
        """
        fold_results = results['fold_results']

        if method == 'best_test':
            # Use parameters from fold with best test score
            best_fold = min(fold_results, key=lambda x: x['test_score'])
            return best_fold['optimal_params']

        # Get all parameter names
        param_names = list(fold_results[0]['optimal_params'].keys())

        # Collect parameter values across folds
        param_values = {name: [] for name in param_names}
        for fold in fold_results:
            for name, value in fold['optimal_params'].items():
                param_values[name].append(value)

        # Ensemble
        if method == 'median':
            return {name: np.median(values) for name, values in param_values.items()}
        elif method == 'mean':
            return {name: np.mean(values) for name, values in param_values.items()}
        else:
            raise ValueError(f"Unknown ensemble method: {method}")


def sharpe_ratio_objective(returns: pd.Series) -> float:
    """
    Objective function for maximizing Sharpe ratio (minimizes negative Sharpe).

    Args:
        returns: Series of returns

    Returns:
        Negative Sharpe ratio (for minimization)
    """
    if len(returns) == 0 or returns.std() == 0:
        return 100.0  # Large penalty for invalid strategies

    sharpe = np.sqrt(252) * (returns.mean() / returns.std())
    return -sharpe  # Negative for minimization


def sortino_ratio_objective(returns: pd.Series) -> float:
    """
    Objective function for maximizing Sortino ratio.

    Args:
        returns: Series of returns

    Returns:
        Negative Sortino ratio
    """
    if len(returns) == 0:
        return 100.0

    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return -10.0  # Good if no downside

    sortino = np.sqrt(252) * (returns.mean() / downside_returns.std())
    return -sortino


def calmar_ratio_objective(returns: pd.Series) -> float:
    """
    Objective function for maximizing Calmar ratio.

    Args:
        returns: Series of returns

    Returns:
        Negative Calmar ratio
    """
    if len(returns) == 0:
        return 100.0

    equity_curve = (1 + returns).cumprod()
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = abs(drawdown.min())

    if max_dd == 0:
        return -10.0  # Good if no drawdown

    annual_return = (equity_curve.iloc[-1] ** (252 / len(returns))) - 1
    calmar = annual_return / max_dd

    return -calmar


def print_walk_forward_report(results: Dict[str, Any]):
    """
    Print formatted walk-forward optimization report.

    Args:
        results: Results from run_walk_forward
    """
    logger.info("=" * 70)
    logger.info("WALK-FORWARD OPTIMIZATION REPORT")
    logger.info("=" * 70)

    logger.info(f"\nNumber of Folds: {results['n_folds']}")
    logger.info(f"Average In-Sample Score: {results['avg_train_score']:.4f}")
    logger.info(f"Average Out-of-Sample Score: {results['avg_test_score']:.4f}")
    logger.info(f"Overfit Ratio: {results['overfit_ratio']:.2f}")

    if results['is_overfitting']:
        logger.info("Status: ⚠️  OVERFITTING DETECTED")
    else:
        logger.info("Status: ✓ ROBUST")

    logger.info("\nFold-by-Fold Results:")
    logger.info("-" * 70)
    logger.info(f"{'Fold':<6} {'Train Score':<15} {'Test Score':<15} {'Ratio':<10}")
    logger.info("-" * 70)

    for fold_result in results['fold_results']:
        fold = fold_result['fold']
        train_score = fold_result['train_score']
        test_score = fold_result['test_score']
        ratio = test_score / train_score if train_score != 0 else np.inf

        logger.info(f"{fold:<6} {train_score:<15.4f} {test_score:<15.4f} {ratio:<10.2f}")

    logger.info("-" * 70)

    # Show optimal parameters for each fold
    logger.info("\nOptimal Parameters by Fold:")
    logger.info("-" * 70)

    for fold_result in results['fold_results']:
        logger.info(f"\nFold {fold_result['fold']}:")
        for param, value in fold_result['optimal_params'].items():
            logger.info(f"  {param}: {value:.4f}")

    logger.info("=" * 70)
