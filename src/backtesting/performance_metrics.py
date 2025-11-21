"""
Performance metrics for backtesting and live trading.

Implements standard trading metrics plus prediction market-specific measures.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from price series."""
    return prices.pct_change().fillna(0)

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of periods per year for annualization

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    return np.sqrt(periods_per_year) * (excess_returns.mean() / excess_returns.std())

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation only).

    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0

    downside_std = downside_returns.std()
    return np.sqrt(periods_per_year) * (excess_returns.mean() / downside_std)

def calculate_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Args:
        equity_curve: Series of cumulative portfolio values

    Returns:
        Maximum drawdown as a decimal (e.g., 0.15 for 15%)
    """
    if len(equity_curve) == 0:
        return 0.0

    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max
    return abs(drawdown.min())

def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annualized return / max drawdown).

    Args:
        returns: Series of returns
        periods_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) == 0:
        return 0.0

    equity_curve = (1 + returns).cumprod()
    max_dd = calculate_max_drawdown(equity_curve)

    if max_dd == 0:
        return 0.0

    annual_return = (equity_curve.iloc[-1] ** (periods_per_year / len(returns))) - 1
    return annual_return / max_dd

def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate percentage of winning trades/periods.

    Args:
        returns: Series of returns

    Returns:
        Win rate as a decimal (e.g., 0.55 for 55%)
    """
    if len(returns) == 0:
        return 0.0

    winning_periods = (returns > 0).sum()
    return winning_periods / len(returns)

def calculate_profit_factor(returns: pd.Series) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        returns: Series of returns

    Returns:
        Profit factor
    """
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())

    if losses == 0:
        return np.inf if profits > 0 else 0.0

    return profits / losses

def calculate_brier_score(forecasts: np.ndarray, outcomes: np.ndarray) -> float:
    """
    Calculate Brier score for probability forecasts.

    Lower is better. Perfect score is 0, worst is 1.

    Args:
        forecasts: Array of probability forecasts [0, 1]
        outcomes: Array of actual outcomes (0 or 1)

    Returns:
        Brier score
    """
    if len(forecasts) == 0:
        return 1.0

    return np.mean((forecasts - outcomes) ** 2)

def calculate_log_loss(forecasts: np.ndarray, outcomes: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate logarithmic loss (cross-entropy loss).

    Lower is better.

    Args:
        forecasts: Array of probability forecasts [0, 1]
        outcomes: Array of actual outcomes (0 or 1)
        eps: Small value to avoid log(0)

    Returns:
        Log loss
    """
    if len(forecasts) == 0:
        return np.inf

    # Clip forecasts to avoid log(0)
    forecasts = np.clip(forecasts, eps, 1 - eps)

    return -np.mean(outcomes * np.log(forecasts) + (1 - outcomes) * np.log(1 - forecasts))

def calculate_avg_trade_return(trade_history: List[Dict]) -> float:
    """
    Calculate average return per trade.

    Args:
        trade_history: List of trade dictionaries with 'pnl' field

    Returns:
        Average trade return
    """
    if not trade_history:
        return 0.0

    returns = [trade.get('pnl', 0) for trade in trade_history]
    return np.mean(returns)

def calculate_avg_win_loss_ratio(trade_history: List[Dict]) -> float:
    """
    Calculate average win / average loss ratio.

    Args:
        trade_history: List of trade dictionaries with 'pnl' field

    Returns:
        Win/loss ratio
    """
    if not trade_history:
        return 0.0

    pnls = [trade.get('pnl', 0) for trade in trade_history]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]

    if not wins or not losses:
        return 0.0

    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return np.inf if avg_win > 0 else 0.0

    return avg_win / avg_loss

def generate_performance_report(
    returns: pd.Series,
    equity_curve: Optional[pd.Series] = None,
    trade_history: Optional[List[Dict]] = None,
    forecasts: Optional[np.ndarray] = None,
    outcomes: Optional[np.ndarray] = None,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Generate comprehensive performance report.

    Args:
        returns: Series of returns
        equity_curve: Series of cumulative portfolio values
        trade_history: List of trade dictionaries
        forecasts: Array of probability forecasts (for prediction markets)
        outcomes: Array of actual outcomes (for prediction markets)
        periods_per_year: Number of periods per year

    Returns:
        Dictionary of performance metrics
    """
    if equity_curve is None and len(returns) > 0:
        equity_curve = (1 + returns).cumprod()

    metrics = {}

    # Basic statistics
    metrics['total_return'] = equity_curve.iloc[-1] - 1 if equity_curve is not None and len(equity_curve) > 0 else 0.0
    metrics['annual_return'] = (equity_curve.iloc[-1] ** (periods_per_year / len(returns))) - 1 if equity_curve is not None and len(equity_curve) > 0 else 0.0
    metrics['volatility'] = returns.std() * np.sqrt(periods_per_year) if len(returns) > 0 else 0.0

    # Risk-adjusted returns
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, periods_per_year=periods_per_year)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, periods_per_year=periods_per_year)
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, periods_per_year=periods_per_year)

    # Drawdown metrics
    if equity_curve is not None and len(equity_curve) > 0:
        metrics['max_drawdown'] = calculate_max_drawdown(equity_curve)
    else:
        metrics['max_drawdown'] = 0.0

    # Trade statistics
    metrics['win_rate'] = calculate_win_rate(returns)
    metrics['profit_factor'] = calculate_profit_factor(returns)

    if trade_history:
        metrics['num_trades'] = len(trade_history)
        metrics['avg_trade_return'] = calculate_avg_trade_return(trade_history)
        metrics['avg_win_loss_ratio'] = calculate_avg_win_loss_ratio(trade_history)

    # Prediction market metrics
    if forecasts is not None and outcomes is not None:
        metrics['brier_score'] = calculate_brier_score(forecasts, outcomes)
        metrics['log_loss'] = calculate_log_loss(forecasts, outcomes)

    return metrics

def print_performance_report(metrics: Dict[str, float]):
    """
    Print formatted performance report.

    Args:
        metrics: Dictionary of performance metrics
    """
    logger.info("=" * 60)
    logger.info("PERFORMANCE REPORT")
    logger.info("=" * 60)

    logger.info("\nReturns:")
    logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
    logger.info(f"  Annual Return: {metrics.get('annual_return', 0):.2%}")
    logger.info(f"  Volatility (Ann.): {metrics.get('volatility', 0):.2%}")

    logger.info("\nRisk-Adjusted Returns:")
    logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
    logger.info(f"  Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
    logger.info(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")

    logger.info("\nDrawdown:")
    logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")

    logger.info("\nTrade Statistics:")
    logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
    logger.info(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")

    if 'num_trades' in metrics:
        logger.info(f"  Number of Trades: {metrics['num_trades']}")
        logger.info(f"  Avg Trade Return: {metrics.get('avg_trade_return', 0):.4f}")
        logger.info(f"  Avg Win/Loss Ratio: {metrics.get('avg_win_loss_ratio', 0):.2f}")

    if 'brier_score' in metrics:
        logger.info("\nPrediction Market Metrics:")
        logger.info(f"  Brier Score: {metrics['brier_score']:.4f}")
        logger.info(f"  Log Loss: {metrics['log_loss']:.4f}")

    logger.info("=" * 60)
