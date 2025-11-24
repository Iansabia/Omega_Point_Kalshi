import numpy as np
from numba import jit


@jit(nopython=True, fastmath=True)
def calculate_fills_optimized(incoming_qty: float, resting_qtys: np.array) -> np.array:
    """
    Optimized calculation of fill quantities for a vector of resting orders.
    Returns array of fill amounts for each resting order.
    """
    fills = np.zeros_like(resting_qtys)
    remaining = incoming_qty

    for i in range(len(resting_qtys)):
        if remaining <= 0:
            break

        fill = min(remaining, resting_qtys[i])
        fills[i] = fill
        remaining -= fill

    return fills


@jit(nopython=True, fastmath=True)
def calculate_vwap(prices: np.array, quantities: np.array) -> float:
    """
    Calculate Volume Weighted Average Price.
    """
    total_vol = np.sum(quantities)
    if total_vol == 0:
        return 0.0
    return np.sum(prices * quantities) / total_vol
