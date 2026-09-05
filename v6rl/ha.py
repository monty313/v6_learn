"""Heikin-Ashi paint and the locked HA door. Door printed ≠ forced enter."""

from __future__ import annotations

import numpy as np


def heikin_ashi(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ha_open, ha_close, green (bool), run length N of same color."""
    o = np.asarray(open_, dtype=np.float64)
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    n = c.size
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty(n, dtype=np.float64)
    if n == 0:
        return ha_open, ha_close, np.zeros(0, dtype=np.bool_), np.zeros(0, dtype=np.int32)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    green = ha_close >= ha_open
    run = np.ones(n, dtype=np.int32)
    for i in range(1, n):
        if green[i] == green[i - 1]:
            run[i] = run[i - 1] + 1
    return ha_open, ha_close, green, run


def door_buy(*, n15: int, green15: bool, n5: int, green5: bool, n1: int, green1: bool) -> bool:
    """Buy lock: 15m HA ≥2 green, 5m HA ≥2 green, 1m HA ≥1 red."""
    return (
        int(n15) >= 2
        and bool(green15)
        and int(n5) >= 2
        and bool(green5)
        and int(n1) >= 1
        and (not bool(green1))
    )


def door_sell(*, n15: int, green15: bool, n5: int, green5: bool, n1: int, green1: bool) -> bool:
    """Sell lock: 15m HA ≥2 red, 5m HA ≥2 red, 1m HA ≥1 green."""
    return (
        int(n15) >= 2
        and (not bool(green15))
        and int(n5) >= 2
        and (not bool(green5))
        and int(n1) >= 1
        and bool(green1)
    )


def door_flags(
    n15: np.ndarray,
    g15: np.ndarray,
    n5: np.ndarray,
    g5: np.ndarray,
    n1: np.ndarray,
    g1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    buy = (n15 >= 2) & g15 & (n5 >= 2) & g5 & (n1 >= 1) & (~g1)
    sell = (n15 >= 2) & (~g15) & (n5 >= 2) & (~g5) & (n1 >= 1) & g1
    return buy.astype(np.bool_), sell.astype(np.bool_)
