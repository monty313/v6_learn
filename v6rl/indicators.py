from __future__ import annotations

import numpy as np

CCI_CONSTANT = 0.015
ATR_PERIOD = 14
FRACTAL_WING = 2


def typical_price(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    return (
        np.asarray(high, dtype=np.float64)
        + np.asarray(low, dtype=np.float64)
        + np.asarray(close, dtype=np.float64)
    ) / 3.0


def _windows(values: np.ndarray, period: int) -> np.ndarray:
    x = np.ascontiguousarray(values, dtype=np.float64)
    shape = (x.size - period + 1, period)
    strides = (x.strides[0], x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def rolling_mean(values: np.ndarray, period: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if period <= 0 or x.size < period:
        return out
    out[period - 1 :] = np.mean(_windows(x, period), axis=1)
    return out


def rolling_mean_abs_dev(values: np.ndarray, period: int, center: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.size < period:
        return out
    mid = np.asarray(center, dtype=np.float64)[period - 1 :]
    out[period - 1 :] = np.mean(np.abs(_windows(x, period) - mid[:, None]), axis=1)
    return out


def rolling_pop_std(values: np.ndarray, period: int, center: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.size < period:
        return out
    mid = np.asarray(center, dtype=np.float64)[period - 1 :]
    out[period - 1 :] = np.sqrt(np.mean((_windows(x, period) - mid[:, None]) ** 2, axis=1))
    return out


def shift_lookback(values: np.ndarray, shift: int) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if shift <= 0:
        return x.copy()
    out = np.full(x.shape, np.nan, dtype=np.float64)
    if x.size > shift:
        out[shift:] = x[:-shift]
    return out


def sma(values: np.ndarray, period: int) -> np.ndarray:
    return rolling_mean(values, period)


def sma_shifted(values: np.ndarray, period: int, shift: int) -> np.ndarray:
    return shift_lookback(sma(values, period), shift)


def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    tp = typical_price(high, low, close)
    mid = rolling_mean(tp, period)
    md = rolling_mean_abs_dev(tp, period, mid)
    out = np.full(tp.shape, np.nan, dtype=np.float64)
    ready = np.isfinite(mid) & np.isfinite(md)
    zero = ready & (md == 0.0)
    valid = ready & (md != 0.0)
    out[valid] = (tp[valid] - mid[valid]) / (CCI_CONSTANT * md[valid])
    out[zero] = 0.0
    return out


def bollinger(
    values: np.ndarray, period: int, deviation: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mid = rolling_mean(values, period)
    std = rolling_pop_std(values, period, mid)
    return mid + deviation * std, mid, mid - deviation * std


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
    h = np.asarray(high, dtype=np.float64)
    l = np.asarray(low, dtype=np.float64)
    c = np.asarray(close, dtype=np.float64)
    out = np.full(c.shape, np.nan, dtype=np.float64)
    if c.size < 2:
        return out
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    if c.size < period + 1:
        return out
    out[period] = float(np.mean(tr[1 : period + 1]))
    for i in range(period, c.size - 1):
        out[i + 1] = (out[i] * (period - 1) + tr[i + 1]) / period
    return out


def fractal_highs(high: np.ndarray, wing: int = FRACTAL_WING) -> np.ndarray:
    h = np.asarray(high, dtype=np.float64)
    out = np.full(h.shape, np.nan, dtype=np.float64)
    n = h.size
    if n < 2 * wing + 1:
        return out
    mid = h[wing : n - wing]
    cond = np.ones(mid.shape, dtype=np.bool_)
    for k in range(1, wing + 1):
        cond &= mid > h[wing - k : n - wing - k]
        cond &= mid > h[wing + k : n - wing + k]
    out[wing : n - wing] = np.where(cond, mid, np.nan)
    return out


def fractal_lows(low: np.ndarray, wing: int = FRACTAL_WING) -> np.ndarray:
    l = np.asarray(low, dtype=np.float64)
    out = np.full(l.shape, np.nan, dtype=np.float64)
    n = l.size
    if n < 2 * wing + 1:
        return out
    mid = l[wing : n - wing]
    cond = np.ones(mid.shape, dtype=np.bool_)
    for k in range(1, wing + 1):
        cond &= mid < l[wing - k : n - wing - k]
        cond &= mid < l[wing + k : n - wing + k]
    out[wing : n - wing] = np.where(cond, mid, np.nan)
    return out


def confirmed_fractal(
    peak_vals: np.ndarray, wing: int = FRACTAL_WING
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """No look-ahead: flag fires at confirmation (peak + wing)."""
    x = np.asarray(peak_vals, dtype=np.float64)
    n = x.size
    on = np.zeros(n, dtype=np.bool_)
    bars_since = np.full(n, float(n), dtype=np.float32)
    last_px = np.full(n, np.nan, dtype=np.float64)
    last_i = -1
    last_p = np.nan
    for i in range(n):
        src = i - wing
        if src >= 0 and np.isfinite(x[src]):
            on[i] = True
            last_i = i
            last_p = float(x[src])
        if last_i >= 0:
            bars_since[i] = float(i - last_i)
            last_px[i] = last_p
    return on, bars_since, last_px


def nz(value: float, default: float = 0.0) -> float:
    if value is None or not np.isfinite(value):
        return default
    return float(value)


def clip_div(num: float, den: float, lo: float = -8.0, hi: float = 8.0) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or abs(den) < 1e-12:
        return 0.0
    v = num / den
    if not np.isfinite(v):
        return 0.0
    return float(np.clip(v, lo, hi))
