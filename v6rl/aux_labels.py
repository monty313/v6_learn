"""Channel B future labels (CARD_10_90_HITGATE). Inputs at t0 only; labels walk bars AFTER t0."""

from __future__ import annotations

import numpy as np

MOMENTUM_HOLD_HORIZON_5M = 12
MOMENTUM_HOLD_EDGE_ATR = 0.25
MOMENTUM_HOLD_ADVERSE_ATR = 1.0
MOMENTUM_HOLD_HTF_N_MIN = 2
PULLBACK_RES_HORIZON_5M = 24
PULLBACK_RES_FAV_ATR = 0.50
PULLBACK_RES_ADV_ATR = 0.75


def _fav_adv(side: int, high: float, low: float, fill: float, fav_d: float, adv_d: float) -> tuple[bool, bool]:
    if side > 0:
        return high >= fill + fav_d, low <= fill - adv_d
    return low <= fill - fav_d, high >= fill + adv_d


def _htf_ok(tape, t: int, inst: int, side: int) -> bool:
    n = int(tape.ha_n["15m"][t, inst])
    if n < MOMENTUM_HOLD_HTF_N_MIN:
        return False
    green = bool(tape.ha_green["15m"][t, inst])
    return green if side > 0 else (not green)


def label_momentum_hold(tape, t0: int, inst: int, side: int) -> float:
    """1 = fav 0.25 ATR first within 12×5m; 0 = adv 1.0 ATR or HTF flip; -1 = no touch / ineligible."""
    if side == 0 or t0 < 0 or t0 >= tape.n:
        return -1.0
    if not _htf_ok(tape, t0, inst, side):
        return -1.0
    atr = float(tape.atr5[t0, inst])
    fill = float(tape.close[t0, inst])
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(fill)):
        return -1.0
    fav_d = MOMENTUM_HOLD_EDGE_ATR * atr
    adv_d = MOMENTUM_HOLD_ADVERSE_ATR * atr
    last5 = float(tape.close5[t0, inst])
    seen = 0
    for t in range(t0 + 1, tape.n):
        c5 = float(tape.close5[t, inst])
        if np.isfinite(c5) and np.isfinite(last5) and c5 != last5:
            seen += 1
            last5 = c5
        if not _htf_ok(tape, t, inst, side):
            return 0.0
        hi, lo = float(tape.high[t, inst]), float(tape.low[t, inst])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            if seen >= MOMENTUM_HOLD_HORIZON_5M:
                break
            continue
        fav, adv = _fav_adv(side, hi, lo, fill, fav_d, adv_d)
        if fav and adv:
            return 0.0
        if adv:
            return 0.0
        if fav:
            return 1.0
        if seen >= MOMENTUM_HOLD_HORIZON_5M:
            break
    return -1.0


def label_pullback_resolution(tape, t0: int, inst: int, side: int, fill_price: float) -> float:
    """1 = fav 0.50 ATR before adv 0.75 ATR in 24×5m; 0 = adv first / same-bar; -1 = no touch."""
    if side == 0 or t0 < 0 or t0 >= tape.n:
        return -1.0
    atr = float(tape.atr5[t0, inst])
    fill = float(fill_price)
    if not (np.isfinite(atr) and atr > 0 and np.isfinite(fill)):
        return -1.0
    fav_d = PULLBACK_RES_FAV_ATR * atr
    adv_d = PULLBACK_RES_ADV_ATR * atr
    last5 = float(tape.close5[t0, inst])
    seen = 0
    for t in range(t0 + 1, tape.n):
        c5 = float(tape.close5[t, inst])
        if np.isfinite(c5) and np.isfinite(last5) and c5 != last5:
            seen += 1
            last5 = c5
        hi, lo = float(tape.high[t, inst]), float(tape.low[t, inst])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            if seen >= PULLBACK_RES_HORIZON_5M:
                break
            continue
        fav, adv = _fav_adv(side, hi, lo, fill, fav_d, adv_d)
        if fav and adv:
            return 0.0
        if adv:
            return 0.0
        if fav:
            return 1.0
        if seen >= PULLBACK_RES_HORIZON_5M:
            break
    return -1.0
