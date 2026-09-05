"""Printability: door-edge MFE path vs typed target. Unprintable → SKIP, not scored."""

from __future__ import annotations

import numpy as np

from v6rl.book import Book
from v6rl.product import MIN_LOT, RISK_PCT_MAX, RISK_PCT_MIN, RISK_WALK, START_EQUITY, TRAIN_BINS
from v6rl.symbols import SPECS, dollars_per_price, max_lot_leverage
from v6rl.tape import Tape


def day_best_path_dollars(
    tape: Tape,
    date,
    risk_pct: float,
    equity: float = START_EQUITY,
) -> float:
    book = Book()
    book.equity = float(equity)
    book.reset_day(1.0, float(risk_pct))
    a, b = tape.day_slice(date)
    sess = np.asarray(tape.session[a:b], dtype=bool)
    risk = float(book.risk_dollars)
    eq = float(equity)
    total = 0.0
    for i, s in enumerate(tape.symbols):
        valid = np.asarray(tape.valid[a:b, i], dtype=bool) & sess
        mid = np.asarray(tape.close[a:b, i], dtype=np.float64)
        atr = np.asarray(tape.atr5[a:b, i], dtype=np.float64)
        door = np.asarray(tape.door_buy[a:b, i], dtype=bool) | np.asarray(tape.door_sell[a:b, i], dtype=bool)
        ok = valid & (mid > 0) & door
        if not bool(ok.any()):
            continue
        spec = SPECS[s]
        heat = np.maximum(atr, spec.spread_price)
        d_heat = dollars_per_price(s, 1.0) * heat
        by_risk = np.where(d_heat <= 1e-9, MIN_LOT, risk / np.maximum(d_heat, 1e-12))
        by_lev = np.array([max_lot_leverage(s, eq, float(m)) if m > 0 else 0.0 for m in mid])
        cap = np.maximum(0.0, np.minimum(by_risk, by_lev))
        pay = np.maximum(tape.mfe_buy_dpl[a:b, i], tape.mfe_sell_dpl[a:b, i]).astype(np.float64) * cap
        pay = np.where(ok, pay, 0.0)
        total += float(pay.max()) if pay.size else 0.0
    return total


def day_can_print(
    tape: Tape,
    date,
    target_pct: float,
    risk_pct: float,
    equity: float = START_EQUITY,
) -> bool:
    target = float(equity) * float(target_pct) / 100.0
    if target <= 1e-9:
        return False
    return day_best_path_dollars(tape, date, float(risk_pct), equity) + 1e-9 >= target


def draw_funded_tr(rng, tape: Tape, date, equity: float, bins=None) -> tuple[float, float, bool]:
    pool = tuple(bins) if bins is not None else tuple(TRAIN_BINS)
    if not pool:
        pool = tuple(TRAIN_BINS)
    for _ in range(8):
        target = float(rng.choice(pool))
        risk = float(rng.uniform(RISK_PCT_MIN, RISK_PCT_MAX))
        if day_can_print(tape, date, target, risk, equity):
            return target, risk, True
    for target in sorted(float(b) for b in pool):
        for risk in RISK_WALK:
            if day_can_print(tape, date, target, risk, equity):
                return target, risk, True
    target = float(min(pool))
    risk = float(RISK_PCT_MAX)
    return target, risk, bool(day_can_print(tape, date, target, risk, equity))
