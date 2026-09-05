from __future__ import annotations

from dataclasses import dataclass

from v6rl.product import LEVERAGE, MIN_LOT, SYMBOLS


@dataclass(frozen=True)
class Spec:
    symbol: str
    contract: float
    pip: float
    pip_value_per_lot: float
    spread_pips: float
    spread_price: float
    spread_cost_per_lot: float


SPECS: dict[str, Spec] = {
    "US30": Spec("US30", 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
}


def spec(symbol: str) -> Spec:
    return SPECS[symbol]


def notional(symbol: str, lots: float, price: float) -> float:
    s = SPECS[symbol]
    return s.contract * price * lots


def pnl_dollars(symbol: str, side: int, entry: float, exit_px: float, lots: float) -> float:
    s = SPECS[symbol]
    return side * (exit_px - entry) / s.pip * s.pip_value_per_lot * lots


def fill_price(symbol: str, side: int, mid: float) -> float:
    half = SPECS[symbol].spread_price / 2.0
    return mid + half if side > 0 else mid - half


def dollars_per_price(symbol: str, lots: float) -> float:
    s = SPECS[symbol]
    return lots * s.pip_value_per_lot / s.pip


def max_lot_leverage(symbol: str, equity: float, price: float) -> float:
    n1 = notional(symbol, 1.0, price)
    if n1 <= 0 or equity <= 0:
        return MIN_LOT
    return max(MIN_LOT, LEVERAGE * equity / n1)


def round_lot(lots: float) -> float:
    if lots < MIN_LOT:
        return 0.0
    steps = round(lots / MIN_LOT)
    return max(MIN_LOT, steps * MIN_LOT)


def assert_watchlist() -> None:
    for name in SYMBOLS:
        if name not in SPECS:
            raise KeyError(name)
