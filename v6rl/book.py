from __future__ import annotations

from dataclasses import dataclass, field

from v6rl.product import MAX_TICKETS_DAY, MIN_LOT, START_EQUITY
from v6rl.symbols import SPECS, dollars_per_price, fill_price, max_lot_leverage, pnl_dollars, round_lot


@dataclass
class Ticket:
    ticket_id: int
    symbol: str
    side: int
    lots: float
    entry: float
    t_index: int
    bars_held: int = 0
    bars_to_first_green: float = float("inf")
    reached_5m: bool = False
    ever_red_after_5m: bool = False
    sma_broken: tuple[bool, bool, bool, bool] = (False, False, False, False)
    bb20_cross_up: bool = False
    bb20_cross_down: bool = False
    bb200_cross_up: bool = False
    bb200_cross_down: bool = False
    bb_up_events: int = 0
    bb_down_events: int = 0
    seen_bb20_mid_up: bool = False
    seen_bb20_up_up: bool = False
    seen_bb200_mid_up: bool = False
    seen_bb200_up_up: bool = False
    seen_bb20_mid_dn: bool = False
    seen_bb20_lo_dn: bool = False
    seen_bb200_mid_dn: bool = False
    seen_bb200_lo_dn: bool = False
    prev_close5: float = float("nan")


@dataclass
class Book:
    equity: float = START_EQUITY
    day_start_equity: float = START_EQUITY
    target_pct: float = 3.0
    risk_pct: float = 2.0
    tickets: list[Ticket] = field(default_factory=list)
    next_id: int = 1
    fills_today: dict[str, int] = field(default_factory=dict)
    tickets_today: int = 0
    realized_day: float = 0.0
    realized_loss_abs: float = 0.0
    dead: str | None = None
    saw_legal: bool = False
    n_trade_closes: int = 0
    n_trade_wins: int = 0
    n_trade_losses: int = 0

    def reset_day(self, target_pct: float, risk_pct: float) -> None:
        self.day_start_equity = self.equity
        self.target_pct = float(target_pct)
        self.risk_pct = float(risk_pct)
        self.tickets.clear()
        self.fills_today.clear()
        self.tickets_today = 0
        self.realized_day = 0.0
        self.realized_loss_abs = 0.0
        self.dead = None
        self.saw_legal = False
        self.n_trade_closes = 0
        self.n_trade_wins = 0
        self.n_trade_losses = 0

    @property
    def target_dollars(self) -> float:
        return self.day_start_equity * self.target_pct / 100.0

    @property
    def risk_dollars(self) -> float:
        return self.day_start_equity * self.risk_pct / 100.0

    @property
    def ticket_floor(self) -> float:
        return self.target_dollars * 0.10

    def open_on(self, symbol: str) -> list[Ticket]:
        return [t for t in self.tickets if t.symbol == symbol]

    def side_on(self, symbol: str) -> int:
        open_t = self.open_on(symbol)
        return open_t[0].side if open_t else 0

    def symbol_pnl(self, symbol: str, mid: float) -> float:
        total = 0.0
        for t in self.open_on(symbol):
            exit_px = fill_price(symbol, -t.side, mid)
            total += pnl_dollars(symbol, t.side, t.entry, exit_px, t.lots)
        return total

    def ticket_pnl(self, ticket: Ticket, mid: float) -> float:
        exit_px = fill_price(ticket.symbol, -ticket.side, mid)
        return pnl_dollars(ticket.symbol, ticket.side, ticket.entry, exit_px, ticket.lots)

    def mark_book(self, mids: dict[str, float]) -> float:
        u = 0.0
        for t in self.tickets:
            mid = mids.get(t.symbol)
            if mid is None:
                continue
            u += self.ticket_pnl(t, mid)
        return self.realized_day + u

    def snapshot(self, mids: dict[str, float]) -> tuple[float, float, float, float]:
        pnl = self.mark_book(mids)
        tgt = self.target_dollars
        risk = self.risk_dollars
        leftover = 0.0 if tgt <= 0 else max(0.0, (tgt - pnl) / tgt)
        used = max(0.0, -pnl, self.realized_loss_abs)
        return pnl, leftover, max(0.0, risk - used), used

    def target_hit(self, pnl: float) -> bool:
        return self.target_dollars > 0 and pnl >= self.target_dollars

    def risk_hit(self, used: float) -> bool:
        return self.risk_dollars > 0 and used >= self.risk_dollars

    def open_ticket(self, symbol: str, side: int, lots: float, mid: float, t_index: int) -> Ticket | None:
        lots = round_lot(lots)
        if lots < MIN_LOT or self.dead or self.tickets_today >= MAX_TICKETS_DAY:
            return None
        existing = self.open_on(symbol)
        if existing and existing[0].side != side:
            return None
        entry = fill_price(symbol, side, mid)
        t = Ticket(self.next_id, symbol, side, lots, entry, t_index)
        self.next_id += 1
        self.tickets.append(t)
        self.tickets_today += 1
        self.fills_today[symbol] = self.fills_today.get(symbol, 0) + 1
        return t

    def close_ticket(self, ticket: Ticket, mid: float) -> float:
        pnl = self.ticket_pnl(ticket, mid)
        self.realized_day += pnl
        self.equity += pnl
        if pnl < 0:
            self.realized_loss_abs += -pnl
            self.n_trade_losses += 1
        self.n_trade_closes += 1
        if pnl > 0.0:
            self.n_trade_wins += 1
        if ticket in self.tickets:
            self.tickets.remove(ticket)
        return pnl

    def flatten(self, mids: dict[str, float]) -> list[tuple[Ticket, float]]:
        out = []
        for t in list(self.tickets):
            mid = mids.get(t.symbol)
            if mid is None or mid <= 0:
                mid = float(t.entry)
            pnl = self.close_ticket(t, float(mid))
            out.append((t, pnl))
        return out

    def risk_cap_lot(self, symbol: str, mid: float, atr: float, risk_left: float | None = None) -> float:
        left = self.risk_dollars if risk_left is None else risk_left
        spec = SPECS[symbol]
        heat = max(atr, spec.spread_price)
        d_heat = dollars_per_price(symbol, 1.0) * heat
        if d_heat <= 1e-9:
            by_risk = MIN_LOT
        else:
            by_risk = left / d_heat
        by_lev = max_lot_leverage(symbol, self.equity, mid)
        return max(0.0, min(by_risk, by_lev))
