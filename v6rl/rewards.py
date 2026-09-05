"""Channel A §§2.1–2.17. No BB candy. SPICE_PER_DAY_CAP clips non-2.1 spices."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from v6rl.product import SPICE_PER_DAY_CAP

INF = float("inf")


def clip(x: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, x)))


def primary_r(pnl_close_after_fees: float, today_target_dollars: float) -> float:
    """2.1 — wait bars with no close stay 0 (caller does not invoke)."""
    tgt = float(today_target_dollars)
    if tgt <= 1e-12:
        return 0.0
    return clip(float(pnl_close_after_fees) / tgt, -1.0, 1.0)


def ha_agreement_spice(*, profit: bool, n5: int, match15: bool, k_htf: int) -> float:
    """2.4 — profit close only; N>=3; 15m must match; k of {30m,1h,4h,1d}."""
    if not profit:
        return 0.0
    n = min(int(n5), 8)
    if n < 3 or not match15:
        return 0.0
    strength = max(n - 2, 0)
    k = int(max(0, min(4, k_htf)))
    if k <= 0:
        return 0.0
    raw = 0.005 * strength if k == 4 else 0.005 * strength * (k / 4.0)
    return min(raw, 0.03)


def trade_streak_spice(pnl: float, streak_before: int) -> tuple[float, int]:
    """2.5A. Returns (pay, new_streak)."""
    if pnl > 0:
        streak = int(streak_before) + 1
        return 0.01 * min(streak, 10), streak
    return 0.0, 0


def day_streak_spice(*, hit: bool, no_trade_or_skip: bool, streak_before: int) -> tuple[float, int]:
    """2.5B. Blank / SKIP / no-trade resets."""
    if no_trade_or_skip:
        return 0.0, 0
    if hit:
        streak = int(streak_before) + 1
        return 0.05 * min(streak, 10), streak
    return 0.0, 0


def stay_green_spice(*, passed: bool, streak_before: int) -> tuple[float, int]:
    """2.6."""
    if passed:
        streak = int(streak_before) + 1
        return min(0.005 * min(streak, 10), 0.05), streak
    return 0.0, 0


def clean_day_spice(
    *,
    day_hit: bool,
    wins: int,
    losses: int,
    streak_before: int,
) -> tuple[float, int]:
    """2.7 — requires HIT + ≥4 winners + 0 losers."""
    if (not day_hit) or int(losses) != 0 or int(wins) < 4:
        return 0.0, 0
    streak = int(streak_before) + 1
    extra = 0.01 * min(int(wins) - 4, 12)
    return min(0.04 * min(streak, 10) + extra, 0.60), streak


def lot_pb_spice(*, new_best: bool, green_close: bool, day_hit: bool) -> float:
    """2.8 — pay at day end only if green close and day HIT."""
    if new_best and green_close and day_hit:
        return 0.02
    return 0.0


def clean_hour_pb_spice(*, day_hit: bool, pending: bool, w_hour: int) -> float:
    """2.9 — defer to day end; requires HIT."""
    if not (day_hit and pending):
        return 0.0
    return min(0.01 + 0.005 * min(int(w_hour), 10), 0.06)


def sma_tunnel_spice(*, held_s: int | None, profit: bool, held_5m: bool) -> float:
    """2.10 — largest S that fully held; one tier only."""
    if (not profit) or (not held_5m) or held_s is None:
        return 0.0
    table = {1: 0.005, 2: 0.010, 3: 0.015, 4: 0.020}
    return float(table.get(int(held_s), 0.0))


def fat_trade_spice(*, fat_closes: int, closes_today: int) -> float:
    """2.11 — defer; pay only if closes_today >= 4. +0.025 each fat, day cap 0.10."""
    if int(closes_today) < 4:
        return 0.0
    return min(0.025 * int(fat_closes), 0.10)


def cci_side_score(cci_val: float, side: int) -> float | None:
    """Buy: CCI>0 score=CCI. Sell: CCI<0 score=-CCI. Wrong side → None."""
    if not np.isfinite(cci_val):
        return None
    if side > 0:
        return float(cci_val) if cci_val > 0 else None
    if side < 0:
        return float(-cci_val) if cci_val < 0 else None
    return None


def cci_pb_spice(score: float | None, best: float, pay: float) -> tuple[float, float, bool]:
    """Returns (reward, new_best, beat). First qualifying beats -inf."""
    if score is None:
        return 0.0, best, False
    if score > best:
        return float(pay), float(score), True
    return 0.0, best, False


def dual_cci_score(cci30: float, cci100: float, side: int) -> float | None:
    s30 = cci_side_score(cci30, side)
    s100 = cci_side_score(cci100, side)
    if s30 is None or s100 is None:
        return None
    return float(min(s30, s100))


def dual_cci_day_pay(*, day_hit: bool, pending_15m: bool, pending_1h: bool) -> float:
    """2.16 — records already updated at close; PAY only if day HIT."""
    if not day_hit:
        return 0.0
    pay = 0.0
    if pending_15m:
        pay += 0.025
    if pending_1h:
        pay += 0.030
    return min(pay, 0.055)


def fast_green_spice(
    *,
    profit: bool,
    bars_to_first_green: float,
    best: float,
) -> tuple[float, float, bool]:
    """2.17 — lower bars beats; profit required; no record update on losers."""
    if (not profit) or not np.isfinite(bars_to_first_green):
        return 0.0, best, False
    bars = float(bars_to_first_green)
    if bars >= best:
        return 0.0, best, False
    speed = clip((30.0 - bars) / 30.0, 0.0, 1.0)
    pay = min(0.010 + 0.010 * speed, 0.020)
    return pay, bars, True


@dataclass
class Career:
    trade_win_streak: int = 0
    day_win_streak: int = 0
    clean_day_streak: int = 0
    stay_green_streak: int = 0
    best_lot_ever: float = 0.0
    best_clean_hour_wins: int = 0
    best_cci30_15m_buy: float = -INF
    best_cci30_15m_sell: float = -INF
    best_cci30_1h_buy: float = -INF
    best_cci30_1h_sell: float = -INF
    best_cci100_15m_buy: float = -INF
    best_cci100_15m_sell: float = -INF
    best_cci100_1h_buy: float = -INF
    best_cci100_1h_sell: float = -INF
    best_dual_cci_15m_buy: float = -INF
    best_dual_cci_15m_sell: float = -INF
    best_dual_cci_1h_buy: float = -INF
    best_dual_cci_1h_sell: float = -INF
    best_bars_to_first_green_buy: float = INF
    best_bars_to_first_green_sell: float = INF


@dataclass
class DayBuckets:
    spice: float = 0.0
    ha: float = 0.0
    trade_streak: float = 0.0
    stay_green: float = 0.0
    sma: float = 0.0
    cci30_15m: float = 0.0
    cci30_1h: float = 0.0
    cci100_15m: float = 0.0
    cci100_1h: float = 0.0
    fast_green: float = 0.0
    clean_hour: float = 0.0
    fat: float = 0.0
    dual: float = 0.0
    lot_pb: float = 0.0
    day_streak: float = 0.0
    clean_day: float = 0.0


@dataclass
class SpiceStats:
    n_closes: int = 0
    n_primary: int = 0
    spice_sum: float = 0.0
    spice_capped_days: int = 0
    last_day_spice: float = 0.0
    aux_n: int = 0
    aux_pos: int = 0


@dataclass
class RewardEngine:
    career: Career = field(default_factory=Career)
    day: DayBuckets = field(default_factory=DayBuckets)
    stats: SpiceStats = field(default_factory=SpiceStats)
    pending_lot_pb_green: bool = False
    pending_clean_hour: bool = False
    pending_clean_hour_w: int = 0
    pending_dual_15m: bool = False
    pending_dual_1h: bool = False
    fat_closes: int = 0
    last_cci_pb_15m: bool = False
    last_cci_pb_1h: bool = False
    last_cci100_pb_15m: bool = False
    last_cci100_pb_1h: bool = False
    last_fast_green_pb: bool = False
    last_fat: bool = False

    def reset_day(self) -> None:
        self.day = DayBuckets()
        self.pending_lot_pb_green = False
        self.pending_clean_hour = False
        self.pending_clean_hour_w = 0
        self.pending_dual_15m = False
        self.pending_dual_1h = False
        self.fat_closes = 0
        self.last_cci_pb_15m = False
        self.last_cci_pb_1h = False
        self.last_cci100_pb_15m = False
        self.last_cci100_pb_1h = False
        self.last_fast_green_pb = False
        self.last_fat = False

    def _take(self, amount: float, event_cap: float, day_attr: str, day_cap: float) -> float:
        amt = min(max(0.0, float(amount)), float(event_cap))
        used = float(getattr(self.day, day_attr))
        amt = min(amt, max(0.0, float(day_cap) - used))
        room = SPICE_PER_DAY_CAP - self.day.spice
        amt = min(amt, max(0.0, room))
        setattr(self.day, day_attr, used + amt)
        self.day.spice += amt
        return amt

    def on_close(
        self,
        *,
        pnl: float,
        target_dollars: float,
        lots: float,
        side: int,
        n5: int,
        match15: bool,
        k_htf: int,
        stay_green_ok: bool,
        held_5m: bool,
        held_s: int | None,
        cci30_15m: float,
        cci30_1h: float,
        cci100_15m: float,
        cci100_1h: float,
        bars_to_first_green: float,
        hour_wins: int,
        hour_losses: int,
    ) -> float:
        self.stats.n_closes += 1
        r = primary_r(pnl, target_dollars)
        self.stats.n_primary += 1
        profit = pnl > 0.0

        ha = ha_agreement_spice(profit=profit, n5=n5, match15=match15, k_htf=k_htf)
        r += self._take(ha, 0.03, "ha", 0.06)

        ts_pay, self.career.trade_win_streak = trade_streak_spice(pnl, self.career.trade_win_streak)
        r += self._take(ts_pay, 0.10, "trade_streak", 0.25)

        sg_pay, self.career.stay_green_streak = stay_green_spice(
            passed=bool(stay_green_ok and profit),
            streak_before=self.career.stay_green_streak,
        )
        r += self._take(sg_pay, 0.05, "stay_green", 0.15)

        tun = sma_tunnel_spice(held_s=held_s, profit=profit, held_5m=held_5m)
        r += self._take(tun, 0.020, "sma", 0.08)

        s15 = cci_side_score(cci30_15m, side) if profit else None
        key15 = "best_cci30_15m_buy" if side > 0 else "best_cci30_15m_sell"
        p15, new15, beat15 = cci_pb_spice(s15, getattr(self.career, key15), 0.015)
        setattr(self.career, key15, new15)
        self.last_cci_pb_15m = beat15
        r += self._take(p15, 0.015, "cci30_15m", 0.06)

        s1h = cci_side_score(cci30_1h, side) if profit else None
        key1h = "best_cci30_1h_buy" if side > 0 else "best_cci30_1h_sell"
        p1h, new1h, beat1h = cci_pb_spice(s1h, getattr(self.career, key1h), 0.020)
        setattr(self.career, key1h, new1h)
        self.last_cci_pb_1h = beat1h
        r += self._take(p1h, 0.020, "cci30_1h", 0.08)

        s100_15 = cci_side_score(cci100_15m, side) if profit else None
        k100_15 = "best_cci100_15m_buy" if side > 0 else "best_cci100_15m_sell"
        p100_15, n100_15, b100_15 = cci_pb_spice(s100_15, getattr(self.career, k100_15), 0.015)
        setattr(self.career, k100_15, n100_15)
        self.last_cci100_pb_15m = b100_15
        r += self._take(p100_15, 0.015, "cci100_15m", 0.06)

        s100_1h = cci_side_score(cci100_1h, side) if profit else None
        k100_1h = "best_cci100_1h_buy" if side > 0 else "best_cci100_1h_sell"
        p100_1h, n100_1h, b100_1h = cci_pb_spice(s100_1h, getattr(self.career, k100_1h), 0.020)
        setattr(self.career, k100_1h, n100_1h)
        self.last_cci100_pb_1h = b100_1h
        r += self._take(p100_1h, 0.020, "cci100_1h", 0.08)

        # 2.16 records update at close; pay later if HIT
        d15 = dual_cci_score(cci30_15m, cci100_15m, side) if profit else None
        dk15 = "best_dual_cci_15m_buy" if side > 0 else "best_dual_cci_15m_sell"
        if d15 is not None and d15 > getattr(self.career, dk15):
            setattr(self.career, dk15, d15)
            self.pending_dual_15m = True
        d1h = dual_cci_score(cci30_1h, cci100_1h, side) if profit else None
        dk1h = "best_dual_cci_1h_buy" if side > 0 else "best_dual_cci_1h_sell"
        if d1h is not None and d1h > getattr(self.career, dk1h):
            setattr(self.career, dk1h, d1h)
            self.pending_dual_1h = True

        best_fg = (
            self.career.best_bars_to_first_green_buy
            if side > 0
            else self.career.best_bars_to_first_green_sell
        )
        fg, new_fg, beat_fg = fast_green_spice(
            profit=profit, bars_to_first_green=bars_to_first_green, best=best_fg
        )
        if side > 0:
            self.career.best_bars_to_first_green_buy = new_fg
        else:
            self.career.best_bars_to_first_green_sell = new_fg
        self.last_fast_green_pb = beat_fg
        r += self._take(fg, 0.020, "fast_green", 0.08)

        if lots > self.career.best_lot_ever + 1e-12:
            self.career.best_lot_ever = float(lots)
            if profit:
                self.pending_lot_pb_green = True

        if hour_losses == 0 and hour_wins > self.career.best_clean_hour_wins:
            self.career.best_clean_hour_wins = int(hour_wins)
            self.pending_clean_hour = True
            self.pending_clean_hour_w = int(hour_wins)

        self.last_fat = pnl + 1e-12 >= 0.10 * float(target_dollars)
        if self.last_fat:
            self.fat_closes += 1

        return r

    def on_day_end(
        self,
        *,
        grade: str,
        day_hit: bool,
        wins: int,
        losses: int,
        closes_today: int,
        had_trade: bool,
    ) -> float:
        skip_or_blank = (grade == "SKIP") or (not had_trade)
        ds, self.career.day_win_streak = day_streak_spice(
            hit=day_hit, no_trade_or_skip=skip_or_blank, streak_before=self.career.day_win_streak
        )
        r = self._take(min(ds, 0.50), 0.50, "day_streak", 0.50)

        cd, self.career.clean_day_streak = clean_day_spice(
            day_hit=day_hit, wins=wins, losses=losses, streak_before=self.career.clean_day_streak
        )
        r += self._take(cd, 0.60, "clean_day", 0.60)

        lot = lot_pb_spice(
            new_best=self.pending_lot_pb_green, green_close=self.pending_lot_pb_green, day_hit=day_hit
        )
        r += self._take(lot, 0.02, "lot_pb", 0.02)

        ch = clean_hour_pb_spice(
            day_hit=day_hit, pending=self.pending_clean_hour, w_hour=self.pending_clean_hour_w
        )
        r += self._take(ch, 0.06, "clean_hour", 0.12)

        fat = fat_trade_spice(fat_closes=self.fat_closes, closes_today=closes_today)
        r += self._take(fat, 0.10, "fat", 0.10)

        dual = dual_cci_day_pay(
            day_hit=day_hit, pending_15m=self.pending_dual_15m, pending_1h=self.pending_dual_1h
        )
        r += self._take(dual, 0.055, "dual", 0.055)

        self.stats.spice_sum += self.day.spice
        self.stats.last_day_spice = self.day.spice
        if self.day.spice + 1e-12 >= SPICE_PER_DAY_CAP:
            self.stats.spice_capped_days += 1
        return r
