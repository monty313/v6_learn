from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from v6rl.book import Book, Ticket
from v6rl.config import (
    BUY,
    CLOSE,
    MAX_LIVE_TICKETS,
    N_ACTIONS,
    RISK_TICKET_FRAC,
    SELL,
    SIZE_FLOOR,
    START_EQUITY,
    TARGET_PCT_MAX,
    WAIT,
    explore_eps,
    size_scale_schedule,
    typed_risk_pct,
    typed_target_pct,
)
from v6rl.geometry import day_can_print, draw_funded_tr
from v6rl.indicators import clip_div, nz
from v6rl.obs import FEATURE_NAMES, OBS_DIM
from v6rl.product import MIN_LOT, RISK_PCT_MAX, RISK_PCT_MIN, draw_typed_target
from v6rl.rewards import INF, RewardEngine, cci_side_score, dual_cci_score
from v6rl.symbols import dollars_per_price, round_lot
from v6rl.tape import Tape

# After WAIT with a ticket open: keep 1m marks (2.17 / SMA / BB) but do not
# re-ask the net every minute. CLOSE stays legal on the next decision.
HOLD_WAIT_EXTRA_BARS = 4


def next_choice_index(
    session: np.ndarray,
    door_buy: np.ndarray,
    door_sell: np.ndarray,
    t: int,
    t_end: int,
    instrument: int = 0,
) -> int:
    """First session bar in [t, t_end) where the HA door is on. Else t_end."""
    if t >= t_end:
        return int(t_end)
    buy = door_buy[t:t_end, instrument] if door_buy.ndim == 2 else door_buy[t:t_end]
    sell = door_sell[t:t_end, instrument] if door_sell.ndim == 2 else door_sell[t:t_end]
    sess = session[t:t_end]
    hit = np.flatnonzero(np.asarray(sess) & (np.asarray(buy) | np.asarray(sell)))
    return t + int(hit[0]) if hit.size else int(t_end)


@dataclass
class Pending:
    lot_frac: np.ndarray = field(default_factory=lambda: np.array([0.8], np.float32))
    size_scale: np.ndarray = field(default_factory=lambda: np.array([1.0], np.float32))


def _cross_up(prev: float, cur: float, rail: float) -> bool:
    if not (np.isfinite(prev) and np.isfinite(cur) and np.isfinite(rail)):
        return False
    return prev < rail <= cur


def _cross_dn(prev: float, cur: float, rail: float) -> bool:
    if not (np.isfinite(prev) and np.isfinite(cur) and np.isfinite(rail)):
        return False
    return prev > rail >= cur


class BookEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, tape: Tape, greedy: bool = False, run_dir: Path | None = None):
        super().__init__()
        self.tape = tape
        self.greedy = greedy
        self.book = Book()
        self.rew = RewardEngine()
        self.run_dir = Path(run_dir) if run_dir is not None else None
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(OBS_DIM,), dtype=np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.pending = Pending()
        self.dates: list = []
        self.date_i = 0
        self.train_day_index = 0
        self.t = 0
        self.t_end = 0
        self._printable = True
        self._mids_t = -1
        self._mids_cache: dict[str, float] = {}
        self._last_mids: dict[str, float] = {}
        self._sess_first = 0
        self._sess_last = 1
        self._mask = np.zeros(N_ACTIONS, dtype=np.bool_)
        self._mask[WAIT] = True
        self.rollout_live = False
        self._last_obs: np.ndarray | None = None
        self._last_info: dict = {}
        self._hour_closes: list[tuple[int, bool]] = []
        self._day_end_paid = False
        self.last_aux = -1.0

    def set_pending(self, lot_frac, size_scale, **_kw) -> None:
        self.pending.lot_frac = np.asarray(lot_frac, dtype=np.float32).reshape(-1)
        self.pending.size_scale = np.asarray(size_scale, dtype=np.float32).reshape(-1)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        opt = options or {}
        self.dates = list(opt.get("dates") or self.tape.train_dates)
        if not self.dates:
            raise RuntimeError("no dates")
        if opt.get("reset_equity", True) or self.book.equity <= MIN_LOT:
            self.book.equity = START_EQUITY
        self.date_i = int(opt.get("date_i", 0)) % len(self.dates)
        date = self.dates[self.date_i]
        self.train_day_index = int(opt.get("train_day_index", self.date_i))
        rng = np.random.default_rng(seed)
        equity = float(self.book.equity)
        self.greedy = bool(opt.get("greedy", self.greedy))
        explicit_t = "target_pct" in opt
        explicit_r = "risk_pct" in opt
        if explicit_t or explicit_r:
            target = typed_target_pct(opt["target_pct"]) if explicit_t else draw_typed_target(rng)
            risk = typed_risk_pct(opt["risk_pct"]) if explicit_r else typed_risk_pct(
                rng.uniform(RISK_PCT_MIN, RISK_PCT_MAX)
            )
            printable = day_can_print(self.tape, date, target, risk, equity)
        else:
            target, risk, printable = draw_funded_tr(rng, self.tape, date, equity)
            target = typed_target_pct(target)
            risk = typed_risk_pct(risk)
        a, b = self.tape.day_slice(np.datetime64(date))
        self.t = a
        self.t_end = b
        self._mids_t = -1
        sess = np.flatnonzero(self.tape.session[a:b])
        if sess.size:
            self._sess_first = a + int(sess[0])
            self._sess_last = a + int(sess[-1])
        else:
            self._sess_first, self._sess_last = a, max(a, b - 1)
        self.book.reset_day(target, risk)
        self.rew.reset_day()
        self._hour_closes.clear()
        self._day_end_paid = False
        self.last_aux = -1.0
        self._printable = bool(printable)
        if not self._printable:
            self.book.dead = "SKIP"
        self._skip_idle()
        self._advance_to_choice()
        obs, info = self._observe()
        info["announce"] = True
        self._last_obs, self._last_info = obs, info
        return obs, info

    def _mids(self) -> dict[str, float]:
        if self._mids_t == self.t:
            return self._mids_cache
        out = dict(self._last_mids)
        t = min(self.t, max(0, self.tape.n - 1))
        for i, s in enumerate(self.tape.symbols):
            if self.tape.valid[t, i]:
                px = float(self.tape.close[t, i])
                if px > 0:
                    out[s] = px
                    self._last_mids[s] = px
        self._mids_cache = out
        self._mids_t = self.t
        return out

    def _skip_idle(self) -> None:
        while self.t < self.t_end:
            if self.tape.session[self.t] or self.book.tickets:
                return
            self.t += 1

    def _advance_to_choice(self) -> None:
        """Jump past 1m bars where WAIT is the only legal action (flat, door off)."""
        if self.book.tickets or self.book.dead:
            return
        self.t = next_choice_index(
            self.tape.session,
            self.tape.door_buy,
            self.tape.door_sell,
            self.t,
            self.t_end,
            self._i(),
        )

    def _fast_hold(self, extra_bars: int = HOLD_WAIT_EXTRA_BARS) -> float:
        """Mark the next 1m bars after WAIT while a ticket is open. No new action."""
        extra_r = 0.0
        for _ in range(int(extra_bars)):
            if self.book.dead or not self.book.tickets:
                return extra_r
            if self.t + 1 >= self.t_end:
                return extra_r
            self.t += 1
            if not bool(self.tape.session[self.t]) and not self.book.tickets:
                self._skip_idle()
                return extra_r
            self._mark_open()
            extra, done = self._halt(self._mids())
            extra_r += extra
            if done:
                return extra_r
        return extra_r

    def _i(self) -> int:
        return 0

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(N_ACTIONS, dtype=np.bool_)
        mask[WAIT] = True
        if self.book.dead or not self._printable:
            self._mask = mask
            return mask
        if self.t < 0 or self.t >= self.tape.n or not bool(self.tape.session[self.t]):
            if self.book.tickets:
                mask[CLOSE] = True
            self._mask = mask
            return mask
        i = self._i()
        if bool(self.tape.door_buy[self.t, i] or self.tape.door_sell[self.t, i]):
            self.book.saw_legal = True
        open_t = self.book.tickets
        side = self.book.side_on(self.tape.symbols[i])
        can_open = len(open_t) < MAX_LIVE_TICKETS
        if can_open and side >= 0 and bool(self.tape.door_buy[self.t, i]):
            mask[BUY] = True
        if can_open and side <= 0 and bool(self.tape.door_sell[self.t, i]):
            mask[SELL] = True
        if open_t:
            mask[CLOSE] = True
        self._mask = mask
        return mask

    def _reserved_heat(self) -> float:
        if self.t < 0 or self.t >= self.tape.n:
            return 0.0
        total = 0.0
        for ticket in self.book.tickets:
            i = self.tape.symbols.index(ticket.symbol)
            atr = float(self.tape.atr5[self.t, i])
            heat = max(atr, 1.0)
            total += float(ticket.lots) * dollars_per_price(ticket.symbol, 1.0) * heat
        return total

    def _lots(self, symbol: str, mid: float, risk_left: float) -> float:
        i = self._i()
        atr = float(self.tape.atr5[self.t, i])
        heat = max(0.0, risk_left * RISK_TICKET_FRAC)
        cap = self.book.risk_cap_lot(symbol, mid, atr, heat)
        scale = float(np.clip(self.pending.size_scale[0] if self.pending.size_scale.size else 1.0, SIZE_FLOOR, 1.0))
        frac = float(np.clip(self.pending.lot_frac[0] if self.pending.lot_frac.size else 0.8, 0.05, 1.0))
        sched = 1.0 if self.greedy else size_scale_schedule(self.train_day_index)
        return round_lot(cap * frac * scale * sched)

    def _mark_open(self) -> None:
        if self.t < 0 or self.t >= self.tape.n:
            return
        i = self._i()
        mid = self._mids().get(self.tape.symbols[i])
        c5 = float(self.tape.close5[self.t, i])
        for ticket in self.book.tickets:
            ticket.bars_held += 1
            if mid is not None:
                upnl = self.book.ticket_pnl(ticket, mid)
                if upnl > 0 and not np.isfinite(ticket.bars_to_first_green):
                    ticket.bars_to_first_green = float(ticket.bars_held)
                if ticket.bars_held >= 5:
                    ticket.reached_5m = True
                    if upnl < 0:
                        ticket.ever_red_after_5m = True
            if ticket.reached_5m and np.isfinite(c5):
                rails = self.tape.sma4_s[self.t, i]
                broken = list(ticket.sma_broken)
                for s in range(4):
                    rail = float(rails[s])
                    if not np.isfinite(rail):
                        broken[s] = True
                        continue
                    if ticket.side > 0 and c5 < rail:
                        broken[s] = True
                    if ticket.side < 0 and c5 > rail:
                        broken[s] = True
                ticket.sma_broken = tuple(broken)  # type: ignore[assignment]
            self._bb_update(ticket, c5, i)
            ticket.prev_close5 = c5

    def _bb_update(self, ticket: Ticket, c5: float, i: int) -> None:
        prev = float(ticket.prev_close5)
        if not np.isfinite(c5):
            return
        mid20 = float(self.tape.bb20_mid[self.t, i])
        up20 = float(self.tape.bb20_up[self.t, i])
        lo20 = float(self.tape.bb20_lo[self.t, i])
        mid200 = float(self.tape.bb200_mid[self.t, i])
        up200 = float(self.tape.bb200_up[self.t, i])
        lo200 = float(self.tape.bb200_lo[self.t, i])
        if _cross_up(prev, c5, mid20) and not ticket.seen_bb20_mid_up:
            ticket.seen_bb20_mid_up = True
            ticket.bb20_cross_up = True
            ticket.bb_up_events += 1
        if _cross_up(prev, c5, up20) and not ticket.seen_bb20_up_up:
            ticket.seen_bb20_up_up = True
            ticket.bb20_cross_up = True
            ticket.bb_up_events += 1
        if _cross_up(prev, c5, mid200) and not ticket.seen_bb200_mid_up:
            ticket.seen_bb200_mid_up = True
            ticket.bb200_cross_up = True
            ticket.bb_up_events += 1
        if _cross_up(prev, c5, up200) and not ticket.seen_bb200_up_up:
            ticket.seen_bb200_up_up = True
            ticket.bb200_cross_up = True
            ticket.bb_up_events += 1
        if _cross_dn(prev, c5, mid20) and not ticket.seen_bb20_mid_dn:
            ticket.seen_bb20_mid_dn = True
            ticket.bb20_cross_down = True
            ticket.bb_down_events += 1
        if _cross_dn(prev, c5, lo20) and not ticket.seen_bb20_lo_dn:
            ticket.seen_bb20_lo_dn = True
            ticket.bb20_cross_down = True
            ticket.bb_down_events += 1
        if _cross_dn(prev, c5, mid200) and not ticket.seen_bb200_mid_dn:
            ticket.seen_bb200_mid_dn = True
            ticket.bb200_cross_down = True
            ticket.bb_down_events += 1
        if _cross_dn(prev, c5, lo200) and not ticket.seen_bb200_lo_dn:
            ticket.seen_bb200_lo_dn = True
            ticket.bb200_cross_down = True
            ticket.bb_down_events += 1

    def _htf_k(self, side: int) -> tuple[int, bool, int]:
        i = self._i()
        g5 = bool(self.tape.ha_green["5m"][self.t, i])
        n5 = int(self.tape.ha_n["5m"][self.t, i])
        want = side > 0
        match15 = bool(self.tape.ha_green["15m"][self.t, i]) == want
        k = 0
        for tf in ("30m", "1h", "4h", "1d"):
            if bool(self.tape.ha_green[tf][self.t, i]) == want:
                k += 1
        return n5, match15, k

    def _hour_wl(self) -> tuple[int, int]:
        t0 = self.t - 60
        w = l = 0
        for tt, win in self._hour_closes:
            if tt >= t0:
                if win:
                    w += 1
                else:
                    l += 1
        return w, l

    def _close_ticket(self, ticket: Ticket, mid: float) -> tuple[float, float]:
        if not np.isfinite(ticket.bars_to_first_green):
            upnl = self.book.ticket_pnl(ticket, mid)
            if upnl > 0:
                ticket.bars_to_first_green = float(max(ticket.bars_held, 1))
        held_s = None
        if ticket.reached_5m:
            for s in (4, 3, 2, 1):
                if not ticket.sma_broken[s - 1]:
                    held_s = s
                    break
        n5, match15, k = self._htf_k(ticket.side)
        i = self._i()
        hw, hl = self._hour_wl()
        stay_ok = ticket.reached_5m and (not ticket.ever_red_after_5m)
        pnl = self.book.close_ticket(ticket, mid)
        r = self.rew.on_close(
            pnl=pnl,
            target_dollars=self.book.target_dollars,
            lots=ticket.lots,
            side=ticket.side,
            n5=n5,
            match15=match15,
            k_htf=k,
            stay_green_ok=stay_ok,
            held_5m=ticket.reached_5m,
            held_s=held_s,
            cci30_15m=float(self.tape.cci30_15m[self.t, i]),
            cci30_1h=float(self.tape.cci30_1h[self.t, i]),
            cci100_15m=float(self.tape.cci100_15m[self.t, i]),
            cci100_1h=float(self.tape.cci100_1h[self.t, i]),
            bars_to_first_green=ticket.bars_to_first_green,
            hour_wins=hw + int(pnl > 0),
            hour_losses=hl + int(pnl <= 0),
        )
        self._hour_closes.append((self.t, pnl > 0))
        ride = 1.0 if (ticket.bb_up_events >= 2 or ticket.bb_down_events >= 2) else 0.0
        self.last_aux = ride
        self.rew.stats.aux_n += 1
        self.rew.stats.aux_pos += int(ride > 0.5)
        return pnl, r

    def _finish_day(self) -> float:
        if self._day_end_paid:
            return 0.0
        if self.t >= self.t_end and self.t_end > 0:
            self.t = self.t_end - 1
        extra = 0.0
        mids = self._mids()
        for ticket, _pnl in []:
            pass
        if self.book.tickets:
            for ticket in list(self.book.tickets):
                mid = mids.get(ticket.symbol) or float(ticket.entry)
                _pnl, r = self._close_ticket(ticket, float(mid))
                extra += r
        if not self.book.dead:
            if not self._printable:
                self.book.dead = "SKIP"
            elif self.book.target_hit(self.book.realized_day) and not self.book.risk_hit(
                self.book.realized_loss_abs
            ):
                self.book.dead = "HIT"
            elif self.book.risk_hit(max(self.book.realized_loss_abs, -self.book.realized_day)):
                self.book.dead = "BLEW"
            elif not self.book.saw_legal:
                self.book.dead = "SKIP"
            else:
                self.book.dead = "MISS"
        extra += self.rew.on_day_end(
            grade=str(self.book.dead),
            day_hit=self.book.dead == "HIT",
            wins=self.book.n_trade_wins,
            losses=self.book.n_trade_losses,
            closes_today=self.book.n_trade_closes,
            had_trade=self.book.tickets_today > 0,
        )
        self._day_end_paid = True
        return extra

    def _halt(self, mids: dict[str, float]) -> tuple[float, bool]:
        if self.book.dead in ("HIT", "BLEW", "MISS", "SKIP"):
            if not self._day_end_paid:
                return self._finish_day(), True
            return 0.0, True
        snap = self.book.snapshot(mids)
        extra = 0.0
        if self.book.target_hit(snap[0]):
            for ticket in list(self.book.tickets):
                mid = mids.get(ticket.symbol) or float(ticket.entry)
                _p, r = self._close_ticket(ticket, float(mid))
                extra += r
            self.book.dead = "HIT"
            extra += self._finish_day()
            return extra, True
        if self.book.risk_hit(snap[3]):
            for ticket in list(self.book.tickets):
                mid = mids.get(ticket.symbol) or float(ticket.entry)
                _p, r = self._close_ticket(ticket, float(mid))
                extra += r
            self.book.dead = "BLEW"
            extra += self._finish_day()
            return extra, True
        return 0.0, False

    def step(self, action: int):
        reward = 0.0
        exec_action = WAIT
        self.last_aux = -1.0
        if self.book.dead == "SKIP" and not self._printable:
            reward += self._finish_day()
            obs, info = self._observe()
            info["exec_action"] = WAIT
            info["aux_y"] = -1.0
            self._last_obs, self._last_info = obs, info
            return obs, float(reward), True, False, info

        self._mark_open()
        mids = self._mids()
        extra, done = self._halt(mids)
        reward += extra
        if done:
            obs, info = self._observe()
            info["exec_action"] = WAIT
            info["aux_y"] = self.last_aux
            self._last_obs, self._last_info = obs, info
            return obs, float(reward), True, False, info

        mask = self.action_masks()
        act = int(action) if 0 <= int(action) < N_ACTIONS and mask[int(action)] else WAIT
        i = self._i()
        s = self.tape.symbols[i]
        mid = mids.get(s)
        snap = self.book.snapshot(mids)
        risk_left = max(0.0, snap[2] - self._reserved_heat())

        if act == BUY and mid is not None and not self.book.dead:
            lots = self._lots(s, mid, risk_left)
            ticket = self.book.open_ticket(s, 1, lots, mid, self.t)
            if ticket is not None:
                ticket.prev_close5 = float(self.tape.close5[self.t, i])
                exec_action = BUY
        elif act == SELL and mid is not None and not self.book.dead:
            lots = self._lots(s, mid, risk_left)
            ticket = self.book.open_ticket(s, -1, lots, mid, self.t)
            if ticket is not None:
                ticket.prev_close5 = float(self.tape.close5[self.t, i])
                exec_action = SELL
        elif act == CLOSE and self.book.tickets and mid is not None:
            for ticket in list(self.book.tickets):
                _p, r = self._close_ticket(ticket, mid)
                reward += r
            exec_action = CLOSE

        extra, done = self._halt(self._mids())
        reward += extra
        terminated = done or self.book.dead in ("HIT", "BLEW", "MISS", "SKIP")
        if not terminated and exec_action == WAIT and self.book.tickets:
            reward += self._fast_hold()
            terminated = self.book.dead in ("HIT", "BLEW", "MISS", "SKIP")
            if self.t >= self.t_end:
                reward += self._finish_day()
                terminated = True
        if not terminated:
            self.t += 1
            self._skip_idle()
            self._advance_to_choice()
            if self.t >= self.t_end:
                reward += self._finish_day()
                terminated = True
        obs, info = self._observe()
        info["exec_action"] = exec_action
        info["aux_y"] = self.last_aux
        self._last_obs, self._last_info = obs, info
        return obs, float(reward), bool(terminated), False, info

    def _primary(self) -> Ticket | None:
        return self.book.tickets[0] if self.book.tickets else None

    def _observe(self) -> tuple[np.ndarray, dict]:
        if self.t >= self.t_end:
            self.t = max(self.t_end - 1, 0)
        mids = self._mids()
        snap = self.book.snapshot(mids)
        day_pnl, leftover, risk_left, used = snap
        tgt = self.book.target_dollars
        risk_d = self.book.risk_dollars
        i = self._i()
        s = self.tape.symbols[i]
        mid = mids.get(s, 0.0)
        ticket = self._primary()
        side = float(ticket.side) if ticket else 0.0
        lots = float(ticket.lots) if ticket else 0.0
        open_pnl = self.book.symbol_pnl(s, mid) if ticket and mid else 0.0
        bars = float(ticket.bars_held) if ticket else 0.0
        cap = self.book.risk_cap_lot(s, mid or 1.0, float(self.tape.atr5[self.t, i]), max(risk_left, 1e-9)) if mid else 0.0
        elapsed = float(
            np.clip((self.t - self._sess_first) / max(1, self._sess_last - self._sess_first), 0.0, 1.0)
        )
        hw, hl = self._hour_wl()
        c30_15 = float(self.tape.cci30_15m[self.t, i])
        c30_1h = float(self.tape.cci30_1h[self.t, i])
        c100_15 = float(self.tape.cci100_15m[self.t, i])
        c100_1h = float(self.tape.cci100_1h[self.t, i])
        side_i = int(side) if side else 0
        sc30_15 = cci_side_score(c30_15, side_i) if side_i else (abs(c30_15) if np.isfinite(c30_15) else 0.0)
        sc30_1h = cci_side_score(c30_1h, side_i) if side_i else (abs(c30_1h) if np.isfinite(c30_1h) else 0.0)
        sc100_15 = cci_side_score(c100_15, side_i) if side_i else (abs(c100_15) if np.isfinite(c100_15) else 0.0)
        sc100_1h = cci_side_score(c100_1h, side_i) if side_i else (abs(c100_1h) if np.isfinite(c100_1h) else 0.0)
        d15 = dual_cci_score(c30_15, c100_15, side_i) if side_i else None
        d1h = dual_cci_score(c30_1h, c100_1h, side_i) if side_i else None
        c5 = float(self.tape.close5[self.t, i])
        atr = float(self.tape.atr5[self.t, i])
        car = self.rew.career

        def fin(x, default=0.0):
            return default if x is None or not np.isfinite(x) else float(x)

        def scale_best(x):
            if x is None or not np.isfinite(x) or abs(x) > 1e12:
                return 0.0
            return float(np.clip(x / 200.0, -8, 8))

        def scale_bars(x):
            if x is None or not np.isfinite(x) or x > 1e8:
                return 0.0
            return float(np.clip(x / 30.0, 0, 8))

        held = [False, False, False, False]
        if ticket and ticket.reached_5m:
            held = [not b for b in ticket.sma_broken]
        bb_up = ticket.bb20_cross_up if ticket else False
        bb_dn = ticket.bb20_cross_down if ticket else False
        bb200_up = ticket.bb200_cross_up if ticket else False
        bb200_dn = ticket.bb200_cross_down if ticket else False
        multi_up = bool(ticket and ticket.bb_up_events >= 2)
        multi_dn = bool(ticket and ticket.bb_down_events >= 2)

        d: dict[str, float] = {
            "target_pct": self.book.target_pct / TARGET_PCT_MAX,
            "risk_pct": self.book.risk_pct / RISK_PCT_MAX,
            "target_dollars": float(np.clip(tgt / START_EQUITY, 0, 2)),
            "risk_dollars": float(np.clip(risk_d / START_EQUITY, 0, 2)),
            "day_pnl": float(np.clip(day_pnl / max(tgt, 1.0), -2, 2)),
            "goal_frac": float(np.clip(day_pnl / max(tgt, 1e-9), -2, 2)),
            "risk_frac": float(np.clip(used / max(risk_d, 1e-9), 0, 2)),
            "leftover_to_target": leftover,
            "risk_left": float(np.clip(risk_left / max(risk_d, 1e-9), 0, 2)),
            "session_on": 1.0 if self.t < self.tape.n and self.tape.session[self.t] else 0.0,
            "session_time_frac": elapsed,
            "flat": 1.0 if ticket is None else 0.0,
            "side": side,
            "lot_size": lots,
            "lot_size_frac_of_max": float(np.clip(lots / max(cap, 1e-9), 0, 2)) if cap > 0 else 0.0,
            "bars_held": float(np.clip(bars / 30.0, 0, 8)),
            "open_pnl": float(np.clip(open_pnl / max(tgt, 1.0), -2, 2)),
            "open_count": min(len(self.book.tickets) / 10.0, 2.0),
            "closes_today": min(self.book.n_trade_closes / 20.0, 2.0),
            "wins_so_far": min(self.book.n_trade_wins / 20.0, 2.0),
            "losses_so_far": min(self.book.n_trade_losses / 20.0, 2.0),
            "trade_win_streak": min(car.trade_win_streak / 10.0, 2.0),
            "day_win_streak": min(car.day_win_streak / 10.0, 2.0),
            "clean_day_streak": min(car.clean_day_streak / 10.0, 2.0),
            "stay_green_streak": min(car.stay_green_streak / 10.0, 2.0),
            "had_trade_today": 1.0 if self.book.tickets_today else 0.0,
            "held_green_after_5m": 1.0 if (ticket and ticket.reached_5m and not ticket.ever_red_after_5m) else 0.0,
            "is_clean_day_so_far": 1.0 if (self.book.n_trade_losses == 0 and self.book.n_trade_wins > 0) else 0.0,
            "best_clean_hour_wins": min(car.best_clean_hour_wins / 10.0, 2.0),
            "hour_wins": min(hw / 10.0, 2.0),
            "hour_losses": min(hl / 10.0, 2.0),
            "best_lot_ever": min(car.best_lot_ever / 10.0, 2.0),
            "lot_is_personal_best": 1.0 if (ticket and lots >= car.best_lot_ever - 1e-12 and lots > 0) else 0.0,
            "sma4_s1_held": 1.0 if held[0] else 0.0,
            "sma4_s2_held": 1.0 if held[1] else 0.0,
            "sma4_s3_held": 1.0 if held[2] else 0.0,
            "sma4_s4_held": 1.0 if held[3] else 0.0,
            "fat_trade_flag": 1.0 if self.rew.last_fat else 0.0,
            "cci30_15m": float(np.clip(nz(c30_15) / 200.0, -8, 8)),
            "cci30_1h": float(np.clip(nz(c30_1h) / 200.0, -8, 8)),
            "cci30_15m_score": float(np.clip(fin(sc30_15) / 200.0, -8, 8)),
            "cci30_1h_score": float(np.clip(fin(sc30_1h) / 200.0, -8, 8)),
            "best_cci30_15m_buy": scale_best(car.best_cci30_15m_buy),
            "best_cci30_15m_sell": scale_best(car.best_cci30_15m_sell),
            "best_cci30_1h_buy": scale_best(car.best_cci30_1h_buy),
            "best_cci30_1h_sell": scale_best(car.best_cci30_1h_sell),
            "cci_pb_15m_flag": 1.0 if self.rew.last_cci_pb_15m else 0.0,
            "cci_pb_1h_flag": 1.0 if self.rew.last_cci_pb_1h else 0.0,
            "cci100_15m": float(np.clip(nz(c100_15) / 200.0, -8, 8)),
            "cci100_1h": float(np.clip(nz(c100_1h) / 200.0, -8, 8)),
            "cci100_15m_score": float(np.clip(fin(sc100_15) / 200.0, -8, 8)),
            "cci100_1h_score": float(np.clip(fin(sc100_1h) / 200.0, -8, 8)),
            "best_cci100_15m_buy": scale_best(car.best_cci100_15m_buy),
            "best_cci100_15m_sell": scale_best(car.best_cci100_15m_sell),
            "best_cci100_1h_buy": scale_best(car.best_cci100_1h_buy),
            "best_cci100_1h_sell": scale_best(car.best_cci100_1h_sell),
            "cci100_pb_15m_flag": 1.0 if self.rew.last_cci100_pb_15m else 0.0,
            "cci100_pb_1h_flag": 1.0 if self.rew.last_cci100_pb_1h else 0.0,
            "dual_cci_15m_score": float(np.clip(fin(d15) / 200.0, -8, 8)),
            "dual_cci_1h_score": float(np.clip(fin(d1h) / 200.0, -8, 8)),
            "best_dual_cci_15m_buy": scale_best(car.best_dual_cci_15m_buy),
            "best_dual_cci_15m_sell": scale_best(car.best_dual_cci_15m_sell),
            "best_dual_cci_1h_buy": scale_best(car.best_dual_cci_1h_buy),
            "best_dual_cci_1h_sell": scale_best(car.best_dual_cci_1h_sell),
            "pending_dual_pb_15m": 1.0 if self.rew.pending_dual_15m else 0.0,
            "pending_dual_pb_1h": 1.0 if self.rew.pending_dual_1h else 0.0,
            "bars_to_first_green": scale_bars(ticket.bars_to_first_green if ticket else INF),
            "best_bars_to_first_green_buy": scale_bars(car.best_bars_to_first_green_buy),
            "best_bars_to_first_green_sell": scale_bars(car.best_bars_to_first_green_sell),
            "fast_green_pb_flag": 1.0 if self.rew.last_fast_green_pb else 0.0,
            "ha_color_1m": 1.0 if bool(self.tape.ha_green["1m"][self.t, i]) else -1.0,
            "ha_n_1m": min(int(self.tape.ha_n["1m"][self.t, i]) / 8.0, 2.0),
            "door_buy": 1.0 if bool(self.tape.door_buy[self.t, i]) else 0.0,
            "door_sell": 1.0 if bool(self.tape.door_sell[self.t, i]) else 0.0,
            "sma4_s1_dist": clip_div(c5 - float(self.tape.sma4_s[self.t, i, 0]), max(atr, 1e-9)),
            "sma4_s2_dist": clip_div(c5 - float(self.tape.sma4_s[self.t, i, 1]), max(atr, 1e-9)),
            "sma4_s3_dist": clip_div(c5 - float(self.tape.sma4_s[self.t, i, 2]), max(atr, 1e-9)),
            "sma4_s4_dist": clip_div(c5 - float(self.tape.sma4_s[self.t, i, 3]), max(atr, 1e-9)),
            "bb20_cross_up": 1.0 if bb_up else 0.0,
            "bb20_cross_down": 1.0 if bb_dn else 0.0,
            "bb200_cross_up": 1.0 if bb200_up else 0.0,
            "bb200_cross_down": 1.0 if bb200_dn else 0.0,
            "bb_multi_channel_up": 1.0 if multi_up else 0.0,
            "bb_multi_channel_down": 1.0 if multi_dn else 0.0,
            "atr5_over_price": clip_div(atr, mid or c5 or 1.0),
        }
        for tf in ("5m", "15m", "30m", "1h", "4h", "1d"):
            d[f"ha_color_{tf}"] = 1.0 if bool(self.tape.ha_green[tf][self.t, i]) else -1.0
            d[f"ha_n_{tf}"] = min(int(self.tape.ha_n[tf][self.t, i]) / 8.0, 2.0)
            d[f"fractal_high_on_{tf}"] = 1.0 if bool(self.tape.frac_high_on[tf][self.t, i]) else 0.0
            d[f"fractal_low_on_{tf}"] = 1.0 if bool(self.tape.frac_low_on[tf][self.t, i]) else 0.0
            d[f"bars_since_fractal_high_{tf}"] = float(
                np.clip(float(self.tape.frac_bars_high[tf][self.t, i]) / 50.0, 0, 8)
            )
            d[f"bars_since_fractal_low_{tf}"] = float(
                np.clip(float(self.tape.frac_bars_low[tf][self.t, i]) / 50.0, 0, 8)
            )
            d[f"dist_to_last_fractal_high_{tf}"] = clip_div(
                (mid or c5) - float(self.tape.frac_px_high[tf][self.t, i]), max(atr, 1e-9)
            )
            d[f"dist_to_last_fractal_low_{tf}"] = clip_div(
                (mid or c5) - float(self.tape.frac_px_low[tf][self.t, i]), max(atr, 1e-9)
            )

        missing = [n for n in FEATURE_NAMES if n not in d]
        if missing:
            raise RuntimeError(f"obs fill missing {missing}")
        obs = np.array([d[n] for n in FEATURE_NAMES], dtype=np.float32)
        np.nan_to_num(obs, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        info = {
            "mask": self.action_masks(),
            "announce": False,
            "printable": bool(self._printable),
            "aux_y": self.last_aux,
            "ideal_lot": cap,
            "risk_cap_lot": cap,
            "size_scale_target": 1.0,
        }
        return obs, info
