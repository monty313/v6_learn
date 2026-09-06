"""§2.18 career-progress bonuses. Unique integer keys. Day-end only. Not 10/90."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from v6rl.product import (
    NEW_PROGRESS_BONUS_PER_DAY_CAP,
    REWARD_PROGRESS_PAIR_A,
    REWARD_PROGRESS_PAIR_B,
    REWARD_PROGRESS_PAIR_C,
    REWARD_PROGRESS_PAIR_D,
    REWARD_PROGRESS_PAIR_E,
    REWARD_PROGRESS_PAIR_F,
)

PAIRS = ("A", "B", "C", "D", "E", "F")
PAIR_BASE = {
    "A": REWARD_PROGRESS_PAIR_A,
    "B": REWARD_PROGRESS_PAIR_B,
    "C": REWARD_PROGRESS_PAIR_C,
    "D": REWARD_PROGRESS_PAIR_D,
    "E": REWARD_PROGRESS_PAIR_E,
    "F": REWARD_PROGRESS_PAIR_F,
}
BPS_DROP = 200


def rate_bps(num: int, den: int) -> int:
    if int(den) <= 0:
        return 0
    return int(round((float(num) / float(den)) * 10000.0))


def mean_cents(total: float, n: int) -> int:
    if int(n) <= 0:
        return 0
    return int(round((float(total) / float(n)) * 100.0))


def pair_mult(consec: int) -> float:
    """1-based consecutive pay count → 1.00 / 1.25 / 1.50 / 1.75 / 2.00."""
    n = max(1, int(consec))
    return float(min(1.0 + 0.25 * (n - 1), 2.0))


def _freeze_key(raw) -> tuple:
    if raw is None:
        return tuple()
    if isinstance(raw, (int, float)):
        return (int(raw),)
    return tuple(int(x) if not isinstance(x, str) else x for x in raw)


@dataclass
class ProgressBook:
    """Career stats = this resume forward, completed printable days only."""

    enabled: bool = True
    n_days: int = 0
    n_hit: int = 0
    n_blew: int = 0
    n_miss: int = 0
    n_no_trade: int = 0
    n_skip: int = 0
    n_closes: int = 0
    n_wins: int = 0
    sum_pnl: float = 0.0
    hit_streak: int = 0
    pos_streak: int = 0
    a_best_hit_bps: int | None = None
    a_last_blew_bps: int | None = None
    b_best_mean_cents: int | None = None
    b_last_wr_bps: int | None = None
    c_best_streak: int = 0
    d_last_blew_bps: int | None = None
    d_last_hit_bps: int | None = None
    e_last_miss_bps: int | None = None
    e_last_wr_bps: int | None = None
    f_best_streak: int = 0
    paid: dict[str, set[tuple]] = field(default_factory=lambda: {p: set() for p in PAIRS})
    consec: dict[str, int] = field(default_factory=lambda: {p: 0 for p in PAIRS})
    last_pay_day: dict[str, int] = field(default_factory=lambda: {p: 0 for p in PAIRS})
    payout_counts: dict[str, int] = field(default_factory=lambda: {p: 0 for p in PAIRS})
    paid_sum: dict[str, float] = field(default_factory=lambda: {p: 0.0 for p in PAIRS})
    dup_blocked: dict[str, int] = field(default_factory=lambda: {p: 0 for p in PAIRS})
    fail_counts: Counter = field(default_factory=Counter)
    used_tr: set[tuple] = field(default_factory=set)
    last_day_total: float = 0.0
    last_day_raw: float = 0.0
    last_paid_pairs: tuple[str, ...] = ()
    last_log: dict = field(default_factory=dict)
    max_day_progress: float = 0.0
    days_over_cap: int = 0

    def reset(self) -> None:
        enabled = self.enabled
        self.__dict__.update(ProgressBook(enabled=enabled).__dict__)
        self.enabled = enabled

    def classify(self, *, grade: str, had_trade: bool, day_pnl: float) -> dict:
        g = str(grade)
        hit = g == "HIT"
        blew = g == "BLEW"
        skip = g == "SKIP"
        no_trade = (not bool(had_trade)) and (not skip)
        miss = g == "MISS" and bool(had_trade)
        positive = bool(hit) or (
            float(day_pnl) > 0.0 and (not blew) and (not no_trade) and (not skip)
        )
        return {
            "hit": hit,
            "blew": blew,
            "miss": miss,
            "no_trade": no_trade,
            "skip": skip,
            "positive": positive,
            "is_break": not hit,
        }

    def rates(self) -> dict:
        n = self.n_days
        return {
            "hit_bps": rate_bps(self.n_hit, n),
            "blew_bps": rate_bps(self.n_blew + self.n_no_trade + self.n_skip, n),
            "miss_bps": rate_bps(self.n_miss + self.n_no_trade + self.n_skip, n),
            "wr_bps": rate_bps(self.n_wins, self.n_closes),
            "mean_cents": mean_cents(self.sum_pnl, n),
            "progress_blew_n": self.n_blew + self.n_no_trade + self.n_skip,
            "progress_miss_n": self.n_miss + self.n_no_trade + self.n_skip,
        }

    def _eval_pairs(self, cls: dict, rates: dict) -> dict[str, dict]:
        hit_bps = rates["hit_bps"]
        blew_bps = rates["blew_bps"]
        miss_bps = rates["miss_bps"]
        wr_bps = rates["wr_bps"]
        mean_c = rates["mean_cents"]
        out: dict[str, dict] = {}

        key_a = (hit_bps, blew_bps)
        a_first = self.a_last_blew_bps is None
        a1 = True if a_first else hit_bps > int(self.a_best_hit_bps)
        a2 = True if a_first else blew_bps <= int(self.a_last_blew_bps)
        fails_a = []
        if not a1:
            fails_a.append("A1_hit_not_higher")
        if not a2:
            fails_a.append("A2_blew_worse")
        dup_a = key_a in self.paid["A"]
        if dup_a:
            fails_a.append("A_dup_key")
        out["A"] = {"ok": (not fails_a) and a1 and a2 and (not dup_a), "key": key_a, "fails": fails_a, "dup": dup_a}

        key_b = (mean_c, wr_bps)
        b_first = self.b_last_wr_bps is None
        b1 = True if b_first else mean_c > int(self.b_best_mean_cents)
        b2 = True if b_first else wr_bps >= int(self.b_last_wr_bps)
        fails_b = []
        if not b1:
            fails_b.append("B1_mean_not_higher")
        if not b2:
            fails_b.append("B2_wr_down")
        dup_b = key_b in self.paid["B"]
        if dup_b:
            fails_b.append("B_dup_key")
        out["B"] = {"ok": (not fails_b), "key": key_b, "fails": fails_b, "dup": dup_b}

        key_c = (int(self.hit_streak),)
        fails_c = []
        if int(self.hit_streak) < 2:
            fails_c.append("C1_streak_lt_2")
        if int(self.hit_streak) <= int(self.c_best_streak):
            fails_c.append("C1_not_longer")
        if not cls["hit"]:
            fails_c.append("C2_not_hit")
        dup_c = key_c in self.paid["C"]
        if dup_c:
            fails_c.append("C_dup_key")
        out["C"] = {"ok": (not fails_c), "key": key_c, "fails": fails_c, "dup": dup_c}

        key_d = (blew_bps, hit_bps)
        d_first = self.d_last_blew_bps is None
        d1 = True if d_first else blew_bps <= int(self.d_last_blew_bps) - BPS_DROP
        d2 = True if d_first else hit_bps >= int(self.d_last_hit_bps)
        fails_d = []
        if not d1:
            fails_d.append("D1_blew_drop_lt_200")
        if not d2:
            fails_d.append("D2_hit_down")
        dup_d = key_d in self.paid["D"]
        if dup_d:
            fails_d.append("D_dup_key")
        out["D"] = {"ok": (not fails_d), "key": key_d, "fails": fails_d, "dup": dup_d}

        key_e = (miss_bps, wr_bps)
        e_first = self.e_last_miss_bps is None
        e1 = True if e_first else miss_bps <= int(self.e_last_miss_bps) - BPS_DROP
        e2 = True if e_first else wr_bps >= int(self.e_last_wr_bps)
        fails_e = []
        if not e1:
            fails_e.append("E1_miss_drop_lt_200")
        if not e2:
            fails_e.append("E2_wr_down")
        dup_e = key_e in self.paid["E"]
        if dup_e:
            fails_e.append("E_dup_key")
        out["E"] = {"ok": (not fails_e), "key": key_e, "fails": fails_e, "dup": dup_e}

        key_f = (int(self.pos_streak),)
        fails_f = []
        if int(self.pos_streak) < 2:
            fails_f.append("F1_streak_lt_2")
        if int(self.pos_streak) <= int(self.f_best_streak):
            fails_f.append("F1_not_longer")
        dup_f = key_f in self.paid["F"]
        if dup_f:
            fails_f.append("F2_dup_key")
        out["F"] = {"ok": (not fails_f), "key": key_f, "fails": fails_f, "dup": dup_f}
        return out

    def _record_pay(self, pair: str, key: tuple, rates: dict) -> None:
        self.paid[pair].add(key)
        self.payout_counts[pair] += 1
        if pair == "A":
            hit_bps, blew_bps = int(key[0]), int(key[1])
            self.a_best_hit_bps = hit_bps if self.a_best_hit_bps is None else max(int(self.a_best_hit_bps), hit_bps)
            self.a_last_blew_bps = blew_bps
        elif pair == "B":
            mean_c, wr_bps = int(key[0]), int(key[1])
            self.b_best_mean_cents = mean_c if self.b_best_mean_cents is None else max(int(self.b_best_mean_cents), mean_c)
            self.b_last_wr_bps = wr_bps
        elif pair == "C":
            self.c_best_streak = max(int(self.c_best_streak), int(key[0]))
        elif pair == "D":
            self.d_last_blew_bps = int(key[0])
            self.d_last_hit_bps = int(key[1])
        elif pair == "E":
            self.e_last_miss_bps = int(key[0])
            self.e_last_wr_bps = int(key[1])
        elif pair == "F":
            self.f_best_streak = max(int(self.f_best_streak), int(key[0]))

    def on_printable_day(
        self,
        *,
        grade: str,
        had_trade: bool,
        day_pnl: float,
        closes: int = 0,
        wins: int = 0,
        printable: bool = True,
    ) -> float:
        self.last_day_total = 0.0
        self.last_day_raw = 0.0
        self.last_paid_pairs = ()
        self.last_log = {}
        if (not self.enabled) or (not printable):
            return 0.0
        cls = self.classify(grade=grade, had_trade=had_trade, day_pnl=day_pnl)
        if cls["hit"]:
            self.hit_streak += 1
        else:
            self.hit_streak = 0
        if cls["positive"]:
            self.pos_streak += 1
        else:
            self.pos_streak = 0
        self.n_days += 1
        self.n_hit += int(cls["hit"])
        self.n_blew += int(cls["blew"])
        self.n_miss += int(cls["miss"])
        self.n_no_trade += int(cls["no_trade"])
        self.n_skip += int(cls["skip"])
        self.n_closes += int(closes)
        self.n_wins += int(wins)
        self.sum_pnl += float(day_pnl)
        rates = self.rates()
        judged = self._eval_pairs(cls, rates)
        is_break = bool(cls["is_break"])
        candidates: dict[str, dict] = {}
        raw_sum = 0.0
        for pair in PAIRS:
            info = judged[pair]
            for reason in info["fails"]:
                self.fail_counts[reason] += 1
            if info["dup"]:
                self.dup_blocked[pair] += 1
            if not info["ok"]:
                candidates[pair] = {
                    "raw": 0.0,
                    "mult": 1.0,
                    "capped": 0.0,
                    "key": info["key"],
                    "new": not info["dup"],
                    "fail": list(info["fails"]),
                    "consec": int(self.consec[pair]),
                }
                continue
            if is_break:
                consec = 1
            elif self.last_pay_day[pair] == self.n_days - 1 and self.consec[pair] > 0:
                consec = self.consec[pair] + 1
            else:
                consec = 1
            mult = pair_mult(consec)
            raw = float(PAIR_BASE[pair]) * mult
            candidates[pair] = {
                "raw": raw,
                "mult": mult,
                "capped": raw,
                "key": info["key"],
                "new": True,
                "fail": [],
                "consec": consec,
            }
            raw_sum += raw
        scale = 1.0
        if raw_sum > NEW_PROGRESS_BONUS_PER_DAY_CAP + 1e-15:
            scale = NEW_PROGRESS_BONUS_PER_DAY_CAP / raw_sum
            self.days_over_cap += 1
        paid_pairs = []
        day_total = 0.0
        for pair in PAIRS:
            c = candidates[pair]
            capped = float(c["raw"]) * scale
            c["capped"] = capped
            if capped > 0.0:
                self._record_pay(pair, c["key"], rates)
                self.paid_sum[pair] += capped
                self.last_pay_day[pair] = self.n_days
                if is_break:
                    self.consec[pair] = 0
                else:
                    self.consec[pair] = int(c["consec"])
                paid_pairs.append(pair)
                day_total += capped
            else:
                if not is_break:
                    self.consec[pair] = 0
            self.last_log[pair] = {
                "raw": c["raw"],
                "multiplier": c["mult"],
                "capped": capped,
                "key": c["key"],
                "new": c["new"],
                "fail": c["fail"],
                "consec": c["consec"],
            }
        if is_break:
            for pair in PAIRS:
                self.consec[pair] = 0
        self.last_day_raw = raw_sum
        self.last_day_total = day_total
        self.last_paid_pairs = tuple(paid_pairs)
        self.max_day_progress = max(self.max_day_progress, day_total)
        return float(min(day_total, NEW_PROGRESS_BONUS_PER_DAY_CAP))

    def state_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "n_days": self.n_days,
            "n_hit": self.n_hit,
            "n_blew": self.n_blew,
            "n_miss": self.n_miss,
            "n_no_trade": self.n_no_trade,
            "n_skip": self.n_skip,
            "n_closes": self.n_closes,
            "n_wins": self.n_wins,
            "sum_pnl": self.sum_pnl,
            "hit_streak": self.hit_streak,
            "pos_streak": self.pos_streak,
            "a_best_hit_bps": self.a_best_hit_bps,
            "a_last_blew_bps": self.a_last_blew_bps,
            "b_best_mean_cents": self.b_best_mean_cents,
            "b_last_wr_bps": self.b_last_wr_bps,
            "c_best_streak": self.c_best_streak,
            "d_last_blew_bps": self.d_last_blew_bps,
            "d_last_hit_bps": self.d_last_hit_bps,
            "e_last_miss_bps": self.e_last_miss_bps,
            "e_last_wr_bps": self.e_last_wr_bps,
            "f_best_streak": self.f_best_streak,
            "paid": {p: [list(k) for k in sorted(self.paid[p])] for p in PAIRS},
            "consec": dict(self.consec),
            "last_pay_day": dict(self.last_pay_day),
            "payout_counts": dict(self.payout_counts),
            "paid_sum": dict(self.paid_sum),
            "dup_blocked": dict(self.dup_blocked),
            "fail_counts": dict(self.fail_counts),
            "used_tr": [list(k) for k in sorted(self.used_tr)],
            "max_day_progress": self.max_day_progress,
            "days_over_cap": self.days_over_cap,
        }

    def load_state(self, blob: dict | None) -> None:
        if not blob:
            return
        self.enabled = bool(blob.get("enabled", True))
        self.n_days = int(blob.get("n_days") or 0)
        self.n_hit = int(blob.get("n_hit") or 0)
        self.n_blew = int(blob.get("n_blew") or 0)
        self.n_miss = int(blob.get("n_miss") or 0)
        self.n_no_trade = int(blob.get("n_no_trade") or 0)
        self.n_skip = int(blob.get("n_skip") or 0)
        self.n_closes = int(blob.get("n_closes") or 0)
        self.n_wins = int(blob.get("n_wins") or 0)
        self.sum_pnl = float(blob.get("sum_pnl") or 0.0)
        self.hit_streak = int(blob.get("hit_streak") or 0)
        self.pos_streak = int(blob.get("pos_streak") or 0)
        self.a_best_hit_bps = blob.get("a_best_hit_bps")
        self.a_last_blew_bps = blob.get("a_last_blew_bps")
        self.b_best_mean_cents = blob.get("b_best_mean_cents")
        self.b_last_wr_bps = blob.get("b_last_wr_bps")
        self.c_best_streak = int(blob.get("c_best_streak") or 0)
        self.d_last_blew_bps = blob.get("d_last_blew_bps")
        self.d_last_hit_bps = blob.get("d_last_hit_bps")
        self.e_last_miss_bps = blob.get("e_last_miss_bps")
        self.e_last_wr_bps = blob.get("e_last_wr_bps")
        self.f_best_streak = int(blob.get("f_best_streak") or 0)
        paid = blob.get("paid") or {}
        self.paid = {p: { _freeze_key(k) for k in (paid.get(p) or []) } for p in PAIRS}
        self.consec = {p: int((blob.get("consec") or {}).get(p) or 0) for p in PAIRS}
        self.last_pay_day = {p: int((blob.get("last_pay_day") or {}).get(p) or 0) for p in PAIRS}
        self.payout_counts = {p: int((blob.get("payout_counts") or {}).get(p) or 0) for p in PAIRS}
        self.paid_sum = {p: float((blob.get("paid_sum") or {}).get(p) or 0.0) for p in PAIRS}
        self.dup_blocked = {p: int((blob.get("dup_blocked") or {}).get(p) or 0) for p in PAIRS}
        self.fail_counts = Counter({str(k): int(v) for k, v in (blob.get("fail_counts") or {}).items()})
        self.used_tr = { _freeze_key(k) for k in (blob.get("used_tr") or []) }
        self.max_day_progress = float(blob.get("max_day_progress") or 0.0)
        self.days_over_cap = int(blob.get("days_over_cap") or 0)

    def report(self) -> dict:
        n = self.n_days
        rates = self.rates() if n else self.rates()
        return {
            "career_days": n,
            "normal_HIT": self.n_hit,
            "normal_BLEW": self.n_blew,
            "normal_MISS": self.n_miss,
            "no_trade": self.n_no_trade,
            "SKIP": self.n_skip,
            "progress_BLEW": rates["progress_blew_n"],
            "progress_MISS": rates["progress_miss_n"],
            "hit_rate": (self.n_hit / n) if n else 0.0,
            "progress_blew_rate": (rates["progress_blew_n"] / n) if n else 0.0,
            "progress_miss_rate": (rates["progress_miss_n"] / n) if n else 0.0,
            "trade_win_rate": (self.n_wins / self.n_closes) if self.n_closes else 0.0,
            "mean_day_dollars": (self.sum_pnl / n) if n else 0.0,
            "n_closes": self.n_closes,
            "n_wins": self.n_wins,
            "max_day_progress": self.max_day_progress,
            "days_over_cap": self.days_over_cap,
            "pairs": {
                p: {
                    "payouts": self.payout_counts[p],
                    "total_paid": self.paid_sum[p],
                    "dup_blocked": self.dup_blocked[p],
                    "consec": self.consec[p],
                    "mult": pair_mult(self.consec[p]) if self.consec[p] else 1.0,
                    "keys": len(self.paid[p]),
                }
                for p in PAIRS
            },
            "fail_counts": dict(self.fail_counts),
            "used_tr": len(self.used_tr),
            "saved": {
                "A": {"best_hit_bps": self.a_best_hit_bps, "last_blew_bps": self.a_last_blew_bps},
                "B": {"best_mean_cents": self.b_best_mean_cents, "last_wr_bps": self.b_last_wr_bps},
                "C": {"best_streak": self.c_best_streak},
                "D": {"last_blew_bps": self.d_last_blew_bps, "last_hit_bps": self.d_last_hit_bps},
                "E": {"last_miss_bps": self.e_last_miss_bps, "last_wr_bps": self.e_last_wr_bps},
                "F": {"best_streak": self.f_best_streak},
            },
        }
