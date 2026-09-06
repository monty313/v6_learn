"""Per-CEST-day target / risk scoreboard. HIT/BLEW rails; printable only for score."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class DayRecord:
    date: str
    n: int
    target_pct: float
    risk_pct: float
    daily_target: float
    daily_risk: float
    realized_pnl: float
    realized_loss_abs: float
    n_fills: int
    grade: str
    equity: float
    fills: dict[str, int]
    n_trade_closes: int = 0
    n_trade_wins: int = 0
    printable: bool = True
    spice: float = 0.0
    imm_candy: float = 0.0
    pend_earned: float = 0.0
    pend_paid: float = 0.0
    pend_forfeit: float = 0.0
    r_day: float = 0.0
    r_step_total: float = 0.0
    progress: float = 0.0
    progress_pairs: str = ""

    @property
    def hit_target(self) -> bool:
        return self.grade == "HIT"

    @property
    def blew_risk(self) -> bool:
        return self.grade == "BLEW"

    @property
    def inside_risk(self) -> bool:
        return not self.blew_risk

    @property
    def target_left(self) -> float:
        return self.daily_target - self.realized_pnl

    @property
    def risk_left(self) -> float:
        return self.daily_risk - self.realized_loss_abs

    @property
    def active(self) -> bool:
        return self.n_fills >= 1


def record_from_env(env) -> DayRecord:
    book = env.book
    date = env.dates[env.date_i] if env.dates else "?"
    return DayRecord(
        date=str(date),
        n=int(env.train_day_index) + 1,
        target_pct=float(book.target_pct),
        risk_pct=float(book.risk_pct),
        daily_target=float(book.target_dollars),
        daily_risk=float(book.risk_dollars),
        realized_pnl=float(book.realized_day),
        realized_loss_abs=float(book.realized_loss_abs),
        n_fills=int(book.tickets_today),
        grade=str(book.dead or "MISS"),
        equity=float(book.equity),
        fills=dict(book.fills_today),
        n_trade_closes=int(book.n_trade_closes),
        n_trade_wins=int(book.n_trade_wins),
        printable=bool(getattr(env, "_printable", True)),
        spice=float(getattr(env.rew.stats, "last_day_spice", 0.0)),
        imm_candy=float(getattr(env.rew.day, "immediate", 0.0)),
        pend_earned=float(getattr(env.rew.day, "pending_earned", 0.0)),
        pend_paid=float(getattr(env.rew.day, "pending_paid", 0.0)),
        pend_forfeit=float(getattr(env.rew.day, "pending_forfeit", 0.0)),
        r_day=float(getattr(env.rew.day, "r_day", 0.0)),
        r_step_total=float(getattr(env.rew.day, "r_step_total", 0.0)),
        progress=float(getattr(env.rew.day, "progress", 0.0)),
        progress_pairs=",".join(getattr(env.rew.progress, "last_paid_pairs", ()) or ()),
    )


def summarize(days: list[DayRecord]) -> dict:
    if not days:
        return {
            "trade_win_rate": 0.0,
            "days_hit_target": 0,
            "mean_dollars": 0.0,
            "n_trade_closes": 0,
            "n_trade_wins": 0,
            "days": 0,
            "active_days": 0,
            "idle_days": 0,
            "days_inside_risk": 0,
            "days_blew_risk": 0,
            "days_miss": 0,
            "days_skip": 0,
            "total_pnl": 0.0,
            "n_fills": 0,
            "days_hit_frac": 0.0,
            "n_printable": 0,
            "days_geometry_skip": 0,
            "mean_spice": 0.0,
        }
    n = len(days)
    scored = [d for d in days if d.printable]
    n_print = len(scored)
    hit = sum(1 for d in scored if d.hit_target)
    blew = sum(1 for d in scored if d.blew_risk)
    miss = sum(1 for d in days if d.grade == "MISS")
    skip = sum(1 for d in days if d.grade == "SKIP")
    geom_skip = sum(1 for d in days if (not d.printable) and d.grade == "SKIP")
    active = sum(1 for d in scored if d.active)
    total_pnl = float(sum(d.realized_pnl for d in scored))
    closes = int(sum(d.n_trade_closes for d in scored))
    wins = int(sum(d.n_trade_wins for d in scored))
    return {
        "trade_win_rate": (wins / closes) if closes else 0.0,
        "n_trade_closes": closes,
        "n_trade_wins": wins,
        "mean_dollars": (total_pnl / n_print) if n_print else 0.0,
        "days": n,
        "n_printable": n_print,
        "active_days": active,
        "idle_days": n - sum(1 for d in days if d.active),
        "days_hit_target": hit,
        "days_inside_risk": n_print - blew,
        "days_blew_risk": blew,
        "days_miss": miss,
        "days_skip": skip,
        "days_geometry_skip": geom_skip,
        "total_pnl": total_pnl,
        "n_fills": int(sum(d.n_fills for d in scored)),
        "days_hit_frac": hit / n_print if n_print else 0.0,
        "equity": days[-1].equity,
        "mean_spice": (sum(d.spice for d in days) / n) if n else 0.0,
    }


def hit_streak(days: list[DayRecord]) -> int:
    """Consecutive printable HIT days, newest first. Unprintable SKIP does not break."""
    n = 0
    for rec in reversed(days):
        if not rec.printable:
            continue
        if rec.hit_target:
            n += 1
        else:
            break
    return n


def window_summary(days: list[DayRecord], n: int = 100) -> dict:
    scored = [d for d in days if d.printable]
    return summarize(scored[-int(n) :] if scored else [])


BOARD_HEADER = f"{'DAY':<12} {'N':>5} {'TGT%':>6} {'TGT$':>10} {'PNL':>10} {'FILLS':>5}  TARGET  RISK"
BOARD_RULE = "-" * 72


def format_row(d: DayRecord) -> str:
    if d.grade == "SKIP" or d.n_fills < 1:
        hit = "idle" if d.grade == "SKIP" else "short"
    elif d.hit_target:
        hit = "HIT"
    else:
        hit = "short"
    risk = "BLEW" if d.blew_risk else "safe"
    return (
        f"{d.date:<12} {d.n:>5} {d.target_pct:>6.1f} {d.daily_target:>10.2f} "
        f"{d.realized_pnl:>+10.2f} {d.n_fills:>5}  {hit:<6}  {risk}"
    )


def format_report_card(s: dict) -> str:
    return (
        f"trade wr {100.0 * float(s.get('trade_win_rate') or 0.0):.1f}% "
        f"({int(s.get('n_trade_wins') or 0)}/{int(s.get('n_trade_closes') or 0)}) | "
        f"days HIT {int(s.get('days_hit_target') or 0)} | "
        f"mean $ {float(s.get('mean_dollars') or 0.0):+.2f}"
    )


def format_footer(days: list[DayRecord]) -> str:
    s = summarize(days)
    return (
        f"{format_report_card(s)} | "
        f"{s['active_days']} active / {s['idle_days']} idle | "
        f"BLEW {s['days_blew_risk']}/{s.get('n_printable') or s['days']} printable | "
        f"total pnl {s['total_pnl']:+.2f}"
    )


def format_day_open(
    rec_n: int,
    n_days: int,
    date,
    target_pct: float,
    risk_pct: float,
    target_dollars: float,
    risk_dollars: float,
) -> str:
    return (
        f"\nDay {rec_n}/{n_days}  {date}  "
        f"tgt ${target_dollars:.2f} ({target_pct:.1f}%)  "
        f"risk ${risk_dollars:.2f} ({risk_pct:.1f}%)"
    )


_CSV_HEADER = [
    "date",
    "n",
    "target_pct",
    "risk_pct",
    "realized_pnl",
    "realized_loss_abs",
    "n_fills",
    "n_trade_closes",
    "n_trade_wins",
    "hit_target",
    "blew_risk",
    "grade",
    "daily_target",
    "daily_risk",
    "equity",
    "printable",
    "spice",
    "progress",
    "progress_pairs",
]


def _csv_row(d: DayRecord) -> list:
    return [
        d.date,
        d.n,
        f"{d.target_pct:.4f}",
        f"{d.risk_pct:.4f}",
        f"{d.realized_pnl:.4f}",
        f"{d.realized_loss_abs:.4f}",
        d.n_fills,
        d.n_trade_closes,
        d.n_trade_wins,
        int(d.hit_target),
        int(d.blew_risk),
        d.grade,
        f"{d.daily_target:.2f}",
        f"{d.daily_risk:.2f}",
        f"{d.equity:.2f}",
        int(d.printable),
        f"{d.spice:.4f}",
        f"{d.progress:.6f}",
        d.progress_pairs,
    ]


class DayBook:
    def __init__(self, run_dir: Path, n_days: int):
        self.run_dir = Path(run_dir)
        self.n_days = int(n_days)
        self.days: list[DayRecord] = []
        self._csv_started = False
        self.best_hit_streak = 0
        self.keep_saves = 0

    def add(self, rec: DayRecord) -> None:
        self.days.append(rec)
        csv_path = self.run_dir / "live_day_book.csv"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        mode = "w" if not self._csv_started else "a"
        with csv_path.open(mode, newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if not self._csv_started:
                w.writerow(_CSV_HEADER)
                self._csv_started = True
            w.writerow(_csv_row(rec))
        payload = {
            "product": "Desktop version 6",
            "last": asdict(rec),
            "summary": self.running(),
            "tail": [asdict(d) for d in self.days[-30:]],
            "update": getattr(self, "last_update", None),
        }
        (self.run_dir / "live.json").write_text(json.dumps(payload, default=str), encoding="utf-8")

    def running(self) -> dict:
        s = summarize(self.days)
        streak = hit_streak(self.days)
        if streak > self.best_hit_streak:
            self.best_hit_streak = streak
        s["hit_streak"] = streak
        s["best_hit_streak"] = self.best_hit_streak
        s["keep_saves"] = self.keep_saves
        return s

    def print_open(self, env) -> None:
        date = env.dates[env.date_i] if env.dates else ""
        print(
            format_day_open(
                len(self.days) + 1,
                self.n_days,
                date,
                env.book.target_pct,
                env.book.risk_pct,
                env.book.target_dollars,
                env.book.risk_dollars,
            ),
            flush=True,
        )

    def print_close(self, rec: DayRecord) -> None:
        if len(self.days) == 1:
            print(BOARD_HEADER, flush=True)
            print(BOARD_RULE, flush=True)
        print(format_row(rec), flush=True)
        print(format_footer(self.days), flush=True)
