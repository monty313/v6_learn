from v6rl.day_book import DayRecord, hit_streak, window_summary
from v6rl.product import card1_keep, is_stamp


def _rec(*, grade: str, printable: bool = True, closes: int = 1, wins: int = 0, pnl: float = 0.0) -> DayRecord:
    return DayRecord(
        date="2020-01-01",
        n=1,
        target_pct=3.0,
        risk_pct=2.0,
        daily_target=300.0,
        daily_risk=200.0,
        realized_pnl=pnl,
        realized_loss_abs=0.0,
        n_fills=closes,
        grade=grade,
        equity=10000.0,
        fills={"US30": closes},
        n_trade_closes=closes,
        n_trade_wins=wins,
        printable=printable,
    )


def test_hit_streak_skips_unprintable():
    days = [
        _rec(grade="HIT"),
        _rec(grade="SKIP", printable=False, closes=0),
        _rec(grade="HIT"),
        _rec(grade="MISS"),
        _rec(grade="HIT"),
        _rec(grade="HIT"),
    ]
    assert hit_streak(days) == 2
    days.append(_rec(grade="BLEW"))
    assert hit_streak(days) == 0


def test_window_summary_last_n_printable():
    days = [_rec(grade="MISS", pnl=-10.0, wins=0) for _ in range(5)]
    days += [_rec(grade="HIT", pnl=300.0, wins=1) for _ in range(3)]
    card = window_summary(days, 3)
    assert card["days_hit_target"] == 3
    assert card["n_printable"] == 3


def test_stamp_and_keep_rule():
    assert is_stamp(wr=0.40, hit=8, base_wr=0.45, base_hit=5)
    assert not is_stamp(wr=0.46, hit=8, base_wr=0.45, base_hit=5)
    assert card1_keep(
        wr=0.46,
        hit=8,
        blew=2,
        mean_dollars=10.0,
        base_wr=0.45,
        base_hit=5,
        base_blew=3,
        base_mean=8.0,
    )
    assert not card1_keep(
        wr=0.40,
        hit=8,
        blew=2,
        mean_dollars=10.0,
        base_wr=0.45,
        base_hit=5,
        base_blew=3,
        base_mean=8.0,
    )
