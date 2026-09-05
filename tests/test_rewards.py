from v6rl.product import SPICE_PER_DAY_CAP
from v6rl.rewards import (
    RewardEngine,
    clean_day_spice,
    day_streak_spice,
    dual_cci_day_pay,
    fast_green_spice,
    primary_r,
)


def test_primary_clip():
    assert primary_r(500.0, 1000.0) == 0.5
    assert primary_r(5000.0, 1000.0) == 1.0
    assert primary_r(-2000.0, 1000.0) == -1.0


def test_clean_day_requires_hit():
    pay, streak = clean_day_spice(day_hit=False, wins=6, losses=0, streak_before=3)
    assert pay == 0.0
    assert streak == 0
    pay, streak = clean_day_spice(day_hit=True, wins=4, losses=0, streak_before=0)
    assert pay == 0.04
    assert streak == 1
    pay, streak = clean_day_spice(day_hit=True, wins=3, losses=0, streak_before=2)
    assert pay == 0.0
    assert streak == 0
    pay, streak = clean_day_spice(day_hit=True, wins=5, losses=1, streak_before=2)
    assert pay == 0.0


def test_dual_cci_pays_only_on_hit():
    assert dual_cci_day_pay(day_hit=False, pending_15m=True, pending_1h=True) == 0.0
    assert dual_cci_day_pay(day_hit=True, pending_15m=True, pending_1h=False) == 0.025
    assert dual_cci_day_pay(day_hit=True, pending_15m=True, pending_1h=True) == 0.055


def test_fast_green_pb():
    pay, best, beat = fast_green_spice(profit=True, bars_to_first_green=1.0, best=float("inf"))
    assert beat is True
    assert best == 1.0
    assert abs(pay - (0.010 + 0.010 * ((30 - 1) / 30))) < 1e-9
    pay2, best2, beat2 = fast_green_spice(profit=False, bars_to_first_green=1.0, best=5.0)
    assert pay2 == 0.0 and best2 == 5.0 and beat2 is False
    pay3, best3, beat3 = fast_green_spice(profit=True, bars_to_first_green=8.0, best=3.0)
    assert pay3 == 0.0 and best3 == 3.0 and beat3 is False


def test_blank_day_resets_day_streak():
    pay, streak = day_streak_spice(hit=True, no_trade_or_skip=True, streak_before=4)
    assert pay == 0.0
    assert streak == 0
    pay, streak = day_streak_spice(hit=False, no_trade_or_skip=False, streak_before=4)
    assert pay == 0.0
    assert streak == 0
    pay, streak = day_streak_spice(hit=True, no_trade_or_skip=False, streak_before=2)
    assert pay == 0.05 * 3
    assert streak == 3


def test_spice_cap_clips_day():
    eng = RewardEngine()
    # dump spices until global cap
    for _ in range(40):
        eng._take(0.10, 0.10, "ha", 10.0)
    assert eng.day.spice == SPICE_PER_DAY_CAP
    extra = eng._take(0.10, 0.10, "ha", 10.0)
    assert extra == 0.0


def test_engine_2_7_and_2_16_hit_gates():
    eng = RewardEngine()
    eng.pending_dual_15m = True
    r = eng.on_day_end(grade="MISS", day_hit=False, wins=4, losses=0, closes_today=4, had_trade=True)
    assert r == 0.0
    assert eng.career.clean_day_streak == 0
    eng.reset_day()
    eng.pending_dual_15m = True
    r = eng.on_day_end(grade="HIT", day_hit=True, wins=4, losses=0, closes_today=4, had_trade=True)
    assert abs(r - (0.05 + 0.04 + 0.025)) < 1e-9
