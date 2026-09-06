from v6rl.product import PHASE_CLOSES, PHASE_DAY, PHASE_FULL, SPICE_PER_DAY_CAP, phase_for_update
from v6rl.rewards import (
    RewardEngine,
    clean_day_spice,
    day_streak_spice,
    dual_cci_day_pay,
    fast_green_spice,
    primary_r,
)


def _close(eng: RewardEngine, pnl: float = 200.0, target: float = 1000.0) -> float:
    return eng.on_close(
        pnl=pnl,
        target_dollars=target,
        lots=1.0,
        side=1,
        n5=8,
        match15=True,
        k_htf=4,
        stay_green_ok=True,
        held_5m=True,
        held_s=4,
        cci30_15m=200.0,
        cci30_1h=200.0,
        cci100_15m=200.0,
        cci100_1h=200.0,
        bars_to_first_green=1.0,
        hour_wins=6,
        hour_losses=0,
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


def test_take_1090_and_terminal():
    eng = RewardEngine()
    paid = eng._take_1090(0.10, 0.10, "ha", 10.0)
    assert abs(paid - 0.01) < 1e-9
    assert abs(eng.day.pending - 0.09) < 1e-9
    hit_extra = eng.settle_pending(day_hit=True)
    assert abs(hit_extra - 0.09) < 1e-9
    eng2 = RewardEngine()
    assert abs(eng2.terminal_day("HIT") - 0.20) < 1e-9
    assert abs(eng2.terminal_day("MISS") - (-0.02)) < 1e-9
    assert abs(eng2.terminal_day("MISS") - (-0.03)) < 1e-9
    blew = RewardEngine()
    assert abs(blew.terminal_day("BLEW") - (-0.10)) < 1e-9
    assert abs(blew.terminal_day("BLEW") - (-0.15)) < 1e-9


def test_spice_cap_clips_day():
    eng = RewardEngine()
    # dump spices until global cap
    for _ in range(40):
        eng._take(0.10, 0.10, "ha", 10.0)
    assert eng.day.spice == SPICE_PER_DAY_CAP
    extra = eng._take(0.10, 0.10, "ha", 10.0)
    assert extra == 0.0


def test_primary_only_strips_spices_terminal_progress():
    eng = RewardEngine()
    eng.set_primary_only(True)
    assert eng.progress.enabled is False
    assert eng.phase == PHASE_CLOSES
    probe = RewardEngine()
    paid = probe._take_1090(0.10, 0.10, "ha", 10.0)
    assert abs(paid - 0.01) < 1e-9
    r = _close(eng)
    assert abs(r - 0.2) < 1e-9
    assert eng.day.spice == 0.0
    assert abs(eng.terminal_day("HIT")) < 1e-12
    assert abs(eng.terminal_day("BLEW")) < 1e-12
    day = eng.on_day_end(
        grade="HIT",
        day_hit=True,
        wins=4,
        losses=0,
        closes_today=4,
        had_trade=True,
        printable=True,
        day_pnl=200.0,
    )
    assert day == 0.0
    assert eng.progress.n_days == 0
    eng.set_primary_only(False)
    assert eng.progress.enabled is True
    assert abs(eng.terminal_day("HIT") - 0.20) < 1e-9
    assert eng.phase == PHASE_FULL


def test_phase_for_update_50_50():
    assert phase_for_update(0, 50, 50) == PHASE_CLOSES
    assert phase_for_update(49, 50, 50) == PHASE_CLOSES
    assert phase_for_update(50, 50, 50) == PHASE_DAY
    assert phase_for_update(99, 50, 50) == PHASE_DAY
    assert phase_for_update(100, 50, 50) == PHASE_FULL


def test_day_phase_harness_paycheck_only():
    eng = RewardEngine()
    eng.set_phase(PHASE_DAY)
    assert eng.progress.enabled is False
    assert _close(eng, pnl=200.0, target=1000.0) == 0.0
    assert eng.day.r_step_total == 0.0
    assert eng.day.spice == 0.0
    assert abs(eng.terminal_day("HIT") - 1.0) < 1e-12
    assert abs(eng.terminal_day("BLEW") - (-0.5)) < 1e-12
    assert abs(eng.terminal_day("MISS") - (-0.15)) < 1e-12
    assert abs(eng.terminal_day("SKIP") - 0.0) < 1e-12
    day = eng.on_day_end(
        grade="HIT",
        day_hit=True,
        wins=4,
        losses=0,
        closes_today=4,
        had_trade=True,
        printable=True,
        day_pnl=400.0,
    )
    assert day == 0.0
    assert eng.progress.n_days == 0
    eng.set_phase(PHASE_FULL)
    assert abs(eng.terminal_day("HIT") - 0.20) < 1e-9


def test_engine_2_7_and_2_16_hit_gates():
    eng = RewardEngine()
    eng.progress.enabled = False
    eng.pending_dual_15m = True
    r = eng.on_day_end(grade="MISS", day_hit=False, wins=4, losses=0, closes_today=4, had_trade=True)
    assert r == 0.0
    assert eng.career.clean_day_streak == 0
    eng.reset_day()
    eng.pending_dual_15m = True
    r = eng.on_day_end(grade="HIT", day_hit=True, wins=4, losses=0, closes_today=4, had_trade=True)
    assert abs(r - (0.05 + 0.04 + 0.025)) < 1e-9