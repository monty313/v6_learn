import numpy as np

from v6rl.geometry import pick_unique_tr, tr_key
from v6rl.product import NEW_PROGRESS_BONUS_PER_DAY_CAP, SPICE_PER_DAY_CAP
from v6rl.progress_bonuses import PAIRS, ProgressBook, pair_mult
from v6rl.rewards import RewardEngine, dual_cci_day_pay


def _day(
    book: ProgressBook,
    grade: str,
    *,
    had_trade: bool = True,
    pnl: float = 100.0,
    closes: int = 2,
    wins: int = 1,
    printable: bool = True,
) -> float:
    return book.on_printable_day(
        grade=grade,
        had_trade=had_trade,
        day_pnl=pnl,
        closes=closes,
        wins=wins,
        printable=printable,
    )


def test_old_rewards_unchanged_when_disabled():
    eng = RewardEngine()
    eng.progress.enabled = False
    eng.pending_dual_15m = True
    r = eng.on_day_end(grade="MISS", day_hit=False, wins=4, losses=0, closes_today=4, had_trade=True)
    assert r == 0.0
    eng.reset_day()
    eng.pending_dual_15m = True
    r = eng.on_day_end(grade="HIT", day_hit=True, wins=4, losses=0, closes_today=4, had_trade=True)
    assert abs(r - (0.05 + 0.04 + 0.025)) < 1e-9
    assert dual_cci_day_pay(day_hit=True, pending_15m=True, pending_1h=False) == 0.025


def test_unique_key_pays_once_and_never_twice():
    book = ProgressBook()
    first = _day(book, "HIT", pnl=300.0)
    assert first > 0.0
    assert book.payout_counts["A"] == 1
    key_a = next(iter(book.paid["A"]))
    second = _day(book, "HIT", pnl=300.0)
    assert key_a in book.paid["A"]
    assert book.dup_blocked["A"] >= 1
    assert book.last_log["A"]["capped"] == 0.0
    assert second >= 0.0


def test_return_to_old_key_pays_zero_new_key_can_pay():
    book = ProgressBook()
    _day(book, "MISS", pnl=10.0, closes=2, wins=1)
    key0 = next(iter(book.paid["A"]))
    assert key0 == (0, 0)
    paid = _day(book, "HIT", pnl=300.0, closes=2, wins=2)
    assert book.last_log["A"]["capped"] > 0.0
    assert paid > 0.0
    replay = ProgressBook()
    replay.load_state(book.state_dict())
    replay.n_days = 0
    replay.n_hit = 0
    replay.n_blew = 0
    replay.n_miss = 0
    replay.n_no_trade = 0
    replay.n_skip = 0
    replay.sum_pnl = 0.0
    replay.n_closes = 0
    replay.n_wins = 0
    again = _day(replay, "MISS", pnl=10.0, closes=2, wins=1)
    assert (0, 0) in replay.paid["A"]
    assert "A_dup_key" in replay.last_log["A"]["fail"]
    assert replay.last_log["A"]["capped"] == 0.0
    assert again >= 0.0


def test_pair_a_fails_if_progress_blew_rises():
    book = ProgressBook()
    book.n_days = 9
    book.n_hit = 4
    book.n_blew = 3
    book.a_best_hit_bps = 1000
    book.a_last_blew_bps = 2000
    book.paid["A"].add((1000, 2000))
    _day(book, "HIT", pnl=100.0)
    assert book.last_log["A"]["capped"] == 0.0
    assert "A2_blew_worse" in book.last_log["A"]["fail"]
    assert book.payout_counts["A"] == 0


def test_pair_b_fails_if_trade_wr_falls():
    book = ProgressBook()
    book.n_days = 9
    book.sum_pnl = 900.0
    book.n_closes = 10
    book.n_wins = 8
    book.b_best_mean_cents = 0
    book.b_last_wr_bps = 9000
    book.paid["B"].add((0, 9000))
    _day(book, "HIT", pnl=100.0, closes=2, wins=1)
    assert book.last_log["B"]["capped"] == 0.0
    assert "B2_wr_down" in book.last_log["B"]["fail"]


def test_pair_c_streak_rules():
    book = ProgressBook()
    _day(book, "HIT", pnl=300.0)
    assert book.last_log["C"]["capped"] == 0.0
    assert "C1_streak_lt_2" in book.last_log["C"]["fail"]
    _day(book, "HIT", pnl=300.0)
    assert book.last_log["C"]["capped"] > 0.0
    assert book.hit_streak == 2
    _day(book, "MISS", pnl=10.0)
    assert book.hit_streak == 0
    _day(book, "HIT", pnl=300.0)
    _day(book, "HIT", pnl=300.0)
    assert book.hit_streak == 2
    assert book.last_log["C"]["capped"] == 0.0
    assert book.dup_blocked["C"] >= 1
    _day(book, "BLEW", pnl=-50.0)
    assert book.hit_streak == 0
    _day(book, "HIT", pnl=300.0)
    _day(book, "MISS", pnl=0.0, had_trade=False, closes=0, wins=0)
    assert book.hit_streak == 0
    _day(book, "HIT", pnl=300.0)
    _day(book, "SKIP", pnl=0.0, had_trade=False, closes=0, wins=0)
    assert book.hit_streak == 0


def test_pair_d_includes_no_trade_skip_and_200bps():
    book = ProgressBook()
    _day(book, "SKIP", pnl=0.0, had_trade=False, closes=0, wins=0)
    assert book.n_skip == 1
    assert book.rates()["progress_blew_n"] == 1
    _day(book, "MISS", pnl=0.0, had_trade=False, closes=0, wins=0)
    assert book.n_no_trade == 1
    assert book.rates()["progress_blew_n"] == 2

    tight = ProgressBook()
    tight.n_days = 99
    tight.n_blew = 48
    tight.n_hit = 20
    tight.d_last_blew_bps = 4999
    tight.d_last_hit_bps = 0
    tight.paid["D"].add((4999, 0))
    _day(tight, "HIT", pnl=100.0)
    assert tight.rates()["blew_bps"] == 4800
    assert 4999 - 4800 == 199
    assert "D1_blew_drop_lt_200" in tight.last_log["D"]["fail"]
    assert tight.last_log["D"]["capped"] == 0.0

    ok = ProgressBook()
    ok.n_days = 99
    ok.n_blew = 48
    ok.n_hit = 20
    ok.d_last_blew_bps = 5000
    ok.d_last_hit_bps = 0
    ok.paid["D"].add((5000, 0))
    _day(ok, "HIT", pnl=100.0)
    assert ok.rates()["blew_bps"] == 4800
    assert 5000 - 4800 == 200
    assert ok.last_log["D"]["capped"] > 0.0

    hit_down = ProgressBook()
    hit_down.n_days = 99
    hit_down.n_blew = 48
    hit_down.n_hit = 20
    hit_down.d_last_blew_bps = 5000
    hit_down.d_last_hit_bps = 9000
    hit_down.paid["D"].add((5000, 9000))
    _day(hit_down, "HIT", pnl=100.0)
    assert "D2_hit_down" in hit_down.last_log["D"]["fail"]


def test_pair_e_includes_no_trade_skip_and_200bps():
    book = ProgressBook()
    _day(book, "SKIP", pnl=0.0, had_trade=False, closes=0, wins=0)
    _day(book, "MISS", pnl=0.0, had_trade=False, closes=0, wins=0)
    assert book.rates()["progress_miss_n"] == 2

    tight = ProgressBook()
    tight.n_days = 99
    tight.n_miss = 48
    tight.n_closes = 100
    tight.n_wins = 80
    tight.e_last_miss_bps = 4999
    tight.e_last_wr_bps = 0
    tight.paid["E"].add((4999, 0))
    _day(tight, "HIT", pnl=100.0, closes=0, wins=0)
    assert tight.rates()["miss_bps"] == 4800
    assert "E1_miss_drop_lt_200" in tight.last_log["E"]["fail"]

    ok = ProgressBook()
    ok.n_days = 99
    ok.n_miss = 48
    ok.n_closes = 100
    ok.n_wins = 80
    ok.e_last_miss_bps = 5000
    ok.e_last_wr_bps = 0
    ok.paid["E"].add((5000, 0))
    _day(ok, "HIT", pnl=100.0, closes=0, wins=0)
    assert ok.last_log["E"]["capped"] > 0.0

    wr_down = ProgressBook()
    wr_down.n_days = 99
    wr_down.n_miss = 48
    wr_down.n_closes = 10
    wr_down.n_wins = 10
    wr_down.e_last_miss_bps = 5000
    wr_down.e_last_wr_bps = 9000
    wr_down.paid["E"].add((5000, 9000))
    _day(wr_down, "HIT", pnl=100.0, closes=2, wins=0)
    assert "E2_wr_down" in wr_down.last_log["E"]["fail"]


def test_pair_f_positive_streak_rules():
    book = ProgressBook()
    _day(book, "HIT", pnl=300.0)
    assert book.pos_streak == 1
    assert book.last_log["F"]["capped"] == 0.0
    _day(book, "MISS", pnl=20.0)
    assert book.pos_streak == 2
    assert book.last_log["F"]["capped"] > 0.0
    _day(book, "MISS", pnl=0.0)
    assert book.pos_streak == 0
    _day(book, "HIT", pnl=300.0)
    _day(book, "MISS", pnl=-1.0)
    assert book.pos_streak == 0
    _day(book, "HIT", pnl=300.0)
    _day(book, "BLEW", pnl=50.0)
    assert book.pos_streak == 0
    _day(book, "HIT", pnl=300.0)
    _day(book, "MISS", pnl=0.0, had_trade=False, closes=0, wins=0)
    assert book.pos_streak == 0
    _day(book, "HIT", pnl=300.0)
    _day(book, "SKIP", pnl=0.0, had_trade=False, closes=0, wins=0)
    assert book.pos_streak == 0
    book2 = ProgressBook()
    _day(book2, "HIT", pnl=300.0)
    _day(book2, "HIT", pnl=300.0)
    assert book2.payout_counts["F"] == 1
    book2.pos_streak = 0
    _day(book2, "HIT", pnl=300.0)
    _day(book2, "HIT", pnl=300.0)
    assert book2.last_log["F"]["capped"] == 0.0
    assert book2.dup_blocked["F"] >= 1


def test_multipliers_and_all_pair_reset():
    book = ProgressBook()

    def all_ok(_cls, _rates):
        n = book.n_days + 1
        return {p: {"ok": True, "key": (n, ord(p)), "fails": [], "dup": False} for p in PAIRS}

    book._eval_pairs = all_ok
    pays = []
    for _ in range(6):
        pays.append(_day(book, "HIT", pnl=300.0))
    assert abs(book.last_log["C"]["multiplier"] - 2.0) < 1e-9
    assert abs(book.last_log["C"]["raw"] - 0.002) < 1e-9
    assert abs(pays[0] - 0.006) < 1e-9
    assert abs(pays[1] - 0.0075) < 1e-9
    assert abs(pays[2] - 0.009) < 1e-9
    assert abs(pays[3] - NEW_PROGRESS_BONUS_PER_DAY_CAP) < 1e-9
    assert abs(pays[4] - NEW_PROGRESS_BONUS_PER_DAY_CAP) < 1e-9
    _day(book, "MISS", pnl=10.0)
    assert all(book.consec[p] == 0 for p in PAIRS)
    nxt = _day(book, "HIT", pnl=300.0)
    assert abs(book.last_log["A"]["multiplier"] - 1.0) < 1e-9
    assert nxt > 0.0
    only_a = ProgressBook()

    def a_only(cls, rates):
        n = only_a.n_days + 1
        out = {p: {"ok": False, "key": (n, ord(p)), "fails": ["x"], "dup": False} for p in PAIRS}
        out["A"] = {"ok": True, "key": (n, 1), "fails": [], "dup": False}
        return out

    only_a._eval_pairs = a_only
    _day(only_a, "HIT", pnl=300.0)
    _day(only_a, "HIT", pnl=300.0)
    assert abs(only_a.last_log["A"]["multiplier"] - 1.25) < 1e-9
    assert only_a.last_log["B"]["multiplier"] == 1.0
    _day(only_a, "HIT", pnl=0.0, closes=0, wins=0)
    # A still pays via monkeypatch; B does not — B consec stays 0
    assert only_a.consec["B"] == 0


def test_ckpt_no_double_pay():
    book = ProgressBook()
    _day(book, "HIT", pnl=300.0)
    blob = book.state_dict()
    assert blob["n_days"] == 1
    assert blob["paid"]["A"]
    other = ProgressBook()
    other.load_state(blob)
    assert other.n_days == 1
    assert other.paid["A"] == book.paid["A"]
    assert other.a_last_blew_bps == book.a_last_blew_bps
    assert other.hit_streak == book.hit_streak
    _day(other, "HIT", pnl=300.0)
    assert other.last_log["A"]["capped"] == 0.0
    assert other.dup_blocked["A"] >= 1


def test_caps_one_pair_once_and_inside_spice():
    book = ProgressBook()
    _day(book, "HIT", pnl=300.0)
    assert book.payout_counts["A"] == 1
    assert book.last_day_total <= NEW_PROGRESS_BONUS_PER_DAY_CAP + 1e-15
    eng = RewardEngine()
    for _ in range(40):
        eng._take(0.10, 0.10, "ha", 10.0)
    assert eng.day.spice == SPICE_PER_DAY_CAP
    extra = eng._take(0.010, 0.010, "progress", 0.010)
    assert extra == 0.0
    eng2 = RewardEngine()
    for _ in range(39):
        eng2._take(0.10, 0.10, "ha", 10.0)
    assert abs(eng2.day.spice - 0.50) < 1e-9 or eng2.day.spice <= SPICE_PER_DAY_CAP
    room = SPICE_PER_DAY_CAP - eng2.day.spice
    got = eng2._take(0.010, 0.010, "progress", 0.010)
    assert got <= room + 1e-15
    assert eng2.day.spice <= SPICE_PER_DAY_CAP + 1e-15


def test_unprintable_does_not_count():
    book = ProgressBook()
    z = _day(book, "SKIP", printable=False, had_trade=False, pnl=0.0, closes=0, wins=0)
    assert z == 0.0
    assert book.n_days == 0


def test_first_pay_vacuous_then_strict():
    book = ProgressBook()
    _day(book, "BLEW", pnl=-20.0, closes=1, wins=0)
    assert book.payout_counts["D"] == 1
    _day(book, "HIT", pnl=300.0, closes=2, wins=2)
    # second day: blew 50% vs last 100% is a 5000bps drop, HIT 50% >= 0
    assert book.payout_counts["D"] == 2


def test_wrap_tr_second_draw_new_key():
    used: set = set()
    rng = np.random.default_rng(0)
    t1, r1, _ = pick_unique_tr("2020-08-20", used, rng, lambda t, r: True)
    t2, r2, _ = pick_unique_tr("2020-08-20", used, rng, lambda t, r: True)
    assert tr_key("2020-08-20", t1, r1) != tr_key("2020-08-20", t2, r2)
    assert len(used) == 2


def test_used_tr_survives_ckpt_and_rejects_old_key():
    book = ProgressBook()
    book.used_tr.add(("2020-08-20", 3, 240))
    other = ProgressBook()
    other.load_state(book.state_dict())
    assert ("2020-08-20", 3, 240) in other.used_tr
    used = set(other.used_tr)
    rng = np.random.default_rng(1)
    t, r, _ = pick_unique_tr("2020-08-20", used, rng, lambda x, y: True)
    assert tr_key("2020-08-20", t, r) != ("2020-08-20", 3, 240)
    assert ("2020-08-20", 3, 240) in used


def test_eval_tr_key_stable_and_funded_draw_untouched():
    assert tr_key("2020-08-20", 3.0, 2.40) == ("2020-08-20", 3, 240)
    assert tr_key("2020-08-20", 10.0, 6.0) == ("2020-08-20", 10, 600)
    from v6rl.geometry import draw_funded_tr

    assert callable(draw_funded_tr)


def test_pair_mult_table():
    assert pair_mult(1) == 1.00
    assert pair_mult(2) == 1.25
    assert pair_mult(3) == 1.50
    assert pair_mult(4) == 1.75
    assert pair_mult(5) == 2.00
    assert pair_mult(9) == 2.00
