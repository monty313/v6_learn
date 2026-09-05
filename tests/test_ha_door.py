from v6rl.env import next_choice_index
from v6rl.ha import door_buy, door_sell, door_flags, heikin_ashi
import numpy as np


def test_door_buy_table():
    assert door_buy(n15=2, green15=True, n5=2, green5=True, n1=1, green1=False)
    assert door_buy(n15=5, green15=True, n5=3, green5=True, n1=2, green1=False)
    assert not door_buy(n15=1, green15=True, n5=2, green5=True, n1=1, green1=False)
    assert not door_buy(n15=2, green15=True, n5=1, green5=True, n1=1, green1=False)
    assert not door_buy(n15=2, green15=True, n5=2, green5=True, n1=1, green1=True)
    assert not door_buy(n15=2, green15=False, n5=2, green5=True, n1=1, green1=False)
    assert not door_buy(n15=2, green15=True, n5=2, green5=False, n1=1, green1=False)


def test_door_sell_is_mirror():
    assert door_sell(n15=2, green15=False, n5=2, green5=False, n1=1, green1=True)
    assert not door_sell(n15=2, green15=False, n5=2, green5=False, n1=1, green1=False)
    assert not door_sell(n15=2, green15=True, n5=2, green5=False, n1=1, green1=True)
    assert not door_sell(n15=1, green15=False, n5=2, green5=False, n1=1, green1=True)


def test_door_flags_vector():
    n15 = np.array([2, 2, 2, 1])
    g15 = np.array([True, False, True, True])
    n5 = np.array([2, 2, 2, 2])
    g5 = np.array([True, False, True, True])
    n1 = np.array([1, 1, 1, 1])
    g1 = np.array([False, True, True, False])
    buy, sell = door_flags(n15, g15, n5, g5, n1, g1)
    assert list(buy) == [True, False, False, False]
    assert list(sell) == [False, True, False, False]


def test_heikin_ashi_run_length():
    o = np.array([1.0, 1.1, 1.2, 1.0, 0.9])
    h = o + 0.2
    l = o - 0.2
    c = np.array([1.2, 1.3, 1.4, 0.8, 0.7])
    _ho, _hc, green, run = heikin_ashi(o, h, l, c)
    assert green[0] in (True, False)
    assert int(run[0]) == 1
    assert run.max() >= 1


def test_next_choice_skips_door_off():
    session = np.array([True, True, True, True, True])
    buy = np.array([False, False, True, False, False])[:, None]
    sell = np.array([False, False, False, False, True])[:, None]
    assert next_choice_index(session, buy, sell, 0, 5) == 2
    assert next_choice_index(session, buy, sell, 3, 5) == 4
    assert next_choice_index(session, buy, sell, 5, 5) == 5
    off = np.zeros((5, 1), dtype=bool)
    assert next_choice_index(session, off, off, 0, 5) == 5
    session[2] = False
    assert next_choice_index(session, buy, sell, 0, 5) == 4
