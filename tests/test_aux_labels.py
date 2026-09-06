import numpy as np

from v6rl.aux_labels import label_momentum_hold, label_pullback_resolution


class _T:
    def __init__(self):
        n = 80
        self.n = n
        self.atr5 = np.full((n, 1), 10.0, np.float32)
        self.close = np.full((n, 1), 100.0, np.float32)
        self.high = np.full((n, 1), 100.0, np.float32)
        self.low = np.full((n, 1), 100.0, np.float32)
        self.close5 = np.zeros((n, 1), np.float32)
        for t in range(n):
            self.close5[t, 0] = t // 5
        self.ha_green = {"15m": np.ones((n, 1), dtype=bool)}
        self.ha_n = {"15m": np.full((n, 1), 3, np.int32)}
        self.high[20, 0] = 103.0
        self.low[40, 0] = 92.0


def test_momentum_fav_and_pullback_adv():
    tape = _T()
    assert label_momentum_hold(tape, 10, 0, 1) == 1.0
    assert label_pullback_resolution(tape, 10, 0, 1, 100.0) == 0.0
    assert label_momentum_hold(tape, 10, 0, 0) == -1.0
