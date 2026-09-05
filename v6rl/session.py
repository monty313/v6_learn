"""London cash + New York cash. New opens only when either is on. Weekends off."""

from __future__ import annotations

import numpy as np
import pandas as pd

LONDON_OPEN_HOUR = 8
LONDON_CLOSE_HOUR = 17
NY_OPEN_HOUR = 9
NY_OPEN_MINUTE = 30
NY_CLOSE_HOUR = 16


def london_or_ny_session(ts: pd.DatetimeIndex) -> np.ndarray:
    idx = pd.DatetimeIndex(ts)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    london = idx.tz_convert("Europe/London")
    ny = idx.tz_convert("America/New_York")
    lh = np.asarray(london.hour)
    nh = np.asarray(ny.hour)
    nm = np.asarray(ny.minute)
    lwd = np.asarray(london.weekday)
    nwd = np.asarray(ny.weekday)
    in_london = (lh >= LONDON_OPEN_HOUR) & (lh < LONDON_CLOSE_HOUR) & (lwd < 5)
    in_ny = (
        ((nh > NY_OPEN_HOUR) | ((nh == NY_OPEN_HOUR) & (nm >= NY_OPEN_MINUTE)))
        & (nh < NY_CLOSE_HOUR)
        & (nwd < 5)
    )
    return in_london | in_ny
