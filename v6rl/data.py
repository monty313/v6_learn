from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from v6rl.config import BERLIN, CSV_DATA_DIR, CSV_FILES, SYMBOLS


def load_symbol_m1(
    symbol: str,
    data_dir: Path | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    path = (data_dir or CSV_DATA_DIR) / CSV_FILES[symbol]
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>"],
        dtype={
            "<DATE>": "string",
            "<TIME>": "string",
            "<OPEN>": np.float64,
            "<HIGH>": np.float64,
            "<LOW>": np.float64,
            "<CLOSE>": np.float64,
        },
    )
    ts = pd.to_datetime(df["<DATE>"] + " " + df["<TIME>"], format="%Y.%m.%d %H:%M:%S")
    out = pd.DataFrame(
        {
            "open": df["<OPEN>"].to_numpy(np.float64),
            "high": df["<HIGH>"].to_numpy(np.float64),
            "low": df["<LOW>"].to_numpy(np.float64),
            "close": df["<CLOSE>"].to_numpy(np.float64),
        },
        index=ts,
    )
    out.index = out.index.tz_localize(BERLIN, ambiguous="infer", nonexistent="shift_forward")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if start:
        out = out.loc[out.index >= pd.Timestamp(start, tz=BERLIN)]
    if end:
        out = out.loc[out.index <= pd.Timestamp(end, tz=BERLIN)]
    return out


def resample_ohlc(m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    return agg.dropna(subset=["open", "high", "low", "close"])


def load_watchlist(
    data_dir: Path | None = None,
    symbols: tuple[str, ...] = SYMBOLS,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    data_dir = data_dir or CSV_DATA_DIR
    return {s: load_symbol_m1(s, data_dir, start=start, end=end) for s in symbols}
