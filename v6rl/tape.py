from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from v6rl.config import CACHE_DIR, CSV_DATA_DIR, CSV_FILES, SYMBOLS, WARMUP_DAYS
from v6rl.data import load_watchlist, resample_ohlc
from v6rl.ha import door_flags, heikin_ashi
from v6rl.indicators import (
    atr_wilder,
    bollinger,
    cci,
    confirmed_fractal,
    fractal_highs,
    fractal_lows,
    sma_shifted,
)
from v6rl.product import BB_DEVIATION, HIDDEN_DAYS_CARD1, MFE_HORIZON_MINUTES
from v6rl.session import london_or_ny_session
from v6rl.symbols import SPECS

# New V6 key — never overwrite v4/v5 pickles.
TAPE_CACHE_VERSION = "v6c1-ha-bb6-cci-20260905"


def _epoch_ns(idx: pd.DatetimeIndex) -> np.ndarray:
    x = pd.DatetimeIndex(idx)
    if x.tz is None:
        x = x.tz_localize("UTC")
    else:
        x = x.tz_convert("UTC")
    return x.as_unit("ns").asi8.astype(np.int64)


def _asof_idx(src_end: np.ndarray, dst_end: np.ndarray) -> np.ndarray:
    return np.searchsorted(src_end, dst_end, side="right") - 1


def _tf_frame(m1: pd.DataFrame, rule: str, minutes: int) -> tuple[pd.DataFrame, np.ndarray]:
    tf = resample_ohlc(m1, rule)
    end = _epoch_ns(tf.index + pd.Timedelta(minutes=minutes))
    return tf, end


def _day_ranges(date_keys: np.ndarray) -> tuple[dict, np.ndarray]:
    n = date_keys.size
    if n == 0:
        return {}, date_keys
    chg = np.flatnonzero(date_keys[1:] != date_keys[:-1]) + 1
    starts = np.concatenate(([0], chg))
    ends = np.concatenate((chg, [n]))
    unique = date_keys[starts]
    return {d: (int(s), int(e)) for d, s, e in zip(unique, starts, ends)}, unique


def _take(arr: np.ndarray, idx: np.ndarray, fill=np.nan):
    ok = idx >= 0
    if arr.ndim == 1:
        if arr.dtype == np.bool_:
            out = np.zeros(idx.shape, dtype=np.bool_)
            out[ok] = arr[idx[ok]]
            return out
        out = np.full(idx.shape, fill, dtype=arr.dtype)
        out[ok] = arr[idx[ok]]
        return out
    out = np.zeros((idx.shape[0],) + arr.shape[1:], dtype=arr.dtype)
    if arr.dtype != np.bool_ and fill != 0:
        out[...] = fill
    out[ok] = arr[idx[ok]]
    return out


@dataclass
class Tape:
    ts: pd.DatetimeIndex
    session: np.ndarray
    date_keys: np.ndarray
    unique_dates: np.ndarray
    day_range: dict
    symbols: tuple[str, ...]
    valid: np.ndarray
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    atr5: np.ndarray
    door_buy: np.ndarray
    door_sell: np.ndarray
    mfe_buy_dpl: np.ndarray
    mfe_sell_dpl: np.ndarray
    close5: np.ndarray
    sma4_s: np.ndarray
    bb20_up: np.ndarray
    bb20_mid: np.ndarray
    bb20_lo: np.ndarray
    bb200_up: np.ndarray
    bb200_mid: np.ndarray
    bb200_lo: np.ndarray
    cci30_15m: np.ndarray
    cci30_1h: np.ndarray
    cci100_15m: np.ndarray
    cci100_1h: np.ndarray
    ha_green: dict
    ha_n: dict
    frac_high_on: dict
    frac_low_on: dict
    frac_bars_high: dict
    frac_bars_low: dict
    frac_px_high: dict
    frac_px_low: dict
    train_dates: list
    hidden_dates: list

    @property
    def n(self) -> int:
        return int(self.ts.size)

    def day_slice(self, date) -> tuple[int, int]:
        return self.day_range[np.datetime64(date, "D")]


def _symbol_native(m1: pd.DataFrame, symbol: str) -> dict:
    frames = {
        "1m": (m1, _epoch_ns(m1.index + pd.Timedelta(minutes=1))),
        "5m": _tf_frame(m1, "5min", 5),
        "15m": _tf_frame(m1, "15min", 15),
        "30m": _tf_frame(m1, "30min", 30),
        "1h": _tf_frame(m1, "1h", 60),
        "4h": _tf_frame(m1, "4h", 240),
        "1d": _tf_frame(m1, "1D", 1440),
    }
    packed: dict[str, dict] = {}
    for name, (df, end) in frames.items():
        o = df["open"].to_numpy(np.float64)
        h = df["high"].to_numpy(np.float64)
        l = df["low"].to_numpy(np.float64)
        c = df["close"].to_numpy(np.float64)
        _ha_o, _ha_c, green, run = heikin_ashi(o, h, l, c)
        fh = fractal_highs(h)
        fl = fractal_lows(l)
        fhon, fhb, fhpx = confirmed_fractal(fh)
        flon, flb, flpx = confirmed_fractal(fl)
        packed[name] = {
            "end": end,
            "o": o,
            "h": h,
            "l": l,
            "c": c,
            "green": green,
            "n": run,
            "fhon": fhon,
            "flon": flon,
            "fhb": fhb,
            "flb": flb,
            "fhpx": fhpx,
            "flpx": flpx,
        }

    p5 = packed["5m"]
    sma4 = np.stack([sma_shifted(p5["c"], 4, s) for s in (1, 2, 3, 4)], axis=1)
    bb20_u, bb20_m, bb20_l = bollinger(p5["c"], 20, BB_DEVIATION)
    bb200_u, bb200_m, bb200_l = bollinger(p5["c"], 200, BB_DEVIATION)
    p15 = packed["15m"]
    p1h = packed["1h"]
    cci30_15 = cci(p15["h"], p15["l"], p15["c"], 30)
    cci100_15 = cci(p15["h"], p15["l"], p15["c"], 100)
    cci30_1h = cci(p1h["h"], p1h["l"], p1h["c"], 30)
    cci100_1h = cci(p1h["h"], p1h["l"], p1h["c"], 100)
    atr5 = atr_wilder(p5["h"], p5["l"], p5["c"])

    end1 = packed["1m"]["end"]
    idx = {tf: _asof_idx(packed[tf]["end"], end1) for tf in packed}

    def g(tf: str, key: str, fill=np.nan):
        return _take(packed[tf][key], idx[tf], fill=fill)

    door_b, door_s = door_flags(
        g("15m", "n", 0).astype(np.int32),
        g("15m", "green", False).astype(np.bool_),
        g("5m", "n", 0).astype(np.int32),
        g("5m", "green", False).astype(np.bool_),
        g("1m", "n", 0).astype(np.int32),
        g("1m", "green", False).astype(np.bool_),
    )

    spec = SPECS[symbol]
    c1 = packed["1m"]["c"]
    h1 = packed["1m"]["h"]
    l1 = packed["1m"]["l"]
    n = c1.size
    half = spec.spread_price / 2.0
    ask = c1 + half
    bid = c1 - half
    pip_v = spec.pip_value_per_lot / spec.pip
    hzn = MFE_HORIZON_MINUTES
    if n == 0:
        mfe_b = np.zeros(0, dtype=np.float32)
        mfe_s = np.zeros(0, dtype=np.float32)
    else:
        pad_h = np.empty(n + hzn, dtype=np.float64)
        pad_h[:n] = h1
        pad_h[n:] = -np.inf
        pad_l = np.empty(n + hzn, dtype=np.float64)
        pad_l[:n] = l1
        pad_l[n:] = np.inf
        hh = sliding_window_view(pad_h[1:], hzn).max(axis=1)
        ll = sliding_window_view(pad_l[1:], hzn).min(axis=1)
        mfe_b = ((hh - ask) * pip_v).astype(np.float32)
        mfe_s = ((bid - ll) * pip_v).astype(np.float32)
        mfe_b[~np.isfinite(mfe_b)] = 0.0
        mfe_s[~np.isfinite(mfe_s)] = 0.0

    i5 = idx["5m"]
    out = {
        "end": end1,
        "close": c1.astype(np.float32),
        "high": h1.astype(np.float32),
        "low": l1.astype(np.float32),
        "atr5": np.nan_to_num(_take(atr5, i5), nan=0.0).astype(np.float32),
        "door_buy": door_b,
        "door_sell": door_s,
        "mfe_buy_dpl": mfe_b,
        "mfe_sell_dpl": mfe_s,
        "close5": _take(p5["c"], i5).astype(np.float32),
        "sma4_s": _take(sma4, i5).astype(np.float32) if sma4.size else np.zeros((n, 4), np.float32),
        "bb20_up": _take(bb20_u, i5).astype(np.float32),
        "bb20_mid": _take(bb20_m, i5).astype(np.float32),
        "bb20_lo": _take(bb20_l, i5).astype(np.float32),
        "bb200_up": _take(bb200_u, i5).astype(np.float32),
        "bb200_mid": _take(bb200_m, i5).astype(np.float32),
        "bb200_lo": _take(bb200_l, i5).astype(np.float32),
        "cci30_15m": _take(cci30_15, idx["15m"]).astype(np.float32),
        "cci30_1h": _take(cci30_1h, idx["1h"]).astype(np.float32),
        "cci100_15m": _take(cci100_15, idx["15m"]).astype(np.float32),
        "cci100_1h": _take(cci100_1h, idx["1h"]).astype(np.float32),
        "ha_green": {},
        "ha_n": {},
        "frac_high_on": {},
        "frac_low_on": {},
        "frac_bars_high": {},
        "frac_bars_low": {},
        "frac_px_high": {},
        "frac_px_low": {},
    }
    if sma4.size == 0:
        out["sma4_s"] = np.zeros((n, 4), dtype=np.float32)
    for tf in ("1m", "5m", "15m", "30m", "1h", "4h", "1d"):
        out["ha_green"][tf] = g(tf, "green", False)
        out["ha_n"][tf] = g(tf, "n", 0).astype(np.int32)
        if tf == "1m":
            continue
        out["frac_high_on"][tf] = g(tf, "fhon", False)
        out["frac_low_on"][tf] = g(tf, "flon", False)
        out["frac_bars_high"][tf] = g(tf, "fhb", 1e6).astype(np.float32)
        out["frac_bars_low"][tf] = g(tf, "flb", 1e6).astype(np.float32)
        out["frac_px_high"][tf] = g(tf, "fhpx").astype(np.float32)
        out["frac_px_low"][tf] = g(tf, "flpx").astype(np.float32)
    return out


def _csv_sig(data_dir: Path, symbols: tuple[str, ...]) -> tuple:
    out = []
    for s in symbols:
        p = Path(data_dir) / CSV_FILES[s]
        if p.exists():
            st = p.stat()
            out.append((p.name, st.st_mtime_ns, st.st_size))
        else:
            out.append((CSV_FILES[s], 0, 0))
    return tuple(out)


def _cache_path(
    cache_dir: Path, hide_last_days: int, symbols: tuple[str, ...], start, end
) -> Path:
    raw = f"{TAPE_CACHE_VERSION}|{hide_last_days}|{start}|{end}|{','.join(symbols)}"
    digest = hashlib.sha1(raw.encode()).hexdigest()[:16]
    return Path(cache_dir) / f"tape_{digest}.pkl"


def build_tape(
    data_dir: Path | None = None,
    cache_dir: Path | None = None,
    hide_last_days: int = HIDDEN_DAYS_CARD1,
    symbols: tuple[str, ...] = SYMBOLS,
    start: str | None = None,
    end: str | None = None,
    use_cache: bool = True,
    rebuild: bool = False,
) -> Tape:
    data_dir = Path(data_dir or CSV_DATA_DIR)
    cache_dir = Path(cache_dir or CACHE_DIR)
    cache_path = _cache_path(cache_dir, hide_last_days, symbols, start, end)
    sig = _csv_sig(data_dir, symbols)
    if use_cache and cache_path.exists() and not rebuild:
        try:
            with cache_path.open("rb") as f:
                blob = pickle.load(f)
            if blob.get("version") == TAPE_CACHE_VERSION:
                print(f"tape cache hit  {cache_path.name}  (no rebuild)", flush=True)
                return blob["tape"]
        except Exception:
            print(f"tape cache unreadable  {cache_path.name}  — building once", flush=True)
    elif cache_path.exists() and rebuild:
        print(f"tape rebuild requested  {cache_path.name}", flush=True)
    else:
        print(f"tape cache miss  {cache_path.name}  — building once", flush=True)

    raw = load_watchlist(data_dir, symbols, start=start, end=end)
    native = {s: _symbol_native(raw[s], s) for s in symbols}
    ends = np.unique(np.concatenate([native[s]["end"] for s in symbols]))
    ends.sort()
    ts = pd.to_datetime(ends, utc=True).tz_convert("Europe/Berlin")
    n = ends.size
    n_s = len(symbols)
    session = london_or_ny_session(ts)
    naive = ts.tz_localize(None) if ts.tz is not None else ts
    date_keys = naive.normalize().to_numpy().astype("datetime64[D]")
    day_range, unique = _day_ranges(date_keys)

    def align_exact(src_end, src, dst_end, fill=0):
        n_d = dst_end.size
        if src.ndim == 1:
            out = np.full(n_d, fill, dtype=src.dtype)
        else:
            out = np.zeros((n_d,) + src.shape[1:], dtype=src.dtype)
            if fill != 0:
                out[...] = fill
        valid = np.zeros(n_d, dtype=np.bool_)
        if src_end.size == 0 or n_d == 0:
            return out, valid
        idx = np.searchsorted(src_end, dst_end)
        in_range = idx < src_end.size
        idx_c = np.minimum(idx, src_end.size - 1)
        ok = in_range & (src_end[idx_c] == dst_end)
        out[ok] = src[idx_c[ok]]
        valid[ok] = True
        return out, valid

    valid = np.zeros((n, n_s), dtype=np.bool_)
    close = np.zeros((n, n_s), dtype=np.float32)
    high = np.zeros((n, n_s), dtype=np.float32)
    low = np.zeros((n, n_s), dtype=np.float32)
    atr5 = np.zeros((n, n_s), dtype=np.float32)
    door_b = np.zeros((n, n_s), dtype=np.bool_)
    door_s = np.zeros((n, n_s), dtype=np.bool_)
    mfe_b = np.zeros((n, n_s), dtype=np.float32)
    mfe_s = np.zeros((n, n_s), dtype=np.float32)
    close5 = np.zeros((n, n_s), dtype=np.float32)
    sma4_s = np.zeros((n, n_s, 4), dtype=np.float32)
    bb20_up = np.zeros((n, n_s), dtype=np.float32)
    bb20_mid = np.zeros((n, n_s), dtype=np.float32)
    bb20_lo = np.zeros((n, n_s), dtype=np.float32)
    bb200_up = np.zeros((n, n_s), dtype=np.float32)
    bb200_mid = np.zeros((n, n_s), dtype=np.float32)
    bb200_lo = np.zeros((n, n_s), dtype=np.float32)
    cci30_15m = np.zeros((n, n_s), dtype=np.float32)
    cci30_1h = np.zeros((n, n_s), dtype=np.float32)
    cci100_15m = np.zeros((n, n_s), dtype=np.float32)
    cci100_1h = np.zeros((n, n_s), dtype=np.float32)
    tfs_ha = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
    tfs_fr = ("5m", "15m", "30m", "1h", "4h", "1d")
    ha_green = {tf: np.zeros((n, n_s), dtype=np.bool_) for tf in tfs_ha}
    ha_n = {tf: np.zeros((n, n_s), dtype=np.int32) for tf in tfs_ha}
    frac_high_on = {tf: np.zeros((n, n_s), dtype=np.bool_) for tf in tfs_fr}
    frac_low_on = {tf: np.zeros((n, n_s), dtype=np.bool_) for tf in tfs_fr}
    frac_bars_high = {tf: np.zeros((n, n_s), dtype=np.float32) for tf in tfs_fr}
    frac_bars_low = {tf: np.zeros((n, n_s), dtype=np.float32) for tf in tfs_fr}
    frac_px_high = {tf: np.zeros((n, n_s), dtype=np.float32) for tf in tfs_fr}
    frac_px_low = {tf: np.zeros((n, n_s), dtype=np.float32) for tf in tfs_fr}

    for k, s in enumerate(symbols):
        d = native[s]
        close[:, k], valid[:, k] = align_exact(d["end"], d["close"], ends)
        high[:, k], _ = align_exact(d["end"], d["high"], ends)
        low[:, k], _ = align_exact(d["end"], d["low"], ends)
        atr5[:, k], _ = align_exact(d["end"], d["atr5"], ends)
        door_b[:, k], _ = align_exact(d["end"], d["door_buy"], ends)
        door_s[:, k], _ = align_exact(d["end"], d["door_sell"], ends)
        mfe_b[:, k], _ = align_exact(d["end"], d["mfe_buy_dpl"], ends)
        mfe_s[:, k], _ = align_exact(d["end"], d["mfe_sell_dpl"], ends)
        close5[:, k], _ = align_exact(d["end"], d["close5"], ends)
        sma4_s[:, k], _ = align_exact(d["end"], d["sma4_s"], ends)
        bb20_up[:, k], _ = align_exact(d["end"], d["bb20_up"], ends)
        bb20_mid[:, k], _ = align_exact(d["end"], d["bb20_mid"], ends)
        bb20_lo[:, k], _ = align_exact(d["end"], d["bb20_lo"], ends)
        bb200_up[:, k], _ = align_exact(d["end"], d["bb200_up"], ends)
        bb200_mid[:, k], _ = align_exact(d["end"], d["bb200_mid"], ends)
        bb200_lo[:, k], _ = align_exact(d["end"], d["bb200_lo"], ends)
        cci30_15m[:, k], _ = align_exact(d["end"], d["cci30_15m"], ends)
        cci30_1h[:, k], _ = align_exact(d["end"], d["cci30_1h"], ends)
        cci100_15m[:, k], _ = align_exact(d["end"], d["cci100_15m"], ends)
        cci100_1h[:, k], _ = align_exact(d["end"], d["cci100_1h"], ends)
        for tf in tfs_ha:
            ha_green[tf][:, k], _ = align_exact(d["end"], d["ha_green"][tf], ends)
            ha_n[tf][:, k], _ = align_exact(d["end"], d["ha_n"][tf], ends)
        for tf in tfs_fr:
            frac_high_on[tf][:, k], _ = align_exact(d["end"], d["frac_high_on"][tf], ends)
            frac_low_on[tf][:, k], _ = align_exact(d["end"], d["frac_low_on"][tf], ends)
            frac_bars_high[tf][:, k], _ = align_exact(d["end"], d["frac_bars_high"][tf], ends)
            frac_bars_low[tf][:, k], _ = align_exact(d["end"], d["frac_bars_low"][tf], ends)
            frac_px_high[tf][:, k], _ = align_exact(d["end"], d["frac_px_high"][tf], ends)
            frac_px_low[tf][:, k], _ = align_exact(d["end"], d["frac_px_low"][tf], ends)

    session_dates = [d for d in unique if bool(session[day_range[d][0] : day_range[d][1]].any())]
    if len(session_dates) <= WARMUP_DAYS + hide_last_days:
        raise RuntimeError(
            f"not enough session days ({len(session_dates)}) for warmup={WARMUP_DAYS} hide={hide_last_days}"
        )
    usable = session_dates[WARMUP_DAYS:]
    hidden = list(usable[-hide_last_days:]) if hide_last_days else []
    train = list(usable[: len(usable) - len(hidden)])
    train_dates = [pd.Timestamp(d).date() for d in train]
    hidden_dates = [pd.Timestamp(d).date() for d in hidden]
    if set(train_dates) & set(hidden_dates):
        raise RuntimeError("train_dates overlap hidden_dates")

    tape = Tape(
        ts=ts,
        session=session,
        date_keys=date_keys,
        unique_dates=unique,
        day_range=day_range,
        symbols=symbols,
        valid=valid,
        close=close,
        high=high,
        low=low,
        atr5=atr5,
        door_buy=door_b,
        door_sell=door_s,
        mfe_buy_dpl=mfe_b,
        mfe_sell_dpl=mfe_s,
        close5=close5,
        sma4_s=sma4_s,
        bb20_up=bb20_up,
        bb20_mid=bb20_mid,
        bb20_lo=bb20_lo,
        bb200_up=bb200_up,
        bb200_mid=bb200_mid,
        bb200_lo=bb200_lo,
        cci30_15m=cci30_15m,
        cci30_1h=cci30_1h,
        cci100_15m=cci100_15m,
        cci100_1h=cci100_1h,
        ha_green=ha_green,
        ha_n=ha_n,
        frac_high_on=frac_high_on,
        frac_low_on=frac_low_on,
        frac_bars_high=frac_bars_high,
        frac_bars_low=frac_bars_low,
        frac_px_high=frac_px_high,
        frac_px_low=frac_px_low,
        train_dates=train_dates,
        hidden_dates=hidden_dates,
    )
    if use_cache and (rebuild or not cache_path.exists()):
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("wb") as f:
                pickle.dump(
                    {"version": TAPE_CACHE_VERSION, "sig": sig, "tape": tape},
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            print(f"tape cache wrote  {cache_path}  version={TAPE_CACHE_VERSION}", flush=True)
        except Exception as exc:
            print(f"tape cache write skipped: {exc}", flush=True)
    return tape


def write_date_split(run_dir: Path, tape: Tape, hide_last_days: int) -> Path:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "hidden_dates.json"
    payload = {
        "rule": "last N CEST session days after warmup — same list eval uses",
        "hide_last_days": int(hide_last_days),
        "train_dates": [str(d) for d in tape.train_dates],
        "hidden_dates": [str(d) for d in tape.hidden_dates],
        "tape_cache_version": TAPE_CACHE_VERSION,
        "account_base": 10_000.0,
        "sma_tunnel_price": "5m_close",
        "bb_deviation": BB_DEVIATION,
        "bb_scheme": "6 bits on 5m only",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_hidden_dates(run_dir: Path) -> list[str]:
    path = Path(run_dir) / "hidden_dates.json"
    blob = json.loads(path.read_text(encoding="utf-8"))
    return [str(d) for d in blob["hidden_dates"]]
