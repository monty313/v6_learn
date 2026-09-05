# INDICATOR_OBS_SETS_V6

This file defines the **indicator sets** that go into OBS for v6. They are the tools the RL model uses to **see and describe market state** inside the HA door.

Timeframes: `1m, 5m, 15m, 1h, 1d`.

- 1m = execution / low timeframe (speed)
- 5m = mid support
- 15m = anchor for intraday trend
- 1h, 1d = higher anchors for regime context

The HA entry door (15m & 5m trend + 1m opposite color) stays fixed. These indicators **do not** create new entries. They are **observations only**.

Rewards and penalties are defined elsewhere. This file does **not** add per-step rewards or penalties. Any illegal-open masks built from these features must be treated as **hard-coded locks**, not learned behavior.

---

## Set 1 — Shifted SMA Envelope (Price Tunnel)

**Indicators**
- SMA(4) shifted forward on **High** and **Low** (envelope) on 1m, 5m, 15m, 1h, 1d.

**OBS features (per TF)**
- `above_top` / `below_bottom` / `inside_envelope` (price vs both rails).
- `full_body_clear` (candle open & close outside envelope) vs `wick_poke_only`.
- `tide_sign` (bull if above both rails; bear if below both rails; neutral inside).

**Principle slots**
- Regime / gravity (force, inertia, regime_gate).
- Full-body clear = breakout; wick-only = weak push.

---

## Set 2 — Dual CCI + Shifted SMA (Fast vs Slow Momentum)

**Indicators**
- Fast CCI (e.g., CCI 30) on all TFs.
- Slow CCI (e.g., CCI 100) on all TFs.
- SMA applied to each CCI with small forward shift (slingshot SMA).

**OBS features (per TF)**
- `fast_sign` / `slow_sign` (above/below 0).
- `fast_vs_sma` / `slow_vs_sma` (above/below shifted SMA).
- `slingshot_state` (categorical):
  - `anchor_load`: slow above SMA, fast below SMA.
  - `anchor_launch`: both above SMA.
  - `anchor_collapse`: both below SMA.

**Principle slots**
- Velocity vs inertia (fast vs slow).
- Load vs launch vs collapse inside HTF gravity.

---

## Set 3 — CCI + Bollinger Bands (Outbreak vs Chop)

**Indicators**
- CCI 10, 30, 100, 300 on all TFs.
- Bollinger Bands on each CCI (e.g., BB(10, dev 1)).

**OBS features (per TF & per CCI)**
- `above_upper_band` / `below_lower_band` / `between_bands`.
- `outbreak_state` (per TF, aggregated over CCIs):
  - `bull_outbreak`: all CCIs above upper band.
  - `bear_outbreak`: all CCIs below lower band.
  - `chop`: all between bands.

**Principle slots**
- Expansion vs consolidation.
- Outbreak vs chop context for continuation vs wait.

---

## Set 4 — Dual RSI + Bands (Trend vs Local Oversold/Overbought)

**Indicators**
- Fast RSI (4) on all TFs.
- Slow RSI (14) on all TFs.
- Bollinger Bands or midline of BB applied to RSI.

**OBS features (per TF)**
- `slow_above_mid` / `slow_below_mid` (trend bias).
- `fast_above_upper` / `fast_below_lower` (local extreme).
- `snap_state` (categorical):
  - `trend_intact`: slow above mid, fast between bands.
  - `tension_load`: slow above mid, fast below lower band in uptrend.
  - `exhaust_revert`: slow crosses below mid against long, or above mid against short.

**Principle slots**
- Trend vs local stretch.
- Pullback vs reversal; exit context when trend breaks.

---

## Set 5 — ADX + ATR Movement Filter (Energy Gate)

**Indicators**
- ADX(14) with SMA(1, shift 2) on all TFs.
- ATR(14) with SMA(1, shift 4) on all TFs.

**OBS features (per TF)**
- `adx_above_sma` / `atr_above_sma`.
- `movement_state` (per TF):
  - `nothing_happening`: both below SMA.
  - `ok_to_trade`: LTF both above SMA, HTF not yet.
  - `great_movement`: LTF & HTF both above SMA.

**Principle slots**
- Volatility and trend strength (expansion).
- Regime survival: energy vs flat chop.

---

## RL Use (Learning, Not Stamping)

- These features go into OBS for all timeframes `1m, 5m, 15m, 1h, 1d`.
- Lower TFs (1m, 5m) are velocity/timing. Higher TFs (15m, 1h, 1d) are force/inertia/regime.
- The RL model is trained on a **separate reward spec** (closed-trade PnL and/or day HIT/BLEW/MISS/Ticket). This file does **not** define reward terms.
- The goal is for the policy to **use** these eyes to forecast which actions improve the money exam, not to maximize "aligned" flags or avoid indicator-based slaps.

If any OBS bit is later used to hard-mask actions ("illegal open"), that mask must be documented as a lock in the harness. It is not counted as learned behavior.