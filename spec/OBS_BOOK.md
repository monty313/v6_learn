# OBS_BOOK — Desktop version 6 (required eyes)

Cursor **must** put these in the observation vector. Fail the card if `target_pct`, `risk_pct`, `goal_frac`, or `risk_frac` are missing.

Work in: `C:\Users\C2K\Desktop\V6`.

---

## 1. Money / day book (mandatory)

| Feature | Meaning |
|---------|---------|
| `target_pct` | Today’s typed target percent |
| `risk_pct` | Today’s typed risk percent |
| `target_dollars` | Target in dollars (normalized ok) |
| `risk_dollars` | Risk in dollars (normalized ok) |
| `day_pnl` | Day dollars so far |
| `goal_frac` | day progress / target (clipped) |
| `risk_frac` | risk used / risk (clipped) |
| `leftover_to_target` | remaining to target (scaled) |
| `risk_left` | remaining risk (scaled) |
| `session_on` | in session |
| `session_time_frac` | session elapsed 0..1 |
| `consecutive_nonwin_days` | career non-HIT streak (eval reset) |
| `consecutive_breach_days` | career BLEW streak (eval reset) |
| `immediate_candy_paid_today` | 10% spice paid so far today |
| `pending_candy_today` | 90% bank (forfeit unless HIT) |
| `remaining_target` | leftover / target (same scale as leftover_to_target) |
| `remaining_risk` | leftover risk / risk |

---

## 2. Position / ticket book (mandatory)

| Feature | Meaning |
|---------|---------|
| `flat` / `side` / `lot_size` / `lot_size_frac_of_max` | position |
| `bars_held` / `open_pnl` / `open_count` | open state |
| `closes_today` / `wins_so_far` / `losses_so_far` | day tape |

---

## 3. Consistency + spice eyes (mandatory)

| Feature | Meaning |
|---------|---------|
| `trade_win_streak` / `day_win_streak` / `clean_day_streak` / `stay_green_streak` | streaks |
| `had_trade_today` / `held_green_after_5m` / `is_clean_day_so_far` | flags |
| `best_clean_hour_wins` / `hour_wins` / `hour_losses` | clean hour |
| `best_lot_ever` / `lot_is_personal_best` | lot PB |
| `sma4_s1_held` … `sma4_s4_held` | 5m SMA4 shift tunnel held after 5m (for open/last trade) |
| `fat_trade_flag` | last close was ≥10% of target |
| `cci30_15m` / `cci30_1h` | CCI(30) on 15m and 1h |
| `cci30_15m_score` / `cci30_1h_score` | distance from 0 on correct side (buy +CCI / sell −CCI) |
| `best_cci30_15m_buy` / `best_cci30_15m_sell` | career PB scores 15m |
| `best_cci30_1h_buy` / `best_cci30_1h_sell` | career PB scores 1h |
| `cci_pb_15m_flag` / `cci_pb_1h_flag` | last close beat CCI30 PB that TF |
| `cci100_15m` / `cci100_1h` | CCI(100) on 15m and 1h |
| `cci100_15m_score` / `cci100_1h_score` | distance from 0 (buy + / sell −) |
| `best_cci100_15m_buy` / `best_cci100_15m_sell` | career PB CCI100 15m |
| `best_cci100_1h_buy` / `best_cci100_1h_sell` | career PB CCI100 1h |
| `cci100_pb_15m_flag` / `cci100_pb_1h_flag` | last close beat CCI100 PB |
| `dual_cci_15m_score` / `dual_cci_1h_score` | min distance of CCI30+CCI100 from 0 |
| `best_dual_cci_15m_buy` / `best_dual_cci_15m_sell` | dual PB records 15m |
| `best_dual_cci_1h_buy` / `best_dual_cci_1h_sell` | dual PB records 1h |
| `pending_dual_pb_15m` / `pending_dual_pb_1h` | dual PB earned today; pays only if day HIT |
| `bars_to_first_green` | M1 bars from open until first unrealized profit > 0 |
| `best_bars_to_first_green_buy` / `best_bars_to_first_green_sell` | career fastest (lower better) |
| `fast_green_pb_flag` | last close beat time-to-green PB |

---

## 4. Heikin Ashi + fractals (mandatory)

Timeframes: **5m, 15m, 30m, 1h, 4h, 1d** (plus 1m HA for door eyes).

**Per TF — Heikin Ashi:** color (or signed), consecutive same-color count `N`.

**Per TF — Fractals (Williams-style, required):**
- `fractal_high_on` / `fractal_low_on`
- `bars_since_fractal_high` / `bars_since_fractal_low` (scaled)
- `dist_to_last_fractal_high` / `dist_to_last_fractal_low` (clipped)

Also expose `door_buy` / `door_sell` (eyes for the HA lock, not pay).

**5m SMA(4) shifted 1..4** rails as eyes (distance above/below each shift), not only reward flags.

**CCI(30) and CCI(100) on 15m and 1h** as eyes (raw + distance-from-0 + dual min-score), required for §§2.12–2.16.

---


## 5. Bollinger channel-cross (mandatory OBS — Mark 2026-09-05)

**Eyes only. Never feed into Channel A reward.**

Bands on price, on each TF v6 already supports (at minimum **5m, 15m, 1h**; also wire 1m / 1d if those HA/door eyes exist):
- **BB20** — period 20
- **BB200** — period 200
- Deviation: pick one stable config used elsewhere in Mark’s stack (default **2.0** if no prior BB config found; log the choice). Do not explode obs with every 0.2–5.0 step on card 1 — one deviation per period is enough; multi-dev is a later card.

**Per open trade (life-of-trade sticky bits; reset on new open):**

| Feature | Meaning |
|---------|---------|
| `bb20_cross_up` | 1 if price ever crossed **up** through BB20 middle or upper during this trade |
| `bb20_cross_down` | 1 if price ever crossed **down** through BB20 middle or lower during this trade |
| `bb200_cross_up` | 1 if price ever crossed **up** through BB200 middle or upper |
| `bb200_cross_down` | 1 if price ever crossed **down** through BB200 middle or lower |
| `bb_multi_channel_up` | 1 if **≥2 distinct upward** BB crossings (across BB20/BB200 events) this trade |
| `bb_multi_channel_down` | 1 if **≥2 distinct downward** BB crossings this trade |

Wire these bits **once per TF** that has BB computed (prefix with TF, e.g. `bb20_cross_up_5m`), **or** aggregate to the primary trade TF (5m) plus a rolled multi-TF OR — pick one scheme, log it, keep it stable. Card-1 minimum: **6 bits on 5m** as named above (count = 6). If per-TF on 5m+15m+1h without rollup: 18 bits.

Optional continuous eyes (not required card 1): distance to BB20/BB200 mid/upper/lower clipped.

**Forbidden:** any reward, spice, or penalty from these bits. If BB ever hard-masks actions, that is a **lock** in the harness — label it as a lock, not “the model learned.”

---
## 6. Thin market extras (optional this card)

5m ATR/price only. Do not dump full INDICATOR_OBS_SETS five-set soup on card 1.

---

## 7. Acceptance

Print `obs_dim` and named dump before train (include BB cross bit names + count). Missing money eyes → fail card, do not train. BB bits missing → fail card.
