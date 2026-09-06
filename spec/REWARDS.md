# REWARDS — Desktop version 6

This file is the **numeric Channel A** money reward for version 6. Cursor must implement these numbers.

Works with:
- `spec/CURSOR_AGENT.md`
- `spec/OBS_BOOK.md` (mandatory eyes; fractals required)
- `INDICATOR_OBS_SETS_V6.md` (thin market eyes only)

Work in: `C:\Users\C2K\Desktop\V6`.

---

## 1. Goal and score order

- One frozen policy. Human types today’s **target %** and **risk %** each morning.
- Hit target without spending risk.

**Keep / revert measurement order:**
1. Trade win rate
2. Day hits (target reached, risk not spent)
3. Mean dollars

Day hits up + trade win rate down = **stamp** → revert.

**First-card keep rule (all required vs same hidden-day untrained baseline):**
1. Trade win rate not down
2. Day HIT count **up**
3. BLEW count not up
4. Mean day $ not more than ~20% worse

Card 1 = smoke (`--hide-days 20`). Card 2 = consistency (`--hide-days 100`). Do not call card 1 Done.

---

## 2. Channel A — money scalar

Wait bars with no close: primary **0.0**.

### Global spice cap
```
SPICE_PER_DAY_CAP = +0.50
```
Sum of all non-2.1 spices in a calendar day cannot exceed this. Primary `r_step` is separate (only its own −1..+1 clip).

### 2.1 Primary — closed trade dollars (always on)

**CARD_CLOSES_THEN_DAY (2026-09-05):** three stages in one occupancy student. Typed target/risk rails stay (printable, HIT/BLEW flatten). Channel B is not money. Do **not** copy this student to `runs/live` or treat it as the official run unless its scorecard beats keep (trade WR → day HIT → BLEW not up → mean $). Ticket leftover pay stays OFF.

| Updates | Channel A money |
|---|---|
| 0–49 (`close_only_updates=50`) | **2.1 only** — `r_step = clip(close_$ / today's_target_$, −1, +1)`. Terminal / spices / 2.18 off. |
| 50–99 (`day_only_updates=50`) | **Day paycheck only** (HARNESS.md): HIT **+1.0**, BLEW **−0.5**, MISS **−0.15**, SKIP **0**. No 2.1. No spices. No 2.18. |
| 100+ | **Full normal** — 2.1 + 2.2 (`+0.20 / −0.02 / −0.10` escalations) + 2.4–2.17 10/90 + 2.18. |

`phase_for_update` in `v6rl/product.py` is the switch. See `spec/LEARN.md`.

**Muscle memory (2026-09-05, Mark):** after a day grades, PPO **actor** step weights are HIT **2.0**, MISS/SKIP **0.15**, BLEW **0.0**. The **critic** still trains on every day (so BLEW teaches “don’t”). No wait lock. No leftover-force. Stamp (HIT up + WR down) means **do not overwrite keep** — it does **not** reload keep into the live net. Winning-day weights stay. Promote keep only if the keep rule passes.
```
target_dollars = account * today's_target_percent
r_step = clip( pnl_close_after_fees / target_dollars , -1.0 , +1.0 )
```

### 2.2 Day terminal (ON — CARD_10_90_HITGATE) + 10/90 spice
```
REWARD_DAY_HIT = +0.20
REWARD_NONWIN_BASE = -0.02
REWARD_BREACH_BASE = -0.10
NONWIN_ESCALATION = -0.01 * min(consecutive_nonwin_days - 1, 5)
BREACH_ESCALATION = -0.05 * min(consecutive_breach_days - 1, 5)
```
HIT resets both streak counters. MISS/SKIP/no-trade increment nonwin only. BLEW increment breach only. Counters persist in ckpt; reset at eval / hide-20. Terminal logged as `r_day`, not spice.

**10/90 candy (temporary):** on-close spices 2.4–2.6 / 2.10 / 2.12–2.15 / 2.17 pay 10% now, bank 90% as `pending_candy`. HIT pays the bank (cap on paid only). MISS/SKIP/BLEW/no-trade forfeit. Day-end HIT-gated 2.5B / 2.7 / 2.8 / 2.9 / 2.11 / 2.16 stay full-on-HIT (already deferred). Flatten rails still use HIT/BLEW/MISS/SKIP.

### 2.3 Ticket / leftover pay — OFF
```
REWARD_TICKET = 0.0
```

### 2.4 Heikin Ashi agreement spice (ON)
Only if close profit. Long=green HA, short=red HA. Evaluate at close.

`N` = consecutive same-color 5m HA looking back. Need `N >= 3` (priors before last 2). Cap N=8. `strength = max(N-2, 0)`.

15m must match or spice=0. `k` = count of {30m,1h,4h,1d} matching.

```
REWARD_HA_FULL    = +0.005 * strength     # k==4
REWARD_HA_PARTIAL = +0.005 * strength * (k/4)   # k in 1..3
HA_SPICE_PER_CLOSE_CAP = +0.03
HA_SPICE_PER_DAY_CAP   = +0.06
```

### 2.5 Bird’s-eye consistency (ON)

**A) Trade-win streak** (on close):
```
if pnl > 0: trade_win_streak += 1; REWARD_TRADE_STREAK = +0.01 * min(trade_win_streak, 10)
else: trade_win_streak = 0; REWARD_TRADE_STREAK = 0
TRADE_STREAK_PER_DAY_CAP = +0.25
```

**B) Day-HIT streak** (day end):
```
if no-trade or SKIP: day_win_streak = 0; REWARD_DAY_STREAK = 0
elif HIT: day_win_streak += 1; REWARD_DAY_STREAK = +0.05 * min(day_win_streak, 10)
else: day_win_streak = 0; REWARD_DAY_STREAK = 0
DAY_STREAK_PER_DAY_CAP = +0.50
```

### 2.6 Stay-in-profit after 5 minutes (ON)
Still open at open+5m; from then until close marked PnL never < 0; close profit.
```
if pass: stay_green_streak += 1; REWARD_STAY_GREEN = +0.005 * min(stay_green_streak, 10)
else: stay_green_streak = 0; REWARD_STAY_GREEN = 0
STAY_GREEN_PER_CLOSE_CAP = +0.05
STAY_GREEN_PER_DAY_CAP   = +0.15
```

### 2.7 Clean day (ON) — **requires day HIT**
At day end, all must hold:
1. Day is **HIT** (target reached, risk not spent)
2. At least **4** winning closes
3. **Zero** losing closes

```
if clean_and_HIT:
    clean_day_streak += 1
    REWARD_CLEAN_DAY = +0.04 * min(clean_day_streak, 10) + 0.01 * min(W - 4, 12)
else:
    clean_day_streak = 0
    REWARD_CLEAN_DAY = 0.0
CLEAN_DAY_PER_DAY_CAP = +0.60
```
No-trade / SKIP / any loser / W<4 / not HIT → reset, pay 0.

### 2.8 Personal-best lot (ON)
New biggest lot ever; pay at day end only if that close was green **and** day HIT.
```
REWARD_LOT_PB = +0.02   # else 0
LOT_PB_PER_DAY_CAP = +0.02
```

### 2.9 Clean hour personal-best (ON)
Rolling 60 minutes; L_hour==0; W_hour new record; **day HIT** (defer to day end).
```
REWARD_CLEAN_HOUR_PB = +0.01 + 0.005 * min(W_hour, 10)   # else 0
CLEAN_HOUR_PB_PER_EVENT_CAP = +0.06
CLEAN_HOUR_PB_PER_DAY_CAP   = +0.12
```

### 2.10 Five-minute SMA4 shift tunnel (ON — Mark 2026-09-05)

**Buys:** after open+5 minutes until close, **price stays at or above** SMA(4) shifted forward by `S` on the **5m** chart the entire time, **and** close in profit.

**Sells:** after open+5 minutes until close, **price stays at or below** SMA(4) shifted forward by `S` on the **5m**, **and** close in profit.

Shifts and pay (only one tier pays — the **largest S that fully held**; do not sum all S):

| Shift S | Reward if tunnel held + profitable close |
|---------|------------------------------------------|
| 1 | +0.005 |
| 2 | +0.010 |
| 3 | +0.015 |
| 4 | +0.020 |

If close before 5m, or dipped through the rail after 5m, or close not profit → **0**.

```
SMA_TUNNEL_PER_CLOSE_CAP = +0.020
SMA_TUNNEL_PER_DAY_CAP   = +0.08
```

Use closed 5m bars only (no look-ahead). “Price” = 5m close (or typical; pick one, log it; default = 5m close).

OBS: flags `sma4_s1_held` … `sma4_s4_held` for the open trade / last close.

### 2.11 Fat trade ≥10% of target (ON — Mark 2026-09-05)

On a close where `pnl_close_after_fees >= 0.10 * today_target_dollars`:

Pay **only if that day ends with at least 4 closes** (wins+losses+scratches that closed). If day has fewer than 4 closes → **0** (defer to day end).

```
if fat_close and closes_today >= 4:
    REWARD_FAT_TRADE = +0.025
else:
    REWARD_FAT_TRADE = 0.0
FAT_TRADE_PER_CLOSE_CAP = +0.025
FAT_TRADE_PER_DAY_CAP   = +0.10
```

Does not require day HIT by itself (primary dollars already score the fat win). Still under `SPICE_PER_DAY_CAP`.

### 2.12 CCI(30) extreme personal-best — 15m (ON — Mark 2026-09-05)

On a **profitable** close, read **CCI(30) on the 15m** at the close bar (closed bar only, no look-ahead).

**Direction gate (must be on the right side of 0):**
- **Buy:** CCI must be **> 0**. Score = CCI (how far above 0).
- **Sell:** CCI must be **< 0**. Score = **−CCI** (how far below 0).

If wrong side of 0, or close not profit → pay **0**, do not update the record.

**Personal best:** keep one career best score for buys and one for sells on this TF (`best_cci30_15m_buy`, `best_cci30_15m_sell`). Pay only if this close’s score **strictly beats** that side’s last best, then update the record.

```
if profit and right_side and score > best_cci30_15m_[side]:
    best_cci30_15m_[side] = score
    REWARD_CCI_PB_15M = +0.015
else:
    REWARD_CCI_PB_15M = 0.0
CCI_PB_15M_PER_CLOSE_CAP = +0.015
CCI_PB_15M_PER_DAY_CAP   = +0.06
```

First ever qualifying close on that side counts as a PB (beat −∞ / unset).

OBS: `cci30_15m`, `cci30_15m_score`, `best_cci30_15m_buy`, `best_cci30_15m_sell`, `cci_pb_15m_flag`.

### 2.13 CCI(30) extreme personal-best — 1h (ON — Mark 2026-09-05)

Same rules as §2.12, but **CCI(30) on the 1 hour** chart.

```
if profit and right_side and score > best_cci30_1h_[side]:
    best_cci30_1h_[side] = score
    REWARD_CCI_PB_1H = +0.020
else:
    REWARD_CCI_PB_1H = 0.0
CCI_PB_1H_PER_CLOSE_CAP = +0.020
CCI_PB_1H_PER_DAY_CAP   = +0.08
```

1h is a little larger than 15m because the chart is slower. 15m and 1h are **independent** records (can both pay on the same close if both beat).

OBS: `cci30_1h`, `cci30_1h_score`, `best_cci30_1h_buy`, `best_cci30_1h_sell`, `cci_pb_1h_flag`.

### 2.14 CCI(100) extreme personal-best — 15m (ON — Mark 2026-09-05)

Same rules as §2.12, but **CCI(100) on the 15m** (not CCI 30).

**Buy:** CCI100 > 0; score = CCI100. **Sell:** CCI100 < 0; score = −CCI100. Profit close required. Pay only if score **strictly beats** `best_cci100_15m_[side]`.

```
if profit and right_side and score > best_cci100_15m_[side]:
    best_cci100_15m_[side] = score
    REWARD_CCI100_PB_15M = +0.015
else:
    REWARD_CCI100_PB_15M = 0.0
CCI100_PB_15M_PER_CLOSE_CAP = +0.015
CCI100_PB_15M_PER_DAY_CAP   = +0.06
```

OBS: `cci100_15m`, `cci100_15m_score`, `best_cci100_15m_buy`, `best_cci100_15m_sell`, `cci100_pb_15m_flag`.

### 2.15 CCI(100) extreme personal-best — 1h (ON — Mark 2026-09-05)

Same as §2.14 on the **1 hour** chart. Independent of CCI30 PBs and of the 15m CCI100 record.

```
if profit and right_side and score > best_cci100_1h_[side]:
    best_cci100_1h_[side] = score
    REWARD_CCI100_PB_1H = +0.020
else:
    REWARD_CCI100_PB_1H = 0.0
CCI100_PB_1H_PER_CLOSE_CAP = +0.020
CCI100_PB_1H_PER_DAY_CAP   = +0.08
```

OBS: `cci100_1h`, `cci100_1h_score`, `best_cci100_1h_buy`, `best_cci100_1h_sell`, `cci100_pb_1h_flag`.

### 2.16 Dual CCI (30 + 100) extreme personal-best — day HIT required (ON — Mark 2026-09-05)

**One spice:** both **CCI(30)** and **CCI(100)** must be far from 0 **at the same time**, on the correct side, on a profitable close. Pays only if that **day ends as HIT**.

Evaluate independently on **15m** and on **1h** (separate career records). A close can qualify on one TF, both, or neither.

**Direction gate (both indicators):**
- **Buy:** CCI30 > 0 **and** CCI100 > 0. Score = `min(CCI30, CCI100)` (bottleneck = the weaker of the two distances above 0).
- **Sell:** CCI30 < 0 **and** CCI100 < 0. Score = `min(−CCI30, −CCI100)`.

If either indicator is on the wrong side of 0, or close not profit → that TF does not qualify.

**Personal best:** keep `best_dual_cci_15m_[side]` and `best_dual_cci_1h_[side]`. Mark a pending dual-PB on that TF only if score **strictly beats** the last best for that side+TF, then update the record at close time.

**Day HIT gate:** do **not** pay on the close. At **day end**, if the day is **HIT** and at least one dual-PB was pending today:

```
REWARD_DUAL_CCI_PB = 0.0
if day_HIT:
    if pending_dual_pb_15m: REWARD_DUAL_CCI_PB += +0.025
    if pending_dual_pb_1h:  REWARD_DUAL_CCI_PB += +0.030
else:
    # day not won — no pay (records still updated so the bar stays high)
    REWARD_DUAL_CCI_PB = 0.0
DUAL_CCI_PB_PER_DAY_CAP = +0.055
```

No-trade / SKIP / MISS / BLEW → pay 0 for this spice (records keep any new highs set during the day).

OBS: `dual_cci_15m_score`, `dual_cci_1h_score`, `best_dual_cci_15m_buy/sell`, `best_dual_cci_1h_buy/sell`, `pending_dual_pb_15m`, `pending_dual_pb_1h`.

### 2.17 Fastest time-to-first-profit personal-best (ON — Mark 2026-09-05)

**Hard gate: the trade must close in profit.** No pay and no record update on losers.

Reward getting **into profit as fast as possible**, then still closing green.

**Clock:** M1 bars (same as the US30 tape). At open, start `bars_held = 0`. Each 1m bar while open, mark unrealized PnL after fees. Record `bars_to_first_green` = bars from open until the **first** bar where unrealized PnL **> 0**. If the trade never goes green before close → no qualify (even if close somehow greens on the last tick without a prior green bar — if close is green, count that close bar as first green).

**Gates:**
1. Close in profit (`pnl_close_after_fees > 0`)
2. `bars_to_first_green` is finite
3. Score beats career best — here **lower is better**: `bars_to_first_green < best_bars_to_first_green` (first ever qualifying sets the record)

Optional side split: one record for buys, one for sells (`best_bars_to_first_green_buy` / `_sell`). Default = **separate by side**.

```
if profit_close and bars_to_first_green < best_bars_to_first_green_[side]:
    best_bars_to_first_green_[side] = bars_to_first_green
    # tiny bonus scales with how fast (1 bar best → full pay; slower PB still pays base)
    speed = clip( (30 - bars_to_first_green) / 30 , 0.0 , 1.0 )   # 30m reference
    REWARD_FAST_GREEN_PB = +0.010 + 0.010 * speed   # +0.010 .. +0.020
else:
    REWARD_FAST_GREEN_PB = 0.0
FAST_GREEN_PB_PER_CLOSE_CAP = +0.020
FAST_GREEN_PB_PER_DAY_CAP   = +0.08
```

If never green until a green close on bar N, `bars_to_first_green = N` (still eligible). Losing closes never update the record and never pay.

OBS: `bars_to_first_green`, `best_bars_to_first_green_buy`, `best_bars_to_first_green_sell`, `fast_green_pb_flag`.

### 2.18 Career progress (ON — additive, unique keys)

Six tiny **day-end** bonuses. They do **not** change §§2.1–2.17, 10/90, or terminal `r_day`. Full pay via `_take` (not 10/90). Clip the six-pair sum to `NEW_PROGRESS_BONUS_PER_DAY_CAP = +0.010` first, then sit inside `SPICE_PER_DAY_CAP = +0.50`.

Career stats = this resume forward, **completed printable days only**. Keep ckpt with no progress book starts empty. Persist paid-key sets, comparison records, streaks, multipliers, career totals, and used T/R keys in the checkpoint. `reset_eval_streaks` clears the book so hide-20 does not leak train career.

Progress-only rates (normal HIT/BLEW/MISS/SKIP labels unchanged):

```
progress_BLEW = normal_BLEW + no_trade + SKIP
progress_MISS = normal_MISS + no_trade + SKIP
```

No-trade / SKIP never progress HIT. Integer keys only (rates → bps, mean $ → cents, streaks → days). Duplicate complete key → 0.

Pairs (each +0.001 base; consecutive same-pair pays 1.00→1.25→1.50→1.75→2.00x; any MISS/BLEW/no-trade/SKIP resets every pair’s consecutive counter):

- **A** HIT rate strictly up vs highest paid A, and progress BLEW ≤ last paid A
- **B** mean day $ strictly up vs highest paid B, and trade WR ≥ last paid B
- **C** HIT streak ≥ 2 and strictly longer than longest paid C, today is HIT
- **D** progress BLEW ≥ 200 bps lower than last paid D, and HIT rate ≥ last paid D
- **E** progress MISS ≥ 200 bps lower than last paid E, and trade WR ≥ last paid E
- **F** positive-day streak ≥ 2 and strictly longer than longest paid F (HIT always positive; traded MISS with PnL>0 extends; zero/neg/BLEW/no-trade/SKIP reset)

First printable-day rule: Pair A/B/D/E may pay once on the first unique key (missing last-paid comparisons are permissive only then). Later pays need a new key and the strict record / 200 bps / not-worse rules.

Train date wrap: a later lap of the same calendar date must not reuse a `(date, target_bin, risk_bps)` already used this resume. Eval T/R seed path is unchanged.

---

## 3. Forbidden as reward

No envelope/RSI/ADX slap. CCI pay is **only** the personal-best extreme spices §§2.12–2.16 (CCI30, CCI100, and dual 30+100; not a continuous CCI candy). No leftover-force. Hard blocks = locks.

---

## 4. Channel B

See `spec/CHANNEL_B.md`. Aux heads are **thinking only** — never added to the money scalar.

Card 1: optional **multi-channel ride** BCE head (weight ~0.05) on BB OBS bits. **Zero** BB reward terms. Primary scalar remains §2.1 (+ spices 2.4–2.17 only).

---

## 5. Observation

See `OBS_BOOK.md`. Fractals mandatory on 5m/15m/30m/1h/4h/1d. Money eyes mandatory.

---

## 6. Scalar sum

On close:
```
r = r_step + HA + TRADE_STREAK + STAY_GREEN + SMA_TUNNEL + CCI_PB_15M + CCI_PB_1H + CCI100_PB_15M + CCI100_PB_1H + FAST_GREEN_PB + (fat deferred)
```
Day end:
```
r += DAY_STREAK + CLEAN_DAY + LOT_PB + CLEAN_HOUR_PB + FAT_TRADE + DUAL_CCI_PB + PROGRESS_2_18
```
Then clip day’s total spice to `SPICE_PER_DAY_CAP`.

---

## 7. First card acceptance

Wire 2.1–2.17 + OBS_BOOK. Hide-20 smoke. Keep only if keep rule in §1 passes. Append LESSONS_BOOK line. Stop and report.
