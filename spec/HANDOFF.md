# Handoff — v5 → v6 (2026-09-05 03:49 ET)

This file is for the next box, **v6**. It is Mark’s law, not a leftover-school photocopy.

Mark confirmed this file matches the vision: same strict HA door, money-only day exam, principles as thinking tools (eyes + names), not reward candy.

v5 is **not Done**. Do not pretend the island is the product. Take the vision, the money exam, the rails that actually kept, and leave the door that is not Mark’s.

---

## 1. What v6 is

**One frozen bot.** Train until a human types **today’s target %** and **today’s risk %** and the same weights trade. Do not retrain because the percent changed.

Each morning:

- **target** 3–35% of **$10k**
- **risk** 1–6% of **$10k**

US30 only. London 08:00–17:00 Europe/London **or** NY cash 09:30–16:00 America/New_York. Weekends off. Many tickets live. HIT or BLEW → flatten all and stop the day.

**Done:** greedy walk of 100 held-out CEST days. HIT every **printable** day. 0 BLEW on those. `n_printable` > 0. Unprintable geometry is SKIP, not scored.

That is the exam. Principles are the thinking tools **inside** the exam. They are not a second exam.

---

## 2. Two locked channels + one missing brain

### Goal channel (exam / paycheck)

Paid only for the day’s money. **Do not pay “the setup printed.”** The setup is already the only legal open.

| Grade | When | r (v5 live; keep unless a scored card beats it) |
|---|---|---|
| HIT | printable, realized ≥ today’s target $, risk not spent | +1.0 |
| BLEW | printable, today’s risk $ spent | −0.5 |
| MISS | printable, session over short, inside risk | −0.15 |
| SKIP | unprintable or never legal | 0.0 |
| Ticket | close pays ≥ `ticket_floor` (10% of **today’s** target $) | +0.4, day cap 0.8 |

HIT / BLEW / MISS / ticket floor are **real OHLC dollars**, not Heikin-Ashi paint.

No candy for align, tide, load, release, “looks like Mark,” or door-on. Align ~93% with WR ~29% already happened. More badge-pay deepens that.

### Entry channel (door)

Fixed. Nothing else may open. Long and short are mirrors.

| Side | 15m HA | 5m HA | 1m HA |
|---|---|---|---|
| **Buy** | ≥2 green | ≥2 green | ≥1 red |
| **Sell** | ≥2 red | ≥2 red | ≥1 green |

- 15m = mass (tide permission). Not “buy the 15m close.”
- 5m = must **agree**. If 5m is not 2-same-color with 15m, there is no tide. No trade.
- 1m = speed only. Opposite color is a **dip/rally against the tide**, not automatically a reverse.
- 1m **never** votes short while 5m and 15m stay 2-green. Mirror for longs vs 2-red HTF.

v5’s live `buy_on` is **not** this door. It is four CCI / band / fractal windows. HA eyes are banned. **v6 replaces that hunt.** Do not wrap Mark’s brain around W1–W4 and call it the HA door. No new entry types.

### Thinking channel (missing — this is v6’s job)

Inside the door the bot still chooses **wait, skip, size, hold, flatten**. Door printed ≠ always enter.

5m/15m color = trend permission. 1m opposite color = short-term move against that trend. The bot must **name** whether that 1m move is still a usable pullback or the whole move is reversing.

Mark will wrap **trading principles** around this as a toolbox (trend vs regime, momentum vs reversal, multi-TF roles, pullback ≠ reversal, wait vs chop, capital preservation, …). Those principles become **eyes and names**, not reward rows. The principles count; exact candle counts are draft until locked.

Student line:

> 15m is mass, 5m must agree, 1m is only speed. One small opposite 1m is a dip. I wait or I take the resume. Fat/multi opposite is dead. HA is paint. Real 1m OHLC is the stop and the day exam. I still have to HIT today’s T without blowing today’s R.

---

## 3. Principles as tools, not paychecks

**Makeup before pay** (standing law):

1. **Define** in the 1m / 5m / 15m stack: trend, pullback, reversal, load, launch, chop, collapse, wait, HA-lag.
2. **Map** each to observables (HA color counts, body vs ATR, opposite wick, real OHLC vs HA, remaining-to-target, remaining risk). A bar can be named without using the reward to invent the name.
3. **RL** optimizes the day exam **conditioned on** those eyes. Do not encode the principle as `+0.05 release`.

Pin numbers before cameras (or the eye is mush). The table below is **draft to lock before train**, not “Mark already said these counts.”

| Name | Geometry (draft — lock before train) |
|---|---|
| Small opposite 1m | 1m HA body ≤ *k* × 1m ATR (pick *k*) |
| Fat opposite 1m | body > *k* or range > prior 5m/15m tide body |
| Load / pullback | exactly one small 1m opposite, 5m+15m still 2-same-color |
| Resume / release | first 1m tide-color after that one small opposite |
| Reverse | 2–3 closed 1m opposite **or** one fat opposite **or** 5m/15m lose 2-same-color |
| Tide weakening | HTF still tide-color but opposite wick growing, or doji |
| HA lag | HA tide-color **and** real 1m close is against the tide → not strength |
| Chop | 5m/15m colors disagree or alternate → no door, wait |

LTF never votes side against the 5m+15m tide.

Official dense set for this product: **`1m | 5m, 15m`**. Do not dump 30m/1h/4h/1d soup into the first card.

---

## 4. Where v5 actually is (honest)

Island keep **h99** on `../v5/runs/h3_keep` (do not overwrite from a live book):

| Metric | Value |
|---|---|
| WR | **31.7%** (84/265) |
| HIT | **26 / 100** printable |
| BLEW | **6** |
| mean $ | **+$40.86** |
| n_printable | 100 |
| flatten | yes |

Bins: **3% 19/50 HIT BLEW 2**; **10% 3/25 HIT BLEW 3**; **20% 4/25 HIT BLEW 1**; **35% n=0**.

Not Done (26 ≠ 100, BLEW 6). Last continues after the keep **got worse** and were discarded:

| Card | Hidden | Call |
|---|---|---|
| h99 80-update no freeze | 31.7 / 26 / 6 | KEEP — this is v5’s bar |
| h100 second 80 | 30.8 / 23 / 7 | REVERT (3% 19→15) |
| h101 40-update | 25.9 / 21 / 9, −$2 | REVERT |

Update-count resumes are **burned** (11-freeze, second-80, 40). Hits↑ + WR↓ = stamp = revert.

Tape cache: `../v4/data/cache/tape_325e25dabb7b8c12.pkl` — do not rebuild unless v6’s HA door cannot be computed from existing 1m/5m OHLC.

---

## 5. Build order in v6 (one change per card)

Do **not** stack door + eyes + reward in one train.

1. **Door** — `buy_on` / `sell_on` = the HA table above. Paycheck untouched. Hidden will jump. New island. Do not sneak W1–W4 back in.
2. **Eyes** — 15m/5m color counts, 1m opposite count, fat vs small, resume, HA-vs-real, remaining target/risk. Pad OBS if resuming any v5 net; prefer a clean v6 obs if the door is new (a new hunt is not a spice continue). No `--from-scratch` wipe of a *v6 keep* once it exists.
3. **Labels / optional aux** — name load | release | reverse | weaken | lag from geometry. Aux heads are tutors, not fatter Credit.
4. **Paycheck last and probably never** — physics `r` stays off until eyes are real **and** money WR is not the 93/29 stamp. No fancy reward tricks. Test one thing at a time.

Keep iff HIT-without-BLEW is up **and** WR did not collapse. Greener cash alone does not keep.

---

## 6. Take from v5 / leave in v5

**Take**

- Typed T/R as observations. One net. Resume a keep; never blank PPO while a keep exists.
- Session rails. Flatten-all on HIT/BLEW. `ticket_floor` = 10% of today’s target $.
- Money grades only. Makeup before pay. Eyes not badges.
- Score order: **WR → day HIT → mean $** (printable only). Stamp = revert.
- FAIL_BANK *mechanisms* (not the leftover Done): unique-side XOR as a mask reprint, sit-after-pay, look-ahead `path[t]`, lot-cut as cash patch, HIT+10, wait tax, fat-only train mix, train 3/10 only, 11-update freeze as a “continue.”
- US30 tape + London/NY. Many live tickets.

**Leave**

- W1–W4 CCI / band / fractal as the hunt.
- Banned-HA as a virtue. v6 **must** see HA.
- Leftover-pay as Done. 2/3/10 as Done. 100/100 including geometry fails.
- Paying HTF spice as the way to teach principles (tiny +0.01/−0.02 close lights). Optional to kill in v6 day one.
- “Door unchanged + HA eyes” as a substitute for the HA door. That names a story that did not open the trade.

---

## 7. Ops (so the next bot does not stall)

- Law files: this folder is v6. `spec/HARNESS.md` is §1–§3 of this file. Do not copy v5 leftover PRODUCT as law.
- One live train. Snapshot keep before a book. Card run-dir only.
- IRAC before any code edit. One invented `Rule: if … then …`.
- Document Learned every tick. REVERT → FAIL_BANK.
- Stop the loop only on **100d forward PASS** (printable table: HIT == `n_printable`, BLEW 0).

Mark: Software engineer. Works in code. Name the student line in LESSONS when the first HA-door card lands.

---

## 8. principle_ids (v6 start)

`P_FROZEN_NET_TR_AS_INPUT`  
`P_HIT_RELATIVE_NOT_DOLLARS`  
`P_HA_DOOR_ONLY`  
`P_1M_IS_SPEED`  
`P_5M_MUST_AGREE`  
`P_PULLBACK_NE_REVERSAL`  
`P_HA_IS_PAINT`  
`P_MAKEUP_BEFORE_PAY`  
`P_EYES_NOT_BADGES`  
`P_PRINCIPLES_ARE_TOOLS`  
`P_RESUME_NOT_FRESH`  
`P_STAMP_IS_REVERT`
