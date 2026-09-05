# Harness

This file is the job. `spec/HANDOFF.md` is the full start file. If PRODUCT or a leftover-school card fights this file, this file wins.

## Learn

**One frozen bot.** Train until a human types today’s target % and today’s risk % and the same weights trade. Do not retrain because the percent changed.

Each morning:

- **target** 3–35% of **$10k**
- **risk** 1–6% of **$10k**

US30 only. London 08:00–17:00 Europe/London **or** NY cash 09:30–16:00 America/New_York. Weekends off. Many tickets live. HIT or BLEW → flatten all and stop the day.

**Done:** greedy walk of 100 held-out CEST days. HIT every **printable** day. 0 BLEW on those. `n_printable` > 0. Unprintable geometry is SKIP, not scored.

That is the exam. Principles are thinking tools inside the exam. They are not a second exam.

## Goal channel (paycheck)

Paid only for the day’s money. **Do not pay “the setup printed.”**

| Grade | When | r |
|---|---|---|
| HIT | printable, realized ≥ today’s target $, risk not spent | +1.0 |
| BLEW | printable, today’s risk $ spent | −0.5 |
| MISS | printable, session over short, inside risk | −0.15 |
| SKIP | unprintable or never legal | 0.0 |
| Ticket | close pays ≥ `ticket_floor` (10% of today’s target $) | +0.4, day cap 0.8 |

HIT / BLEW / MISS / ticket are **real OHLC dollars**, not Heikin-Ashi paint. No candy for align, tide, load, release, or “looks like Mark.”

## Entry channel (door)

Fixed. Nothing else may open. Long and short are mirrors.

| Side | 15m HA | 5m HA | 1m HA |
|---|---|---|---|
| **Buy** | ≥2 green | ≥2 green | ≥1 red |
| **Sell** | ≥2 red | ≥2 red | ≥1 green |

- 15m = mass. Not “buy the 15m close.”
- 5m = must agree. No 2-same-color with 15m → no tide → no trade.
- 1m = speed only. Opposite color is a dip/rally, not automatically a reverse.
- 1m never votes short while 5m and 15m stay 2-green. Mirror for 2-red HTF.

Door printed ≠ always enter. Inside the door: wait, skip, size, hold, flatten.

## Thinking channel

Define trend / pullback / reversal / load / launch / chop / collapse / wait / HA-lag on **`1m | 5m, 15m`**. Map to eyes. RL optimizes the day exam conditioned on those eyes. Do not encode principles as reward candy.

Makeup before pay. Exact small/fat counts are draft until locked. See `spec/HANDOFF.md` §3.

## Build order

One change per card: **door → eyes → labels → paycheck last (probably never).**

Keep iff HIT-without-BLEW is up and WR did not collapse. Hits↑ + WR↓ = stamp = revert.
