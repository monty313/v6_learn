Build Desktop V6 first card. Specs are law. Do not invent rewards or obs beyond the files.

## Paths
Work only in: `C:\Users\C2K\Desktop\V6`

Read in this order (later wins on conflict for money/obs):
1. `spec\CURSOR_AGENT.md`
2. `spec\REWARDS.md` — Channel A §§2.1–2.17 (implement every ON section)
3. `spec\CHANNEL_B.md` — BB multi-channel ride aux only (NOT money)
4. `spec\OBS_BOOK.md` — mandatory eyes (fail card if missing)
5. `spec\HARNESS.md` + `spec\HANDOFF.md` — product, HA door, sessions (if paycheck rows conflict with REWARDS.md, **REWARDS.md wins**)
6. `LESSONS_BOOK.md` — do not repeat failed patterns
7. `INDICATOR_OBS_SETS_V6*.md` — thin market eyes only; ignore hard-penalty bindings

Leave alone:
- Box keep: `C:\Users\C2K\meta-learn-rl-model-mask-2026-08-20`
- Desktop `v5` training runs / weights (you may **read** v5 code + data wiring; do not train inside v5)
Never `--fresh` unless I type that word.

## Price data (use this — do not scrape random web)
Primary M1 OHLC (same tape v5 uses):
`C:\Users\C2K\Desktop\v4\data\US30_M1_202007231046_202605262359.csv`

v5 already points at `../v4/data` (US30 M1). Reuse that loader pattern / tape-cache idea from `C:\Users\C2K\Desktop\v5\setuprl\` (read-only reference). Build V6 code under `C:\Users\C2K\Desktop\V6` (e.g. `v6rl\` or `setuprl\`), do not mutate v5.

MT5 is also installed under MetaQuotes Terminal on this PC. Prefer the CSV tape above for train/eval. If you need a refresh path, document an MT5→CSV export into `v4\data` or `V6\data` — do not block card 1 on live MT5.

Symbol card 1: **US30** only. Sessions: London 08:00–17:00 Europe/London **or** NY 09:30–16:00 America/New_York (same as v5/HANDOFF).

## Product
- One frozen policy. Human types daily **target %** and **risk %** (runtime inputs; do not bake into weights).
- Account base for $ math: **$10k** unless REWARDS/HANDOFF say otherwise — log the choice.
- Score / keep order: **trade WR → day hits → mean $**. Hits up + WR down = stamp → revert.
- Card-1 keep vs same hidden-day **untrained** baseline: WR not down AND day HIT count **up** AND BLEW not up AND mean $ not ~20%+ worse.
- Card 1 = `--hide-days 20` smoke. Card 2 later = `--hide-days 100`. Do **not** claim 100-day Done on card 1.
- HIT/BLEW → flatten-all stop day (rails). Unprintable geometry = SKIP (not scored as fail).

## Entry door (from HANDOFF — fixed lock, not reward)
Nothing else may open. Long/short mirrors:
- Buy: 15m HA ≥2 green, 5m HA ≥2 green, 1m HA ≥1 red
- Sell: 15m HA ≥2 red, 5m HA ≥2 red, 1m HA ≥1 green
Door printed ≠ forced enter. Inside door: wait / skip / size / hold / flatten.

## Channel A — money (REWARDS.md)
- Primary: `clip(pnl_close_after_fees / today_target_dollars, -1, +1)`. Wait bars with no close = 0.
- Large day HIT/BLEW/MISS candy **OFF**. Ticket/leftover pay **OFF** (even if old HANDOFF tables show +1.0 / ticket — ignore those for the scalar).
- Spices ON exactly as written (2.17 = fastest time-to-first-profit PB; **must close in profit**): 2.4 HA streak, 2.5 trade/day streaks, 2.6 stay-green after 5m, 2.7 clean day (requires HIT), 2.8 lot PB, 2.9 clean-hour PB, 2.10 SMA4 shift tunnel 5m, 2.11 fat ≥10% target with ≥4 closes, 2.12–2.13 CCI30 PB 15m/1h, 2.14–2.15 CCI100 PB 15m/1h, 2.16 dual CCI30+100 PB (pay only if day HIT).
- `SPICE_PER_DAY_CAP = +0.50`.
- **No Bollinger reward / penalty.**

## Channel B — thinking (CHANNEL_B.md)
- OBS: BB20 + BB200 life-of-trade cross bits + multi_channel up/down (OBS_BOOK §5). Card-1 minimum 6 bits on 5m.
- Aux head: predict `multi_channel_ride`, BCE, weight ~0.05. Not money.
- Report bit count, aux stability, WR/HIT impact on short run.

## Observation (OBS_BOOK.md)
Wire fully before train: money/book, position, streaks, HA + Williams fractals (5m/15m/30m/1h/4h/1d), SMA4 shift eyes, CCI30+CCI100 + dual PB eyes, BB channel-cross bits, door flags. Print `obs_dim` + named dump. Missing `target_pct` / `risk_pct` / `goal_frac` / `risk_frac` or BB bits → fail, do not train.

## First card deliverable
1. Scaffold package under `C:\Users\C2K\Desktop\V6` (env, PPO, tape from v4 US30 CSV, reuse v5 patterns where useful).
2. Wire REWARDS 2.1–2.17 + OBS_BOOK + CHANNEL_B aux + HA door lock.
3. Short train + hide-20 eval vs untrained baseline.
4. Report: WR / HIT / BLEW / mean $, BB bit count, aux loss behavior, keep rule pass/fail, data path used.
5. Append one plain-English line to `LESSONS_BOOK.md`.
6. Stop and report. No overnight spray. No fifth window. One change family only if you must cut scope — prefer full wire then short smoke.
