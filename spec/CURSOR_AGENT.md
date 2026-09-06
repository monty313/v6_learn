# Cursor agent instructions — Desktop version 6

Read and obey:
- `spec/CURSOR_AGENT.md` (this file)
- `spec/REWARDS.md` — Channel A §§2.1–2.18 (money only; **2.18** = tiny unique-key day-end career-progress bonuses inside +0.50 spice). **CARD_CLOSES_THEN_DAY:** updates 0–49 = 2.1 closes only; 50–99 = HARNESS day paycheck (HIT +1.0 / BLEW −0.5 / MISS −0.15 / SKIP 0); 100+ = full 2.1–2.18. Actor weights HIT days extra, MISS low, BLEW critic-only. Stamp does **not** reload keep into live (winning-day memory stays). Do not copy to `runs/live` unless the book beats keep. See `spec/LEARN.md`.
- `spec/CHANNEL_B.md` — aux / thinking only (BB ride + momentum_hold + pullback_resolution)
- `spec/OBS_BOOK.md` — mandatory eyes (fractals + money + SMA + CCI + **BB channel-cross**)
- `INDICATOR_OBS_SETS_V6*.md` — thin only; ignore hard-penalty bindings
- `LESSONS_BOOK.md`

Work in `C:\Users\C2K\Desktop\V6`. Never `--fresh` unless Mark types that word. Leave Box keep and v5 `h3_keep` alone.

## Product
Typed daily target/risk. Score: trade win rate → day hits → mean $. Stamp = do not promote keep; live weights stay (winning days are muscle memory).

## Channel A (see REWARDS.md)
1. Primary: clipped close dollars / today’s target.
2. Spices 2.4–2.17 as written (incl. **2.17 fast green PB**: profit close required; M1 bars to first unrealized >0; lower beats last best).
3. **2.7 clean day requires day HIT.**
4. **SPICE_PER_DAY_CAP = +0.50.**
5. **Do not add BB / channel-cross candy or penalties.** Money file stays money.

## Channel B (see CHANNEL_B.md)
- Aux heads: `multi_channel_ride` ≤0.05 + `momentum_hold` 0.025 + `pullback_resolution` 0.025. Total ≤0.10. Not a reward.

## Keep rule (card 1)
Vs untrained hide-20: WR not down **and** HIT up **and** BLEW not up **and** mean $ not ~20%+ worse **and** trade count not >1.75×.

Card 1 = `--hide-days 20` smoke. Card 2 = `--hide-days 100`. Do not claim 100-day Done on card 1.

## Observation
Wire **OBS_BOOK.md** first (money, position, streaks, fractals, HA, SMA4, CCI30/100, **BB20/BB200 life-of-trade cross bits**, door flags). Fail if target/risk/goal_frac/risk_frac or BB bits missing.

## First card
Scaffold `v6rl`, wire rewards 2.1–2.17 + OBS_BOOK + CHANNEL_B aux + HA door lock, short train/eval, report numbers (incl. BB bit count + aux stability), append one LESSONS_BOOK line, stop.
