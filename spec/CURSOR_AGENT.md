# Cursor agent instructions — Desktop version 6

Read and obey:
- `spec/CURSOR_AGENT.md` (this file)
- `spec/REWARDS.md` — Channel A §§2.1–2.17 (money only)
- `spec/CHANNEL_B.md` — aux / thinking only (BB multi-channel ride)
- `spec/OBS_BOOK.md` — mandatory eyes (fractals + money + SMA + CCI + **BB channel-cross**)
- `INDICATOR_OBS_SETS_V6*.md` — thin only; ignore hard-penalty bindings
- `LESSONS_BOOK.md`

Work in `C:\Users\C2K\Desktop\V6`. Never `--fresh` unless Mark types that word. Leave Box keep and v5 `h3_keep` alone.

## Product
Typed daily target/risk. Score: trade win rate → day hits → mean $. Stamp = revert.

## Channel A (see REWARDS.md)
1. Primary: clipped close dollars / today’s target.
2. Spices 2.4–2.17 as written (incl. **2.17 fast green PB**: profit close required; M1 bars to first unrealized >0; lower beats last best).
3. **2.7 clean day requires day HIT.**
4. **SPICE_PER_DAY_CAP = +0.50.**
5. **Do not add BB / channel-cross candy or penalties.** Money file stays money.

## Channel B (see CHANNEL_B.md)
- Optional aux head: predict `multi_channel_ride` from BB OBS bits.
- BCE loss, weight **~0.05** vs main loss.
- Not a reward. Report bit count, aux stability, WR/HIT impact on short run.

## Keep rule (card 1)
Vs untrained baseline on same hidden days: WR not down **and** day HIT count **up** **and** BLEW not up **and** mean $ not ~20%+ worse.

Card 1 = `--hide-days 20` smoke. Card 2 = `--hide-days 100`. Do not claim 100-day Done on card 1.

## Observation
Wire **OBS_BOOK.md** first (money, position, streaks, fractals, HA, SMA4, CCI30/100, **BB20/BB200 life-of-trade cross bits**, door flags). Fail if target/risk/goal_frac/risk_frac or BB bits missing.

## First card
Scaffold `v6rl`, wire rewards 2.1–2.17 + OBS_BOOK + CHANNEL_B aux + HA door lock, short train/eval, report numbers (incl. BB bit count + aux stability), append one LESSONS_BOOK line, stop.
