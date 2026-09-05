# Desktop version 6

One frozen day-trading policy. A human types today’s **target %** (3–35) and **risk %** (1–6) of a **$10,000** account. Those numbers are observations, not weights.

US30 only. Session: London 08:00–17:00 Europe/London **or** NY cash 09:30–16:00 America/New_York.

## Card 1 smoke (not 100-day Done)

```
python -m v6rl train --updates 5 --hide-days 20
```

Prints `obs_dim` and the named OBS dump **before** train. Evaluates greedy hide-20 vs the same-day untrained greedy baseline.

## Live board (does not sit on the train path)

Train writes one small `live.json` when a day closes (and once per PPO update). The page only polls that file.

```
python -m v6rl serve --run-dir runs/live
python -m v6rl train --updates 20 --hide-days 20 --run-dir runs/live --resume runs/card1/ckpt.pt --no-eval
```

Open `http://127.0.0.1:8768/live.html` (V6 only). Port **8766** is the other bot’s Box keep board — do not use it for V6. `--no-eval` skips the hide-20 double walk so watching does not redo the exam. Tape cache is reused. Never `--rebuild-tape` unless the CSV changed.

## Price data

Primary M1 CSV:

`C:\Users\C2K\Desktop\v4\data\US30_M1_202007231046_202605262359.csv`

Tape cache (new version key, never overwrites v4/v5 pickles):

`C:\Users\C2K\Desktop\V6\data\cache\`

### Optional MT5 → CSV refresh (does not block card 1)

1. In MetaTrader 5: View → Symbols → US30 → Bars.
2. Export M1 as tab-separated `<DATE> <TIME> <OPEN> <HIGH> <LOW> <CLOSE>`.
3. Replace the file above (or point `CSV_DATA_DIR`).
4. Rebuild the V6 tape only: `python -m v6rl train --rebuild-tape --updates 5 --hide-days 20`.

## Locks

- Entry door is HA only (15m/5m same color ≥2, 1m opposite ≥1). Door printed ≠ forced enter.
- HIT / BLEW flatten-all and stop the day.
- No leftover-force, leftover-bank, leftover-cut, ADD, or W1–W4 hunt.
- Never `--from-scratch`. Resume `runs/keep/ckpt.pt` once it exists. Island birth only when no keep exists.
