# LEARN — training schedule (plan, not a scored lesson)

This is the card Mark asked for on 2026-09-05. It is a **plan**. It is not Done. It is not a keep. Do not write “it worked” into `LESSONS_BOOK.md` until a book exists and is compared.

## Official run vs student

| Path | What it is |
|---|---|
| `runs/keep/` | The only mind you would take live. Promote **only** if keep rule passes (WR → HIT → BLEW → mean $). |
| `runs/live/` | Official occupancy student. **Do not copy this card here** unless that book beats keep **and** the prior occupancy books. |
| `runs/card_muscle/` | This card’s student (three stages + HIT-day actor weight). Watch it here. |

Prior bar to beat (keep vault): trade WR **48.9%** (308/630), HIT **3/22**, BLEW **9**, mean **+$30.74**. Never `--fresh`.

**Stamp** = last-100 HIT up and trade WR down vs keep → **do not save keep**. Live weights **stay**. Winning days are muscle memory. Do not reload keep into the live net.

## Muscle memory (actor)

After the day grades (HIT = target made, risk not spent):

| Day | Actor weight | Critic |
|---|---|---|
| HIT | **2.0** (extra) | 1.0 |
| MISS | **0.15** | 1.0 |
| SKIP | **0.15** | 1.0 |
| BLEW | **0.0** (policy silent) | 1.0 (still learns “don’t”) |

No wait lock. No leftover-force. No forced close.

## Three stages (one resume from keep)

1. **Updates 0–49 — closing trades in profit.**  
   `r_step = clip(close_$ / today’s_target_$, −1, +1)`. Wait with no close = 0. No day HIT/BLEW points. No spices. No progress.

2. **Updates 50–99 — the day.**

   | Day | Pay |
   |---|---|
   | HIT | +1.0 |
   | BLEW | −0.5 |
   | MISS | −0.15 |
   | SKIP | 0 |

   No points per close. No extras.

3. **Update 100 on — everything on.**  
   Normal Channel A: 2.1 + 2.2 (`+0.20 / −0.02 / −0.10`) + 2.4–2.18.

Rails the whole time: typed target/risk, printable, HA door, HIT/BLEW flatten. Channel B aux is not money.

```
python -m v6rl train --updates 0 --until-hit-streak 100 --hide-days 20 \
  --run-dir runs/card_muscle --resume runs/keep/ckpt.pt --no-eval \
  --close-only-updates 50 --day-only-updates 50
```

Board: `http://127.0.0.1:8768/live.html` reading `runs/card_muscle/live.json`.
