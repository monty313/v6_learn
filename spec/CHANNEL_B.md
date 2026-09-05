# CHANNEL_B — Desktop version 6 (thinking / aux only)

Channel B teaches **concepts**. It does **not** pay dollars. Channel A money stays in `REWARDS.md` only.

Work in: `C:\Users\C2K\Desktop\V6`.

---

## 1. Rule

- Aux heads may predict labels from OBS.
- Aux loss weight must stay small so it cannot dominate the PPO / money objective.
- **No** aux prediction may be copied into the scalar reward.

---

## 2. Multi-channel ride aux (ON — Mark 2026-09-05)

**Input eyes:** Bollinger channel-cross bits from `OBS_BOOK.md` §5 (`bb20_*`, `bb200_*`, `bb_multi_channel_*`).

**Label (per closed trade):**
```
multi_channel_ride = 1 if (bb_multi_channel_up == 1 or bb_multi_channel_down == 1) else 0
```
(Use the same TF scheme as OBS — if multi-TF bits exist, OR them for the label unless logged otherwise.)

**Head:** binary logit → sigmoid.

**Loss:** binary cross-entropy on `multi_channel_ride`.

**Weight:** `AUX_MULTI_CHANNEL_WEIGHT = 0.05` relative to the main money / policy loss. Do not raise without a measured keep on hide-days.

**Report after first short run:**
1. How many new OBS BB bits were added
2. Whether aux BCE trains stably (no NaN, loss not exploding, weight not dominating)
3. Impact (if any) on trade WR and day hits vs same short run without the aux head (or vs prior baseline)

---

## 3. Forbidden

- No reward bonus for multi-channel ride
- No BB crossing candy or slap in Channel A
- BB hard masks = locks only, named in harness
