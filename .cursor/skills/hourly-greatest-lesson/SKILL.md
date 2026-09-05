---
name: hourly-greatest-lesson
description: >-
  Once per hour, append the single greatest verified lesson from building
  Desktop V6 to LESSONS_BOOK.md. Use when Mark says hourly lesson, greatest
  learned lesson, /hourly-lesson, or AGENT_LOOP_TICK_hourly_lesson fires.
disable-model-invocation: true
---

# Hourly greatest lesson (Desktop V6)

## Do

1. Read `LESSONS_BOOK.md` (last 30 lines + latest `L##` number).
2. Read only what changed this hour if you can: `v6rl/`, `runs/`, `spec/`, recent cards. Prefer measured numbers over essays.
3. Pick **one** lesson: the greatest thing learned while creating this bot in the last hour (or since the last hourly line).
4. If that claim is already in the book (same mechanism, same product), **write nothing**.
5. If nothing new and verified happened this hour, **write nothing**. Do not invent locks or fake metrics.
6. If new: append the next `L##` in section 4 (after the last `L##`, before `## 5. Island`). Update the footer date. Do not rewrite old lessons.

## Lesson shape (plain English, same as the book)

```markdown
#### LNN. One-line claim
- **Worked / Did not / Mixed:** ...
- **Evidence:** full words, real numbers, file or card if you have them
- **Product:** Desktop version 6
```

## What “greatest” means here

Score order is still trade win rate → day hits → mean dollars. Prefer a lesson that protects 100-day consistency or stops stamp/spray/sit. Not a changelog. Not “we added a file.”

## Never

- Invent locks
- Call hide-20 Done
- Mix Box keep student (`meta-learn-rl-model-mask-2026-08-20`) into a V6 lesson
- Promote leftover-force, leftover-bank/cut, or W1–W4 as the fix
