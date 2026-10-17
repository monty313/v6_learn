# v6_learn — IRAC handoff (2026-10-17)

## Issue
Desktop V6: one frozen US30 day-trading PPO policy taking target% and risk% as observations.
Central question: Not hide-20 Done; L73 diagnosis: actor weight 0 on BLEW days removes the losing contrast.

## Rule
README: target 3-35%, risk 1-6%, US30, London/NY sessions; spec/. Doctrine: scalping-only; 5m CCI 30/100 burst entry; 1R=0.25% equity; max 4 concurrent; daily floor -3R; pyramid winners OK, add-to-losers banned.

## Application
2 commits, last 0273517 2026-09-05. LESSONS_BOOK.md L72/L73. First vault BLEW 9/20 printable. Data at C:\Users\C2K\Desktop\v4\data (local, not in repo). Risk 1-6% conflicts with 0.25% R.

## Conclusion
Status: in progress (card 1 smoke). Blocker: policy-loss weighting bug (L73); data local-only.
Next: 1) Restore nonzero actor weight on BLEW days. 2) Rerun hide-20 eval.
