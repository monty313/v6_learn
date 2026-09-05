from __future__ import annotations

import argparse
from pathlib import Path

from v6rl.config import RUN_DIR, TrainConfig
from v6rl.eval import eval_hidden
from v6rl.serve import serve
from v6rl.train import train


def main() -> None:
    p = argparse.ArgumentParser(prog="v6rl")
    sub = p.add_subparsers(dest="cmd", required=True)
    tr = sub.add_parser("train", help="card-1 smoke: train then eval hide-days vs untrained")
    tr.add_argument("--updates", type=int, default=5)
    tr.add_argument("--rollout", type=int, default=2048)
    tr.add_argument("--max-days", type=int, default=None)
    tr.add_argument("--hide-days", type=int, default=20)
    tr.add_argument("--greedy", action="store_true")
    tr.add_argument("--seed", type=int, default=7)
    tr.add_argument("--since", dest="start", default=None)
    tr.add_argument("--until", dest="end", default=None)
    tr.add_argument("--run-dir", dest="run_dir", default=None)
    tr.add_argument("--device", default="auto")
    tr.add_argument("--rebuild-tape", action="store_true")
    tr.add_argument("--resume", dest="resume_ckpt", default=None)
    tr.add_argument(
        "--from-scratch",
        action="store_true",
        help="refused — V6 never blanks a keep; island birth only when no keep exists",
    )
    tr.add_argument(
        "--no-eval",
        action="store_true",
        help="train only — skip hide-20 untrained+trained eval (use this while watching live.html)",
    )
    tr.add_argument(
        "--until-hit-streak",
        type=int,
        default=0,
        help="keep training until this many printable HIT days in a row (0 = use --updates only)",
    )
    ev = sub.add_parser("eval", help="greedy walk of last-N hidden CEST days")
    ev.add_argument("--hide-days", type=int, default=20)
    ev.add_argument("--seed", type=int, default=7)
    ev.add_argument("--since", dest="start", default=None)
    ev.add_argument("--until", dest="end", default=None)
    ev.add_argument("--run-dir", dest="run_dir", default=None)
    ev.add_argument("--untrained", action="store_true")
    sv = sub.add_parser("serve", help="live.html on 8766 — reads live.json, does not train")
    sv.add_argument("--run-dir", dest="run_dir", default=None)
    sv.add_argument("--port", type=int, default=8768)
    sv.add_argument("--no-browser", action="store_true")
    args = p.parse_args()
    if getattr(args, "from_scratch", False):
        raise SystemExit("refuse --from-scratch. resume runs/keep/ckpt.pt or island-birth if none.")
    if args.cmd == "serve":
        serve(
            run_dir=Path(args.run_dir) if args.run_dir else (RUN_DIR / "live"),
            port=int(args.port),
            open_browser=not args.no_browser,
        )
        return
    if args.cmd == "eval":
        eval_hidden(
            TrainConfig(
                hide_last_days=args.hide_days,
                greedy=True,
                seed=args.seed,
                start=args.start,
                end=args.end,
                run_dir=Path(args.run_dir) if args.run_dir else TrainConfig().run_dir,
            ),
            untrained=bool(args.untrained),
        )
        return
    if args.cmd == "train":
        train(
            TrainConfig(
                total_updates=args.updates,
                rollout_steps=args.rollout,
                max_train_days=args.max_days,
                hide_last_days=args.hide_days,
                greedy=args.greedy,
                seed=args.seed,
                start=args.start,
                end=args.end,
                device=args.device,
                rebuild_tape=args.rebuild_tape,
                run_dir=Path(args.run_dir) if args.run_dir else (RUN_DIR / "live"),
                resume_ckpt=Path(args.resume_ckpt) if args.resume_ckpt else None,
                skip_eval=bool(args.no_eval),
                until_hit_streak=int(args.until_hit_streak),
            )
        )


if __name__ == "__main__":
    main()
