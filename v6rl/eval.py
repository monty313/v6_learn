from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from v6rl.config import KEEP_CKPT, START_EQUITY, TrainConfig
from v6rl.day_book import DayBook, record_from_env
from v6rl.env import BookEnv
from v6rl.obs import dump_obs_names
from v6rl.ppo import PPO
from v6rl.product import card1_keep, harness_complete, log_product_choices
from v6rl.tape import Tape, build_tape, load_hidden_dates


def _day_seed(base: int, date_i: int) -> int:
    return int(base) * 1_000_003 + int(date_i) + 17


def walk_dates(
    cfg: TrainConfig,
    tape: Tape,
    dates: list,
    ppo: PPO,
    run_dir: Path,
    *,
    greedy: bool = True,
) -> dict:
    env = BookEnv(tape, greedy=greedy, run_dir=run_dir)
    env.dates = dates
    env.book.equity = START_EQUITY
    board = DayBook(run_dir, n_days=len(dates))
    for i, _date in enumerate(dates):
        obs, info = env.reset(
            seed=_day_seed(cfg.seed, i),
            options={
                "dates": dates,
                "date_i": i,
                "train_day_index": 10**9,
                "greedy": greedy,
                "reset_equity": True,
            },
        )
        ppo.reset_hidden()
        terminated = False
        steps = 0
        while not terminated and steps < 20000:
            mask = info["mask"]
            step = ppo.act(obs, mask, greedy=greedy)
            env.set_pending(step["lot_frac"], step["size_scale"])
            action = int(step["action"]) if mask[int(step["action"])] else int(np.flatnonzero(mask)[0])
            obs, _, terminated, _, info = env.step(action)
            steps += 1
        rec = record_from_env(env)
        board.add(rec)
        board.print_close(rec)
    return board.running()


def eval_hidden(
    cfg: TrainConfig,
    tape: Tape | None = None,
    *,
    untrained: bool = False,
    ckpt: Path | None = None,
) -> dict:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = cfg or TrainConfig()
    cfg.greedy = True
    log_product_choices()
    dump_obs_names()
    if tape is None:
        tape = build_tape(
            cfg.data_dir,
            cache_dir=cfg.cache_dir,
            hide_last_days=cfg.hide_last_days,
            start=cfg.start,
            end=cfg.end,
            rebuild=False,
        )
    overlap = set(tape.train_dates) & set(tape.hidden_dates)
    if overlap:
        raise SystemExit(f"train/hidden overlap {len(overlap)} — abort eval")
    split = cfg.run_dir / "hidden_dates.json"
    if split.is_file() and not untrained:
        saved = [str(d) for d in load_hidden_dates(cfg.run_dir)]
        now = [str(d) for d in tape.hidden_dates]
        if saved != now:
            raise SystemExit("hidden_dates.json is not the last-N CEST list this tape built — abort")
    dates = list(tape.hidden_dates)
    if not dates:
        raise SystemExit("no hidden_dates — eval has nothing to walk")
    ppo = PPO(cfg)
    tag = "untrained" if untrained else "trained"
    if not untrained:
        path = Path(ckpt) if ckpt is not None else (cfg.run_dir / "ckpt.pt")
        if not path.is_file() and KEEP_CKPT.is_file():
            path = KEEP_CKPT
        if not path.is_file():
            raise SystemExit("ckpt.pt missing — abort eval")
        blob = torch.load(path, map_location=cfg.device, weights_only=False)
        ppo.net.load_state_dict(blob["net"])
        print(f"eval load {path}", flush=True)
    else:
        print("eval untrained greedy (same hidden days, same T/R seeds)", flush=True)
    ppo.reset_hidden()
    card = walk_dates(cfg, tape, dates, ppo, cfg.run_dir / f"eval_{tag}", greedy=True)
    print(json.dumps({"eval": tag, "n_hidden": len(dates), **card}, default=str), flush=True)
    done = harness_complete(
        hit_days=int(card.get("days_hit_target") or 0),
        blew_days=int(card.get("days_blew_risk") or 0),
        printable_days=int(card.get("n_printable") or 0),
    )
    print(
        f"card-1 smoke hide={cfg.hide_last_days} — not 100-day Done. "
        f"printable={int(card.get('n_printable') or 0)} harness_complete={done}.",
        flush=True,
    )
    return card


def compare_keep(trained: dict, baseline: dict) -> dict:
    keep = card1_keep(
        wr=float(trained.get("trade_win_rate") or 0.0),
        hit=int(trained.get("days_hit_target") or 0),
        blew=int(trained.get("days_blew_risk") or 0),
        mean_dollars=float(trained.get("mean_dollars") or 0.0),
        base_wr=float(baseline.get("trade_win_rate") or 0.0),
        base_hit=int(baseline.get("days_hit_target") or 0),
        base_blew=int(baseline.get("days_blew_risk") or 0),
        base_mean=float(baseline.get("mean_dollars") or 0.0),
    )
    out = {
        "keep": keep,
        "trained": {
            "wr": trained.get("trade_win_rate"),
            "hit": trained.get("days_hit_target"),
            "blew": trained.get("days_blew_risk"),
            "mean": trained.get("mean_dollars"),
            "closes": trained.get("n_trade_closes"),
            "wins": trained.get("n_trade_wins"),
            "printable": trained.get("n_printable"),
            "spice": trained.get("mean_spice"),
        },
        "untrained": {
            "wr": baseline.get("trade_win_rate"),
            "hit": baseline.get("days_hit_target"),
            "blew": baseline.get("days_blew_risk"),
            "mean": baseline.get("mean_dollars"),
            "closes": baseline.get("n_trade_closes"),
            "wins": baseline.get("n_trade_wins"),
            "printable": baseline.get("n_printable"),
        },
        "rule": "WR not down AND HIT count UP AND BLEW not up AND mean $ not ~20%+ worse",
        "not_100_day_done": True,
    }
    print(json.dumps({"card1_keep": out}, default=str), flush=True)
    return out
