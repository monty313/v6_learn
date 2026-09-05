from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from v6rl.config import KEEP_CKPT, START_EQUITY, TrainConfig, explore_eps
from v6rl.day_book import DayBook, hit_streak, record_from_env, window_summary
from v6rl.env import BookEnv
from v6rl.eval import compare_keep, eval_hidden
from v6rl.obs import OBS_DIM, dump_obs_names
from v6rl.ppo import PPO
from v6rl.product import card1_keep, is_stamp, log_product_choices
from v6rl.tape import Tape, build_tape, write_date_split

KEEP_META = KEEP_CKPT.parent / "best.json"


def resolve_resume(cfg: TrainConfig) -> Path | None:
    live = Path(cfg.run_dir) / "ckpt.pt"
    if live.is_file() and live.stat().st_size > 0:
        return live
    if KEEP_CKPT.is_file() and KEEP_CKPT.stat().st_size > 0:
        return KEEP_CKPT
    if cfg.resume_ckpt is not None and Path(cfg.resume_ckpt).is_file():
        return Path(cfg.resume_ckpt)
    return None


def load_resume(ppo: PPO, path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"resume ckpt missing: {path}")
    blob = torch.load(path, map_location=ppo.device, weights_only=False)
    ppo.net.load_state_dict(blob["net"])
    ppo.reset_hidden()
    print(f"resume {path}", flush=True)


def write_ckpt(ppo: PPO, cfg: TrainConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"net": ppo.net.state_dict(), "cfg": cfg.__dict__, "obs_dim": OBS_DIM}, tmp)
    tmp.replace(path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise SystemExit(f"{path} missing or empty — abort")


def _promote_keep(live_ckpt: Path, window: dict, update: int, board: DayBook, reason: str) -> None:
    KEEP_CKPT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live_ckpt, KEEP_CKPT)
    board.keep_saves += 1
    payload = {
        "product": "Desktop version 6",
        "reason": reason,
        "update": update,
        "hit_streak": hit_streak(board.days),
        "window": window,
        "occupancy_not_hide100": True,
        "keep_saves": board.keep_saves,
    }
    KEEP_META.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    print(f"KEEP — {reason} → {KEEP_CKPT}  window HIT {window.get('days_hit_target')} WR {window.get('trade_win_rate')}", flush=True)


def maybe_keep(ppo: PPO, cfg: TrainConfig, board: DayBook, live_ckpt: Path, update: int) -> str:
    """Save runs/keep/ckpt.pt only on scorecard improve. Stamp = reload keep, do not overwrite it."""
    scored = [d for d in board.days if d.printable]
    if len(scored) < int(cfg.keep_min_printable):
        return "wait"
    now = window_summary(board.days, int(cfg.keep_window_days))
    if int(now.get("n_trade_closes") or 0) < 10:
        return "wait"
    if not KEEP_CKPT.is_file() or not KEEP_META.is_file():
        _promote_keep(live_ckpt, now, update, board, "first vault after 20 printable occupancy days")
        return "first"
    prev = json.loads(KEEP_META.read_text(encoding="utf-8"))
    base = prev.get("window") or {}
    if is_stamp(
        wr=float(now.get("trade_win_rate") or 0.0),
        hit=int(now.get("days_hit_target") or 0),
        base_wr=float(base.get("trade_win_rate") or 0.0),
        base_hit=int(base.get("days_hit_target") or 0),
    ):
        load_resume(ppo, KEEP_CKPT)
        write_ckpt(ppo, cfg, live_ckpt)
        print(
            f"STAMP REVERT — window HIT {now.get('days_hit_target')} vs keep {base.get('days_hit_target')} "
            f"but WR {now.get('trade_win_rate')} < {base.get('trade_win_rate')}. keep file left alone.",
            flush=True,
        )
        return "stamp"
    if card1_keep(
        wr=float(now.get("trade_win_rate") or 0.0),
        hit=int(now.get("days_hit_target") or 0),
        blew=int(now.get("days_blew_risk") or 0),
        mean_dollars=float(now.get("mean_dollars") or 0.0),
        base_wr=float(base.get("trade_win_rate") or 0.0),
        base_hit=int(base.get("days_hit_target") or 0),
        base_blew=int(base.get("days_blew_risk") or 0),
        base_mean=float(base.get("mean_dollars") or 0.0),
    ):
        _promote_keep(live_ckpt, now, update, board, "scorecard improve vs last keep (WR / HIT / BLEW / mean $)")
        return "keep"
    return "hold"


def _pick_action(mask: np.ndarray, net_action: int, eps: float, rng: np.random.Generator, greedy: bool) -> int:
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return 0
    if greedy or eps <= 0:
        return int(net_action) if mask[int(net_action)] else int(legal[0])
    if rng.random() < eps:
        return int(rng.choice(legal))
    return int(net_action) if mask[int(net_action)] else int(legal[0])


def collect_rollout(env: BookEnv, ppo: PPO, cfg: TrainConfig, rng: np.random.Generator, board: DayBook):
    n = cfg.rollout_steps
    obs_dim = env.observation_space.shape[0]
    n_act = env.action_space.n
    obs_b = np.empty((n, obs_dim), np.float32)
    mask_b = np.empty((n, n_act), np.bool_)
    act_b = np.empty(n, np.int64)
    logp_b = np.empty(n, np.float32)
    val_b = np.empty(n, np.float32)
    rew_b = np.empty(n, np.float32)
    done_b = np.empty(n, np.float32)
    aux_y = np.full(n, -1.0, np.float32)
    size_y = np.empty(n, np.float32)
    lot_y = np.full(n, -1.0, np.float32)

    if env.rollout_live and env._last_obs is not None:
        obs, info = env._last_obs, env._last_info
    else:
        obs, info = env.reset(
            seed=int(rng.integers(1e9)),
            options={
                "dates": env.dates or env.tape.train_dates,
                "date_i": env.date_i,
                "train_day_index": env.train_day_index,
                "greedy": cfg.greedy,
                "reset_equity": True,
            },
        )
        env.rollout_live = True
        env.dates = list(env.dates or env.tape.train_dates)
        ppo.reset_hidden()

    hits = 0
    days = 0
    for i in range(n):
        if info.get("announce"):
            board.print_open(env)
            info["announce"] = False
            ppo.reset_hidden()
        mask = info["mask"]
        eps = 0.0 if cfg.greedy else explore_eps(env.train_day_index)
        step = ppo.act(obs, mask, greedy=cfg.greedy)
        action = _pick_action(mask, step["action"], eps, rng, cfg.greedy)
        env.set_pending(step["lot_frac"], step["size_scale"])
        next_obs, reward, terminated, _, next_info = env.step(action)
        exec_action = int(next_info.get("exec_action", action))
        obs_b[i] = obs
        mask_b[i] = mask
        act_b[i] = exec_action
        logp_b[i] = PPO.logp_of(step["actor_logits"], exec_action)
        val_b[i] = step["value"]
        rew_b[i] = reward
        done_b[i] = float(terminated)
        aux_y[i] = float(next_info.get("aux_y", -1.0))
        size_y[i] = float(info.get("size_scale_target") or 1.0)
        cap = float(info.get("risk_cap_lot") or 0.0)
        if cap > 0:
            lot_y[i] = float(np.clip(step["lot_frac"][0], 0.0, 1.0))
        if terminated:
            days += 1
            rec = record_from_env(env)
            if rec.hit_target:
                hits += 1
            board.add(rec)
            board.print_close(rec)
            env.date_i = (env.date_i + 1) % max(1, len(env.dates))
            env.train_day_index += 1
            ppo.reset_hidden()
            obs, info = env.reset(
                options={
                    "dates": env.dates,
                    "date_i": env.date_i,
                    "train_day_index": env.train_day_index,
                    "greedy": cfg.greedy,
                    "reset_equity": True,
                }
            )
        else:
            obs, info = next_obs, next_info

    env._last_obs, env._last_info = obs, info
    last = ppo.value(obs)
    return {
        "obs": obs_b,
        "masks": mask_b,
        "actions": act_b,
        "logp": logp_b,
        "values": val_b,
        "rewards": rew_b,
        "dones": done_b,
        "aux_y": aux_y,
        "size_y": size_y,
        "lot_y": lot_y,
        "last_value": last,
        "hits": hits,
        "days": days,
    }


def train(cfg: TrainConfig | None = None, tape: Tape | None = None) -> PPO:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cfg = cfg or TrainConfig()
    if cfg.device == "auto":
        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    threads = max(1, min(8, os.cpu_count() or 4))
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    log_product_choices()
    dump_obs_names()
    print(f"obs_dim={OBS_DIM} (printed before train)", flush=True)
    rng = np.random.default_rng(cfg.seed)
    if tape is None:
        tape = build_tape(
            cfg.data_dir,
            cache_dir=cfg.cache_dir,
            hide_last_days=cfg.hide_last_days,
            start=cfg.start,
            end=cfg.end,
            rebuild=bool(cfg.rebuild_tape),
        )
        print(
            f"train days={len(tape.train_dates)} hidden={len(tape.hidden_dates)} bars={tape.n} "
            f"csv={cfg.data_dir}",
            flush=True,
        )
    overlap = set(tape.train_dates) & set(tape.hidden_dates)
    if overlap:
        raise SystemExit(f"train/hidden overlap {len(overlap)} dates — abort")
    write_date_split(Path(cfg.run_dir), tape, cfg.hide_last_days)
    dates = tape.train_dates
    if cfg.max_train_days:
        dates = dates[: cfg.max_train_days]
    env = BookEnv(tape, greedy=cfg.greedy, run_dir=cfg.run_dir)
    env.dates = dates
    env.book.equity = START_EQUITY
    ppo = PPO(cfg)
    resume = resolve_resume(cfg)
    if resume is not None:
        load_resume(ppo, resume)
    else:
        print("island birth — new V6 mind (HA door + OBS_BOOK). no runs/keep/ckpt.pt yet.", flush=True)
    if int(cfg.total_updates) <= 0 and int(cfg.until_hit_streak) <= 0:
        raise SystemExit("need --updates N or --until-hit-streak N")
    board = DayBook(Path(cfg.run_dir), n_days=len(dates))
    print(f"device={cfg.device}  updates={cfg.total_updates}  hide={cfg.hide_last_days}  torch_threads={threads}", flush=True)
    print("speed: skip door-off WAIT bars; 1m marks in a ticket, net ~every 5m on WAIT", flush=True)
    if int(cfg.until_hit_streak) > 0:
        print(
            f"run until {cfg.until_hit_streak} printable HIT days in a row — "
            f"auto-save {KEEP_CKPT} when scorecard improves (stamp = revert). "
            "occupancy streak is not hide-100 Done.",
            flush=True,
        )
    print("watch  http://127.0.0.1:8768/live.html  (V6 only — 8766 is the other bot)", flush=True)
    aux_rows = []
    u = 0
    while True:
        if int(cfg.total_updates) > 0 and u >= int(cfg.total_updates):
            break
        batch = collect_rollout(env, ppo, cfg, rng, board)
        stats = ppo.update(batch)
        aux_rows.append(stats)
        card = board.running()
        line = {
            "update": u,
            "trade_win_rate": card["trade_win_rate"],
            "days_hit_target": card["days_hit_target"],
            "mean_dollars": card["mean_dollars"],
            "n_trade_closes": card["n_trade_closes"],
            "n_trade_wins": card["n_trade_wins"],
            "hits": batch["hits"],
            "days": batch["days"],
            "hit_streak": card["hit_streak"],
            "best_hit_streak": card["best_hit_streak"],
            "keep_saves": card["keep_saves"],
            "equity": env.book.equity,
            "eps": explore_eps(env.train_day_index),
            "spice_sum": env.rew.stats.spice_sum,
            "spice_capped_days": env.rew.stats.spice_capped_days,
            "aux_n": env.rew.stats.aux_n,
            "aux_pos": env.rew.stats.aux_pos,
            **stats,
        }
        print(
            json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in line.items()}),
            flush=True,
        )
        board.last_update = line
        ckpt = Path(cfg.run_dir) / "ckpt.pt"
        write_ckpt(ppo, cfg, ckpt)
        print(f"wrote {ckpt} ({ckpt.stat().st_size} bytes)", flush=True)
        verdict = maybe_keep(ppo, cfg, board, ckpt, u)
        card = board.running()
        line["keep"] = verdict
        line["keep_saves"] = card["keep_saves"]
        line["hit_streak"] = card["hit_streak"]
        board.last_update = line
        if board.days:
            (cfg.run_dir / "live.json").write_text(
                json.dumps(
                    {
                        "product": "Desktop version 6",
                        "last": asdict(board.days[-1]),
                        "summary": card,
                        "tail": [asdict(d) for d in board.days[-30:]],
                        "update": line,
                        "keep": str(KEEP_CKPT) if KEEP_CKPT.is_file() else None,
                    },
                    default=str,
                ),
                encoding="utf-8",
            )
        streak = int(card["hit_streak"])
        print(
            f"hit_streak={streak}/{int(cfg.until_hit_streak) or '—'}  keep={verdict}  "
            f"saves={card['keep_saves']}  best_streak={card['best_hit_streak']}",
            flush=True,
        )
        if int(cfg.until_hit_streak) > 0 and streak >= int(cfg.until_hit_streak):
            print(
                f"occupancy HIT streak {streak} — stopping train. "
                "this is not hide-100 Done; run --hide-days 100 before calling the exam complete.",
                flush=True,
            )
            break
        u += 1

    if cfg.skip_eval:
        print("skip eval — watch live.html; hide-20 exam is a later command", flush=True)
        return ppo

    print("=== card-1 eval: untrained greedy then trained greedy on same hide-20 ===", flush=True)
    base = eval_hidden(TrainConfig(**{**cfg.__dict__, "run_dir": cfg.run_dir}), tape=tape, untrained=True)
    trained = eval_hidden(TrainConfig(**{**cfg.__dict__, "run_dir": cfg.run_dir}), tape=tape, untrained=False)
    verdict = compare_keep(trained, base)
    (cfg.run_dir / "card1_report.json").write_text(
        json.dumps(
            {
                "trained": trained,
                "untrained": base,
                "keep": verdict,
                "aux": aux_rows,
                "spice": {
                    "sum": env.rew.stats.spice_sum,
                    "capped_days": env.rew.stats.spice_capped_days,
                    "aux_n": env.rew.stats.aux_n,
                    "aux_pos": env.rew.stats.aux_pos,
                },
                "hide_days": cfg.hide_last_days,
                "not_100_day_done": True,
            },
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )
    if verdict["keep"]:
        KEEP_CKPT.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cfg.run_dir / "ckpt.pt", KEEP_CKPT)
        print(f"KEEP — wrote {KEEP_CKPT}", flush=True)
    else:
        print("NO KEEP — card-1 vs untrained failed the keep rule. not promoted.", flush=True)
    return ppo
