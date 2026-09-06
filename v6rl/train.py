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
from v6rl.product import (
    PHASE_CLOSES,
    PHASE_DAY,
    PHASE_FULL,
    actor_day_weight,
    card1_keep,
    is_stamp,
    log_product_choices,
    phase_for_update,
)
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


def load_net_flexible(net, state: dict) -> list[str]:
    own = net.state_dict()
    notes = []
    merged = dict(own)
    for k, v in state.items():
        if k not in own:
            notes.append(f"skip {k}")
            continue
        if own[k].shape == v.shape:
            merged[k] = v
            continue
        if k.endswith("enc.0.weight") or k == "enc.0.weight":
            w = own[k].clone()
            rows = min(w.size(0), v.size(0))
            cols = min(w.size(1), v.size(1))
            w[:rows, :cols] = v[:rows, :cols]
            merged[k] = w
            notes.append(f"{k} {tuple(v.shape)}→{tuple(w.shape)}")
            continue
        notes.append(f"fresh {k} {tuple(v.shape)}→{tuple(own[k].shape)}")
    net.load_state_dict(merged)
    return notes


def load_resume(ppo: PPO, path: Path, env=None) -> int:
    if not path.is_file():
        raise SystemExit(f"resume ckpt missing: {path}")
    blob = torch.load(path, map_location=ppo.device, weights_only=False)
    notes = load_net_flexible(ppo.net, blob["net"])
    ppo.reset_hidden()
    if env is not None:
        env.rew.career.consecutive_nonwin_days = int(blob.get("consecutive_nonwin_days") or 0)
        env.rew.career.consecutive_breach_days = int(blob.get("consecutive_breach_days") or 0)
        if blob.get("progress"):
            env.rew.progress.load_state(blob["progress"])
    print(f"resume {path}" + (f"  ({'; '.join(notes)})" if notes else ""), flush=True)
    return int(blob.get("updates_done") or 0)


def write_ckpt(ppo: PPO, cfg: TrainConfig, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    extra = {}
    env = getattr(ppo, "_env_ref", None)
    if env is not None:
        extra["consecutive_nonwin_days"] = env.rew.career.consecutive_nonwin_days
        extra["consecutive_breach_days"] = env.rew.career.consecutive_breach_days
        extra["progress"] = env.rew.progress.state_dict()
    extra["updates_done"] = int(getattr(ppo, "_updates_done", 0))
    torch.save({"net": ppo.net.state_dict(), "cfg": cfg.__dict__, "obs_dim": OBS_DIM, **extra}, tmp)
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
    mile = live_ckpt.parent / "milestones" / f"u{update:04d}_k{board.keep_saves}"
    mile.mkdir(parents=True, exist_ok=True)
    shutil.copy2(live_ckpt, mile / "ckpt.pt")
    (mile / "best.json").write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    print(
        f"KEEP — {reason} → {KEEP_CKPT}  milestone {mile}  "
        f"window HIT {window.get('days_hit_target')} WR {window.get('trade_win_rate')}",
        flush=True,
    )


def maybe_keep(ppo: PPO, cfg: TrainConfig, board: DayBook, live_ckpt: Path, update: int) -> str:
    """Save runs/keep/ckpt.pt only on scorecard improve. Stamp = reload keep, do not overwrite it."""
    scored = [d for d in board.days if d.printable]
    if len(scored) < int(cfg.keep_min_printable):
        return "wait"
    now = window_summary(board.days, int(cfg.keep_window_days))
    if int(now.get("n_trade_closes") or 0) < 10:
        return "wait"
    if not getattr(cfg, "occupancy_keep", True):
        return "card"
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
        print(
            f"STAMP — last-100 HIT {now.get('days_hit_target')} vs keep {base.get('days_hit_target')} "
            f"and WR {now.get('trade_win_rate')} < {base.get('trade_win_rate')}. "
            "keep file left alone. live weights kept (winning-day memory).",
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
    aux_mom_y = np.full(n, -1.0, np.float32)
    aux_pb_y = np.full(n, -1.0, np.float32)
    size_y = np.empty(n, np.float32)
    lot_y = np.full(n, -1.0, np.float32)
    actor_w = np.full(n, actor_day_weight("MISS"), np.float32)

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
    day_start = 0
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
        aux_mom_y[i] = float(next_info.get("aux_mom_y", -1.0))
        aux_pb_y[i] = float(next_info.get("aux_pb_y", -1.0))
        size_y[i] = float(info.get("size_scale_target") or 1.0)
        cap = float(info.get("risk_cap_lot") or 0.0)
        if cap > 0:
            lot_y[i] = float(np.clip(step["lot_frac"][0], 0.0, 1.0))
        if terminated:
            days += 1
            rec = record_from_env(env)
            if rec.hit_target:
                hits += 1
            actor_w[day_start : i + 1] = actor_day_weight(rec.grade)
            day_start = i + 1
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
        "aux_mom_y": aux_mom_y,
        "aux_pb_y": aux_pb_y,
        "size_y": size_y,
        "lot_y": lot_y,
        "actor_w": actor_w,
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
    ppo._env_ref = env
    start_u = 0
    if resume is not None:
        start_u = load_resume(ppo, resume, env)
        if resume.resolve() == KEEP_CKPT.resolve() and Path(cfg.run_dir).resolve() != KEEP_CKPT.parent.resolve():
            start_u = 0
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
            f"auto-save {KEEP_CKPT} when scorecard improves (stamp = do not promote; live stays). "
            "occupancy streak is not hide-100 Done.",
            flush=True,
        )
    print(f"watch  http://127.0.0.1:8768/live.html  (this run-dir {cfg.run_dir})", flush=True)
    close_n = int(getattr(cfg, "close_only_updates", 0) or 0)
    day_n = int(getattr(cfg, "day_only_updates", 0) or 0)
    if close_n == 0 and day_n == 0:
        close_n = int(getattr(cfg, "primary_only_updates", 0) or 0)
    if close_n > 0 or day_n > 0:
        bits = []
        if close_n > 0:
            bits.append(f"0..{close_n - 1} closes (2.1)")
        if day_n > 0:
            bits.append(
                f"{close_n}..{close_n + day_n - 1} day paycheck "
                "(HIT +1.0 / BLEW −0.5 / MISS −0.15 / SKIP 0)"
            )
        bits.append(f"{close_n + day_n}+ full 2.1–2.18")
        print(
            "CARD_CLOSES_THEN_DAY — " + "; ".join(bits) + ". target/risk rails stay. "
            "do not copy to runs/live unless this book beats keep.",
            flush=True,
        )
    aux_rows = []
    u = int(start_u)
    while True:
        if int(cfg.total_updates) > 0 and u >= int(cfg.total_updates):
            break
        prev_phase = str(env.rew.phase)
        now_phase = phase_for_update(u, close_n, day_n) if (close_n > 0 or day_n > 0) else PHASE_FULL
        env.rew.set_phase(now_phase)
        if prev_phase != now_phase:
            if now_phase == PHASE_DAY:
                print(
                    f"PHASE 2 — update {u}: day paycheck only (HIT +1.0 / BLEW −0.5 / MISS −0.15 / SKIP 0)",
                    flush=True,
                )
            elif now_phase == PHASE_FULL:
                print(
                    f"PHASE 3 — update {u}: full Channel A on (2.1 + 2.2 terminal + 2.4–2.17 + 10/90 + 2.18)",
                    flush=True,
                )
            elif now_phase == PHASE_CLOSES:
                print(f"PHASE 1 — update {u}: closed-trade 2.1 only", flush=True)
        ppo._updates_done = u + 1
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
            "aux_mom_n": env.rew.stats.aux_mom_n,
            "aux_mom_pos": env.rew.stats.aux_mom_pos,
            "aux_pb_n": env.rew.stats.aux_pb_n,
            "aux_pb_pos": env.rew.stats.aux_pb_pos,
            "imm_candy": env.rew.stats.imm_candy,
            "pend_HIT": env.rew.stats.pend_HIT,
            "pend_forfeit": env.rew.stats.pend_forfeit,
            "nonwin_streak": env.rew.career.consecutive_nonwin_days,
            "breach_streak": env.rew.career.consecutive_breach_days,
            "progress_sum": env.rew.stats.progress_sum,
            "progress_max_day": env.rew.stats.progress_max_day,
            "progress": env.rew.progress.report(),
            "primary_only": bool(env.rew.primary_only),
            "reward_phase": str(env.rew.phase),
            "actor_w_mean": float(np.mean(batch.get("actor_w", np.ones(1)))),
            "actor_w_hit_frac": float(np.mean(np.asarray(batch.get("actor_w", np.ones(1))) >= 1.99)),
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
                    "aux_mom_n": env.rew.stats.aux_mom_n,
                    "aux_mom_pos": env.rew.stats.aux_mom_pos,
                    "aux_pb_n": env.rew.stats.aux_pb_n,
                    "aux_pb_pos": env.rew.stats.aux_pb_pos,
                    "imm_candy": env.rew.stats.imm_candy,
                    "pend_HIT": env.rew.stats.pend_HIT,
                    "pend_forfeit": env.rew.stats.pend_forfeit,
                    "max_nonwin": env.rew.stats.max_nonwin,
                    "max_breach": env.rew.stats.max_breach,
                    "progress_sum": env.rew.stats.progress_sum,
                    "progress_max_day": env.rew.stats.progress_max_day,
                    "progress": env.rew.progress.report(),
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
