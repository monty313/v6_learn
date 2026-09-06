from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from v6rl.product import (
    AUX_MOMENTUM_HOLD_WEIGHT,
    AUX_MULTI_CHANNEL_WEIGHT,
    AUX_PULLBACK_RES_WEIGHT,
    BERLIN,
    HIDDEN_DAYS_CARD1,
    LEVERAGE,
    LOT_STEP,
    MAX_LIVE_TICKETS,
    MAX_TICKETS_DAY,
    MIN_LOT,
    RISK_PCT_MAX,
    RISK_PCT_MIN,
    RISK_TICKET_FRAC,
    RISK_WALK,
    SPICE_PER_DAY_CAP,
    START_EQUITY,
    SYMBOLS,
    TARGET_BINS,
    TARGET_PCT_MAX,
    TARGET_PCT_MIN,
    TICKET_FLOOR_FRAC,
    TRAIN_BINS,
    WARMUP_DAYS,
    draw_typed_target,
    harness_complete,
    typed_risk_pct,
    typed_target_pct,
)

ROOT = Path(__file__).resolve().parents[1]
_V4_DATA = ROOT.parent / "v4" / "data"
CSV_DATA_DIR = _V4_DATA if _V4_DATA.is_dir() else (ROOT / "data")
CACHE_DIR = ROOT / "data" / "cache"
RUN_DIR = ROOT / "runs"
KEEP_CKPT = RUN_DIR / "keep" / "ckpt.pt"

CSV_FILES = {
    "US30": "US30_M1_202007231046_202605262359.csv",
}

GRU_HIDDEN = 128
N_SYMBOLS = len(SYMBOLS)
ENTROPY_FLOOR = 0.15
SIZE_FLOOR = 0.5
EXPLORE_TAPER_DAYS = 100

ACTIONS = ("WAIT", "BUY", "SELL", "CLOSE")
N_ACTIONS = 4
WAIT, BUY, SELL, CLOSE = range(N_ACTIONS)


@dataclass
class TrainConfig:
    data_dir: Path = CSV_DATA_DIR
    cache_dir: Path = CACHE_DIR
    run_dir: Path = RUN_DIR / "card1"
    seed: int = 7
    device: str = "auto"
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.08
    entropy_floor: float = ENTROPY_FLOOR
    value_coef: float = 0.5
    aux_coef: float = AUX_MULTI_CHANNEL_WEIGHT
    aux_momentum_coef: float = AUX_MOMENTUM_HOLD_WEIGHT
    aux_pullback_coef: float = AUX_PULLBACK_RES_WEIGHT
    occupancy_keep: bool = True
    lot_coef: float = 0.1
    size_coef: float = 0.1
    rollout_steps: int = 2048
    ppo_epochs: int = 4
    minibatch: int = 256
    total_updates: int = 5
    max_train_days: int | None = None
    hide_last_days: int = HIDDEN_DAYS_CARD1
    greedy: bool = False
    start: str | None = None
    end: str | None = None
    rebuild_tape: bool = False
    resume_ckpt: Path | None = None
    skip_eval: bool = False
    until_hit_streak: int = 0
    keep_window_days: int = 100
    keep_min_printable: int = 20
    primary_only_updates: int = 0
    close_only_updates: int = 0
    day_only_updates: int = 0


def explore_eps(train_day_index: int) -> float:
    if train_day_index <= 0:
        return 1.0
    return float(max(0.0, 1.0 - train_day_index / EXPLORE_TAPER_DAYS))


def size_scale_schedule(train_day_index: int) -> float:
    if train_day_index <= 0:
        return 1.0
    t = min(1.0, train_day_index / EXPLORE_TAPER_DAYS)
    return float(1.0 + t * (0.50 - 1.0))


__all__ = [
    "ACTIONS",
    "BERLIN",
    "BUY",
    "CACHE_DIR",
    "CLOSE",
    "CSV_DATA_DIR",
    "CSV_FILES",
    "ENTROPY_FLOOR",
    "EXPLORE_TAPER_DAYS",
    "GRU_HIDDEN",
    "KEEP_CKPT",
    "LEVERAGE",
    "LOT_STEP",
    "MAX_LIVE_TICKETS",
    "MAX_TICKETS_DAY",
    "MIN_LOT",
    "N_ACTIONS",
    "N_SYMBOLS",
    "RISK_PCT_MAX",
    "RISK_PCT_MIN",
    "RISK_TICKET_FRAC",
    "RISK_WALK",
    "ROOT",
    "RUN_DIR",
    "SELL",
    "SIZE_FLOOR",
    "SPICE_PER_DAY_CAP",
    "START_EQUITY",
    "SYMBOLS",
    "TARGET_BINS",
    "TARGET_PCT_MAX",
    "TARGET_PCT_MIN",
    "TICKET_FLOOR_FRAC",
    "TRAIN_BINS",
    "TrainConfig",
    "WAIT",
    "WARMUP_DAYS",
    "draw_typed_target",
    "explore_eps",
    "harness_complete",
    "size_scale_schedule",
    "typed_risk_pct",
    "typed_target_pct",
]
