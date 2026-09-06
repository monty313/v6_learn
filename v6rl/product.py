"""Desktop version 6 product law. REWARDS.md wins on money numbers."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

HARNESS_PATH = Path(__file__).resolve().parents[1] / "spec" / "HARNESS.md"
REWARDS_PATH = Path(__file__).resolve().parents[1] / "spec" / "REWARDS.md"
OBS_BOOK_PATH = Path(__file__).resolve().parents[1] / "spec" / "OBS_BOOK.md"

BERLIN = ZoneInfo("Europe/Berlin")

SYMBOLS = ("US30",)
START_EQUITY = 10_000.0
LEVERAGE = 200.0
MIN_LOT = 0.01
LOT_STEP = 0.01

TARGET_BINS = (3.0, 10.0, 20.0, 35.0)
TRAIN_BINS = TARGET_BINS
TARGET_PCT_MIN = 3.0
TARGET_PCT_MAX = 35.0
RISK_PCT_MIN = 1.0
RISK_PCT_MAX = 6.0
RISK_WALK = (6.0, 5.0, 4.0, 3.0, 2.0, 1.0)

MAX_TICKETS_DAY = 400
MAX_LIVE_TICKETS = 400
TICKET_FLOOR_FRAC = 0.10
RISK_TICKET_FRAC = 0.5

# Channel A 2.2 — CARD_10_90_HITGATE terminal (day-level). Ticket leftover still OFF.
REWARD_DAY_HIT = 0.20
REWARD_NONWIN_BASE = -0.02
REWARD_BREACH_BASE = -0.10
REWARD_NONWIN_ESC = -0.01
REWARD_BREACH_ESC = -0.05
REWARD_DAY_BLEW = 0.0
REWARD_DAY_MISS = 0.0
REWARD_DAY_SKIP = 0.0
REWARD_TICKET = 0.0
# HARNESS.md goal-channel paycheck — used only in CARD_CLOSES_THEN_DAY phase 2.
HARNESS_DAY_HIT = 1.0
HARNESS_DAY_BLEW = -0.5
HARNESS_DAY_MISS = -0.15
HARNESS_DAY_SKIP = 0.0
PHASE_CLOSES = "closes"
PHASE_DAY = "day"
PHASE_FULL = "full"
CANDY_IMMEDIATE_FRAC = 0.10
CANDY_PENDING_FRAC = 0.90

SPICE_PER_DAY_CAP = 0.50
REWARD_PROGRESS_PAIR_A = 0.001
REWARD_PROGRESS_PAIR_B = 0.001
REWARD_PROGRESS_PAIR_C = 0.001
REWARD_PROGRESS_PAIR_D = 0.001
REWARD_PROGRESS_PAIR_E = 0.001
REWARD_PROGRESS_PAIR_F = 0.001
NEW_PROGRESS_BONUS_PER_DAY_CAP = 0.010
SMA_TUNNEL_PRICE = "5m_close"
BB_DEVIATION = 2.0
BB_SCHEME = "6 bits on 5m only"
BB_BIT_COUNT = 6
AUX_MULTI_CHANNEL_WEIGHT = 0.05
AUX_MOMENTUM_HOLD_WEIGHT = 0.025
AUX_PULLBACK_RES_WEIGHT = 0.025

HIDDEN_DAYS_CARD1 = 20
WARMUP_DAYS = 20
MFE_HORIZON_MINUTES = 30
TRADE_COUNT_MAX_RATIO = 1.75

# Actor step weights after the day grades (critic stays 1.0 on every day).
# HIT extra so a 13-close hit is not drowned by a 48-close miss. No wait lock.
ACTOR_W_HIT = 2.0
ACTOR_W_MISS = 0.15
ACTOR_W_BLEW = 0.0
ACTOR_W_SKIP = 0.15


def phase_for_update(update: int, close_only: int, day_only: int) -> str:
    """0 .. close_only-1 = closes; next day_only = day paycheck; then full."""
    u = int(update)
    c = max(0, int(close_only))
    d = max(0, int(day_only))
    if u < c:
        return PHASE_CLOSES
    if u < c + d:
        return PHASE_DAY
    return PHASE_FULL


def typed_target_pct(target_pct: float) -> float:
    raw = float(target_pct)
    if raw < TARGET_PCT_MIN - 1e-12 or raw > TARGET_PCT_MAX + 1e-12:
        raise ValueError(f"type target in {TARGET_PCT_MIN:g}–{TARGET_PCT_MAX:g}, got {raw}")
    return raw


def typed_risk_pct(risk_pct: float) -> float:
    raw = float(risk_pct)
    if raw < RISK_PCT_MIN - 1e-12 or raw > RISK_PCT_MAX + 1e-12:
        raise ValueError(f"type risk in {RISK_PCT_MIN:g}–{RISK_PCT_MAX:g}, got {raw}")
    return raw


def draw_typed_target(rng) -> float:
    return float(rng.choice(TRAIN_BINS))


def target_bin(target_pct: float) -> float:
    return min(TARGET_BINS, key=lambda b: abs(b - float(target_pct)))


def harness_complete(*, hit_days: int, blew_days: int, printable_days: int) -> bool:
    n = int(printable_days)
    return n > 0 and int(hit_days) == n and int(blew_days) == 0


def mean_not_much_worse(trained: float, baseline: float, frac: float = 0.20) -> bool:
    """Mean $ not ~20%+ worse than the same-day untrained baseline."""
    t = float(trained)
    b = float(baseline)
    if abs(b) < 1.0:
        return t + 1e-9 >= b - 20.0
    if b > 0:
        return t + 1e-9 >= (1.0 - frac) * b
    return t + 1e-9 >= b * (1.0 + frac)


def card1_keep(
    *,
    wr: float,
    hit: int,
    blew: int,
    mean_dollars: float,
    base_wr: float,
    base_hit: int,
    base_blew: int,
    base_mean: float,
    n_closes: int | None = None,
    base_closes: int | None = None,
) -> bool:
    """WR not down AND HIT up AND BLEW not up AND mean $ not ~20%+ worse AND no spray jump."""
    if not (
        float(wr) + 1e-12 >= float(base_wr)
        and int(hit) > int(base_hit)
        and int(blew) <= int(base_blew)
        and mean_not_much_worse(mean_dollars, base_mean)
    ):
        return False
    if n_closes is not None and base_closes is not None and int(base_closes) > 0:
        if int(n_closes) > TRADE_COUNT_MAX_RATIO * int(base_closes) + 1e-12:
            return False
    return True


def is_stamp(*, wr: float, hit: int, base_wr: float, base_hit: int) -> bool:
    """Hits up and win rate down — do not promote keep. Do not reload live."""
    return int(hit) > int(base_hit) and float(wr) + 1e-12 < float(base_wr)


def actor_day_weight(grade: str) -> float:
    """HIT extra; MISS/SKIP low; BLEW 0 on the actor (critic still sees the day)."""
    g = str(grade)
    if g == "HIT":
        return float(ACTOR_W_HIT)
    if g == "BLEW":
        return float(ACTOR_W_BLEW)
    return float(ACTOR_W_MISS)


def log_product_choices() -> None:
    print(f"product=Desktop version 6", flush=True)
    print(f"account_base=${START_EQUITY:.0f}", flush=True)
    print(f"sma_tunnel_price={SMA_TUNNEL_PRICE}", flush=True)
    print(f"bb_deviation={BB_DEVIATION}", flush=True)
    print(f"bb_scheme={BB_SCHEME}", flush=True)
    print(f"spice_per_day_cap={SPICE_PER_DAY_CAP}", flush=True)
    print(
        f"progress_pair=0.001 cap={NEW_PROGRESS_BONUS_PER_DAY_CAP} (2.18 unique-key day-end)",
        flush=True,
    )
    print(f"aux_multi_channel_weight={AUX_MULTI_CHANNEL_WEIGHT}", flush=True)
    print(f"aux_momentum_hold_weight={AUX_MOMENTUM_HOLD_WEIGHT}", flush=True)
    print(f"aux_pullback_res_weight={AUX_PULLBACK_RES_WEIGHT}", flush=True)
    print(f"reward_day_hit={REWARD_DAY_HIT} candy=10/90", flush=True)
    print(
        "channel_a=CARD_CLOSES_THEN_DAY: 2.1 closes, then harness day paycheck, then full 2.2-2.18",
        flush=True,
    )
    print(
        f"actor_day_w HIT={ACTOR_W_HIT} MISS/SKIP={ACTOR_W_MISS} BLEW={ACTOR_W_BLEW} "
        "(critic 1.0; stamp does not reload keep into live)",
        flush=True,
    )
