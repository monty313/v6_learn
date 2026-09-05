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

# Channel A 2.2 / 2.3 — rails still use grades; candy is OFF.
REWARD_DAY_HIT = 0.0
REWARD_DAY_BLEW = 0.0
REWARD_DAY_MISS = 0.0
REWARD_DAY_SKIP = 0.0
REWARD_TICKET = 0.0

SPICE_PER_DAY_CAP = 0.50
SMA_TUNNEL_PRICE = "5m_close"
BB_DEVIATION = 2.0
BB_SCHEME = "6 bits on 5m only"
BB_BIT_COUNT = 6
AUX_MULTI_CHANNEL_WEIGHT = 0.05

HIDDEN_DAYS_CARD1 = 20
WARMUP_DAYS = 20
MFE_HORIZON_MINUTES = 30


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
) -> bool:
    """WR not down AND day HIT count UP AND BLEW not up AND mean $ not ~20%+ worse."""
    return (
        float(wr) + 1e-12 >= float(base_wr)
        and int(hit) > int(base_hit)
        and int(blew) <= int(base_blew)
        and mean_not_much_worse(mean_dollars, base_mean)
    )


def is_stamp(*, wr: float, hit: int, base_wr: float, base_hit: int) -> bool:
    """Hits up and win rate down — occupancy stamp, revert, do not keep."""
    return int(hit) > int(base_hit) and float(wr) + 1e-12 < float(base_wr)


def log_product_choices() -> None:
    print(f"product=Desktop version 6", flush=True)
    print(f"account_base=${START_EQUITY:.0f}", flush=True)
    print(f"sma_tunnel_price={SMA_TUNNEL_PRICE}", flush=True)
    print(f"bb_deviation={BB_DEVIATION}", flush=True)
    print(f"bb_scheme={BB_SCHEME}", flush=True)
    print(f"spice_per_day_cap={SPICE_PER_DAY_CAP}", flush=True)
    print(f"aux_multi_channel_weight={AUX_MULTI_CHANNEL_WEIGHT}", flush=True)
