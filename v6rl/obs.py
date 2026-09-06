"""OBS_BOOK named vector. Fail the card if money eyes or BB bits are missing."""

from __future__ import annotations

from v6rl.product import BB_BIT_COUNT

HA_TFS = ("5m", "15m", "30m", "1h", "4h", "1d")
FRAC_TFS = HA_TFS

MONEY_NAMES: tuple[str, ...] = (
    "target_pct",
    "risk_pct",
    "target_dollars",
    "risk_dollars",
    "day_pnl",
    "goal_frac",
    "risk_frac",
    "leftover_to_target",
    "risk_left",
    "session_on",
    "session_time_frac",
    "consecutive_nonwin_days",
    "consecutive_breach_days",
    "immediate_candy_paid_today",
    "pending_candy_today",
    "remaining_target",
    "remaining_risk",
)

POSITION_NAMES: tuple[str, ...] = (
    "flat",
    "side",
    "lot_size",
    "lot_size_frac_of_max",
    "bars_held",
    "open_pnl",
    "open_count",
    "closes_today",
    "wins_so_far",
    "losses_so_far",
)

SPICE_NAMES: tuple[str, ...] = (
    "trade_win_streak",
    "day_win_streak",
    "clean_day_streak",
    "stay_green_streak",
    "had_trade_today",
    "held_green_after_5m",
    "is_clean_day_so_far",
    "best_clean_hour_wins",
    "hour_wins",
    "hour_losses",
    "best_lot_ever",
    "lot_is_personal_best",
    "sma4_s1_held",
    "sma4_s2_held",
    "sma4_s3_held",
    "sma4_s4_held",
    "fat_trade_flag",
    "cci30_15m",
    "cci30_1h",
    "cci30_15m_score",
    "cci30_1h_score",
    "best_cci30_15m_buy",
    "best_cci30_15m_sell",
    "best_cci30_1h_buy",
    "best_cci30_1h_sell",
    "cci_pb_15m_flag",
    "cci_pb_1h_flag",
    "cci100_15m",
    "cci100_1h",
    "cci100_15m_score",
    "cci100_1h_score",
    "best_cci100_15m_buy",
    "best_cci100_15m_sell",
    "best_cci100_1h_buy",
    "best_cci100_1h_sell",
    "cci100_pb_15m_flag",
    "cci100_pb_1h_flag",
    "dual_cci_15m_score",
    "dual_cci_1h_score",
    "best_dual_cci_15m_buy",
    "best_dual_cci_15m_sell",
    "best_dual_cci_1h_buy",
    "best_dual_cci_1h_sell",
    "pending_dual_pb_15m",
    "pending_dual_pb_1h",
    "bars_to_first_green",
    "best_bars_to_first_green_buy",
    "best_bars_to_first_green_sell",
    "fast_green_pb_flag",
)

HA_NAMES: tuple[str, ...] = tuple(
    name
    for tf in HA_TFS
    for name in (f"ha_color_{tf}", f"ha_n_{tf}")
) + ("ha_color_1m", "ha_n_1m", "door_buy", "door_sell")

FRAC_NAMES: tuple[str, ...] = tuple(
    name
    for tf in FRAC_TFS
    for name in (
        f"fractal_high_on_{tf}",
        f"fractal_low_on_{tf}",
        f"bars_since_fractal_high_{tf}",
        f"bars_since_fractal_low_{tf}",
        f"dist_to_last_fractal_high_{tf}",
        f"dist_to_last_fractal_low_{tf}",
    )
)

SMA_EYE_NAMES: tuple[str, ...] = (
    "sma4_s1_dist",
    "sma4_s2_dist",
    "sma4_s3_dist",
    "sma4_s4_dist",
)

BB_NAMES: tuple[str, ...] = (
    "bb20_cross_up",
    "bb20_cross_down",
    "bb200_cross_up",
    "bb200_cross_down",
    "bb_multi_channel_up",
    "bb_multi_channel_down",
)

THIN_NAMES: tuple[str, ...] = ("atr5_over_price",)

FEATURE_NAMES: tuple[str, ...] = (
    MONEY_NAMES
    + POSITION_NAMES
    + SPICE_NAMES
    + HA_NAMES
    + FRAC_NAMES
    + SMA_EYE_NAMES
    + BB_NAMES
    + THIN_NAMES
)
OBS_DIM = len(FEATURE_NAMES)

MANDATORY_MONEY = ("target_pct", "risk_pct", "goal_frac", "risk_frac")


def assert_obs_book() -> None:
    missing = [n for n in MANDATORY_MONEY if n not in FEATURE_NAMES]
    bb_missing = [n for n in BB_NAMES if n not in FEATURE_NAMES]
    if missing or bb_missing:
        raise SystemExit(
            f"OBS_BOOK missing money={missing} bb={bb_missing} — fail card, do not train"
        )
    if len(BB_NAMES) != BB_BIT_COUNT:
        raise SystemExit(f"BB bit count {len(BB_NAMES)} != {BB_BIT_COUNT} — fail card")


def dump_obs_names() -> None:
    assert_obs_book()
    print(f"obs_dim={OBS_DIM}", flush=True)
    print(f"bb_bit_count={len(BB_NAMES)} names={list(BB_NAMES)}", flush=True)
    print("obs_names=" + ",".join(FEATURE_NAMES), flush=True)
    print(
        "money_eyes=" + ",".join(n for n in MANDATORY_MONEY + ("target_dollars", "risk_dollars")),
        flush=True,
    )
