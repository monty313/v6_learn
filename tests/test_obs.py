from v6rl.obs import (
    BB_NAMES,
    FEATURE_NAMES,
    MANDATORY_MONEY,
    OBS_DIM,
    assert_obs_book,
    dump_obs_names,
)
from v6rl.product import BB_BIT_COUNT


def test_obs_names_include_money_and_bb():
    assert_obs_book()
    for name in MANDATORY_MONEY:
        assert name in FEATURE_NAMES, name
    for name in BB_NAMES:
        assert name in FEATURE_NAMES, name
    assert len(BB_NAMES) == BB_BIT_COUNT == 6
    assert OBS_DIM == len(FEATURE_NAMES)
    assert OBS_DIM > 50
    for name in (
        "consecutive_nonwin_days",
        "consecutive_breach_days",
        "immediate_candy_paid_today",
        "pending_candy_today",
        "remaining_target",
        "remaining_risk",
    ):
        assert name in FEATURE_NAMES, name


def test_dump_prints(capsys):
    dump_obs_names()
    out = capsys.readouterr().out
    assert "obs_dim=" in out
    assert "bb_bit_count=6" in out
    assert "target_pct" in out
    assert "goal_frac" in out
    assert "bb20_cross_up" in out
    assert "bb_multi_channel_up" in out