import numpy as np
import pytest

from xau import risk


def test_units_scale_with_equity_and_inversely_with_stop():
    a = risk.units_from_risk(10_000, 0.01, stop_distance_usd=10.0)
    b = risk.units_from_risk(20_000, 0.01, stop_distance_usd=10.0)
    c = risk.units_from_risk(10_000, 0.01, stop_distance_usd=20.0)
    assert b == pytest.approx(2 * a)
    assert c == pytest.approx(a / 2)
    # 1% of $10k, $10 stop, 100 oz/lot -> 0.1 lots risks exactly $100.
    assert a * 100 * 10.0 == pytest.approx(100.0)

    with pytest.raises(ValueError):
        risk.units_from_risk(10_000, 0.01, stop_distance_usd=0.0)


def test_risk_of_ruin_rises_with_bet_size():
    ruins = [risk.risk_of_ruin(0.4, 2.0, f, n_trades=500, n_paths=4000) for f in (0.01, 0.05, 0.15)]
    assert ruins == sorted(ruins)
    assert ruins[-1] > ruins[0]


def test_positive_edge_still_ruins_at_a_large_enough_bet():
    # 40% hit rate at 2:1 is a genuinely profitable system.
    assert risk.risk_of_ruin(0.4, 2.0, 0.25, n_trades=500, n_paths=4000) > 0.5


def test_kelly_matches_mean_over_variance():
    assert risk.kelly_fraction(0.01, 0.04) == pytest.approx(0.25)
    assert risk.kelly_fraction(0.01, 0.0) == 0.0


def test_target_paths_reports_a_median_below_its_mean():
    out = risk.target_paths(10_000, 1_000_000, per_step_mean=0.001, per_step_std=0.02,
                            n_steps=500, n_paths=4000)
    assert 0.0 <= out["p_reach_target"] <= 1.0
    assert out["p5_final"] < out["median_final"] < out["p95_final"]
    # Multiplicative compounding is right-skewed: the mean always leads the median.
    assert out["mean_final"] > out["median_final"]
    assert out["curves"].shape == (4000, 500)


def test_a_negative_edge_rarely_reaches_the_target():
    out = risk.target_paths(10_000, 1_000_000, per_step_mean=-0.0005, per_step_std=0.01,
                            n_steps=1000, n_paths=4000)
    assert out["p_reach_target"] < 0.01
    assert out["median_final"] < 10_000
