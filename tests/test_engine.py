import numpy as np
import pandas as pd
import pytest

from xau.backtest import Backtest, apply_atr_stop
from xau.costs import CostModel
from xau.data import bars_per_year, load_csv, resample_ohlcv
from xau.metrics import max_drawdown, sharpe, summary, trades_from_position
from xau.strategies import atr, ma_crossover, zscore_reversion
from xau.validate import expanding_folds, multiple_testing_hurdle, permute_prices, split


@pytest.fixture(scope="module")
def prices():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from make_synthetic import make
    return make(n=800, seed=3)


def test_no_lookahead_position_is_lagged_signal(prices):
    bt = Backtest(prices, costs=CostModel(0, 0, 0), vol_target=None)
    sig = pd.Series(np.where(np.arange(len(prices)) % 2 == 0, 1.0, -1.0), index=prices.index)
    res = bt.run(sig)
    # Position held over bar t must be the signal formed at bar t-1.
    assert res.position.iloc[1:].equals(sig.shift(1).iloc[1:].reindex(res.position.index[1:]))


def test_perfect_foresight_is_only_available_with_a_future_signal(prices):
    """Sanity check on the lag: a signal built from the *next* bar's return
    should be highly profitable, and the same signal without the peek should
    not be. If both score the same, the engine is leaking."""
    bt = Backtest(prices, costs=CostModel(0, 0, 0), vol_target=None)
    fwd = np.sign(prices["close"].pct_change().shift(-1)).fillna(0.0)
    cheating = bt.run(fwd)          # signal formed at t-1 already knows bar t's return
    honest = bt.run(np.sign(prices["close"].pct_change()).fillna(0.0))
    assert sharpe(cheating.returns, cheating.ann) > 10
    assert sharpe(honest.returns, honest.ann) < 5


def test_costs_reduce_returns_and_scale_with_turnover(prices):
    free = Backtest(prices, costs=CostModel(0, 0, 0), vol_target=None)
    paid = Backtest(prices, costs=CostModel(spread_usd=0.5, slippage_usd=0.2), vol_target=None)
    flippy = pd.Series(np.where(np.arange(len(prices)) % 2 == 0, 1.0, -1.0), index=prices.index)
    calm = pd.Series(1.0, index=prices.index)

    assert paid.run(flippy).returns.sum() < free.run(flippy).returns.sum()
    assert paid.run(flippy).costs.sum() > paid.run(calm).costs.sum() * 50


def test_flip_costs_two_units_of_turnover(prices):
    bt = Backtest(prices, costs=CostModel(spread_usd=1.0, slippage_usd=0.0), vol_target=None)
    sig = pd.Series(1.0, index=prices.index)
    sig.iloc[400:] = -1.0
    res = bt.run(sig)
    unit = bt.costs.per_unit_turnover(prices["close"])
    charged = res.costs.loc[prices.index[401]]
    assert charged == pytest.approx(2.0 * unit.loc[prices.index[401]], rel=1e-9)


def test_buy_and_hold_matches_asset_returns(prices):
    bt = Backtest(prices, costs=CostModel(0, 0, 0), vol_target=None)
    res = bt.buy_and_hold()
    expected = prices["close"].pct_change().dropna()
    assert res.returns.iloc[1:].round(12).equals(expected.iloc[1:].round(12))


def test_vol_targeting_uses_only_trailing_data(prices):
    bt = Backtest(prices, vol_target=0.10, vol_lookback=30)
    sized = bt.size(pd.Series(1.0, index=prices.index))
    # First lookback bars have no trailing estimate, so no position is taken.
    assert (sized.iloc[: bt.vol_lookback - 1] == 0).all()
    assert sized.abs().max() <= bt.max_leverage + 1e-9


def test_signals_are_bounded_and_aligned(prices):
    for sig in (ma_crossover(prices), zscore_reversion(prices)):
        assert sig.index.equals(prices.index)
        assert sig.abs().max() <= 1.0


def test_atr_stop_flattens_and_never_increases_exposure(prices):
    raw = pd.Series(1.0, index=prices.index)
    stopped = apply_atr_stop(prices, raw, atr(prices, 14), atr_mult=1.0)
    assert stopped.abs().sum() < raw.abs().sum()
    assert set(np.unique(stopped)).issubset({-1.0, 0.0, 1.0})


def test_metrics_on_a_known_series():
    idx = pd.date_range("2020-01-01", periods=4, freq="D", tz="UTC")
    r = pd.Series([0.10, -0.20, 0.10, 0.0], index=idx)
    # equity: 1.10, 0.88, 0.968, 0.968 -> trough 0.88 against a 1.10 peak
    assert max_drawdown(r) == pytest.approx(0.88 / 1.10 - 1.0)
    assert summary(r, 252)["TotalReturn"] == pytest.approx(0.968 - 1.0)


def test_trades_are_grouped_by_sign():
    idx = pd.date_range("2020-01-01", periods=6, freq="D", tz="UTC")
    pos = pd.Series([0, 1, 1, 0, -1, -1], index=idx, dtype=float)
    trades = trades_from_position(pos)
    assert list(trades["side"]) == [1, -1]
    assert list(trades["bars"]) == [2, 2]


def test_permutation_keeps_shape_and_positive_prices(prices):
    fake = permute_prices(prices, np.random.default_rng(0))
    assert len(fake) == len(prices)
    assert (fake[["open", "high", "low", "close"]] > 0).all().all()
    assert (fake["high"] >= fake["low"]).all()
    # Same starting point, different path.
    assert fake["close"].iloc[0] == pytest.approx(prices["close"].iloc[0])
    assert not np.allclose(fake["close"], prices["close"])


def test_splits_are_chronological(prices):
    train, test = split(prices, 0.6)
    assert train.index[-1] < test.index[0]
    for tr, te in expanding_folds(prices, 4):
        assert tr.index[-1] < te.index[0]


def test_hurdle_grows_with_the_number_of_trials():
    a = multiple_testing_hurdle(10, 2000, 252)
    b = multiple_testing_hurdle(1000, 2000, 252)
    assert 0 < a < b


def test_csv_roundtrip_and_resample(prices, tmp_path):
    path = tmp_path / "px.csv"
    prices.to_csv(path)
    loaded = load_csv(str(path))
    assert loaded.shape == prices.shape
    assert bars_per_year(loaded) == pytest.approx(252, rel=0.02)
    weekly = resample_ohlcv(loaded, "1W")
    assert len(weekly) < len(loaded)
    assert weekly["high"].iloc[0] >= weekly["close"].iloc[0]


def test_loader_accepts_mt5_style_columns(tmp_path):
    path = tmp_path / "mt5.csv"
    path.write_text(
        "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
        "2024.01.02\t00:00:00\t2062.5\t2078.1\t2059.0\t2073.4\t1200\n"
        "2024.01.03\t00:00:00\t2073.4\t2080.0\t2035.2\t2041.9\t1500\n"
    )
    df = load_csv(str(path), sep="\t")
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df["close"].iloc[-1] == pytest.approx(2041.9)


def test_session_breakout_needs_intraday_bars(prices):
    with pytest.raises(ValueError, match="intraday"):
        from xau.strategies import session_breakout
        session_breakout(prices)


def test_session_breakout_only_trades_inside_its_window():
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from make_synthetic import make
    from xau.strategies import session_breakout

    intraday = make(n=3000, freq="1h", seed=11)
    sig = session_breakout(intraday, range_start="00:00", range_end="07:00", close_at="20:00")

    assert sig.index.equals(intraday.index)
    assert set(np.unique(sig)).issubset({-1.0, 0.0, 1.0})
    tod = sig.index.strftime("%H:%M")
    # Nothing may be held while the range is still being built, or after the cutoff.
    assert (sig[(tod >= "00:00") & (tod < "07:00")] == 0).all()
    assert (sig[tod >= "20:00"] == 0).all()
    assert (sig != 0).any()
