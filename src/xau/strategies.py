"""Candidate signals for XAUUSD.

Every function takes an OHLCV frame and returns a signal series in [-1, 1]
aligned to the input index, using only data available at each bar's close.
They are starting points to attack, not recommendations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- indicators

def true_range(prices: pd.DataFrame) -> pd.Series:
    prev_close = prices["close"].shift(1)
    ranges = pd.concat(
        [
            prices["high"] - prices["low"],
            (prices["high"] - prev_close).abs(),
            (prices["low"] - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(prices: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(prices).ewm(alpha=1 / n, adjust=False).mean()


def realised_vol(prices: pd.DataFrame, n: int = 20) -> pd.Series:
    return prices["close"].pct_change().rolling(n).std(ddof=1)


# ---------------------------------------------------------------- strategies

def ma_crossover(prices: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """Classic trend filter. Long when the fast EMA is above the slow EMA."""
    if fast >= slow:
        raise ValueError("fast must be shorter than slow")
    close = prices["close"]
    f = close.ewm(span=fast, adjust=False).mean()
    s = close.ewm(span=slow, adjust=False).mean()
    return np.sign(f - s).fillna(0.0)


def donchian_breakout(prices: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.Series:
    """Turtle-style channel breakout: enter on an N-bar extreme, exit on the
    opposite M-bar extreme. Gold trends in long regimes, which is why this
    family is the usual first thing to test on it."""
    high, low, close = prices["high"], prices["low"], prices["close"]
    upper = high.rolling(entry).max().shift(1)
    lower = low.rolling(entry).min().shift(1)
    exit_up = high.rolling(exit).max().shift(1)
    exit_dn = low.rolling(exit).min().shift(1)

    side = np.zeros(len(close))
    c, u, l, xu, xd = (x.to_numpy(dtype=float) for x in (close, upper, lower, exit_up, exit_dn))
    state = 0.0
    for i in range(len(c)):
        if state > 0 and c[i] < xd[i]:
            state = 0.0
        elif state < 0 and c[i] > xu[i]:
            state = 0.0
        if state == 0.0:
            if np.isfinite(u[i]) and c[i] > u[i]:
                state = 1.0
            elif np.isfinite(l[i]) and c[i] < l[i]:
                state = -1.0
        side[i] = state
    return pd.Series(side, index=prices.index)


def ts_momentum(prices: pd.DataFrame, lookback: int = 120) -> pd.Series:
    """Time-series momentum: sign of the trailing return."""
    return np.sign(prices["close"].pct_change(lookback)).fillna(0.0)


def zscore_reversion(
    prices: pd.DataFrame, n: int = 20, entry: float = 2.0, exit: float = 0.5
) -> pd.Series:
    """Fade stretches away from a rolling mean. Mean reversion and trend are
    opposite bets on the same series - if both look profitable on the same
    data and timeframe, one of them is noise."""
    close = prices["close"]
    z = (close - close.rolling(n).mean()) / close.rolling(n).std(ddof=1)
    side = np.zeros(len(close))
    zz = z.to_numpy(dtype=float)
    state = 0.0
    for i in range(len(zz)):
        if not np.isfinite(zz[i]):
            side[i] = 0.0
            continue
        if state != 0.0 and abs(zz[i]) < exit:
            state = 0.0
        if state == 0.0:
            if zz[i] > entry:
                state = -1.0
            elif zz[i] < -entry:
                state = 1.0
        side[i] = state
    return pd.Series(side, index=prices.index)


def trend_filtered_reversion(
    prices: pd.DataFrame, trend: int = 200, n: int = 20, entry: float = 2.0
) -> pd.Series:
    """Only take the reversion trade in the direction of the long trend."""
    rev = zscore_reversion(prices, n=n, entry=entry)
    bias = np.sign(prices["close"] - prices["close"].rolling(trend).mean()).fillna(0.0)
    return rev.where(np.sign(rev) == bias, 0.0)


def session_breakout(
    prices: pd.DataFrame,
    range_start: str = "00:00",
    range_end: str = "07:00",
    close_at: str = "20:00",
) -> pd.Series:
    """Intraday only: build a range during the Asian session, trade the break
    of it during London/New York, flatten before the close.

    Gold's volatility is heavily clustered around the London open and the US
    session, so a session-aware signal is a different bet from a symmetric
    daily one. Needs intraday bars and a UTC index.
    """
    step = prices.index.to_series().diff().median()
    if step is pd.NaT or step >= pd.Timedelta("1D"):
        raise ValueError("session_breakout needs intraday bars")

    idx = prices.index
    day = idx.normalize()
    tod = idx.tz_convert("UTC").strftime("%H:%M")
    in_range = (tod >= range_start) & (tod < range_end)
    tradable = (tod >= range_end) & (tod < close_at)

    frame = pd.DataFrame(
        {"high": prices["high"], "low": prices["low"], "close": prices["close"], "day": day}
    )
    hi = frame["high"].where(in_range).groupby(frame["day"]).transform("max")
    lo = frame["low"].where(in_range).groupby(frame["day"]).transform("min")
    # Only the range built *before* the current bar is usable.
    hi = hi.groupby(frame["day"]).ffill()
    lo = lo.groupby(frame["day"]).ffill()

    sig = np.where(
        tradable & (frame["close"] > hi), 1.0,
        np.where(tradable & (frame["close"] < lo), -1.0, 0.0),
    )
    return pd.Series(sig, index=idx).fillna(0.0)


REGISTRY = {
    "ma_crossover": ma_crossover,
    "donchian_breakout": donchian_breakout,
    "ts_momentum": ts_momentum,
    "zscore_reversion": zscore_reversion,
    "trend_filtered_reversion": trend_filtered_reversion,
    "session_breakout": session_breakout,
}
