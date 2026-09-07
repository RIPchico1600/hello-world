"""Generate synthetic XAUUSD-shaped bars.

Purpose: let the notebook and tests run end-to-end with no data file and no
network. It is a stochastic model with trend regimes and volatility
clustering, NOT gold. Any result computed on it is meaningless as research -
it only proves the plumbing works.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def make(n: int = 3000, start_price: float = 1200.0, seed: int = 7,
         freq: str = "1D") -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # Volatility clustering via a slow-moving log-vol process.
    log_vol = np.zeros(n)
    log_vol[0] = np.log(0.009)
    for i in range(1, n):
        log_vol[i] = 0.985 * log_vol[i - 1] + 0.015 * np.log(0.009) + 0.06 * rng.normal()
    vol = np.exp(log_vol)

    # Persistent drift regimes, so trend strategies have something to find.
    drift = np.zeros(n)
    state = 0.0
    for i in range(n):
        if rng.random() < 1 / 180:
            state = rng.normal(0.0, 0.0006)
        drift[i] = state

    rets = drift + vol * rng.standard_t(df=4, size=n) / np.sqrt(2.0)
    close = start_price * np.exp(np.cumsum(rets))

    wick = vol * close * rng.uniform(0.4, 1.6, size=n)
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.0004, n))
    high = np.maximum(open_, close) + wick * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - wick * rng.uniform(0.2, 1.0, n)

    idx = pd.date_range("2013-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.integers(5_000, 60_000, n).astype(float)},
        index=idx,
    ).rename_axis("time")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--freq", default="1D")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic_xauusd_d1.csv")
    args = ap.parse_args()
    make(n=args.n, seed=args.seed, freq=args.freq).to_csv(args.out)
    print(f"wrote {args.out}")
