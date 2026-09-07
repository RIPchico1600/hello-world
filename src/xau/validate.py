"""Tools for finding out whether a result is an edge or a coincidence.

The default failure mode in strategy research is not a bad backtest, it is a
good one. Search enough parameters on one price series and something will look
excellent purely by chance. Everything here exists to price that in.
"""

from __future__ import annotations

import itertools
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .backtest import Backtest
from .metrics import sharpe

SignalFn = Callable[..., pd.Series]


# ------------------------------------------------------------------- splits

def split(prices: pd.DataFrame, train_frac: float = 0.6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological hold-out. Never split a time series at random."""
    cut = int(len(prices) * train_frac)
    return prices.iloc[:cut], prices.iloc[cut:]


def expanding_folds(
    prices: pd.DataFrame, n_splits: int = 5, min_train: float = 0.3
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Anchored walk-forward folds: training window grows, test window rolls."""
    n = len(prices)
    start = int(n * min_train)
    if start < 2 or n_splits < 1:
        raise ValueError("Not enough data for the requested folds.")
    edges = np.linspace(start, n, n_splits + 1).astype(int)
    folds = []
    for i in range(n_splits):
        train = prices.iloc[: edges[i]]
        test = prices.iloc[edges[i] : edges[i + 1]]
        if len(test) > 1:
            folds.append((train, test))
    return folds


# --------------------------------------------------------------- param sweep

def _grid(space: Mapping[str, Sequence]) -> list[dict]:
    keys = list(space)
    return [dict(zip(keys, combo)) for combo in itertools.product(*(space[k] for k in keys))]


def param_sweep(
    prices: pd.DataFrame,
    signal_fn: SignalFn,
    space: Mapping[str, Sequence],
    metric: str = "Sharpe",
    **bt_kwargs,
) -> pd.DataFrame:
    """Score every parameter combination on one dataset.

    Read the resulting surface for a broad plateau, not for its single peak.
    An isolated peak surrounded by bad neighbours is a fitting artefact: the
    parameter you would actually have chosen in advance is the middle of a
    region that works, because that region survives a regime it has not seen.
    """
    bt = Backtest(prices, **bt_kwargs)
    rows = []
    for params in _grid(space):
        try:
            sig = signal_fn(prices, **params)
        except (ValueError, ZeroDivisionError):
            continue
        stats = bt.run(sig).stats()
        rows.append({**params, **stats.to_dict()})
    out = pd.DataFrame(rows)
    return out.sort_values(metric, ascending=False).reset_index(drop=True) if metric in out else out


def sweep_surface(sweep: pd.DataFrame, x: str, y: str, metric: str = "Sharpe") -> pd.DataFrame:
    return sweep.pivot_table(index=y, columns=x, values=metric)


# ------------------------------------------------------------- walk forward

def walk_forward(
    prices: pd.DataFrame,
    signal_fn: SignalFn,
    space: Mapping[str, Sequence],
    n_splits: int = 5,
    metric: str = "Sharpe",
    min_train: float = 0.3,
    **bt_kwargs,
) -> tuple[pd.Series, pd.DataFrame]:
    """Refit parameters on each training window, trade them on the next window.

    Returns the stitched out-of-sample return series and a per-fold report.
    This number, not the in-sample sweep maximum, is your honest estimate.
    """
    oos_parts, report = [], []
    for i, (train, test) in enumerate(expanding_folds(prices, n_splits, min_train)):
        sweep = param_sweep(train, signal_fn, space, metric=metric, **bt_kwargs)
        if sweep.empty:
            continue
        keys = list(space)
        best = sweep.iloc[0]
        params = {k: best[k] for k in keys}
        params = {k: (int(v) if float(v).is_integer() else float(v)) for k, v in params.items()}

        # Warm up indicators on the tail of the training window so the test
        # window does not start with a blind spot, then keep only test bars.
        warmup = min(len(train), 500)
        joined = pd.concat([train.iloc[len(train) - warmup :], test])
        sig = signal_fn(joined, **params)
        res = Backtest(joined, **bt_kwargs).run(sig)
        oos = res.returns.loc[test.index[0] :]
        oos_parts.append(oos)

        report.append(
            {
                "fold": i,
                "train_end": train.index[-1],
                "test_start": test.index[0],
                "test_end": test.index[-1],
                **params,
                "IS_" + metric: float(best[metric]),
                "OOS_Sharpe": sharpe(oos, res.ann),
                "OOS_Return": float((1 + oos).prod() - 1),
            }
        )
    if not oos_parts:
        return pd.Series(dtype=float), pd.DataFrame()
    return pd.concat(oos_parts).sort_index(), pd.DataFrame(report)


# ---------------------------------------------------------- significance

def permute_prices(prices: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle bar-to-bar returns, keeping each bar's shape.

    Destroys the serial structure a strategy claims to exploit while leaving
    the volatility and bar geometry intact. A signal that scores as well on
    permuted gold as on real gold has found nothing.
    """
    close = prices["close"].to_numpy(dtype=float)
    rets = np.diff(np.log(close))
    shuffled = rng.permutation(rets)
    new_close = close[0] * np.exp(np.concatenate([[0.0], np.cumsum(shuffled)]))
    scale = new_close / close
    out = prices.copy()
    for col in ("open", "high", "low", "close"):
        out[col] = prices[col].to_numpy(dtype=float) * scale
    return out


def permutation_test(
    prices: pd.DataFrame,
    signal_fn: SignalFn,
    params: Mapping | None = None,
    n_permutations: int = 200,
    seed: int = 0,
    **bt_kwargs,
) -> dict:
    """Monte Carlo p-value for an observed Sharpe."""
    params = dict(params or {})
    real = Backtest(prices, **bt_kwargs).run(signal_fn(prices, **params))
    observed = sharpe(real.returns, real.ann)

    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_permutations):
        fake = permute_prices(prices, rng)
        res = Backtest(fake, **bt_kwargs).run(signal_fn(fake, **params))
        null.append(sharpe(res.returns, res.ann))
    null = np.array([x for x in null if np.isfinite(x)])
    p = float((null >= observed).sum() + 1) / (len(null) + 1)
    return {
        "observed_sharpe": float(observed),
        "null_mean": float(null.mean()),
        "null_p95": float(np.percentile(null, 95)),
        "p_value": p,
        "null": null,
    }


def block_bootstrap_sharpe(
    returns: pd.Series, ann: float, block: int = 20, n: int = 1000, seed: int = 0
) -> pd.Series:
    """Confidence interval for Sharpe under a moving-block bootstrap.

    Blocks preserve short-horizon autocorrelation, so the interval is wider -
    and more honest - than an i.i.d. resample.
    """
    rng = np.random.default_rng(seed)
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) <= block:
        raise ValueError("Series shorter than the block size.")
    n_blocks = int(np.ceil(len(values) / block))
    out = np.empty(n)
    for i in range(n):
        starts = rng.integers(0, len(values) - block, size=n_blocks)
        sample = np.concatenate([values[s : s + block] for s in starts])[: len(values)]
        sd = sample.std(ddof=1)
        out[i] = sample.mean() / sd * np.sqrt(ann) if sd > 0 else np.nan
    return pd.Series(out).dropna()


def multiple_testing_hurdle(n_trials: int, n_bars: int, ann: float) -> float:
    """Sharpe you should expect from the best of `n_trials` worthless strategies.

    Approximation of the expected maximum of n independent standard normals,
    scaled by the standard error of a Sharpe estimate. If your best sweep
    result does not clear this, you have measured your search, not an edge.
    """
    if n_trials < 2:
        return 0.0
    euler = 0.5772156649
    z = (1 - euler) * _norm_ppf(1 - 1 / n_trials) + euler * _norm_ppf(1 - 1 / (n_trials * np.e))
    se = np.sqrt(ann / n_bars)  # SE of an annualised Sharpe under the null
    return float(z * se)


def _norm_ppf(p: float) -> float:
    """Acklam's inverse normal CDF - avoids a scipy dependency."""
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = np.sqrt(-2 * np.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = np.sqrt(-2 * np.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
