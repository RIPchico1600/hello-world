"""Position sizing and what an edge actually implies for an account.

The backtest answers "does this signal have an edge". This module answers the
question that decides whether you keep the edge: how much to bet, and what the
distribution of outcomes looks like at that bet size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def units_from_risk(
    equity: float, risk_frac: float, stop_distance_usd: float, contract_size: float = 100.0
) -> float:
    """Lots to trade so that hitting the stop costs `risk_frac` of the account.

    XAUUSD standard lot = 100 oz, so a $1 move is $100 per lot.
    """
    if stop_distance_usd <= 0:
        raise ValueError("stop_distance_usd must be positive")
    return equity * risk_frac / (stop_distance_usd * contract_size)


def kelly_fraction(mean: float, variance: float) -> float:
    """Continuous Kelly. This is the *maximum growth* bet and it is far too
    aggressive to trade: it assumes your estimates of mean and variance are
    exact. Halve it at least; a quarter is common in practice."""
    if variance <= 0:
        return 0.0
    return mean / variance


def risk_of_ruin(win_rate: float, payoff: float, risk_frac: float, ruin_frac: float = 0.5,
                 n_trades: int = 1000, n_paths: int = 20_000, seed: int = 0) -> float:
    """Probability of losing `ruin_frac` of the account within n_trades.

    A strategy with a genuine positive expectancy still ruins you at the wrong
    bet size. This is the number that decides survivability, not the CAGR.
    """
    rng = np.random.default_rng(seed)
    wins = rng.random((n_paths, n_trades)) < win_rate
    step = np.where(wins, risk_frac * payoff, -risk_frac)
    curve = np.cumprod(1.0 + step, axis=1)
    return float((curve.min(axis=1) <= (1.0 - ruin_frac)).mean())


def target_paths(
    start: float,
    target: float,
    per_step_mean: float,
    per_step_std: float,
    n_steps: int = 1000,
    n_paths: int = 20_000,
    ruin_frac: float = 0.9,
    seed: int = 0,
) -> dict:
    """Monte Carlo: chance of reaching `target` before losing `ruin_frac`.

    A "step" is whatever unit the moments were measured in - one bar or one
    trade - so keep the two consistent.

    Feed it the moments your *walk-forward* produced, not your in-sample ones.
    Driving this from an in-sample fit is how a strategy that cannot beat
    shuffled data still shows a cheerful probability of reaching a million.
    The gap between the median path and the mean path is the part people skip.
    """
    rng = np.random.default_rng(seed)
    steps = rng.normal(per_step_mean, per_step_std, size=(n_paths, n_steps))
    curve = start * np.cumprod(1.0 + steps, axis=1)

    hit = (curve >= target).argmax(axis=1)
    reached = (curve >= target).any(axis=1)
    ruined = (curve <= start * (1 - ruin_frac)).any(axis=1)
    # Ruin only counts if it happens before the target is reached.
    ruin_first = ruined & (~reached | ((curve <= start * (1 - ruin_frac)).argmax(axis=1) < hit))

    return {
        "p_reach_target": float(reached.mean()),
        "p_ruin_first": float(ruin_first.mean()),
        "median_final": float(np.median(curve[:, -1])),
        "mean_final": float(curve[:, -1].mean()),
        "p5_final": float(np.percentile(curve[:, -1], 5)),
        "p95_final": float(np.percentile(curve[:, -1], 95)),
        "median_steps_to_target": float(np.median(hit[reached])) if reached.any() else np.nan,
        "curves": curve,
    }


def per_trade_moments(returns: pd.Series, position: pd.Series) -> tuple[float, float]:
    """Extract per-trade mean and std from a backtest, for `target_paths`."""
    from .metrics import trades_from_position

    trades = trades_from_position(position)
    pnl = [
        float((1 + returns.loc[r.start : r.end].fillna(0.0)).prod() - 1)
        for r in trades.itertuples()
    ]
    arr = np.array(pnl)
    if arr.size < 2:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1))
