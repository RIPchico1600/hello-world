"""Vectorised bar-close backtester.

Execution convention (the part that decides whether a backtest is honest):

  * A signal computed from data up to and including the close of bar t is
    filled at the close of bar t and earns the return of bar t+1.
    That is the single `.shift(1)` in `run()` - remove it and you are
    trading on information you did not have.
  * Costs are charged on |position change|, priced off the close.
  * Position is a float in [-max_leverage, +max_leverage]: 1.0 means fully
    invested in one unit of notional, -0.5 means half size short.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .costs import CostModel
from .data import bars_per_year
from .metrics import drawdown, equity_curve, summary


@dataclass
class BacktestResult:
    returns: pd.Series          # net per-bar returns
    gross_returns: pd.Series
    costs: pd.Series
    position: pd.Series
    ann: float
    label: str = "strategy"

    @property
    def equity(self) -> pd.Series:
        return equity_curve(self.returns)

    @property
    def drawdown(self) -> pd.Series:
        return drawdown(self.returns)

    def stats(self) -> pd.Series:
        s = summary(self.returns, self.ann, self.position)
        s["CostDrag"] = float(self.costs.sum())
        s.name = self.label
        return s

    def plot(self, ax=None, benchmark: pd.Series | None = None):
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(11, 4))
        self.equity.plot(ax=ax, label=self.label, lw=1.4)
        if benchmark is not None:
            equity_curve(benchmark).plot(ax=ax, label="buy & hold", lw=1.0, alpha=0.6)
        ax.set_yscale("log")
        ax.set_ylabel("equity (log)")
        ax.legend()
        ax.grid(alpha=0.3)
        return ax


@dataclass
class Backtest:
    """Turns a signal series into a costed equity curve.

    signal        target position per bar, in [-1, 1] before sizing.
    vol_target    if set, scale the position so trailing realised volatility
                  matches this annual figure. Gold's vol regime shifts a lot;
                  without this, position risk is 3x larger in a crisis than a
                  drift, and drawdowns are dominated by whenever vol spiked.
    """

    prices: pd.DataFrame
    costs: CostModel = field(default_factory=CostModel)
    vol_target: float | None = 0.10
    vol_lookback: int = 60
    max_leverage: float = 3.0
    ann: float | None = None

    def __post_init__(self) -> None:
        if self.ann is None:
            self.ann = bars_per_year(self.prices)
        self.asset_returns = self.prices["close"].pct_change()

    def size(self, signal: pd.Series) -> pd.Series:
        """Apply volatility targeting to a raw signal. Uses trailing data only."""
        sig = signal.reindex(self.prices.index).fillna(0.0).clip(-1.0, 1.0)
        if self.vol_target is None:
            return sig
        realised = self.asset_returns.rolling(self.vol_lookback).std(ddof=1) * np.sqrt(self.ann)
        scale = (self.vol_target / realised.replace(0.0, np.nan)).clip(upper=self.max_leverage)
        return (sig * scale).fillna(0.0)

    def run(self, signal: pd.Series, label: str = "strategy") -> BacktestResult:
        target = self.size(signal)
        position = target.shift(1).fillna(0.0)          # <- the no-lookahead lag
        gross = position * self.asset_returns
        turnover = position.diff().abs().fillna(position.abs())
        cost = turnover * self.costs.per_unit_turnover(self.prices["close"])
        net = (gross - cost).fillna(0.0)
        valid = self.asset_returns.notna()
        return BacktestResult(
            returns=net[valid],
            gross_returns=gross[valid].fillna(0.0),
            costs=cost[valid].fillna(0.0),
            position=position[valid],
            ann=self.ann,
            label=label,
        )

    def buy_and_hold(self) -> BacktestResult:
        return self.run(pd.Series(1.0, index=self.prices.index), label="buy & hold")


def apply_atr_stop(
    prices: pd.DataFrame,
    signal: pd.Series,
    atr: pd.Series,
    atr_mult: float = 2.5,
) -> pd.Series:
    """Flatten a position once price trades `atr_mult` ATRs against the entry.

    After a stop-out the position stays flat until the signal changes away
    from the stopped side - otherwise a permanently-on signal would re-enter
    on the very bar it was stopped, and the stop would do nothing.

    This is a bar-resolution approximation: the stop is checked against each
    bar's low (long) or high (short). It cannot see the path inside a bar, so
    on a gap or a spike the real fill is worse than this implies. Treat the
    result as an upper bound on how well the stop performs.
    """
    sig = signal.reindex(prices.index).fillna(0.0).to_numpy(dtype=float)
    close = prices["close"].to_numpy(dtype=float)
    high = prices["high"].to_numpy(dtype=float)
    low = prices["low"].to_numpy(dtype=float)
    band = (atr.reindex(prices.index).to_numpy(dtype=float)) * atr_mult

    out = np.zeros_like(sig)
    side = 0.0
    stop = np.nan
    blocked = 0.0          # side we were stopped out of; blocks re-entry
    for i in range(len(sig)):
        if side != 0.0:
            hit = (low[i] <= stop) if side > 0 else (high[i] >= stop)
            if hit:
                blocked, side, stop = side, 0.0, np.nan
            elif np.sign(sig[i]) != np.sign(side):
                side, stop = 0.0, np.nan
        if blocked != 0.0 and np.sign(sig[i]) != np.sign(blocked):
            blocked = 0.0  # signal moved on, re-entry allowed again
        if side == 0.0 and sig[i] != 0.0 and blocked == 0.0 and np.isfinite(band[i]):
            side = sig[i]
            stop = close[i] - band[i] if side > 0 else close[i] + band[i]
        out[i] = side
    return pd.Series(out, index=prices.index, name="position")
