"""Performance and risk statistics.

Everything takes a series of *net* per-bar returns and an annualisation factor
(bars per year) so the same code works on 5-minute and daily bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(returns: pd.Series, initial: float = 1.0) -> pd.Series:
    return initial * (1.0 + returns.fillna(0.0)).cumprod()


def cagr(returns: pd.Series, ann: float) -> float:
    curve = equity_curve(returns)
    years = len(returns) / ann
    if years <= 0 or curve.iloc[-1] <= 0:
        return np.nan
    return curve.iloc[-1] ** (1.0 / years) - 1.0


def ann_vol(returns: pd.Series, ann: float) -> float:
    return float(returns.std(ddof=1) * np.sqrt(ann))


def sharpe(returns: pd.Series, ann: float, rf: float = 0.0) -> float:
    """Annualised Sharpe. rf is an annual rate, converted to per-bar."""
    excess = returns - rf / ann
    sd = excess.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return np.nan
    return float(excess.mean() / sd * np.sqrt(ann))


def sortino(returns: pd.Series, ann: float) -> float:
    downside = returns[returns < 0]
    dd = downside.std(ddof=1)
    if dd == 0 or np.isnan(dd):
        return np.nan
    return float(returns.mean() / dd * np.sqrt(ann))


def drawdown(returns: pd.Series) -> pd.Series:
    curve = equity_curve(returns)
    return curve / curve.cummax() - 1.0


def max_drawdown(returns: pd.Series) -> float:
    return float(drawdown(returns).min())


def calmar(returns: pd.Series, ann: float) -> float:
    mdd = abs(max_drawdown(returns))
    if mdd == 0:
        return np.nan
    return cagr(returns, ann) / mdd


def time_under_water(returns: pd.Series) -> int:
    """Longest run of bars spent below a previous equity high."""
    below = drawdown(returns) < 0
    longest = current = 0
    for flag in below:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return int(longest)


def trades_from_position(position: pd.Series) -> pd.DataFrame:
    """Group consecutive bars of constant sign into round-turn trades."""
    sign = np.sign(position.fillna(0.0))
    block = (sign != sign.shift()).cumsum()
    rows = []
    for _, chunk in position.groupby(block):
        s = np.sign(chunk.iloc[0])
        if s == 0:
            continue
        rows.append({"start": chunk.index[0], "end": chunk.index[-1], "side": int(s), "bars": len(chunk)})
    return pd.DataFrame(rows)


def trade_stats(returns: pd.Series, position: pd.Series) -> dict:
    trades = trades_from_position(position)
    if trades.empty:
        return {"n_trades": 0, "hit_rate": np.nan, "avg_trade": np.nan, "profit_factor": np.nan}
    pnl = []
    for row in trades.itertuples():
        window = returns.loc[row.start : row.end]
        pnl.append(float((1.0 + window.fillna(0.0)).prod() - 1.0))
    pnl = np.array(pnl)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "n_trades": int(len(pnl)),
        "hit_rate": float((pnl > 0).mean()),
        "avg_trade": float(pnl.mean()),
        "profit_factor": float(wins.sum() / abs(losses.sum())) if losses.size and losses.sum() != 0 else np.inf,
    }


def summary(returns: pd.Series, ann: float, position: pd.Series | None = None) -> pd.Series:
    returns = returns.dropna()
    stats = {
        "CAGR": cagr(returns, ann),
        "AnnVol": ann_vol(returns, ann),
        "Sharpe": sharpe(returns, ann),
        "Sortino": sortino(returns, ann),
        "MaxDD": max_drawdown(returns),
        "Calmar": calmar(returns, ann),
        "TimeUnderWater_bars": time_under_water(returns),
        "TotalReturn": float((1.0 + returns).prod() - 1.0),
        "Bars": len(returns),
    }
    if position is not None:
        pos = position.reindex(returns.index).fillna(0.0)
        stats["Exposure"] = float((pos != 0).mean())
        stats["AnnTurnover"] = float(pos.diff().abs().sum() / len(pos) * ann)
        stats.update(trade_stats(returns, pos))
    return pd.Series(stats)
