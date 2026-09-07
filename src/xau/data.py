"""Data loading for XAUUSD.

Two paths:
  load_csv      - broker/platform exports (MT5, TradingView, Dukascopy, generic)
  load_yfinance - convenience path for daily bars; needs network access

Both return a DataFrame indexed by UTC timestamp with columns
open/high/low/close/volume, sorted, de-duplicated, no NaN in OHLC.
"""

from __future__ import annotations

import pandas as pd

OHLCV = ["open", "high", "low", "close", "volume"]

# Column spellings seen across MT5, TradingView, Dukascopy and generic exports.
_ALIASES = {
    "open": {"open", "o", "<open>", "open_price"},
    "high": {"high", "h", "<high>", "high_price"},
    "low": {"low", "l", "<low>", "low_price"},
    "close": {"close", "c", "<close>", "close_price", "price", "adj close", "adj_close"},
    "volume": {"volume", "v", "<vol>", "<tickvol>", "tickvol", "tick_volume", "real_volume"},
}
_TIME_KEYS = {"time", "date", "datetime", "timestamp", "<date>", "gmt time", "local time"}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {}
    for col in df.columns:
        key = str(col).strip().lower()
        for canon, spellings in _ALIASES.items():
            if key in spellings and canon not in lookup.values():
                lookup[col] = canon
                break
    return df.rename(columns=lookup)


def _build_index(df: pd.DataFrame, time_col: str | None) -> pd.DataFrame:
    """MT5 splits the stamp across <DATE> and <TIME>; everything else uses one column."""
    cols = {str(c).strip().lower(): c for c in df.columns}

    if time_col is not None:
        stamp = pd.to_datetime(df[time_col], utc=True, errors="coerce", format="mixed")
    elif "<date>" in cols and "<time>" in cols:
        joined = df[cols["<date>"]].astype(str) + " " + df[cols["<time>"]].astype(str)
        stamp = pd.to_datetime(joined, utc=True, errors="coerce", format="mixed")
    else:
        found = next((cols[k] for k in _TIME_KEYS if k in cols), None)
        if found is None:
            raise ValueError(
                f"No timestamp column found. Columns were: {list(df.columns)}. "
                "Pass time_col= explicitly."
            )
        stamp = pd.to_datetime(df[found], utc=True, errors="coerce", format="mixed")

    df = df.assign(_stamp=stamp).dropna(subset=["_stamp"]).set_index("_stamp")
    df.index.name = "time"
    return df


def _finalise(df: pd.DataFrame) -> pd.DataFrame:
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[OHLCV].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])
    if df.empty:
        raise ValueError("No usable rows after cleaning.")
    return df


def load_csv(path: str, time_col: str | None = None, sep: str | None = None) -> pd.DataFrame:
    """Load an OHLCV export. `sep=None` lets pandas sniff comma/tab/semicolon."""
    df = pd.read_csv(path, sep=sep, engine="python")
    df = _build_index(df, time_col)
    df = _normalise_columns(df)
    missing = [c for c in ["open", "high", "low", "close"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLC columns {missing}; got {list(df.columns)}")
    return _finalise(df)


def load_yfinance(symbol: str = "XAUUSD=X", period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """Daily spot gold. 'GC=F' (COMEX futures) has cleaner history than the FX quote."""
    import yfinance as yf

    raw = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"yfinance returned nothing for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename(columns=str.lower)
    raw.index = pd.to_datetime(raw.index, utc=True)
    raw.index.name = "time"
    return _finalise(_normalise_columns(raw))


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate to a coarser timeframe, e.g. '4h', '1D', '1W'."""
    out = df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open", "high", "low", "close"])


def bars_per_year(df: pd.DataFrame) -> float:
    """Annualisation factor inferred from the median bar spacing."""
    if len(df) < 3:
        raise ValueError("Need at least 3 bars to infer a sampling frequency.")
    step = df.index.to_series().diff().dropna().median()
    seconds = step.total_seconds()
    if seconds <= 0:
        raise ValueError("Non-monotonic index.")
    if seconds >= 86_400:  # daily or slower: gold trades ~252 sessions a year
        return 365.25 * 86_400 / seconds * (252 / 365.25) if seconds < 86_400 * 3 else 365.25 * 86_400 / seconds
    # intraday: spot gold runs ~23h/day, 5 days a week
    return 252 * (23 * 3600) / seconds
