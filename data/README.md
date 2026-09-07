# XAUUSD data

`.gitignore` excludes `data/*.csv` except the synthetic files, so your own
history never gets committed by accident.

## Where to get bars

**Your own broker (best).** Only this data has the spread, session boundaries,
and rollover behaviour of the account you would actually trade.

- *MetaTrader 5*: `View -> Symbols -> Bars`, or from Python:
  ```python
  import MetaTrader5 as mt5, pandas as pd
  mt5.initialize()
  rates = mt5.copy_rates_range("XAUUSD", mt5.TIMEFRAME_D1, start, end)
  pd.DataFrame(rates).to_csv("data/XAUUSD_D1.csv", index=False)
  ```
- *TradingView*: chart menu -> Export chart data (paid plans only).

**Dukascopy** publishes free XAUUSD tick and bar history back to ~2003.

**yfinance**: `GC=F` (COMEX futures, cleaner history) or `XAUUSD=X` (spot FX).
Free and fine for validating logic; its spot quote has gaps and no volume.
Set `USE_YFINANCE = True` in the notebook.

## Format

`load_csv` sniffs the delimiter and accepts the usual column spellings —
MT5's `<DATE>`/`<TIME>`/`<OPEN>`, TradingView's `time`/`open`, and plain
`date,open,high,low,close,volume`. It returns a UTC-indexed frame with
`open, high, low, close, volume`, sorted and de-duplicated.

```python
from xau.data import load_csv, resample_ohlcv
prices = load_csv("data/XAUUSD_M15.csv")
h4 = resample_ohlcv(prices, "4h")
```

## Two things that quietly ruin gold backtests

- **Timezone and the daily close.** A "daily" gold bar depends on where your
  broker cuts the day (New York 17:00 is common). Shifting that cut moves every
  daily open, high, and low, and strategies that look robust often are not
  robust to it. Test your idea on a couple of different cut times before
  trusting it.
- **The Sunday reopen.** Spot gold gaps at the weekly open and spreads are wide
  for the first minutes. If your signal fires there, model the cost of that fill
  specifically — `CostModel.with_stress` is the blunt version.
