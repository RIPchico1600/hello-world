# XAUUSD strategy research

A research notebook and a small library for testing gold trading ideas honestly:
with real transaction costs, out-of-sample validation, and a correction for the
fact that you searched for the result.

## What this is not

There is no setting here that produces a risk-free path to a million. The point
of the notebook is the opposite — it is built to *reject* ideas quickly, because
almost all of them deserve rejecting. An idea that survives every section is
worth forward-testing on a demo account. That is the strongest claim this repo
makes about anything.

## Quick start

```bash
pip install -r requirements.txt
jupyter notebook notebooks/xauusd_research.ipynb
```

The notebook runs with no data file and no network — it falls back to synthetic
bars so every cell executes. **Synthetic results are plumbing checks, not
research.** Point it at real data with:

```bash
export XAU_DATA_PATH=data/XAUUSD_D1.csv   # your broker's export
```

See `data/README.md` for where to get XAUUSD history.

## Layout

```
src/xau/
  data.py        loaders (MT5 / TradingView / Dukascopy / generic CSV, yfinance), resampling
  costs.py       spread + slippage + commission, charged on turnover
  backtest.py    vectorised bar-close engine, vol targeting, ATR stop
  strategies.py  candidate signals: MA cross, Donchian, momentum, z-reversion, session breakout
  metrics.py     CAGR, Sharpe, Sortino, drawdown, Calmar, per-trade stats
  validate.py    walk-forward, parameter sweeps, permutation test, bootstrap CI, search hurdle
  risk.py        position sizing, Kelly, risk of ruin, Monte Carlo to a target
notebooks/
  xauusd_research.ipynb    the research workflow, sections 1-13
scripts/
  make_synthetic.py        generates the fallback data
  build_notebook.py        regenerates the notebook from source
tests/
```

## The execution convention

A signal computed from data up to and including the close of bar `t` is filled at
that close and earns the return of bar `t+1`. That is the single `.shift(1)` in
`Backtest.run` — remove it and you are trading on information you did not have.
`tests/test_engine.py::test_no_lookahead_position_is_lagged_signal` pins it.

Costs are charged on `|position change|`, so flipping long to short pays the
spread twice.

## The four hurdles

The notebook is organised around what actually kills strategies, cheapest test
first:

1. **Costs** (§5) — Sharpe must survive 2x your assumed spread and slippage.
2. **Search** (§6-7) — the best of N parameter sets must beat what the best of N
   *worthless* strategies would score. `validate.multiple_testing_hurdle`.
3. **Out-of-sample** (§8-9) — walk-forward Sharpe, and a permutation test against
   shuffled gold. `validate.walk_forward`, `validate.permutation_test`.
4. **Path risk** (§10-11) — a bootstrap CI on Sharpe, then a Monte Carlo of what
   the honest moments imply for an account. `risk.target_paths`.

Running the notebook on synthetic data walks all four: the in-sample sweep finds
a Sharpe that fails the search hurdle, walk-forward turns it negative, and the
permutation test cannot distinguish it from noise. That is the expected outcome
and the notebook is written to make that outcome legible rather than to hide it.

## Adding your own signal

Write a function taking an OHLCV frame and returning a series in `[-1, 1]`:

```python
def my_signal(prices: pd.DataFrame, lookback: int = 50) -> pd.Series:
    ...  # every value at bar t uses only data up to bar t's close
```

Any `.shift(-n)`, centred rolling window, or full-sample statistic (a mean over
all history, a fitted scaler) leaks the future. Section 12 of the notebook has
the template and the gauntlet to run it through.

## Tests

```bash
pytest
```

## Regenerating the notebook

`notebooks/xauusd_research.ipynb` is generated so it stays reviewable as code:

```bash
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace notebooks/xauusd_research.ipynb
```
