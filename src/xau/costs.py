"""Transaction costs for XAUUSD.

Retail gold is quoted in USD per troy ounce. The three things that eat an edge:

  spread_usd     round-turn quoted spread in USD/oz. Typical retail: 0.20-0.35
                 on a quiet London session, 0.60+ around NFP and the US open,
                 and wider still on the Sunday reopen.
  slippage_usd   extra USD/oz lost to the fill you actually get vs the close
                 you backtested on. Never assume zero on a breakout strategy.
  commission_bps commission charged on notional, in basis points per side.

Costs are charged on *turnover*: moving position from -1 to +1 crosses the
spread twice and is charged as 2.0 units of turnover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CostModel:
    spread_usd: float = 0.30
    slippage_usd: float = 0.10
    commission_bps: float = 0.0

    def per_unit_turnover(self, price: pd.Series) -> pd.Series:
        """Cost of one unit of turnover, as a fraction of notional.

        Half the spread is paid on entry and half on exit, so one unit of
        turnover (a one-way trade) carries half the round-turn spread.
        """
        price = pd.Series(np.asarray(price, dtype=float), index=price.index)
        crossing = (0.5 * self.spread_usd + self.slippage_usd) / price
        return crossing + self.commission_bps / 10_000.0

    def with_stress(self, factor: float) -> "CostModel":
        """Scale the frictions. Run every result through with_stress(2.0)."""
        return CostModel(
            spread_usd=self.spread_usd * factor,
            slippage_usd=self.slippage_usd * factor,
            commission_bps=self.commission_bps * factor,
        )


# Rough presets. Measure your own broker instead of trusting these.
RETAIL_TIGHT = CostModel(spread_usd=0.20, slippage_usd=0.05, commission_bps=0.0)
RETAIL_TYPICAL = CostModel(spread_usd=0.30, slippage_usd=0.10, commission_bps=0.0)
RETAIL_WIDE = CostModel(spread_usd=0.50, slippage_usd=0.25, commission_bps=0.0)
