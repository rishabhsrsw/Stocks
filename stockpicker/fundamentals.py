"""Basic fundamentals (current snapshot from Yahoo Finance) used as a quality overlay."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .universe import yahoo_symbol

log = logging.getLogger(__name__)

FIELDS = {
    "trailingPE": "pe", "priceToBook": "pb", "returnOnEquity": "roe", "debtToEquity": "debt_to_equity",
    "earningsGrowth": "earnings_growth", "revenueGrowth": "revenue_growth", "profitMargins": "profit_margin",
    "marketCap": "market_cap", "sector": "yf_sector",
}


def fetch_fundamentals(symbols: list[str]) -> pd.DataFrame:
    import yfinance as yf

    rows = []
    for s in symbols:
        try:
            info = yf.Ticker(yahoo_symbol(s)).info or {}
        except Exception as exc:
            log.warning("Fundamentals failed for %s: %s", s, exc)
            info = {}
        rows.append({"symbol": s, **{v: info.get(k) for k, v in FIELDS.items()}})
    return pd.DataFrame(rows).set_index("symbol")


def quality_score(f: pd.DataFrame) -> pd.Series:
    """0..1 percentile blend: profitable, growing, not over-levered, not absurdly priced."""
    if f.empty:
        return pd.Series(dtype=float)
    num = f.drop(columns=["yf_sector"], errors="ignore").apply(pd.to_numeric, errors="coerce")
    parts = [
        num["roe"].rank(pct=True),
        num["earnings_growth"].rank(pct=True),
        num["revenue_growth"].rank(pct=True),
        num["profit_margin"].rank(pct=True),
        1 - num["debt_to_equity"].rank(pct=True),
        1 - num["pe"].where(num["pe"] > 0).rank(pct=True),  # negative P/E = losses -> NaN -> neutral
    ]
    score = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return score.fillna(0.5).clip(0, 1).astype(float)
