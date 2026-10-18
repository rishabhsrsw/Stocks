"""Price data download (Yahoo Finance via yfinance) with a local daily cache."""
from __future__ import annotations

import datetime as dt
import logging
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .universe import yahoo_symbol

log = logging.getLogger(__name__)


@dataclass
class Prices:
    """Wide frames indexed by date, one column per NSE symbol."""
    close: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    benchmark: pd.Series  # benchmark close

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def _download_chunk(tickers: list[str], start: str) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        tickers, start=start, auto_adjust=True, group_by="ticker",
        threads=True, progress=False,
    )


def download_prices(symbols: list[str], benchmark: str, years: int, use_cache: bool = True) -> Prices:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"prices_{dt.date.today():%Y%m%d}_{len(symbols)}.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    start = (dt.date.today() - dt.timedelta(days=int(365.25 * years))).isoformat()
    fields = {"Close": {}, "High": {}, "Low": {}, "Volume": {}}
    tickers = [yahoo_symbol(s) for s in symbols]
    for i in range(0, len(tickers), 50):
        chunk = tickers[i:i + 50]
        raw = _download_chunk(chunk, start)
        for t in chunk:
            if t not in raw.columns.get_level_values(0):
                continue
            frame = raw[t]
            if frame["Close"].notna().sum() < 300:  # need >~1y of history
                continue
            sym = t[:-3]
            for f in fields:
                fields[f][sym] = frame[f]
        log.info("Downloaded %d/%d tickers", min(i + 50, len(tickers)), len(tickers))

    bench_raw = _download_chunk([benchmark], start)
    bench = bench_raw[benchmark]["Close"] if benchmark in bench_raw.columns.get_level_values(0) else bench_raw["Close"]

    close = pd.DataFrame(fields["Close"]).sort_index()
    prices = Prices(
        close=close,
        high=pd.DataFrame(fields["High"]).reindex(close.index),
        low=pd.DataFrame(fields["Low"]).reindex(close.index),
        volume=pd.DataFrame(fields["Volume"]).reindex(close.index),
        benchmark=bench.reindex(close.index).ffill(),
    )
    missing = sorted(set(symbols) - set(close.columns))
    if missing:
        log.warning("No usable price history for %d symbols: %s", len(missing), ", ".join(missing[:20]))
    with open(cache_file, "wb") as fh:
        pickle.dump(prices, fh)
    return prices


def synthetic_prices(n_symbols: int = 60, n_days: int = 2600, seed: int = 0) -> Prices:
    """Random-walk market with a small persistent momentum effect, for offline tests/demo."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=dt.date.today(), periods=n_days)
    market = rng.normal(0.0004, 0.010, n_days)
    drift = np.zeros((n_days, n_symbols))
    regime = rng.normal(0, 0.0008, n_symbols)
    for t in range(n_days):  # slowly mean-reverting stock-specific drift -> learnable momentum
        if t % 63 == 0:
            regime = 0.7 * regime + rng.normal(0, 0.0008, n_symbols)
        drift[t] = regime
    beta = rng.uniform(0.6, 1.4, n_symbols)
    idio = rng.normal(0, 0.016, (n_days, n_symbols))
    rets = market[:, None] * beta + drift + idio
    close = 100 * np.exp(np.cumsum(rets, axis=0))
    syms = [f"SYN{i:03d}" for i in range(n_symbols)]
    close_df = pd.DataFrame(close, index=dates, columns=syms)
    spread = np.abs(rng.normal(0, 0.01, close.shape))
    vol = pd.DataFrame(rng.lognormal(13, 0.5, close.shape), index=dates, columns=syms)
    return Prices(
        close=close_df,
        high=close_df * (1 + spread),
        low=close_df * (1 - spread),
        volume=vol,
        benchmark=pd.Series(100 * np.exp(np.cumsum(market)), index=dates),
    )
