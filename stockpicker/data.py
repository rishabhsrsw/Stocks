"""Price data download (Yahoo Finance via yfinance) with a local daily cache."""
from __future__ import annotations

import datetime as dt
import logging
import pickle
import time
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


def _download_chunk(tickers: list[str], start: str | None = None, period: str | None = None,
                    retries: int = 3) -> pd.DataFrame:
    import yfinance as yf

    for attempt in range(retries):
        try:
            raw = yf.download(tickers, start=start, period=period, auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
            if raw is not None and not raw.empty:
                return raw
        except Exception as exc:  # rate limits and network blips: back off and retry
            log.warning("Download failed (%s), attempt %d/%d", exc, attempt + 1, retries)
        time.sleep(5 * (attempt + 1))
    return pd.DataFrame()


def download_prices(symbols: list[str], benchmark: str, years: int, use_cache: bool = True,
                    tickers: dict[str, str] | None = None, min_rows: int = 300) -> Prices:
    """Daily OHLCV for `symbols`. `tickers` maps symbol -> Yahoo ticker (default SYMBOL.NS);
    symbols with fewer than `min_rows` days of data are dropped."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"prices_{dt.date.today():%Y%m%d}_{len(symbols)}_{years}y.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    start = (dt.date.today() - dt.timedelta(days=int(365.25 * years))).isoformat()
    fields = {"Close": {}, "High": {}, "Low": {}, "Volume": {}}
    to_sym = {(tickers or {}).get(s, yahoo_symbol(s)): s for s in symbols}
    ticker_list = list(to_sym)
    for i in range(0, len(ticker_list), 50):
        chunk = ticker_list[i:i + 50]
        raw = _download_chunk(chunk, start)
        if raw.empty:
            continue
        for t in chunk:
            if t not in raw.columns.get_level_values(0):
                continue
            frame = raw[t]
            if frame["Close"].notna().sum() < min_rows:
                continue
            for f in fields:
                fields[f][to_sym[t]] = frame[f]
        log.info("Downloaded %d/%d tickers", min(i + 50, len(ticker_list)), len(ticker_list))

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


def download_last_close(tickers: dict[str, str]) -> dict[str, float]:
    """Latest closing price per symbol (symbol -> Yahoo ticker), from the last week of trading."""
    to_sym = {t: s for s, t in tickers.items()}
    ticker_list = list(to_sym)
    out: dict[str, float] = {}
    for i in range(0, len(ticker_list), 100):
        chunk = ticker_list[i:i + 100]
        raw = _download_chunk(chunk, period="7d")
        if raw.empty:
            continue
        for t in chunk:
            if t in raw.columns.get_level_values(0):
                c = raw[t]["Close"].dropna()
                if len(c):
                    out[to_sym[t]] = round(float(c.iloc[-1]), 2)
    log.info("Latest close for %d/%d tickers", len(out), len(ticker_list))
    return out
