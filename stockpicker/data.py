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


def _paced_download(tickers: list[str], chunk: int = 20, pause: float = 1.5, retry_after: float = 60,
                    deadline: float | None = None, **kw) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Download in small paced batches, then retry once whatever came back empty.

    Yahoo rate-limits bulk downloads and reports it per ticker as empty data, so a ticker
    that fails twice is treated as unavailable (e.g. a small BSE stock Yahoo doesn't carry).
    Stops at `deadline` (a time.monotonic() value); tickers not tried by then are in neither result.
    Returns (data per ticker, tickers that failed)."""
    got: dict[str, pd.DataFrame] = {}
    todo, failed = tickers, []
    for attempt in range(2):
        failed = []
        for i in range(0, len(todo), chunk):
            if deadline is not None and time.monotonic() > deadline:
                log.info("Time budget used up: stopping with %d tickers left for the next run", len(todo) - i)
                break
            part = todo[i:i + chunk]
            raw = _download_chunk(part, **kw)
            for t in part:
                if not raw.empty and t in raw.columns.get_level_values(0) and raw[t]["Close"].notna().any():
                    got[t] = raw[t]
                else:
                    failed.append(t)
            if (i // chunk) % 25 == 0:
                log.info("Prices: %d/%d tickers (pass %d)", min(i + chunk, len(todo)), len(todo), attempt + 1)
            time.sleep(pause)
        if not failed or attempt == 1 or (deadline is not None and time.monotonic() + retry_after > deadline):
            break
        log.info("Retrying %d tickers that came back empty in %.0fs", len(failed), retry_after)
        time.sleep(retry_after)
        todo = failed
    log.info("Prices for %d/%d tickers", len(got), len(tickers))
    return got, failed


def download_prices(symbols: list[str], benchmark: str, years: int, use_cache: bool = True,
                    tickers: dict[str, str] | None = None, min_rows: int = 300, suffix: str = ".NS") -> Prices:
    """Daily OHLCV for `symbols`. `tickers` maps symbol -> Yahoo ticker (default SYMBOL + `suffix`,
    i.e. SYMBOL.NS for India, the bare symbol for the US); symbols with fewer than `min_rows` days of
    data are dropped."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    bench_tag = "".join(ch for ch in benchmark if ch.isalnum())  # keeps one market's cache from serving another's
    cache_file = CACHE_DIR / f"prices_{bench_tag}_{dt.date.today():%Y%m%d}_{len(symbols)}_{years}y.pkl"
    if use_cache and cache_file.exists():
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)

    start = (dt.date.today() - dt.timedelta(days=int(365.25 * years))).isoformat()
    fields = {"Close": {}, "High": {}, "Low": {}, "Volume": {}}
    to_sym = {(tickers or {}).get(s, yahoo_symbol(s, suffix)): s for s in symbols}
    for t, frame in _paced_download(list(to_sym), start=start)[0].items():
        if frame["Close"].notna().sum() >= min_rows:
            for f in fields:
                fields[f][to_sym[t]] = frame[f]

    bench_raw = _download_chunk([benchmark], start=start)
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

