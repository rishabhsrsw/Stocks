"""Price history for the whole market, kept between runs and extended from the exchange files.

Downloading two years of history for ~4,500 stocks from Yahoo every week trips its rate limit
(a run took over 90 minutes and was cancelled). Instead the history is stored (CI keeps it in
the Actions cache) and each run only:
  * appends the days since the last run from the official NSE/BSE bhavcopies (2 files a day),
  * fetches from Yahoo the stocks it doesn't have yet, within a time budget; the rest wait
    for the next run, so the first full seed may take a couple of runs.
Bhavcopy prices aren't adjusted for splits and bonuses, so a stock whose price jumps like a
split is dropped and refetched (adjusted) from Yahoo.
"""
from __future__ import annotations

import datetime as dt
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from . import bhavcopy
from .config import CACHE_DIR
from .data import Prices, _paced_download

log = logging.getLogger(__name__)

STORE_FILE = CACHE_DIR / "market_history.pkl"
FIELDS = ("close", "high", "low", "volume")
YAHOO_FIELDS = {"close": "Close", "high": "High", "low": "Low", "volume": "Volume"}
SPLIT_JUMP = (0.55, 1.8)  # one-day close ratio outside this range looks like a split/bonus, not a move
RETRY_UNAVAILABLE_DAYS = 28  # a stock Yahoo didn't have is asked for again after this long


@dataclass
class Store:
    frames: dict[str, pd.DataFrame] = field(default_factory=lambda: {f: pd.DataFrame() for f in FIELDS})
    unavailable: dict[str, str] = field(default_factory=dict)  # symbol -> ISO date Yahoo last had nothing

    @property
    def close(self) -> pd.DataFrame:
        return self.frames["close"]

    def drop(self, symbols) -> None:
        for f in FIELDS:
            self.frames[f] = self.frames[f].drop(columns=list(symbols), errors="ignore")


def load(path: Path = STORE_FILE) -> Store:
    if path.exists():
        try:
            with open(path, "rb") as fh:
                store = pickle.load(fh)
            log.info("Market history: %d stocks up to %s", store.close.shape[1],
                     store.close.index.max().date() if len(store.close) else "-")
            return store
        except Exception as exc:  # a corrupt or outdated cache just means starting over
            log.warning("Ignoring unreadable market history (%s)", exc)
    return Store()


def save(store: Store, path: Path = STORE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(store, fh)
    tmp.replace(path)


def append_bhavcopies(store: Store, directory: pd.DataFrame, today: dt.date | None = None,
                      max_days: int = 30, fetch=bhavcopy.ohlcv_by_isin) -> int:
    """Add each trading day since the last stored date from the exchange files. Returns days added."""
    if store.close.empty:
        return 0
    today = today or dt.date.today()
    last = store.close.index.max().date()
    days = [d for d in pd.bdate_range(last + dt.timedelta(days=1), today).date][-max_days:]
    isin_of = {s: i for s, i in zip(directory["symbol"], directory["isin"]) if isinstance(i, str)}
    added = 0
    for day in days:
        rows = fetch(day)
        if rows.empty:
            continue  # holiday, or not published yet: picked up by the next run
        syms = [s for s in store.close.columns if isin_of.get(s) in rows.index]
        sub = rows.loc[[isin_of[s] for s in syms]]
        ts = pd.Timestamp(day)
        for f in FIELDS:
            new = pd.DataFrame([sub[f].to_numpy()], index=[ts], columns=syms)
            store.frames[f] = pd.concat([store.frames[f], new]).sort_index()
        added += 1
    if added:
        recent = store.close.iloc[-(added + 1):].ffill()
        ratio = (recent / recent.shift(1)).iloc[1:]
        jumped = ratio.columns[((ratio < SPLIT_JUMP[0]) | (ratio > SPLIT_JUMP[1])).any()]
        if len(jumped):
            log.info("Refetching %d stocks with split-like jumps: %s", len(jumped), ", ".join(jumped[:10]))
            store.drop(jumped)
        log.info("Market history: added %d day(s) from the exchange files, now up to %s",
                 added, store.close.index.max().date())
    return added


def fill_from_yahoo(store: Store, tickers: dict[str, str], years: int, budget_minutes: float,
                    today: dt.date | None = None, download=_paced_download) -> int:
    """Fetch the stocks in `tickers` (symbol -> Yahoo ticker) that the store lacks. Returns stocks added."""
    today = today or dt.date.today()
    cutoff = (today - dt.timedelta(days=RETRY_UNAVAILABLE_DAYS)).isoformat()
    have = set(store.close.columns)
    todo = {s: t for s, t in tickers.items() if s not in have and store.unavailable.get(s, "") < cutoff}
    if not todo:
        return 0
    log.info("Fetching history for %d new stocks from Yahoo (budget %.0f min)", len(todo), budget_minutes)
    start = (today - dt.timedelta(days=int(365.25 * years))).isoformat()
    to_sym = {t: s for s, t in todo.items()}
    got, failed = download(list(to_sym), start=start, deadline=time.monotonic() + 60 * budget_minutes)
    for t in failed:
        store.unavailable[to_sym[t]] = today.isoformat()
    if not got:
        return 0
    last = store.close.index.max() if not store.close.empty else None
    for f, yf in YAHOO_FIELDS.items():
        new = pd.DataFrame({to_sym[t]: fr[yf] for t, fr in got.items()})
        if new.index.tz is not None:
            new.index = new.index.tz_localize(None)
        new.index = new.index.normalize()
        new = new[new.index.date < today]  # today's bar may be intraday; the exchange file adds it later
        if last is not None:  # never run ahead of the other stocks, or they'd get a gap on that day
            new = new[new.index <= last]
        store.frames[f] = store.frames[f].join(new, how="outer") if not store.frames[f].empty else new
    for t in got:
        store.unavailable.pop(to_sym[t], None)
    return len(got)


def trim(store: Store, years: int, today: dt.date | None = None) -> None:
    start = pd.Timestamp((today or dt.date.today()) - dt.timedelta(days=int(365.25 * years) + 7))
    for f in FIELDS:
        store.frames[f] = store.frames[f][store.frames[f].index >= start]


def to_prices(store: Store, symbols, benchmark: pd.Series, min_rows: int, max_gap: int = 10) -> Prices | None:
    """Prices on the market's trading calendar. Small stocks skip days with no trades, and a gap
    anywhere in a 200-day window would blank out the moving averages, liquidity and volatility
    features, so the last close is carried over no-trade days (up to `max_gap` sessions, so a stock
    that stopped trading still drops out) with zero volume."""
    cols = [s for s in symbols if s in store.close.columns]
    raw = store.close[cols]
    raw = raw.loc[:, raw.notna().sum() >= min_rows]
    if raw.empty:
        return None
    lo, hi = raw.index.min(), raw.index.max()
    bench_days = benchmark.dropna().index
    idx = bench_days[(bench_days >= lo) & (bench_days <= hi)].union(raw.index[raw.index > bench_days.max()])
    observed = raw.reindex(idx).notna()
    close = raw.reindex(idx).ffill(limit=max_gap)
    filled = close.notna() & ~observed
    high = store.frames["high"][close.columns].reindex(idx).where(observed).fillna(close)
    low = store.frames["low"][close.columns].reindex(idx).where(observed).fillna(close)
    volume = store.frames["volume"][close.columns].reindex(idx).where(~filled, 0.0)
    return Prices(
        close=close, high=high, low=low, volume=volume,
        benchmark=benchmark.reindex(benchmark.index.union(idx)).ffill().reindex(idx),
    )
