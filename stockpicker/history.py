"""Price history for the whole market, built from the official NSE/BSE daily files and kept between runs.

Yahoo can't serve this: it rate-limits downloads of thousands of stocks, and doesn't carry most
BSE-only companies at all. The exchanges publish one "bhavcopy" per day with the open/high/low/
close and volume of every traded stock, in the current (UDiFF) format since July 2024. So the
store (kept in the Actions cache by CI) is filled one trading day at a time:
  * newest days first, so recent history is usable while the 2-year backfill is still running,
  * within a time budget per run; days not reached yet are fetched by the next run,
  * a day is only marked done once both exchanges' files were read (or it is clearly a holiday).
Exchange prices aren't adjusted for splits and bonuses. Stocks outside the F&O segment can't move
more than 20% in a day (price bands), so a bigger one-day jump is a corporate action, and the
earlier history is scaled to remove it (`adjust_corporate_actions`).
"""
from __future__ import annotations

import datetime as dt
import logging
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import bhavcopy
from .config import CACHE_DIR
from .data import Prices

log = logging.getLogger(__name__)

STORE_FILE = CACHE_DIR / "market_history.pkl"
FIELDS = ("close", "high", "low", "volume")
ACTION_JUMP = (0.7, 1.43)  # one-day close ratio outside this range is a split/bonus, not a trade
HOLIDAY_AFTER_DAYS = 5  # a weekday with no file this long ago was a holiday, not a late upload
UDIFF_START = dt.date(2024, 7, 8)  # first day both exchanges published in the current format


STORE_VERSION = 2  # 2: columns are ISINs, every instrument in the files is kept


@dataclass
class Store:
    """Daily close/high/low/volume, one column per ISIN (so new listings or a changed market list
    never need old days read again)."""
    frames: dict[str, pd.DataFrame] = field(default_factory=lambda: {f: pd.DataFrame() for f in FIELDS})
    done: set = field(default_factory=set)  # trading days already read from both exchanges
    version: int = STORE_VERSION

    @property
    def close(self) -> pd.DataFrame:
        return self.frames["close"]


def load(path: Path = STORE_FILE) -> Store:
    if path.exists():
        try:
            with open(path, "rb") as fh:
                store = pickle.load(fh)
            if not isinstance(store, Store) or getattr(store, "version", 1) != STORE_VERSION:
                raise ValueError("older format")
            log.info("Market history: %d instruments, %d days read (%s to %s)", store.close.shape[1], len(store.done),
                     min(store.done, default="-"), max(store.done, default="-"))
            return store
        except Exception as exc:  # a corrupt or outdated cache just means starting over
            log.warning("Starting a new market history (%s)", exc)
    return Store()


def save(store: Store, path: Path = STORE_FILE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(store, fh)
    tmp.replace(path)


def _read_day(day: dt.date) -> tuple[pd.DataFrame, bool]:
    """(rows by ISIN, whether both exchanges answered) without keeping the files in memory."""
    frames, complete = [], True
    for ex in ("NSE", "BSE"):
        df = bhavcopy.fetch.__wrapped__(ex, day)
        if df is None:
            complete = False
        else:
            frames.append(df)
    return bhavcopy.ohlcv_from_frames(frames), complete


def sync_bhavcopies(store: Store, years: int, budget_minutes: float,
                    today: dt.date | None = None, read_day=_read_day, pause: float = 0.3) -> int:
    """Read every trading day of the last `years` not read yet, newest first, within the budget.
    Returns the number of days read."""
    today = today or dt.date.today()
    start = max(today - dt.timedelta(days=int(365.25 * years) + 7), UDIFF_START)
    todo = [d for d in pd.bdate_range(start, today).date[::-1] if d not in store.done]
    if not todo:
        return 0
    log.info("Market history: %d trading days to read from the exchange files (budget %.0f min)",
             len(todo), budget_minutes)
    deadline = time.monotonic() + 60 * budget_minutes
    batch: dict[str, dict[pd.Timestamp, pd.Series]] = {f: {} for f in FIELDS}
    read = 0
    for day in todo:
        if time.monotonic() > deadline:
            log.info("Time budget used up: %d days left for the next run", len(todo) - read)
            break
        rows, complete = read_day(day)
        read += 1
        if not rows.empty:
            rows = rows[~rows.index.duplicated()]
            for f in FIELDS:
                batch[f][pd.Timestamp(day)] = rows[f]
        if complete or (today - day).days > HOLIDAY_AFTER_DAYS:
            store.done.add(day)
        time.sleep(pause)
    if any(batch["close"]):
        for f in FIELDS:
            new = pd.DataFrame(batch[f]).T.sort_index()
            old = store.frames[f]
            store.frames[f] = new if old.empty else old.combine_first(new)
        log.info("Market history: now %d instruments from %s to %s", store.close.shape[1],
                 store.close.index.min().date(), store.close.index.max().date())
    return read


def trim(store: Store, years: int, today: dt.date | None = None) -> None:
    start = pd.Timestamp((today or dt.date.today()) - dt.timedelta(days=int(365.25 * years) + 7))
    for f in FIELDS:
        store.frames[f] = store.frames[f][store.frames[f].index >= start]
    store.done = {d for d in store.done if pd.Timestamp(d) >= start}


def adjust_corporate_actions(close: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    """Factor to multiply each price by so that splits/bonuses don't show up as crashes.
    `observed` marks real (not carried-over) prices; jumps are measured between real prices."""
    prev = close.where(observed).ffill().shift(1)
    ratio = (close / prev).where(observed)
    jumps = ratio.where((ratio < ACTION_JUMP[0]) | (ratio > ACTION_JUMP[1])).fillna(1.0)
    # Each price is scaled by every jump that comes after it.
    after = jumps.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    return after


def to_prices(store: Store, isin_of: dict[str, str], benchmark: pd.Series, min_rows: int,
              max_gap: int = 10) -> Prices | None:
    """Adjusted prices on the market's trading calendar. Small stocks skip days with no trades, and a
    gap anywhere in a 200-day window would blank out the moving averages, liquidity and volatility
    features, so the last close is carried over no-trade days (up to `max_gap` sessions, so a stock
    that stopped trading still drops out) with zero volume. `isin_of` maps symbol -> ISIN; the
    result has one column per symbol."""
    pairs = {s: i for s, i in isin_of.items() if i in store.close.columns}
    sym_frames = {f: store.frames[f][list(pairs.values())].set_axis(list(pairs), axis=1) for f in FIELDS}
    raw = sym_frames["close"]
    raw = raw.loc[:, raw.notna().sum() >= min_rows]
    if raw.empty:
        return None
    lo, hi = raw.index.min(), raw.index.max()
    bench_days = benchmark.dropna().index
    idx = bench_days[(bench_days >= lo) & (bench_days <= hi)].union(raw.index[raw.index > bench_days.max()])
    observed = raw.reindex(idx).notna()
    factor = adjust_corporate_actions(raw.reindex(idx), observed)
    close = (raw.reindex(idx) * factor).ffill(limit=max_gap)
    filled = close.notna() & ~observed
    high = (sym_frames["high"][close.columns].reindex(idx) * factor).where(observed).fillna(close)
    low = (sym_frames["low"][close.columns].reindex(idx) * factor).where(observed).fillna(close)
    volume = (sym_frames["volume"][close.columns].reindex(idx) / factor).where(~filled, 0.0)
    return Prices(
        close=close, high=high, low=low, volume=volume.replace([np.inf, -np.inf], np.nan),
        benchmark=benchmark.reindex(benchmark.index.union(idx)).ffill().reindex(idx),
    )
