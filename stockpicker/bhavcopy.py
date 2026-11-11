"""Official end-of-day files ("bhavcopies") from NSE and BSE.

One download per exchange gives the closing price of every traded stock, which is far more
reliable than thousands of Yahoo requests (Yahoo rate-limits, and lacks many small BSE stocks).
"""
from __future__ import annotations

import datetime as dt
import functools
import io
import logging
import zipfile

import pandas as pd
import requests

log = logging.getLogger(__name__)

NSE_URL = "https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{date:%Y%m%d}_F_0000.csv.zip"
BSE_URL = "https://www.bseindia.com/download/BhavCopy/Equity/BhavCopy_BSE_CM_0_0_0_{date:%Y%m%d}_F_0000.CSV"
# NSE and BSE reject requests that don't look like a browser.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
}


def parse(content: bytes) -> pd.DataFrame:
    """UDiFF bhavcopy (CSV, or a zip holding one) -> stock rows."""
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            content = z.read(z.namelist()[0])
    df = pd.read_csv(io.BytesIO(content), dtype=str)
    df.columns = [c.strip() for c in df.columns]
    if "FinInstrmTp" in df:
        df = df[df["FinInstrmTp"].str.strip() == "STK"]
    df["ClsPric"] = pd.to_numeric(df["ClsPric"], errors="coerce")
    return df


@functools.lru_cache(maxsize=None)
def fetch(exchange: str, day: dt.date) -> pd.DataFrame | None:
    """The bhavcopy of `exchange` ("NSE" or "BSE") for `day`, or None if there isn't one."""
    url = NSE_URL if exchange == "NSE" else BSE_URL
    headers = HEADERS | {"Referer": "https://www.nseindia.com/" if exchange == "NSE" else "https://www.bseindia.com/"}
    try:
        resp = requests.get(url.format(date=day), timeout=60, headers=headers)
        # Before a file is published BSE answers 200 with an HTML page, so check it is the data.
        if resp.status_code == 200 and (resp.content[:2] == b"PK" or b"FinInstrmId" in resp.content[:2000]):
            df = parse(resp.content)
            log.info("%s bhavcopy for %s: %d stocks", exchange, day, len(df))
            return df
        log.info("No %s bhavcopy for %s (HTTP %s)", exchange, day, resp.status_code)
    except Exception as exc:
        log.warning("%s bhavcopy for %s failed: %s", exchange, day, exc)
    return None


@functools.lru_cache(maxsize=None)
def latest(exchange: str) -> tuple[dt.date | None, pd.DataFrame]:
    """The most recent bhavcopy of `exchange` from the last week."""
    for back in range(8):
        day = dt.date.today() - dt.timedelta(days=back)
        if day.weekday() < 5 and (df := fetch(exchange, day)) is not None:
            return day, df
    return None, pd.DataFrame(columns=["ISIN", "ClsPric"])


OHLCV = {"ClsPric": "close", "HghPric": "high", "LwPric": "low", "TtlTradgVol": "volume"}


def ohlcv_by_isin(day: dt.date) -> pd.DataFrame:
    """ISIN -> close/high/low/volume for `day` (NSE row preferred for dual-listed stocks);
    empty if neither exchange traded that day."""
    frames = []
    for ex in ("NSE", "BSE"):
        df = fetch(ex, day)
        if df is not None and "ISIN" in df:
            out = pd.DataFrame({"isin": df["ISIN"].str.strip()})
            for src, dst in OHLCV.items():
                out[dst] = pd.to_numeric(df[src], errors="coerce").values if src in df else float("nan")
            out["high"] = out["high"].fillna(out["close"])
            out["low"] = out["low"].fillna(out["close"])
            frames.append(out.dropna(subset=["isin", "close"]))
    if not frames:
        return pd.DataFrame(columns=["close", "high", "low", "volume"])
    return pd.concat(frames).drop_duplicates("isin").set_index("isin")


def closes_by_isin() -> dict[str, float]:
    """ISIN -> latest close, NSE first (a dual-listed stock gets its NSE price)."""
    out: dict[str, float] = {}
    for ex in ("BSE", "NSE"):
        _, df = latest(ex)
        for isin, close in zip(df["ISIN"], df["ClsPric"]):
            if isinstance(isin, str) and pd.notna(close):
                out[isin.strip()] = round(float(close), 2)
    return out


def closing_prices(directory: pd.DataFrame) -> dict[str, float]:
    """Latest close for every company in the market directory, matched by ISIN."""
    by_isin = closes_by_isin()
    return {s: by_isin[i] for s, i in zip(directory["symbol"], directory["isin"]) if i in by_isin}
