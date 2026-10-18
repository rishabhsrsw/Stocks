"""Chart reading: technical features computed from price/volume history.

Every feature at date t uses only data up to and including t. Targets look forward.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import Prices

# Features that get an extra cross-sectional percentile-rank version (rank within the
# universe on that date), which is more stable across bull and bear markets.
RANKED = ["ret_21", "ret_63", "ret_126", "mom_12_1", "rs_126", "vol_63", "dist_52w_high", "vol_ratio"]


def _rsi(close: pd.DataFrame, n: int = 14) -> pd.DataFrame:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def stock_features(p: Prices) -> dict[str, pd.DataFrame]:
    c, v = p.close, p.volume
    r = c.pct_change(fill_method=None)
    b = p.benchmark
    br = b.pct_change(fill_method=None)

    f: dict[str, pd.DataFrame] = {}
    # Momentum / trend
    for n in (5, 21, 63, 126, 252):
        f[f"ret_{n}"] = c / c.shift(n) - 1
    f["mom_12_1"] = c.shift(21) / c.shift(252) - 1
    f["rs_63"] = f["ret_63"].sub(b / b.shift(63) - 1, axis=0)
    f["rs_126"] = f["ret_126"].sub(b / b.shift(126) - 1, axis=0)
    sma20, sma50, sma200 = c.rolling(20).mean(), c.rolling(50).mean(), c.rolling(200).mean()
    f["dist_sma20"] = c / sma20 - 1
    f["dist_sma50"] = c / sma50 - 1
    f["dist_sma200"] = c / sma200 - 1
    f["sma50_200"] = sma50 / sma200 - 1
    f["sma200_slope"] = sma200 / sma200.shift(21) - 1
    hi252, lo252 = p.high.rolling(252, min_periods=200).max(), p.low.rolling(252, min_periods=200).min()
    f["dist_52w_high"] = c / hi252 - 1
    f["dist_52w_low"] = c / lo252 - 1
    # Oscillators
    f["rsi_14"] = _rsi(c, 14)
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    f["macd_hist"] = (macd - macd.ewm(span=9, adjust=False).mean()) / c
    # Risk
    f["vol_21"] = r.rolling(21).std() * np.sqrt(252)
    f["vol_63"] = r.rolling(63).std() * np.sqrt(252)
    f["vol_ratio_21_63"] = f["vol_21"] / f["vol_63"]
    f["downside_vol_63"] = r.clip(upper=0).rolling(63).std() * np.sqrt(252)
    f["max_dd_126"] = c / c.rolling(126).max() - 1
    cov = r.rolling(252, min_periods=126).cov(br)
    f["beta_252"] = cov.div(br.rolling(252, min_periods=126).var(), axis=0)
    tr = pd.concat({"a": p.high - p.low, "b": (p.high - c.shift()).abs(), "c": (p.low - c.shift()).abs()}).groupby(level=1).max()
    f["atr_pct"] = tr.reindex(c.index).rolling(14).mean() / c
    # Volume / liquidity
    traded = c * v
    f["vol_ratio"] = v.rolling(20).mean() / v.rolling(120).mean()
    f["log_traded_value"] = np.log1p(traded.rolling(60).median())
    up_vol = (v * (r > 0)).rolling(50).sum()
    f["up_down_vol"] = up_vol / v.rolling(50).sum()
    return f


def market_features(p: Prices) -> pd.DataFrame:
    b = p.benchmark
    br = b.pct_change(fill_method=None)
    breadth = (p.close > p.close.rolling(200).mean()).where(p.close.notna()).mean(axis=1)
    return pd.DataFrame({
        "mkt_ret_63": b / b.shift(63) - 1,
        "mkt_ret_252": b / b.shift(252) - 1,
        "mkt_dist_sma200": b / b.rolling(200).mean() - 1,
        "mkt_vol_63": br.rolling(63).std() * np.sqrt(252),
        "mkt_breadth_200": breadth,
    })


def forward_returns(p: Prices, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Total forward return and forward return in excess of the benchmark."""
    fwd = p.close.shift(-horizon) / p.close - 1
    bfwd = p.benchmark.shift(-horizon) / p.benchmark - 1
    return fwd, fwd.sub(bfwd, axis=0)


def build_panel(p: Prices, horizons: dict[str, int], dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Long panel indexed by (date, symbol) with features and forward targets."""
    feats = stock_features(p)
    mkt = market_features(p)
    if dates is None:
        dates = p.close.index
    cols = {}
    for name, frame in feats.items():
        frame = frame.loc[dates]
        cols[name] = frame.stack(future_stack=True)
        if name in RANKED:
            cols[f"{name}_rank"] = frame.rank(axis=1, pct=True).stack(future_stack=True)
    for h_name, h in horizons.items():
        fwd, excess = forward_returns(p, h)
        cols[f"y_ret_{h_name}"] = fwd.loc[dates].stack(future_stack=True)
        cols[f"y_exc_{h_name}"] = excess.loc[dates].stack(future_stack=True)
    panel = pd.DataFrame(cols)
    panel.index.names = ["date", "symbol"]
    panel = panel.join(mkt.loc[dates], on="date")
    return panel.replace([np.inf, -np.inf], np.nan)


def feature_columns(panel: pd.DataFrame) -> list[str]:
    return [c for c in panel.columns if not c.startswith("y_")]
