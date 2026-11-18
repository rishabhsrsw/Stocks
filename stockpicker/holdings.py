"""Hold / sell verdicts for any covered stock, so the apps can review the user's own holdings.

The verdict is forward-looking: it uses the model's outlook and the chart, never the user's
buy price (what was paid doesn't change what the stock does next). The apps add P&L and
tax timing on the device, where the holdings are stored.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config

VERDICTS = ["strong_hold", "hold", "review", "exit"]


def verdict_frame(df: pd.DataFrame) -> pd.Series:
    """Vectorised verdict. Needs model_score, p_beat, dist_sma200, sma200_slope (+ optional red_flag)."""
    red = df["red_flag"].fillna(False).astype(bool) if "red_flag" in df else pd.Series(False, index=df.index)
    downtrend = (df["dist_sma200"] < -0.05) & (df["sma200_slope"] < 0)
    v = np.select(
        [
            red,
            (df["model_score"] < 0.2) & (df["p_beat"] < 0.5),
            (df["model_score"] < 0.4) | downtrend,
            (df["model_score"] >= 0.7) & (df["p_beat"] >= 0.55),
        ],
        ["exit", "exit", "review", "strong_hold"],
        default="hold",
    )
    return pd.Series(v, index=df.index)


def money_per_day(x: float, cfg: Config) -> str:
    """Daily turnover in the unit people use in that market: lakh of rupees, or millions of dollars."""
    if cfg.currency == "₹":
        return f"₹{x / 1e5:.1f} lakh"
    return f"{cfg.currency}{x / 1e6:.1f}M"


def warnings(row: pd.Series, p5: pd.Series, cfg: Config | None = None) -> list[str]:
    """Plain-language concerns from the chart and the forecast. `cfg` names the market (India by default)."""
    cfg = cfg or Config()
    bench = cfg.benchmark_name
    out = []
    traded = np.expm1(row.get("log_traded_value", np.nan))  # median daily turnover, in the market's currency
    if traded < cfg.thin_traded_value:
        out.append(f"Thinly traded (about {money_per_day(traded, cfg)} a day): may be hard to sell quickly")
    if row.get("dist_sma200", 0) < 0 and row.get("sma200_slope", 0) < 0:
        out.append(f"Below a falling 200-day average ({row['dist_sma200']:+.0%}): long-term downtrend")
    if row.get("ret_126_rank", 1) <= 0.2:
        out.append(f"Weak 6-month momentum ({row['ret_126']:+.0%}, bottom {row['ret_126_rank'] * 100:.0f}% of stocks)")
    if row.get("rs_63", 0) < -0.08:
        out.append(f"Lagging {bench} by {-row['rs_63']:.0%} over 3 months")
    if row.get("max_dd_126", 0) < -0.25:
        out.append(f"{-row['max_dd_126']:.0%} below its 6-month high")
    if row.get("rsi_14", 50) > 75:
        out.append(f"RSI {row['rsi_14']:.0f}: overbought, a pullback is possible")
    if row.get("vol_63_rank", 0) >= 0.85:
        out.append("More volatile than most stocks: expect big swings")
    if p5["p_beat"] < 0.45:
        out.append(f"Only {p5['p_beat']:.0%} chance of beating the {bench} over 5 months")
    if p5["q10"] < -0.25:
        out.append(f"Bad-case 5-month outcome is {p5['q10']:.0%}")
    return out[:5]


def spark(series: pd.Series, days: int = 126, step: int = 3) -> list[float]:
    s = series.dropna().iloc[-days:]
    pts = s.iloc[::-1][::step][::-1]  # sample backwards so the latest close is always included
    return [round(float(x), 2) for x in pts]


def bucket_stats(pred: pd.DataFrame, ret_col: str, exc_col: str) -> dict:
    """How each verdict actually performed in the walk-forward backtest."""
    v = verdict_frame(pred)
    g = pred.assign(verdict=v).groupby("verdict")
    out = {}
    for k in VERDICTS:
        if k in g.groups:
            grp = g.get_group(k)
            out[k] = {
                "avg_return": float(grp[ret_col].mean()),
                "avg_excess": float(grp[exc_col].mean()),
                "share_positive": float((grp[ret_col] > 0).mean()),
                "share": float(len(grp) / len(pred)),
            }
    return out
