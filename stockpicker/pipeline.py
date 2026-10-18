"""Weekly run: train -> score universe -> overlay news & fundamentals -> pick top N -> write site data."""
from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import news as news_mod
from .config import SITE_DATA_DIR, Config
from .data import Prices
from .features import build_panel, feature_columns
from .fundamentals import fetch_fundamentals, quality_score
from .model import HorizonModel, eligible, walk_forward_backtest

log = logging.getLogger(__name__)


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def explain(row: pd.Series) -> list[str]:
    """Plain-language reasons from the chart features (what the model is looking at)."""
    why = []
    if row.get("ret_126_rank", 0) >= 0.8:
        why.append(f"Strong 6-month momentum ({row['ret_126']:+.0%}, top {100 - row['ret_126_rank'] * 100:.0f}% of universe)")
    if row.get("rs_63", 0) > 0.05:
        why.append(f"Outperforming Nifty by {row['rs_63']:+.0%} over 3 months")
    if row.get("dist_sma200", -1) > 0 and row.get("sma50_200", -1) > 0:
        why.append("Price above rising 200-day average (uptrend, golden-cross alignment)")
    if row.get("dist_52w_high", -1) > -0.05:
        why.append("Trading within 5% of its 52-week high")
    if row.get("vol_ratio", 0) > 1.3:
        why.append(f"Volume {row['vol_ratio']:.1f}x its 6-month average (accumulation)")
    if 45 <= row.get("rsi_14", 0) <= 70:
        why.append(f"RSI {row['rsi_14']:.0f}: momentum without being overbought")
    elif row.get("rsi_14", 0) > 75:
        why.append(f"Caution: RSI {row['rsi_14']:.0f} is overbought, a pullback is possible")
    if row.get("vol_63_rank", 1) <= 0.3:
        why.append("Lower volatility than most peers")
    return why[:5]


def run_weekly(prices: Prices, universe: pd.DataFrame, cfg: Config, *, with_news: bool = True,
               with_fundamentals: bool = True, with_backtest: bool = True, out_dir: Path = SITE_DATA_DIR) -> dict:
    universe = universe[universe["symbol"].isin(prices.symbols)].reset_index(drop=True)
    days = prices.close.index
    asof = days[-1]
    sample = days[::-1][:: cfg.sample_every_n_days][::-1]  # weekly snapshots ending today
    log.info("Building feature panel on %d dates x %d symbols", len(sample), len(prices.symbols))
    panel = build_panel(prices, cfg.horizons, sample)
    feats = feature_columns(panel)

    backtest = walk_forward_backtest(panel, days, cfg) if with_backtest else None

    today = panel.xs(asof, level="date", drop_level=False)
    preds = {}
    for h in cfg.horizons:
        model = HorizonModel(h, feats).fit(panel)
        log.info("Trained %s model on %d rows", h, model.n_train)
        preds[h] = model.predict(today).droplevel("date")
    today = today.droplevel("date")
    main = preds[cfg.rank_horizon]

    scores = pd.DataFrame({"model_score": main["model_score"]})
    scores["eligible"] = eligible(today, cfg)
    meta = universe.set_index("symbol")
    scores = scores.join(meta[["name", "sector"]])

    # Overlays are computed for a shortlist only (keeps the weekly run fast and polite).
    shortlist = scores[scores["eligible"]].nlargest(max(cfg.llm_shortlist, cfg.top_n * 6), "model_score").index.tolist()
    news_res: dict[str, news_mod.NewsResult] = {}
    if with_news:
        news_res = news_mod.collect_news(universe[universe["symbol"].isin(shortlist)], cfg)
        for s in shortlist[: cfg.llm_shortlist]:
            nr = news_res.get(s)
            if nr and nr.headlines:
                nr.llm = news_mod.llm_analyse(s, meta.loc[s, "name"], nr.headlines, cfg.llm_model)
    news_score = pd.Series({
        s: (0.5 * r.score + 0.5 * r.llm["sentiment"]) if r.llm else r.score for s, r in news_res.items()
    }, dtype=float)
    red = {s for s, r in news_res.items() if r.red_flags or (r.llm and r.llm.get("red_flag"))}

    fund = fetch_fundamentals(shortlist) if with_fundamentals else pd.DataFrame()
    quality = quality_score(fund)

    sl = scores.loc[shortlist].copy()
    sl["news"] = news_score.reindex(sl.index).fillna(0.0)
    sl["quality"] = quality.reindex(sl.index).fillna(0.5)
    sl["final"] = (cfg.w_model * sl["model_score"] + cfg.w_news * (sl["news"] + 1) / 2 + cfg.w_fundamentals * sl["quality"])
    sl.loc[sl.index.isin(red), "final"] = -1  # never pick a stock with a governance/fraud red flag

    picks, per_sector = [], {}
    for s, row in sl.sort_values("final", ascending=False).iterrows():
        if row["final"] < 0:
            break
        sec = row["sector"] or "Unknown"
        if per_sector.get(sec, 0) >= cfg.max_per_sector:
            continue
        per_sector[sec] = per_sector.get(sec, 0) + 1
        picks.append(s)
        if len(picks) == cfg.top_n:
            break

    def card(s: str) -> dict:
        px = prices.close[s].dropna()
        nr = news_res.get(s)
        f = fund.loc[s].to_dict() if s in fund.index else {}
        return {
            "symbol": s,
            "name": meta.loc[s, "name"],
            "sector": meta.loc[s, "sector"],
            "price": _r(px.iloc[-1], 2),
            "final_score": _r(sl.loc[s, "final"]),
            "model_score": _r(sl.loc[s, "model_score"]),
            "news_score": _r(sl.loc[s, "news"]),
            "quality_score": _r(sl.loc[s, "quality"]),
            "forecast": {
                h: {"q10": _r(p.loc[s, "q10"]), "q50": _r(p.loc[s, "q50"]), "q90": _r(p.loc[s, "q90"]),
                    "p_beat_nifty": _r(p.loc[s, "p_beat"])}
                for h, p in preds.items()
            },
            "reasons": explain(today.loc[s]),
            "risk": {"volatility_63d": _r(today.loc[s, "vol_63"]), "max_drawdown_6m": _r(today.loc[s, "max_dd_126"]),
                     "beta": _r(today.loc[s, "beta_252"])},
            "fundamentals": {k: (_r(v) if isinstance(v, (int, float)) else v) for k, v in f.items()},
            "news": {
                "n_articles": nr.n_articles if nr else 0,
                "headlines": nr.headlines[:5] if nr else [],
                "llm": nr.llm if nr else None,
            },
            "chart": [_r(x, 2) for x in px.iloc[-126:].tolist()],
        }

    result = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(asof.date()),
        "horizon_months": {h: d / 21 for h, d in cfg.horizons.items()},
        "rank_horizon": cfg.rank_horizon,
        "universe_size": int(len(today)),
        "eligible_size": int(scores["eligible"].sum()),
        "llm_news": any(r.llm for r in news_res.values()),
        "picks": [card(s) for s in picks],
        "watchlist": [
            {"symbol": s, "name": meta.loc[s, "name"], "final_score": _r(sl.loc[s, "final"]),
             "q50_5m": _r(main.loc[s, "q50"]), "red_flag": s in red}
            for s in sl.sort_values("final", ascending=False).index[:20] if s not in picks
        ],
        "backtest": backtest,
    }
    write_outputs(result, prices, out_dir)
    return result


def update_track_record(out_dir: Path, prices: Prices, horizon_days: int) -> list[dict]:
    """Mark every past week's picks to market so the site shows the live (out-of-sample) record."""
    rows = []
    for f in sorted((out_dir / "history").glob("*.json")):
        past = json.loads(f.read_text())
        d = pd.Timestamp(past["as_of"])
        idx = prices.close.index
        if d not in idx:
            continue
        i0 = idx.get_loc(d)
        i1 = min(i0 + horizon_days, len(idx) - 1)
        b0, b1 = prices.benchmark.iloc[i0], prices.benchmark.iloc[i1]
        for p in past["picks"]:
            s = p["symbol"]
            if s not in prices.close.columns:
                continue
            c0, c1 = prices.close[s].iloc[i0], prices.close[s].iloc[i1]
            rows.append({
                "week": past["as_of"], "symbol": s, "entry": _r(c0, 2), "latest": _r(c1, 2),
                "return": _r(c1 / c0 - 1), "nifty_return": _r(b1 / b0 - 1),
                "days_held": int(i1 - i0), "complete": bool(i0 + horizon_days < len(idx)),
                "predicted_q50": p["forecast"].get("5m", {}).get("q50"),
            })
    return rows


def write_outputs(result: dict, prices: Prices, out_dir: Path) -> None:
    (out_dir / "history").mkdir(parents=True, exist_ok=True)
    (out_dir / "history" / f"{result['as_of']}.json").write_text(json.dumps(result, indent=1, default=str))
    result["track_record"] = update_track_record(out_dir, prices, 105)
    (out_dir / "latest.json").write_text(json.dumps(result, indent=1, default=str))
    index = sorted(p.stem for p in (out_dir / "history").glob("*.json"))
    (out_dir / "index.json").write_text(json.dumps(index))
