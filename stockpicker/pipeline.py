"""Weekly run: train -> score universe -> overlay news & fundamentals -> pick top N -> write site data."""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from . import holdings
from . import news as news_mod
from .config import SITE_DATA_DIR, Config
from .data import Prices
from .features import RANKED, build_panel, feature_columns, stock_features
from .fundamentals import fetch_fundamentals, quality_score
from .model import HorizonModel, eligible, walk_forward_backtest

log = logging.getLogger(__name__)


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), n)


def explain(row: pd.Series, bench: str = "Nifty") -> list[str]:
    """Plain-language reasons from the chart features (what the model is looking at)."""
    why = []
    if row.get("ret_126_rank", 0) >= 0.8:
        why.append(f"Strong 6-month momentum ({row['ret_126']:+.0%}, top {100 - row['ret_126_rank'] * 100:.0f}% of universe)")
    if row.get("rs_63", 0) > 0.05:
        why.append(f"Outperforming {bench} by {row['rs_63']:+.0%} over 3 months")
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


def explain_short(row: pd.Series) -> list[str]:
    """Reasons that matter for a 2-week view: recent move, short-term momentum, volume."""
    why = []
    r5, rsi = row.get("ret_5", 0), row.get("rsi_14", 50)
    if r5 > 0.03:
        why.append(f"Up {r5:+.1%} over the last week")
    elif r5 < -0.03 and row.get("dist_sma200", -1) > 0:
        why.append(f"Short-term dip ({r5:+.1%} this week) inside a long-term uptrend")
    if row.get("macd_hist", 0) > 0:
        why.append("MACD positive: short-term momentum turning up")
    if row.get("vol_ratio", 0) > 1.3:
        why.append(f"Volume {row['vol_ratio']:.1f}x its 6-month average")
    if rsi < 35:
        why.append(f"RSI {rsi:.0f}: oversold, bounce candidate")
    elif rsi > 75:
        why.append(f"Caution: RSI {rsi:.0f} is overbought")
    if -0.02 <= row.get("dist_sma20", 1) <= 0.03 and row.get("sma50_200", -1) > 0:
        why.append("Holding near its 20-day average in an uptrend")
    return why


def news_overlay(shortlist: list[str], meta: pd.DataFrame, cfg: Config, n_llm: int):
    """Headlines + sentiment for the shortlist, Claude's read for the top `n_llm`, and red flags."""
    universe = meta.loc[shortlist, ["name"]].reset_index()
    news_res = news_mod.collect_news(universe, cfg)
    for s in shortlist[:n_llm]:
        nr = news_res.get(s)
        if nr and nr.headlines:
            nr.llm = news_mod.llm_analyse(s, meta.loc[s, "name"], nr.headlines, cfg.llm_model, cfg.market)
    news_score = pd.Series({
        s: (0.5 * r.score + 0.5 * r.llm["sentiment"]) if r.llm else r.score for s, r in news_res.items()
    }, dtype=float)
    red = {s for s, r in news_res.items() if r.red_flags or (r.llm and r.llm.get("red_flag"))}
    return news_res, news_score, red


def news_block(nr) -> dict:
    return {
        "n_articles": nr.n_articles if nr else 0,
        "headlines": nr.headlines[:5] if nr else [],
        "llm": nr.llm if nr else None,
    }


def select_picks(sl: pd.DataFrame, top_n: int, max_per_sector: int) -> list[str]:
    """Best `final` scores first, skipping red-flagged stocks (final < 0) and capping each sector."""
    picks, per_sector = [], {}
    for s, row in sl.sort_values("final", ascending=False).iterrows():
        if row["final"] < 0:
            break
        sec = row["sector"] or "Unknown"
        if per_sector.get(sec, 0) >= max_per_sector:
            continue
        per_sector[sec] = per_sector.get(sec, 0) + 1
        picks.append(s)
        if len(picks) == top_n:
            break
    return picks


def run_weekly(prices: Prices, universe: pd.DataFrame, cfg: Config, *, coverage: pd.DataFrame | None = None,
               extended: Prices | None = None, directory: pd.DataFrame | None = None,
               funds: Prices | None = None,
               with_news: bool = True, with_fundamentals: bool = True, with_backtest: bool = True,
               out_dir: Path = SITE_DATA_DIR) -> dict:
    """`universe` is where picks come from (and what the model trains and is backtested on);
    `coverage` is the wider list scored for the holdings review; `extended` holds shorter price
    histories for every other listed company in `directory` (scored against `coverage`)."""
    universe = universe[universe["symbol"].isin(prices.symbols)].reset_index(drop=True)
    coverage = universe if coverage is None else coverage[coverage["symbol"].isin(prices.symbols)]
    meta = pd.concat([universe, coverage]).drop_duplicates("symbol").set_index("symbol")
    pick_syms = set(universe["symbol"])
    days = prices.close.index
    asof = days[-1]
    sample = days[::-1][:: cfg.sample_every_n_days][::-1]  # weekly snapshots ending today
    log.info("Building feature panel on %d dates x %d symbols", len(sample), len(prices.symbols))
    panel = build_panel(prices, cfg.horizons, sample)
    feats = feature_columns(panel)
    train_panel = panel[panel.index.get_level_values("symbol").isin(pick_syms)]

    backtest = walk_forward_backtest(train_panel, days, cfg) if with_backtest else None

    today = panel.xs(asof, level="date", drop_level=False)
    preds, models = {}, {}
    for h in cfg.horizons:
        models[h] = HorizonModel(h, feats).fit(train_panel)
        log.info("Trained %s model on %d rows", h, models[h].n_train)
        preds[h] = models[h].predict(today).droplevel("date")
    today = today.droplevel("date")
    main = preds[cfg.rank_horizon]

    scores = pd.DataFrame({"model_score": main["model_score"]})
    scores["eligible"] = eligible(today, cfg) & scores.index.isin(pick_syms)
    scores = scores.join(meta[["name", "sector"]])

    # Overlays are computed for a shortlist only (keeps the weekly run fast and polite).
    shortlist = scores[scores["eligible"]].nlargest(max(cfg.llm_shortlist, cfg.top_n * 6), "model_score").index.tolist()
    news_res, news_score, red = news_overlay(shortlist, meta, cfg, cfg.llm_shortlist) if with_news else ({}, pd.Series(dtype=float), set())

    fund = fetch_fundamentals(shortlist, cfg.yahoo_suffix) if with_fundamentals else pd.DataFrame()
    quality = quality_score(fund)

    sl = scores.loc[shortlist].copy()
    sl["news"] = news_score.reindex(sl.index).fillna(0.0)
    sl["quality"] = quality.reindex(sl.index).fillna(0.5)
    sl["final"] = (cfg.w_model * sl["model_score"] + cfg.w_news * (sl["news"] + 1) / 2 + cfg.w_fundamentals * sl["quality"])
    sl.loc[sl.index.isin(red), "final"] = -1  # never pick a stock with a governance/fraud red flag
    # Only pick what the holdings review would also say to hold: in the backtest, top-ranked stocks
    # below a falling 200-day average beat the Nifty by ~2% over 5 months, against ~18% for the rest.
    verdict = holdings.verdict_frame(
        main.loc[sl.index, ["model_score", "p_beat"]].join(today[["dist_sma200", "sma200_slope"]])
        .assign(red_flag=sl.index.isin(red)))
    sl.loc[~verdict.isin(["hold", "strong_hold"]), "final"] = -1

    picks = select_picks(sl, cfg.top_n, cfg.max_per_sector)

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
            "reasons": explain(today.loc[s], cfg.benchmark_name),
            "risk": {"volatility_63d": _r(today.loc[s, "vol_63"]), "max_drawdown_6m": _r(today.loc[s, "max_dd_126"]),
                     "beta": _r(today.loc[s, "beta_252"])},
            "fundamentals": {k: (_r(v) if isinstance(v, (int, float)) else v) for k, v in f.items()},
            "news": news_block(nr),
            "chart": [_r(x, 2) for x in px.iloc[-126:].tolist()],
        }

    result = {
        "market": cfg.market,
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

    # Holdings review data goes in its own compact file: it covers thousands of stocks.
    stocks = stock_scores(prices, today, preds, meta, news_res, red, cfg)
    if extended is not None and not extended.close.empty and directory is not None:
        ext_today, ext_preds = score_extended(extended, today, preds, models, feats, asof, cfg)
        ext_meta = directory.drop_duplicates("symbol").set_index("symbol")
        stocks += stock_scores(extended, ext_today, ext_preds, ext_meta, {}, set(), cfg, tested=False)
    fund_list = fund_rows(funds, directory) if funds is not None and directory is not None else []
    listed = {s["symbol"] for s in stocks} | {f["symbol"] for f in fund_list}
    unscored = [] if directory is None else [
        {"symbol": r.symbol, "name": r.name, "exchange": r.exchange, "kind": getattr(r, "kind", "stock"),
         "yahoo": r.yahoo}
        for r in directory.itertuples(index=False) if r.symbol not in listed
    ]
    (out_dir / "stocks.json").write_text(json.dumps(
        {"market": cfg.market, "as_of": result["as_of"], "generated_at": result["generated_at"], "stocks": stocks, "funds": fund_list,
         "unscored": unscored}, separators=(",", ":"), default=str))
    log.info("Holdings review: %d stocks scored, %d funds/gold bonds, %d listed without a verdict",
             len(stocks), len(fund_list), len(unscored))
    return result | {"stocks": stocks, "funds": fund_list, "unscored": unscored}


def fund_rows(funds: Prices, directory: pd.DataFrame) -> list[dict]:
    """Price, trend and risk for ETFs and gold bonds. The stock model doesn't rate these (it was
    trained on company shares against the Nifty), so they get facts, not a hold/sell verdict."""
    meta = directory.drop_duplicates("symbol").set_index("symbol")
    out = []
    for sym in funds.close.columns:
        c = funds.close[sym].dropna()
        if c.empty or sym not in meta.index:
            continue
        last = float(c.iloc[-1])
        ret = lambda n: _r(last / c.iloc[-1 - n] - 1) if len(c) > n else None
        r = c.pct_change().dropna()
        sma200 = c.rolling(200).mean().iloc[-1] if len(c) >= 200 else np.nan
        hi = float(c.iloc[-252:].max())
        notes = []
        if np.isfinite(sma200):
            notes.append(f"{'Above' if last >= sma200 else 'Below'} its 200-day average ({last / sma200 - 1:+.0%})")
        if last / hi - 1 < -0.1:
            notes.append(f"{-(last / hi - 1):.0%} below its 1-year high")
        traded = (funds.close[sym] * funds.volume[sym]).iloc[-60:].median()
        if np.isfinite(traded) and traded < 1e6:
            notes.append(f"Thinly traded (about ₹{traded / 1e5:.1f} lakh a day): may be hard to sell at a fair price")
        out.append({
            "symbol": sym, "name": meta.loc[sym, "name"], "sector": meta.loc[sym, "sector"],
            "exchange": meta.loc[sym, "exchange"], "kind": meta.loc[sym, "kind"], "yahoo": meta.loc[sym, "yahoo"], "price": _r(last, 2),
            "returns": {"1m": ret(21), "6m": ret(126), "1y": ret(252)},
            "vol": _r(r.iloc[-63:].std() * np.sqrt(252)) if len(r) >= 63 else None,
            "notes": notes, "spark": holdings.spark(c),
        })
    return out


def _pct_vs(ref: pd.Series, values: pd.Series) -> np.ndarray:
    """Percentile of each value within the reference distribution (NaN stays NaN)."""
    r = np.sort(ref.dropna().to_numpy())
    v = values.to_numpy(dtype=float)
    out = np.searchsorted(r, v, side="right") / max(len(r), 1)
    return np.where(np.isnan(v), np.nan, out)


def score_extended(ext: Prices, ref_today: pd.DataFrame, ref_preds: dict, models: dict, feats: list[str],
                   asof: pd.Timestamp, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Score stocks outside the training universe with the trained models.

    Cross-sectional features and the model score are measured against the covered stocks
    (`ref_today` / `ref_preds`), so a score means the same thing for every stock."""
    raw = stock_features(ext)
    close = ext.close.loc[:asof]
    recent = close.index[-cfg.stale_days:]
    live = close.columns[close.loc[recent].notna().any()]  # traded in the last `stale_days` sessions
    X = pd.DataFrame({name: frame.loc[:asof].ffill(limit=cfg.stale_days).iloc[-1] for name, frame in raw.items()})
    X = X.loc[X.index.intersection(live)]
    for name in RANKED:
        X[f"{name}_rank"] = _pct_vs(ref_today[name], X[name])
    for c in ref_today.columns:
        if c.startswith("mkt_"):
            X[c] = ref_today[c].iloc[0]
    X = X.reindex(columns=feats).replace([np.inf, -np.inf], np.nan)
    X.index = pd.MultiIndex.from_product([[asof], X.index], names=["date", "symbol"])
    preds = {}
    for h, model in models.items():
        p = model.predict(X).droplevel("date")
        ref = ref_preds[h]
        p["model_score"] = (_pct_vs(ref["rank_pred"], p["rank_pred"]) + _pct_vs(ref["p_beat"], p["p_beat"])
                            + _pct_vs(ref["q50"], p["q50"])) / 3
        preds[h] = p
    return X.droplevel("date"), preds


def stock_scores(prices: Prices, today: pd.DataFrame, preds: dict, meta: pd.DataFrame,
                 news_res: dict, red: set, cfg: Config, tested: bool = True) -> list[dict]:
    """Outlook and hold/sell verdict for every scored stock (used by the holdings review).
    `tested` is False for stocks outside the universe the model was trained and backtested on."""
    main = preds[cfg.rank_horizon]
    frame = main.join(today[["dist_sma200", "sma200_slope"]])
    frame["red_flag"] = frame.index.isin(red)
    verdict = holdings.verdict_frame(frame)
    if not tested:  # not backtested on stocks this small, so don't claim the strongest verdict
        verdict = verdict.replace("strong_hold", "hold")
    out = []
    for s in main.index:
        if s not in meta.index or pd.isna(main.loc[s, "q50"]):
            continue
        row = today.loc[s]
        nr = news_res.get(s)
        out.append({
            "symbol": s,
            "name": meta.loc[s, "name"],
            "sector": meta.loc[s, "sector"],
            "exchange": meta.loc[s, "exchange"] if "exchange" in meta else cfg.default_exchange,
            "tested": tested,
            "yahoo": meta.loc[s, "yahoo"] if "yahoo" in meta else None,  # ticker for live quotes (BSE: 543320.BO)
            "price": _r(prices.close[s].dropna().iloc[-1], 2),
            "model_score": _r(main.loc[s, "model_score"]),
            "verdict": verdict[s],
            "forecast": {h: {"q10": _r(p.loc[s, "q10"]), "q50": _r(p.loc[s, "q50"]), "q90": _r(p.loc[s, "q90"]),
                             "p_beat_nifty": _r(p.loc[s, "p_beat"])} for h, p in preds.items()},
            "positives": explain(row, cfg.benchmark_name),
            "concerns": holdings.warnings(row, main.loc[s], cfg),
            "red_flag": s in red,
            "news_summary": (nr.llm or {}).get("summary") if nr else None,
            "spark": holdings.spark(prices.close[s]),
        })
    return out


def update_track_record(history_dir: Path, prices: Prices, horizon_days: int, horizon: str = "5m") -> list[dict]:
    """Mark every past run's picks to market so the apps show the live (out-of-sample) record."""
    rows = []
    for f in sorted(history_dir.glob("*.json")):
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
                "predicted_q50": p["forecast"].get(horizon, {}).get("q50"),
            })
    return rows


def write_outputs(result: dict, prices: Prices, out_dir: Path) -> None:
    (out_dir / "history").mkdir(parents=True, exist_ok=True)
    (out_dir / "history" / f"{result['as_of']}.json").write_text(json.dumps(result, indent=1, default=str))
    result["track_record"] = update_track_record(out_dir / "history", prices, 105)
    (out_dir / "latest.json").write_text(json.dumps(result, indent=1, default=str))
    index = sorted(p.stem for p in (out_dir / "history").glob("*.json"))
    (out_dir / "index.json").write_text(json.dumps(index))


def run_daily(prices: Prices, universe: pd.DataFrame, cfg: Config, *, coverage: pd.DataFrame | None = None,
              extra_prices: dict[str, float] | None = None, with_news: bool = True, with_backtest: bool = True,
              out_dir: Path = SITE_DATA_DIR, force: bool = False) -> dict | None:
    """Short-term picks for the next ~2 weeks, plus today's closing prices for the holdings review.

    Returns None (and writes nothing) when there is no new trading day since the last run."""
    days = prices.close.index
    asof = days[-1]
    prev = out_dir / "daily.json"
    if not force and prev.exists():
        last = json.loads(prev.read_text())
        # A run during market hours (India 09:15-15:30 IST, US 09:30-16:00 New York) saw intraday
        # prices: redo it after the close.
        after_close = pd.Timestamp(last.get("generated_at", "1970-01-01T00:00:00+00:00")) >= \
            pd.Timestamp(f"{asof.date()}T{cfg.market_close_utc}:00+00:00")
        if last.get("as_of") == str(asof.date()) and after_close:
            log.info("No new trading day since %s: daily picks unchanged", asof.date())
            return None

    universe = universe[universe["symbol"].isin(prices.symbols)].reset_index(drop=True)
    coverage = universe if coverage is None else coverage[coverage["symbol"].isin(prices.symbols)]
    meta = pd.concat([universe, coverage]).drop_duplicates("symbol").set_index("symbol")
    pick_syms = set(universe["symbol"])

    h = "2w"
    dcfg = replace(cfg, horizons={h: cfg.daily_horizon_days}, rank_horizon=h, top_n=cfg.daily_top_n,
                   retrain_every_n_days=cfg.daily_retrain_every_n_days)
    sample = days[::-1][:: cfg.sample_every_n_days][::-1]
    panel = build_panel(prices, dcfg.horizons, sample)
    feats = feature_columns(panel)
    train_panel = panel[panel.index.get_level_values("symbol").isin(pick_syms)]

    backtest = None
    if with_backtest:
        backtest = walk_forward_backtest(train_panel, days, dcfg)
        backtest.pop("weekly")
        backtest.pop("verdict_stats", None)
        backtest["round_trip_cost"] = cfg.round_trip_cost

    model = HorizonModel(h, feats).fit(train_panel)
    today = panel.xs(asof, level="date", drop_level=False)
    pred = model.predict(today).droplevel("date")
    today = today.droplevel("date")

    scores = pd.DataFrame({"model_score": pred["model_score"]})
    scores["eligible"] = eligible(today, cfg) & scores.index.isin(pick_syms)
    scores = scores.join(meta[["name", "sector"]])
    shortlist = scores[scores["eligible"]].nlargest(cfg.daily_top_n * 5, "model_score").index.tolist()
    news_res, news_score, red = (news_overlay(shortlist, meta, cfg, cfg.daily_llm_shortlist) if with_news
                                 else ({}, pd.Series(dtype=float), set()))

    sl = scores.loc[shortlist].copy()
    sl["news"] = news_score.reindex(sl.index).fillna(0.0)
    sl["final"] = cfg.w_daily_model * sl["model_score"] + cfg.w_daily_news * (sl["news"] + 1) / 2
    sl.loc[sl.index.isin(red), "final"] = -1
    picks = select_picks(sl, cfg.daily_top_n, cfg.daily_max_per_sector)

    def card(s: str) -> dict:
        px = prices.close[s].dropna()
        row = today.loc[s]
        return {
            "symbol": s,
            "name": meta.loc[s, "name"],
            "sector": meta.loc[s, "sector"],
            "price": _r(px.iloc[-1], 2),
            "final_score": _r(sl.loc[s, "final"]),
            "model_score": _r(sl.loc[s, "model_score"]),
            "news_score": _r(sl.loc[s, "news"]),
            "forecast": {h: {"q10": _r(pred.loc[s, "q10"]), "q50": _r(pred.loc[s, "q50"]),
                             "q90": _r(pred.loc[s, "q90"]), "p_beat_nifty": _r(pred.loc[s, "p_beat"])}},
            "reasons": (explain_short(row) + explain(row, cfg.benchmark_name))[:5],
            "risk": {"volatility_63d": _r(row["vol_63"]), "atr_pct": _r(row["atr_pct"])},
            "news": news_block(news_res.get(s)),
            "chart": [_r(x, 2) for x in px.iloc[-63:].tolist()],
        }

    result = {
        "market": cfg.market,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "as_of": str(asof.date()),
        "horizon_days": cfg.daily_horizon_days,
        "eligible_size": int(scores["eligible"].sum()),
        "picks": [card(s) for s in picks],
        "backtest": backtest,
    }
    hist = out_dir / "daily-history"
    hist.mkdir(parents=True, exist_ok=True)
    (hist / f"{result['as_of']}.json").write_text(json.dumps(result, indent=1, default=str))
    result["track_record"] = update_track_record(hist, prices, cfg.daily_horizon_days, h)
    # Latest closing prices so the holdings review stays current between weekly runs.
    result["prices"] = (extra_prices or {}) | {s: _r(prices.close[s].dropna().iloc[-1], 2) for s in meta.index}
    (out_dir / "daily.json").write_text(json.dumps(result, indent=1, default=str))
    return result
