import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from stockpicker import holdings, news
from stockpicker.config import Config
from stockpicker.data import Prices, synthetic_prices
from stockpicker.features import build_panel, stock_features
from stockpicker.model import HorizonModel, train_cutoff
from stockpicker.pipeline import run_daily, run_weekly
from stockpicker.universe import parse_bse_bhavcopy, parse_bse_json, parse_nse_csv


@pytest.fixture(scope="module")
def prices():
    return synthetic_prices(n_symbols=30, n_days=1500, seed=1)


def test_features_have_no_lookahead(prices):
    """Changing future prices must not change features at an earlier date."""
    t = prices.close.index[1000]
    base = {k: v.loc[t] for k, v in stock_features(prices).items()}
    prices2 = synthetic_prices(n_symbols=30, n_days=1500, seed=1)
    prices2.close.iloc[1001:] *= 3
    prices2.high.iloc[1001:] *= 3
    prices2.low.iloc[1001:] *= 3
    prices2.volume.iloc[1001:] *= 5
    after = {k: v.loc[t] for k, v in stock_features(prices2).items()}
    for k in base:
        pd.testing.assert_series_equal(base[k], after[k], check_names=False)


def test_train_cutoff_purges_horizon(prices):
    days = prices.close.index
    asof = days[1200]
    cut = train_cutoff(days, asof, 105)
    assert days.get_loc(asof) - days.get_loc(cut) > 105


def test_model_predictions_shape_and_order(prices):
    cfg = Config()
    panel = build_panel(prices, {"5m": 105}, prices.close.index[::5])
    feats = [c for c in panel.columns if not c.startswith("y_")]
    m = HorizonModel("5m", feats).fit(panel)
    last = panel.xs(panel.index.get_level_values("date").max(), level="date", drop_level=False)
    out = m.predict(last)
    assert len(out) == 30
    assert (out["q10"] <= out["q50"]).all() and (out["q50"] <= out["q90"]).all()
    assert out["p_beat"].between(0, 1).all()
    assert out["model_score"].between(0, 1).all()
    assert m.widen >= 0


def test_news_lexicon_and_red_flags():
    assert news.score_headline("Company bags Rs 500 crore order, shares surge") > 0.3
    assert news.score_headline("Shares plunge after Q2 profit misses estimates; brokerage downgrades") < -0.3
    assert news.find_red_flags("ED raid at promoter premises over alleged fraud")
    assert not news.find_red_flags("Company reports record quarterly profit")
    now = dt.datetime(2026, 1, 10, tzinfo=dt.timezone.utc)
    hl = [{"title": "Firm wins major order", "published": "2026-01-09T00:00:00+00:00"},
          {"title": "Firm shares plunge on fraud probe", "published": "2025-11-01T00:00:00+00:00"}]
    r = news.aggregate("X", hl, half_life_days=5, now=now)
    assert r.score > 0  # the recent positive headline dominates the stale negative one
    assert r.red_flags


def test_weekly_run_end_to_end(prices, tmp_path):
    cfg = Config(min_median_traded_value_cr=0, backtest_years=2, retrain_every_n_days=126)
    universe = pd.DataFrame({"symbol": prices.symbols, "name": prices.symbols,
                             "sector": [f"S{i % 6}" for i in range(len(prices.symbols))]})
    picks_universe = universe.iloc[:20]  # the other 10 are only scored for the holdings review
    res = run_weekly(prices, picks_universe, cfg, coverage=universe, with_news=False, with_fundamentals=False,
                     out_dir=tmp_path)
    assert len(res["picks"]) == 5
    assert {p["symbol"] for p in res["picks"]} <= set(picks_universe["symbol"])
    stocks = {s["symbol"]: s for s in res["stocks"]}
    assert set(stocks) == set(universe["symbol"])
    assert all(s["verdict"] in holdings.VERDICTS for s in stocks.values())
    assert set(res["backtest"]["verdict_stats"]) <= set(holdings.VERDICTS)
    sectors = [p["sector"] for p in res["picks"]]
    assert max(sectors.count(s) for s in sectors) <= cfg.max_per_sector
    saved = json.loads((tmp_path / "latest.json").read_text())
    assert saved["backtest"]["n_weeks"] > 50
    assert set(saved["picks"][0]["forecast"]) == {"3m", "4m", "5m"}


def test_verdicts():
    df = pd.DataFrame({
        "model_score": [0.9, 0.5, 0.3, 0.1, 0.9, 0.9],
        "p_beat":      [0.7, 0.5, 0.5, 0.4, 0.7, 0.7],
        "dist_sma200": [0.1, 0.0, 0.0, -0.2, -0.2, 0.1],
        "sma200_slope": [0.02, 0.0, 0.0, -0.03, -0.03, 0.02],
        "red_flag":    [False, False, False, False, False, True],
    })
    assert holdings.verdict_frame(df).tolist() == ["strong_hold", "hold", "review", "exit", "review", "exit"]


def test_spark_keeps_latest_close():
    s = pd.Series(np.arange(300, dtype=float))
    pts = holdings.spark(s)
    assert pts[-1] == 299.0 and len(pts) == 42


def test_daily_run(prices, tmp_path):
    cfg = Config(min_median_traded_value_cr=0, backtest_years=2, daily_retrain_every_n_days=252)
    universe = pd.DataFrame({"symbol": prices.symbols, "name": prices.symbols,
                             "sector": [f"S{i % 6}" for i in range(len(prices.symbols))]})
    res = run_daily(prices, universe.iloc[:20], cfg, coverage=universe, with_news=False, out_dir=tmp_path)
    assert len(res["picks"]) == cfg.daily_top_n
    assert len({p["sector"] for p in res["picks"]}) == cfg.daily_top_n  # one per sector
    assert set(res["prices"]) == set(universe["symbol"])
    assert res["backtest"]["round_trip_cost"] == cfg.round_trip_cost
    saved = json.loads((tmp_path / "daily.json").read_text())
    assert saved["as_of"] == res["as_of"] and "2w" in saved["picks"][0]["forecast"]
    assert (tmp_path / "daily-history" / f"{res['as_of']}.json").exists()
    # Same trading day again (weekend, holiday): nothing is rewritten.
    assert run_daily(prices, universe.iloc[:20], cfg, coverage=universe, with_news=False, out_dir=tmp_path) is None


def test_market_directory_parsing():
    nse = parse_nse_csv(
        "SYMBOL,NAME OF COMPANY, SERIES, DATE OF LISTING, ISIN NUMBER\n"
        "TCS,Tata Consultancy Services Limited,EQ,25-AUG-2004,INE467B01029\n"
        "ABC,ABC Limited,BE,01-JAN-2020,INE000A01011\n"
        "XYZ,XYZ Limited,N1,01-JAN-2020,INE000A01099\n")  # debt series: excluded
    assert nse["symbol"].tolist() == ["TCS", "ABC"] and nse["yahoo"].tolist() == ["TCS.NS", "ABC.NS"]
    bse = parse_bse_json([
        {"SCRIP_CD": "532540", "Scrip_Name": "TCS Ltd", "Status": "Active", "ISIN_NUMBER": "INE467B01029", "scrip_id": "TCS"},
        {"SCRIP_CD": "530001", "Scrip_Name": "Small Co", "Status": "Active", "ISIN_NUMBER": "INE111A01011", "scrip_id": "SMALL", "INDUSTRY": "Textiles"},
        {"SCRIP_CD": "530002", "Scrip_Name": "Other ABC", "Status": "Active", "ISIN_NUMBER": "INE222A01011", "scrip_id": "ABC"},
        {"SCRIP_CD": "530003", "Scrip_Name": "Gone Ltd", "Status": "Delisted", "ISIN_NUMBER": "INE333A01011", "scrip_id": "GONE"},
    ], set(nse["isin"]), set(nse["symbol"]))
    # TCS is already on NSE (same ISIN), ABC clashes with a different NSE company, GONE is delisted.
    assert bse["symbol"].tolist() == ["SMALL", "ABC-BSE"]
    assert bse["yahoo"].tolist() == ["530001.BO", "530002.BO"] and bse["sector"].iloc[0] == "Textiles"


def test_weekly_scores_stocks_outside_coverage(prices, tmp_path):
    cfg = Config(min_median_traded_value_cr=0, stale_days=10)
    universe = pd.DataFrame({"symbol": prices.symbols, "name": prices.symbols,
                             "sector": [f"S{i % 6}" for i in range(len(prices.symbols))]})
    other = synthetic_prices(n_symbols=8, n_days=1500, seed=7)
    rename = {s: f"EXT{i}" for i, s in enumerate(other.symbols)}
    window = slice(-504, None)  # the extended download only has ~2 years
    ext = Prices(*(f.iloc[window].rename(columns=rename) for f in (other.close, other.high, other.low, other.volume)),
                 benchmark=other.benchmark.iloc[window])
    ext.close.iloc[-20:, 0] = np.nan  # EXT0 stopped trading: no verdict
    directory = pd.DataFrame({"symbol": list(rename.values()) + ["NEWIPO"], "name": "Co", "exchange": "BSE",
                              "yahoo": "x.BO", "isin": "", "sector": ""})
    res = run_weekly(prices, universe, cfg, extended=ext, directory=directory, with_news=False,
                     with_fundamentals=False, with_backtest=False, out_dir=tmp_path)
    by_sym = {s["symbol"]: s for s in res["stocks"]}
    for i in range(1, 8):
        s = by_sym[f"EXT{i}"]
        assert s["tested"] is False and s["exchange"] == "BSE" and s["verdict"] in holdings.VERDICTS
        assert 0 <= s["model_score"] <= 1
    assert by_sym[prices.symbols[0]]["tested"] is True
    assert {u["symbol"] for u in res["unscored"]} == {"EXT0", "NEWIPO"}
    saved = json.loads((tmp_path / "stocks.json").read_text())
    assert len(saved["stocks"]) == 30 + 7
    assert "stocks" not in json.loads((tmp_path / "latest.json").read_text())


def test_thin_trading_warning():
    row = pd.Series({"log_traded_value": np.log1p(3e5)})
    p5 = pd.Series({"p_beat": 0.5, "q10": -0.1})
    assert holdings.warnings(row, p5)[0].startswith("Thinly traded (about ₹3.0 lakh a day)")


def test_sme_list_and_bhavcopy_parsing():
    sme = parse_nse_csv("SYMBOL,NAME_OF_COMPANY,SERIES,DATE_OF_LISTING,ISIN_NUMBER\n"
                        "SMEONE,SME One Limited,SM,01-JAN-2023,INE999A01011\n"
                        "DUMMYSME,Dummy,SM,01-JAN-2023,INE999A01099\n")
    assert sme["symbol"].tolist() == ["SMEONE"] and sme["name"].tolist() == ["SME One Limited"]
    rows = parse_bse_bhavcopy(
        "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,FinInstrmNm,ClsPric\n"
        "2026-11-10,2026-11-10,CM,BSE,STK,530001,INE111A01011,SMALL,X,Small Co Ltd,12.5\n"
        "2026-11-10,2026-11-10,CM,BSE,ETF,590001,INF000A01011,GOLDETF,E,Gold ETF,50\n")
    bse = parse_bse_json(rows, set(), set())
    assert bse["symbol"].tolist() == ["SMALL"] and bse["yahoo"].tolist() == ["530001.BO"]
