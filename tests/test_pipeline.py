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
from stockpicker import bhavcopy
from stockpicker.universe import bse_rows_from_bhavcopy, parse_bse_json, parse_nse_csv


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
    # Same trading day again (weekend, holiday): nothing is rewritten...
    asof = pd.Timestamp(res["as_of"])
    saved["generated_at"] = f"{asof.date()}T12:00:00+00:00"
    (tmp_path / "daily.json").write_text(json.dumps(saved))
    assert run_daily(prices, universe.iloc[:20], cfg, coverage=universe, with_news=False, out_dir=tmp_path) is None
    # ...unless the earlier run was during market hours and saw intraday prices.
    saved["generated_at"] = f"{asof.date()}T06:00:00+00:00"
    (tmp_path / "daily.json").write_text(json.dumps(saved))
    assert run_daily(prices, universe.iloc[:20], cfg, coverage=universe, with_news=False, out_dir=tmp_path) is not None


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
    csv = (b"TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,FinInstrmNm,ClsPric\n"
           b"2026-11-10,2026-11-10,CM,BSE,STK,530001,INE111A01011,SMALL,X,Small Co Ltd,12.5\n"
           b"2026-11-10,2026-11-10,CM,BSE,ETF,590001,INF000A01011,GOLDETF,E,Gold ETF,50\n")
    bse = parse_bse_json(bse_rows_from_bhavcopy(bhavcopy.parse(csv)), set(), set())
    assert bse["symbol"].tolist() == ["SMALL"] and bse["yahoo"].tolist() == ["530001.BO"]


def test_bhavcopy_prices_by_isin(monkeypatch):
    import io, zipfile
    nse_csv = (b"TradDt,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,ClsPric\n"
               b"2026-11-10,STK,2885,INE002A01018,RELIANCE,EQ,1500.5\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:  # NSE ships its bhavcopy zipped
        z.writestr("BhavCopy_NSE_CM.csv", nse_csv)
    bse_csv = (b"TradDt,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,ClsPric\n"
               b"2026-11-10,STK,500325,INE002A01018,RELIANCE,A,1501.0\n"
               b"2026-11-10,STK,530001,INE111A01011,SMALL,X,12.5\n")
    files = {"NSE": bhavcopy.parse(buf.getvalue()), "BSE": bhavcopy.parse(bse_csv)}
    monkeypatch.setattr(bhavcopy, "latest", lambda ex: (None, files[ex]))
    directory = pd.DataFrame({"symbol": ["RELIANCE", "SMALL", "NOTRADE"],
                              "isin": ["INE002A01018", "INE111A01011", "INE999Z01011"]})
    # Dual-listed stocks take the NSE close; untraded ones have no price.
    assert bhavcopy.closing_prices(directory) == {"RELIANCE": 1500.5, "SMALL": 12.5}


def test_market_history_store(tmp_path):
    from stockpicker import history
    today = dt.date(2026, 11, 11)
    calls = []

    def read_day(day):  # a holiday on 10 Nov; BSE's file for today isn't out yet
        calls.append(day)
        if day == dt.date(2026, 11, 10):
            return pd.DataFrame(columns=["close", "high", "low", "volume"]), False
        n = (day - dt.date(2024, 1, 1)).days
        rows = pd.DataFrame({"close": [100 + n * 0.01, 5.0], "high": [101 + n * 0.01, 5.1],
                             "low": [99 + n * 0.01, 4.9], "volume": [1e5, 1e3]}, index=pd.Index(["IN1", "IN2"]))
        return rows, day != today

    store = history.Store()
    history.sync_bhavcopies(store, 2, budget_minutes=1, today=today, read_day=read_day, pause=0)
    assert calls[0] == today  # newest first
    assert set(store.close.columns) == {"IN1", "IN2"} and len(store.close) > 450  # keyed by ISIN
    assert today not in store.done and dt.date(2026, 11, 10) not in store.done  # retried next run
    calls.clear()
    history.sync_bhavcopies(store, 2, budget_minutes=1, today=today, read_day=read_day, pause=0)
    assert sorted(calls) == [dt.date(2026, 11, 10), today]
    # Out of time: nothing read, and the store is untouched.
    calls.clear()
    history.sync_bhavcopies(store, 2, budget_minutes=-1, today=dt.date(2026, 11, 12),
                            read_day=read_day, pause=0)
    assert calls == []

    history.save(store, tmp_path / "h.pkl")
    again = history.load(tmp_path / "h.pkl")
    assert again.close.shape == store.close.shape
    old = history.Store()  # a store saved before the ISIN format must be rebuilt, not reused
    del old.__dict__["version"]
    history.save(old, tmp_path / "old.pkl")
    assert history.load(tmp_path / "old.pkl").close.empty and history.load(tmp_path / "old.pkl").done == set()
    bench = pd.Series(1.0, index=pd.bdate_range("2024-01-01", "2026-11-11"))
    p = history.to_prices(again, {"AAA": "IN1", "BSEONLY": "IN2", "GONE": "IN9"}, bench, min_rows=252)
    assert list(p.close.columns) == ["AAA", "BSEONLY"] and p.benchmark.notna().all()

    # A 1:1 bonus halves the price: earlier history is scaled so it isn't a 50% crash.
    days = pd.bdate_range("2025-01-01", periods=300)
    px = pd.Series(100.0, index=days)
    px.iloc[200:] = 50.0
    split = history.Store()
    for f in history.FIELDS:
        split.frames[f] = pd.DataFrame({"S": px if f != "volume" else 1000.0}, index=days)
    p = history.to_prices(split, {"S": "S"}, pd.Series(1.0, index=days), min_rows=252)
    assert p.close["S"].nunique() == 1 and p.close["S"].iloc[0] == 50.0
    assert p.volume["S"].iloc[0] == 2000.0 and p.volume["S"].iloc[-1] == 1000.0

def test_market_history_gaps():
    from stockpicker import history
    # A small stock with no trades on some days: carried over on the market calendar, volume 0.
    gappy = history.Store()
    cal = pd.bdate_range("2024-01-01", "2026-11-06")
    traded = cal.delete([100, 101, 300])
    for f in history.FIELDS:
        gappy.frames[f] = pd.DataFrame({"SMALL": np.linspace(10, 20, len(traded))}, index=traded)
    p = history.to_prices(gappy, {"SMALL": "SMALL"}, pd.Series(1.0, index=cal), min_rows=252)
    assert p.close.index.equals(cal) and p.close["SMALL"].notna().all()
    assert p.volume["SMALL"].iloc[100] == 0 and p.close["SMALL"].iloc[100] == p.close["SMALL"].iloc[99]
    feats = stock_features(p)
    assert feats["dist_sma200"]["SMALL"].iloc[-1] == feats["dist_sma200"]["SMALL"].iloc[-1]  # not NaN
    assert np.isfinite(feats["log_traded_value"]["SMALL"].iloc[-1])


def test_etfs_and_gold_bonds():
    from stockpicker.pipeline import fund_rows
    from stockpicker.universe import instrument_kind, parse_nse_etf_csv
    etf = parse_nse_etf_csv(
        "Symbol,Underlying Asset,SecurityName,DateofListing,MarketLot,ISINNumber,FaceValue,ETF Underlying,Underlying Key\n"
        "GOLDBEES,Gold,NIPINDETFGOLDBEES,19-Mar-07,1,INF204KB17I5,1,COMMODITY,GOL\n"
        "SILVERBEES,Silver,NIPINDETFSILVERBEES,01-Feb-22,1,INF204KC1402,1,COMMODITY,SIL\n"
        "LIQUIDBEES,Government Securities,NIPINDETFLIQUIDBEES,16-Jul-03,1,INF732E01037,1000,DEBT,Overnight\n",
        {"INF204KB17I5": "NIP IND ETF GOLD BEES", "INF204KC1402": "NIPPONAMC - NETFSILVER"})
    assert etf["sector"].tolist() == ["ETF · Gold", "ETF · Silver", "ETF · Debt"]
    assert etf["name"].tolist() == ["Nippon India Gold ETF", "Nippon India Silver ETF", "Nippon India Government Securities ETF"]
    from stockpicker.universe import fund_name
    assert fund_name("TATA SILVER EXCHANGE TRADED FUND", "Silver", "Silver", "TATSILV") == "Tata Silver ETF"
    assert fund_name("QUANT_ - SILVER", "Silver", "Silver", "SILVER001") == "Quant Silver ETF"
    assert set(etf["kind"]) == {"etf"}
    from stockpicker.universe import sgb_name
    assert sgb_name("2.5%GOLDBONDS2028SR-III") == "Sovereign Gold Bond 2028 Series III (2.5%)"
    # BSE rows are classified by ISIN: fund units are ETFs, not stocks; government bonds other than SGBs are left out.
    rows = [{"scrip_cd": "1", "scrip_name": "Kotak Nifty 50 ETF", "isin_number": "INF174K01Z71", "scrip_id": "NIFTY1"},
            {"scrip_cd": "2", "scrip_name": "2.50%GOLDBONDS2028SR-I", "isin_number": "IN0020200062", "scrip_id": "SGBAPR28I"},
            {"scrip_cd": "3", "scrip_name": "7.26% GOI 2033", "isin_number": "IN0020230011", "scrip_id": "GS2033"},
            {"scrip_cd": "4", "scrip_name": "Small Co", "isin_number": "INE111A01011", "scrip_id": "SMALL"}]
    bse = parse_bse_json(rows, set(), set())
    assert dict(zip(bse["symbol"], bse["kind"])) == {"NIFTY1": "etf", "SGBAPR28I": "sgb", "SMALL": "stock"}
    assert bse.set_index("symbol").loc["NIFTY1", "sector"] == "ETF · Equity"
    assert instrument_kind("INE002A01018") == "stock"

    days = pd.bdate_range("2025-06-01", periods=300)
    close = pd.DataFrame({"GOLDBEES": np.linspace(80, 120, 300), "SILVERBEES": np.linspace(150, 120, 300)}, index=days)
    funds = Prices(close=close, high=close, low=close, volume=pd.DataFrame(1e6, index=days, columns=close.columns),
                   benchmark=pd.Series(1.0, index=days))
    directory = etf.assign(exchange="NSE")
    out = {f["symbol"]: f for f in fund_rows(funds, directory)}
    g, s = out["GOLDBEES"], out["SILVERBEES"]
    assert g["price"] == 120.0 and g["returns"]["1y"] > 0 and "verdict" not in g and g["kind"] == "etf"
    assert any(n.startswith("Above its 200-day") for n in g["notes"])
    assert any(n.startswith("Below its 200-day") for n in s["notes"]) and any("below its 1-year high" in n for n in s["notes"])
