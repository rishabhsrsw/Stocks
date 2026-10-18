import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from stockpicker import news
from stockpicker.config import Config
from stockpicker.data import synthetic_prices
from stockpicker.features import build_panel, stock_features
from stockpicker.model import HorizonModel, train_cutoff
from stockpicker.pipeline import run_weekly


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
    res = run_weekly(prices, universe, cfg, with_news=False, with_fundamentals=False, out_dir=tmp_path)
    assert len(res["picks"]) == 5
    sectors = [p["sector"] for p in res["picks"]]
    assert max(sectors.count(s) for s in sectors) <= cfg.max_per_sector
    saved = json.loads((tmp_path / "latest.json").read_text())
    assert saved["backtest"]["n_weeks"] > 50
    assert set(saved["picks"][0]["forecast"]) == {"3m", "4m", "5m"}
