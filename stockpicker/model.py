"""Gradient-boosted forecasting models and a purged walk-forward backtest."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from .config import Config
from .holdings import bucket_stats, verdict_frame

log = logging.getLogger(__name__)
QUANTILES = (0.1, 0.5, 0.9)


def _gbr(**kw) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=250, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=200,
        l2_regularization=1.0, random_state=42, **kw,
    )


@dataclass
class HorizonModel:
    """Models for one horizon:
    - ranker: predicts the cross-sectional percentile of the stock's excess return
    - beat:   P(stock beats Nifty over the horizon)
    - q10/q50/q90: quantiles of the stock's total return (a forecast range)
    """
    horizon: str
    features: list[str]

    def fit(self, train: pd.DataFrame) -> "HorizonModel":
        y_ret, y_exc = train[f"y_ret_{self.horizon}"], train[f"y_exc_{self.horizon}"]
        ok = y_ret.notna() & y_exc.notna()
        X, y_ret, y_exc = train.loc[ok, self.features], y_ret[ok], y_exc[ok]
        y_rank = y_exc.groupby(level="date").rank(pct=True)
        self.ranker = _gbr().fit(X, y_rank)
        self.beat = HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.04, max_leaf_nodes=24, min_samples_leaf=200,
            l2_regularization=1.0, random_state=42,
        ).fit(X, (y_exc > 0).astype(int))
        self.quantile = {q: _gbr(loss="quantile", quantile=q).fit(X, y_ret) for q in QUANTILES}
        self.widen = self._conformal_widen(X, y_ret)
        self.n_train = int(ok.sum())
        return self

    def _conformal_widen(self, X: pd.DataFrame, y: pd.Series, holdout: float = 0.2, purge: int = 21) -> float:
        """Conformalized quantile regression: fit q10/q90 on older dates, measure how far the
        newest dates fall outside the band, and widen the band by that amount so that the
        q10..q90 range really holds ~80% of outcomes."""
        dates = X.index.get_level_values("date")
        uniq = dates.unique().sort_values()
        split = int(len(uniq) * (1 - holdout))
        if split - purge < 50:
            return 0.0
        fit_mask, cal_mask = dates < uniq[split - purge], dates >= uniq[split]
        lo = _gbr(loss="quantile", quantile=QUANTILES[0]).fit(X[fit_mask], y[fit_mask])
        hi = _gbr(loss="quantile", quantile=QUANTILES[-1]).fit(X[fit_mask], y[fit_mask])
        Xc, yc = X[cal_mask], y[cal_mask].to_numpy()
        err = np.maximum(lo.predict(Xc) - yc, yc - hi.predict(Xc))
        level = QUANTILES[-1] - QUANTILES[0]
        return float(max(0.0, np.quantile(err, min(1.0, level * (1 + 1 / len(err))))))

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        Xf = X[self.features]
        out = pd.DataFrame(index=X.index)
        out["rank_pred"] = self.ranker.predict(Xf)
        out["p_beat"] = self.beat.predict_proba(Xf)[:, 1]
        qs = np.sort(np.column_stack([self.quantile[q].predict(Xf) for q in QUANTILES]), axis=1)
        out["q10"], out["q50"], out["q90"] = qs[:, 0] - self.widen, qs[:, 1], qs[:, 2] + self.widen
        # Combined model score in [0, 1]: blend of rank, beat-probability and median, ranked per date.
        g = out.groupby(level="date")
        out["model_score"] = (g["rank_pred"].rank(pct=True) + g["p_beat"].rank(pct=True) + g["q50"].rank(pct=True)) / 3
        return out


def eligible(X: pd.DataFrame, cfg: Config) -> pd.Series:
    """Liquidity and risk filters (as of the prediction date)."""
    min_tv = np.log1p(cfg.min_traded_value)
    return (X["log_traded_value"] >= min_tv) & (X["vol_63"] <= cfg.max_annual_volatility)


def train_cutoff(dates: pd.DatetimeIndex, asof: pd.Timestamp, horizon_days: int) -> pd.Timestamp:
    """Last date whose forward label is fully observed at `asof` (purges label overlap)."""
    pos = dates.searchsorted(asof)
    return dates[max(pos - horizon_days - 1, 0)]


def walk_forward_backtest(panel: pd.DataFrame, trading_days: pd.DatetimeIndex, cfg: Config) -> dict:
    h = cfg.rank_horizon
    hd = cfg.horizons[h]
    features = [c for c in panel.columns if not c.startswith("y_")]
    panel_dates = panel.index.get_level_values("date").unique().sort_values()
    last_labeled = trading_days[-hd - 1]
    start = last_labeled - pd.DateOffset(years=cfg.backtest_years)
    test_dates = panel_dates[(panel_dates >= start) & (panel_dates <= last_labeled)]
    if len(test_dates) == 0:
        raise ValueError("Not enough history for a backtest")

    picks_rows, ic_rows, preds_all = [], [], []
    model, model_date = None, None
    for d in test_dates:
        if model is None or trading_days.searchsorted(d) - trading_days.searchsorted(model_date) >= cfg.retrain_every_n_days:
            cut = train_cutoff(trading_days, d, hd)
            train = panel[panel.index.get_level_values("date") <= cut]
            if train[f"y_exc_{h}"].notna().sum() < 2000:
                continue
            model, model_date = HorizonModel(h, features).fit(train), d
            log.info("Backtest: retrained at %s on %d rows", d.date(), model.n_train)
        day = panel.xs(d, level="date", drop_level=False)
        day = day[day[f"y_exc_{h}"].notna() & eligible(day, cfg)]
        if len(day) < cfg.top_n * 3:
            continue
        pred = model.predict(day).join(day[[f"y_ret_{h}", f"y_exc_{h}", "dist_sma200", "sma200_slope"]])
        preds_all.append(pred)
        ic_rows.append({"date": d, "ic": pred["model_score"].corr(pred[f"y_exc_{h}"], method="spearman")})
        # Same rule as the live picks: only stocks the holdings review would also say to hold.
        ok = verdict_frame(pred).isin(["hold", "strong_hold"])
        top = pred[ok].nlargest(cfg.top_n, "model_score")
        picks_rows.append({
            "date": d,
            "pick_ret": top[f"y_ret_{h}"].mean(),
            "pick_exc": top[f"y_exc_{h}"].mean(),
            "univ_ret": pred[f"y_ret_{h}"].mean(),
            "bench_ret": (top[f"y_ret_{h}"] - top[f"y_exc_{h}"]).mean(),
            "hit_rate": (top[f"y_ret_{h}"] > 0).mean(),
            "beat_rate": (top[f"y_exc_{h}"] > 0).mean(),
        })

    res = pd.DataFrame(picks_rows).set_index("date")
    ic = pd.DataFrame(ic_rows).set_index("date")["ic"]
    preds = pd.concat(preds_all)
    coverage = ((preds[f"y_ret_{h}"] >= preds["q10"]) & (preds[f"y_ret_{h}"] <= preds["q90"])).mean()

    # Non-overlapping chain of holding periods for a realistic equity curve.
    chain = res.iloc[:: max(1, hd // cfg.sample_every_n_days)]
    years = hd / 252 * len(chain)
    eq_picks = float((1 + chain["pick_ret"]).prod())
    eq_bench = float((1 + chain["bench_ret"]).prod())

    # Decile spread: does a higher score really mean a higher return?
    preds["decile"] = preds.groupby(level="date")["model_score"].transform(lambda s: pd.qcut(s.rank(method="first"), 10, labels=False))
    deciles = preds.groupby("decile")[f"y_exc_{h}"].mean()

    # Does a long-term downtrend (below a falling 200-day average) change how highly rated stocks do?
    down = (preds["dist_sma200"] < -0.05) & (preds["sma200_slope"] < 0)
    top5 = preds.groupby(level="date")["model_score"].rank(ascending=False, method="first") <= cfg.top_n
    trend_check = {}
    for name, mask in {"top5": top5, "high_score": preds["model_score"] >= 0.7}.items():
        trend_check[name] = {}
        for label, sub in {"downtrend": preds[mask & down], "no_downtrend": preds[mask & ~down]}.items():
            trend_check[name][label] = {
                "n": int(len(sub)),
                "avg_excess": float(sub[f"y_exc_{h}"].mean()) if len(sub) else None,
                "avg_return": float(sub[f"y_ret_{h}"].mean()) if len(sub) else None,
                "share_positive": float((sub[f"y_ret_{h}"] > 0).mean()) if len(sub) else None,
                "p10_return": float(sub[f"y_ret_{h}"].quantile(0.1)) if len(sub) else None,
            }

    return {
        "horizon": h,
        "period_start": str(test_dates[0].date()),
        "period_end": str(test_dates[-1].date()),
        "n_weeks": int(len(res)),
        "avg_pick_return": float(res["pick_ret"].mean()),
        "avg_pick_excess": float(res["pick_exc"].mean()),
        "avg_universe_return": float(res["univ_ret"].mean()),
        "avg_benchmark_return": float(res["bench_ret"].mean()),
        "pick_hit_rate": float(res["hit_rate"].mean()),
        "pick_beat_benchmark_rate": float(res["beat_rate"].mean()),
        "weeks_top5_beat_benchmark": float((res["pick_exc"] > 0).mean()),
        "mean_ic": float(ic.mean()),
        "ic_t_stat": float(ic.mean() / (ic.std() + 1e-12) * np.sqrt(len(ic) / max(1, hd // cfg.sample_every_n_days))),
        "q10_q90_coverage": float(coverage),
        "chained_cagr_picks": float(eq_picks ** (1 / years) - 1) if years > 0 else None,
        "chained_cagr_benchmark": float(eq_bench ** (1 / years) - 1) if years > 0 else None,
        "decile_excess": {int(k): float(v) for k, v in deciles.items()},
        "verdict_stats": bucket_stats(preds, f"y_ret_{h}", f"y_exc_{h}"),
        "trend_check": trend_check,
        "weekly": [
            {"date": str(d.date()), "pick_ret": round(float(r.pick_ret), 4), "bench_ret": round(float(r.bench_ret), 4)}
            for d, r in res.iterrows()
        ],
    }
