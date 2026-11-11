"""CLI: python -m stockpicker [run|daily|backtest|demo]"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import SITE_DATA_DIR, Config
from .data import download_last_close, download_prices, synthetic_prices
from .features import build_panel
from .model import walk_forward_backtest
from .pipeline import run_daily, run_weekly
from .universe import load_market_directory, load_universe


def main() -> None:
    ap = argparse.ArgumentParser(prog="stockpicker", description=__doc__)
    ap.add_argument("command", choices=["run", "daily", "backtest", "demo"])
    ap.add_argument("--index", default="nifty200", help="nifty50|nifty100|nifty200|nifty500")
    ap.add_argument("--coverage", default="nifty500", help="wider list scored for the holdings review")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-fundamentals", action="store_true")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--force", action="store_true", help="daily: rerun even if there is no new trading day")
    ap.add_argument("--out", default=str(SITE_DATA_DIR))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = Config()

    if args.command == "demo":  # offline, synthetic market: verifies the whole pipeline end to end
        import pandas as pd
        prices = synthetic_prices()
        universe = pd.DataFrame({"symbol": prices.symbols, "name": prices.symbols,
                                 "sector": [f"Sector{i % 8}" for i in range(len(prices.symbols))]})
        cfg.min_median_traded_value_cr = 0
        res = run_weekly(prices, universe, cfg, with_news=False, with_fundamentals=False, out_dir=Path(args.out))
        run_daily(prices, universe, cfg, with_news=False, out_dir=Path(args.out), force=True)
        print(json.dumps({k: res[k] for k in ("as_of", "universe_size")} | {"picks": [p["symbol"] for p in res["picks"]]}, indent=1))
        return

    universe = load_universe(args.index)
    if args.command == "backtest":
        prices = download_prices(universe["symbol"].tolist(), cfg.benchmark, cfg.history_years, use_cache=not args.no_cache)
        days = prices.close.index
        panel = build_panel(prices, cfg.horizons, days[::-1][:: cfg.sample_every_n_days][::-1])
        bt = walk_forward_backtest(panel, days, cfg)
        bt.pop("weekly")
        print(json.dumps(bt, indent=1))
        return

    coverage = load_universe(args.coverage) if args.coverage != args.index else universe
    symbols = sorted(set(universe["symbol"]) | set(coverage["symbol"]))
    prices = download_prices(symbols, cfg.benchmark, cfg.history_years, use_cache=not args.no_cache)
    # Every other listed company (NSE + BSE-only), for the holdings review.
    directory = load_market_directory()
    rest = directory[~directory["symbol"].isin(symbols)]
    other = dict(zip(rest["symbol"], rest["yahoo"]))
    if args.command == "daily":
        extra = download_last_close(other) if other else {}
        res = run_daily(prices, universe, cfg, coverage=coverage, extra_prices=extra, with_news=not args.no_news,
                        with_backtest=not args.no_backtest, out_dir=Path(args.out), force=args.force)
        if res is None:
            print("No new trading day: nothing to update.")
            return
        print(f"\nDaily picks for {res['as_of']} (next {res['horizon_days']} trading days):")
        for p in res["picks"]:
            f = p["forecast"]["2w"]
            print(f"  {p['symbol']:<12} {p['name'][:32]:<32} median {f['q50']:+.1%}  "
                  f"range {f['q10']:+.0%}..{f['q90']:+.0%}  P(beat Nifty) {f['p_beat_nifty']:.0%}")
        return

    extended = (download_prices(list(other), cfg.benchmark, cfg.extended_years, use_cache=not args.no_cache,
                                tickers=other, min_rows=cfg.extended_min_rows) if other else None)
    res = run_weekly(prices, universe, cfg, coverage=coverage, extended=extended, directory=directory,
                     with_news=not args.no_news,
                     with_fundamentals=not args.no_fundamentals, with_backtest=not args.no_backtest, out_dir=Path(args.out))
    print(f"\nTop {len(res['picks'])} picks for the week of {res['as_of']}:")
    for p in res["picks"]:
        f = p["forecast"][cfg.rank_horizon]
        print(f"  {p['symbol']:<12} {p['name'][:32]:<32} median 5m {f['q50']:+.1%}  "
              f"range {f['q10']:+.0%}..{f['q90']:+.0%}  P(beat Nifty) {f['p_beat_nifty']:.0%}")


if __name__ == "__main__":
    main()
