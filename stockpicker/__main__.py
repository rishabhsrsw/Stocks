"""CLI: python -m stockpicker [run|backtest|demo]"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import SITE_DATA_DIR, Config
from .data import download_prices, synthetic_prices
from .features import build_panel
from .model import walk_forward_backtest
from .pipeline import run_weekly
from .universe import load_universe


def main() -> None:
    ap = argparse.ArgumentParser(prog="stockpicker", description=__doc__)
    ap.add_argument("command", choices=["run", "backtest", "demo"])
    ap.add_argument("--index", default="nifty200", help="nifty50|nifty100|nifty200|nifty500")
    ap.add_argument("--no-news", action="store_true")
    ap.add_argument("--no-fundamentals", action="store_true")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
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
        print(json.dumps({k: res[k] for k in ("as_of", "universe_size")} | {"picks": [p["symbol"] for p in res["picks"]]}, indent=1))
        return

    universe = load_universe(args.index)
    prices = download_prices(universe["symbol"].tolist(), cfg.benchmark, cfg.history_years, use_cache=not args.no_cache)
    if args.command == "backtest":
        days = prices.close.index
        panel = build_panel(prices, cfg.horizons, days[::-1][:: cfg.sample_every_n_days][::-1])
        bt = walk_forward_backtest(panel, days, cfg)
        bt.pop("weekly")
        print(json.dumps(bt, indent=1))
        return

    res = run_weekly(prices, universe, cfg, with_news=not args.no_news, with_fundamentals=not args.no_fundamentals,
                     with_backtest=not args.no_backtest, out_dir=Path(args.out))
    print(f"\nTop {len(res['picks'])} picks for the week of {res['as_of']}:")
    for p in res["picks"]:
        f = p["forecast"][cfg.rank_horizon]
        print(f"  {p['symbol']:<12} {p['name'][:32]:<32} median 5m {f['q50']:+.1%}  "
              f"range {f['q10']:+.0%}..{f['q90']:+.0%}  P(beat Nifty) {f['p_beat_nifty']:.0%}")


if __name__ == "__main__":
    main()
