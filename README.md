# NSE Weekly Picks

A self-updating site that tracks Indian stocks (Nifty 200 by default) and, every week, publishes the
**5 stocks most likely to do well over the next 5 months**, with 3-, 4- and 5-month forecasts for each.

It combines three sources:

| Source | What it does | Backtested? |
|---|---|---|
| **Charts** (price & volume) | ~35 technical features (momentum, relative strength vs Nifty, moving-average trend, 52-week high/low, RSI, MACD, volatility, drawdown, beta, volume accumulation) plus market regime. Gradient-boosted trees learn from 12 years of weekly snapshots. | ✅ walk-forward |
| **News** | Google News headlines per company, scored by a finance-tuned sentiment model. Optional: Claude reads the headlines like an analyst and flags material events. Fraud, raid, default and regulator red flags exclude a stock. | ❌ (no point-in-time history) |
| **Fundamentals** | ROE, earnings and revenue growth, margins, debt/equity and P/E from Yahoo Finance, as a quality tilt. | ❌ |

Final score = 70% model + 15% news + 15% quality, with at most 2 picks per sector and liquidity and volatility filters.

## What you get for each pick

* Median forecast for 3, 4 and 5 months, plus an **80% range** (10th–90th percentile, conformally calibrated)
* Probability of beating the Nifty 50
* Plain-language chart reasons ("Outperforming Nifty by +12% over 3 months", …)
* Latest headlines with sentiment, and risk metrics (volatility, drawdown, beta)
* A **backtest** section: average 5-month return of past weekly top-5s vs Nifty, hit rate, rank IC, decile chart
* A **live track record**: every published pick marked to market each week

## Quick start (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python -m stockpicker demo        # offline smoke test on a synthetic market
python -m stockpicker backtest    # real data: honest walk-forward backtest only
python -m stockpicker run         # real data: this week's picks -> docs/data/latest.json

python -m http.server -d docs 8000   # open http://localhost:8000
```

Options: `--index nifty500` (wider universe, slower), `--no-news`, `--no-fundamentals`, `--no-backtest`.
A full run on Nifty 200 takes about 10–20 minutes, mostly model training.

## Run it automatically every week (free, on GitHub)

1. Push this repo to GitHub.
2. **Settings → Pages**: Source = *Deploy from a branch*, Branch = your default branch, folder = `/docs`.
3. (Optional) **Settings → Secrets and variables → Actions**: add `ANTHROPIC_API_KEY` to turn on the Claude news analyst.
   It reads headlines for the top 20 candidates each week, which costs a few cents.
4. **Actions → Weekly picks → Run workflow** to run it once now. After that it runs every Saturday morning (IST)
   and commits the new picks, and the site updates itself.

## Tuning

Everything lives in `stockpicker/config.py`: horizons, number of picks, blend weights, liquidity and volatility filters,
sector cap and the news window.

## Honest limitations

* No model can predict stock prices precisely. A 5-month forecast for one stock is very noisy, which is why every
  number comes with a range. Judge the system by its **backtest and live track record**, not by any single pick.
* Backtest returns are before brokerage, STT, slippage and taxes, and use today's index constituents
  (survivorship bias flatters results somewhat).
* Yahoo Finance and Google News are free, unofficial sources. They occasionally break or rate-limit.
* This is a research tool, **not investment advice**.

## Layout

```
stockpicker/
  universe.py      index constituents (niftyindices.com, with built-in fallback)
  data.py          price download + cache, synthetic market for tests
  features.py      chart features and forward-return targets
  model.py         gradient-boosted ranker / classifier / quantile models, walk-forward backtest
  news.py          headlines, sentiment, red flags, optional Claude analyst
  fundamentals.py  quality overlay
  pipeline.py      weekly run -> docs/data/*.json
docs/index.html    the dashboard (static, no build step)
.github/workflows/weekly-picks.yml
```
