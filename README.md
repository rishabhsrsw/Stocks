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
A pick must also get Hold or Strong hold in the holdings review, so it is never in a long-term downtrend (below a
falling 200-day average): in the 2021–26 backtest such top-ranked stocks beat the Nifty by ~2% over 5 months,
against ~18% for the rest. The backtest applies the same rule.

## What you get for each pick

* Median forecast for 3, 4 and 5 months, plus an **80% range** (10th–90th percentile, conformally calibrated)
* Probability of beating the Nifty 50
* Plain-language chart reasons ("Outperforming Nifty by +12% over 3 months", …)
* Latest headlines with sentiment, and risk metrics (volatility, drawdown, beta)
* A **backtest** section: average 5-month return of past weekly top-5s vs Nifty, hit rate, rank IC, decile chart
* A **live track record**: every published pick marked to market each week

## Daily picks (short-term)

Every weekday evening (18:13 IST, after NSE closing prices are out) a separate model, trained to predict the
**next 2 weeks**, picks the **top 3** stocks (one per sector, no news red flags). It uses the same chart features with
more weight on recent moves, and is walk-forward backtested **after trading costs** (0.4% per round trip, set in
`config.py`). The same run refreshes closing prices, so the holdings review shows up-to-date profit/loss between weekly runs.

Short-term moves are mostly noise, so expect a much weaker edge than the 5-month model and keep these positions small.

```bash
python -m stockpicker daily          # writes docs/data/daily.json (skips if no new trading day)
python -m stockpicker daily --force  # rerun anyway
```

## My holdings: hold or sell?

The **My holdings** tab (website, iPhone and Windows apps) lets you enter the stocks you own, with quantity,
average buy price and optionally the buy date. For each one it shows:

* a verdict: **Strong hold**, **Hold**, **Review** or **Consider selling**, with the reasons (chart strengths and concerns)
* its 3- and 5-month outlook, likely range and chance of beating Nifty
* your profit/loss, and short- vs long-term tax timing if you gave a buy date
* this week's picks as alternatives when something looks weak
* how each verdict actually performed in the backtest, so you can judge how much to trust it

Verdicts are forward-looking: the buy price is only used for P&L and tax, because what you paid doesn't change what the
stock does next. Picks still come from the Nifty 200, but the holdings review covers **every NSE- and BSE-listed company**:

* Nifty 500 stocks (`--coverage`) are scored with 12 years of history, as the model was trained and tested on.
* Every other listed company (the full NSE list including SME, plus BSE-only companies, matched by ISIN) is scored each
  week from about 2 years of prices, measured against the Nifty 500. The app labels these as outside the tested range,
  caps their verdict at Hold (the backtest never covered stocks this small), and flags thinly traded stocks.
* Companies with under a year of trading (recent IPOs), or that haven't traded for two weeks, get no verdict, but still
  show price and P&L from the daily closing prices.
* The whole-market price history is built from the official NSE/BSE daily files ("bhavcopies"), not Yahoo
  (which rate-limits big downloads and doesn't carry most BSE-only companies). It is kept in the GitHub Actions
  cache (`stockpicker/history.py`): the first runs backfill 2 years, newest days first, within a time budget,
  and each daily run adds that day. Splits and bonuses are adjusted automatically: outside F&O, stocks can't
  move more than 20% in a day, so a bigger jump is a corporate action.

* ETFs (gold, silver, index, debt) and Sovereign Gold Bonds can be added too. The stock model rates company
  shares against the Nifty, so funds get price, P&L, 1-month/6-month/1-year returns, trend and volatility
  instead of a hold/sell verdict, plus the tax rules for their type.

Holdings are stored only on the device (with a copy/restore backup), never uploaded.

## US stocks

The site has an **India | US** switch (top right; the choice is remembered on the device). The US side is the same model
and pipeline pointed at a different market (`us_config()` in `config.py`):

* **Universe:** S&P 500 plus the Nasdaq-100 members not in it (about 520 stocks), from public constituent lists
  (`universe.py`), with the last good list kept in `docs/data/us/universe/`.
* **Benchmark:** S&P 500 (`^GSPC`) in place of Nifty; prices in dollars; liquidity filter $20M median daily turnover.
  Daily picks assume 0.2% round-trip trading cost (no commission, but spreads and slippage).
* **News:** Google News US edition, with US red flags (SEC/DOJ action, Chapter 11, going-concern doubts, restatements).
* **My holdings:** type a US ticker (AAPL, BRK-B…) and get the same verdict, outlook and profit/loss, in dollars. US and
  India holdings are stored separately. Unlike India there is no whole-market exchange file, so only the ~520 stocks
  above are covered (no ETFs or small caps), and the tax note gives holding periods, not rates, since the rules depend on
  where you are tax resident.
* **Data:** written to `docs/data/us/` (`latest.json`, `daily.json`, `stocks.json`, `history/`), refreshed by
  the *US weekly picks* (Saturday) and *US daily picks* (weekdays after the US close) workflows.

```bash
python -m stockpicker run --market us        # weekly picks -> docs/data/us/
python -m stockpicker daily --market us
```

The JSON field `p_beat_nifty` keeps its name for both markets (it means "probability of beating the benchmark").

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

1. Merge into your default branch. Private repos are fine; the weekly run uses about 30 of the 2,000 free Actions minutes per month.
2. (Optional) **Settings → Secrets and variables → Actions**: add `ANTHROPIC_API_KEY` to turn on the Claude news analyst.
3. **Actions → Weekly picks → Run workflow** to run it once now. After that it runs every Saturday morning (IST)
   and commits the new picks to `docs/data/`.

## Apps

### Windows

1. **Actions → Windows app → Run workflow** (it also rebuilds by itself whenever the app or dashboard changes).
2. Open **Releases** in the repo, download `NSE Weekly Picks Setup x.y.z.exe` and run it.
   The installer isn't code-signed, so Windows SmartScreen will warn you: click **More info → Run anyway**.
3. On first launch, paste a **read-only GitHub token** (the app explains how to make one:
   fine-grained token → only this repo → *Contents: Read-only*). It is stored encrypted with your Windows account.

The app reads the picks straight from the private repo, so it needs no website. It keeps the last picks for offline use.

### iPhone (free home-screen app)

Apple only allows App Store/TestFlight apps without a paid developer account, so the iPhone version is an
installable web app. It has its own icon, opens full-screen and works offline. It needs hosting that can read a private repo,
locked to you:

1. **Cloudflare Pages** (free): *Workers & Pages → Create → Pages → Connect to Git*, pick this repo,
   production branch = your default branch, framework = None, build command empty, **build output directory = `docs`**.
2. **Lock it to you**: *Zero Trust → Access → Applications → Add → Self-hosted*, domain = your `*.pages.dev` URL,
   policy *Allow → Emails → your email*. Opening the site then asks for a one-time code sent to your email.
3. On the iPhone, open the URL in **Safari → Share → Add to Home Screen**, then open it from the icon and sign in once.

Cloudflare redeploys automatically after each weekly commit. The same URL also works in any desktop browser.

### Developing the apps

```bash
cd desktop && npm install
NSE_PICKS_DIR=../docs/data npm start   # preview the Windows app with local data
```

All three (website, iPhone app, Windows app) render the same `docs/index.html`.

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
  universe.py      index constituents (niftyindices.com; S&P 500 + Nasdaq-100 for the US), built-in fallbacks
  data.py          price download + cache, synthetic market for tests
  features.py      chart features and forward-return targets
  model.py         gradient-boosted ranker / classifier / quantile models, walk-forward backtest
  news.py          headlines, sentiment, red flags, optional Claude analyst
  fundamentals.py  quality overlay
  pipeline.py      weekly run -> docs/data/*.json
docs/index.html    the dashboard (static, no build step), also an installable app (manifest, sw.js, icons/)
desktop/           Windows app (Electron) that shows the same dashboard
.github/workflows/weekly-picks.yml   weekly model run
.github/workflows/daily-picks.yml    weekday short-term picks + fresh prices
.github/workflows/us-weekly-picks.yml, us-daily-picks.yml   the same two runs for US stocks (docs/data/us/)
.github/workflows/windows-app.yml    builds the Windows installer into Releases
```
