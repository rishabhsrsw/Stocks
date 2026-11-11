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

## My holdings: hold or sell?

The **My holdings** tab (website, iPhone and Windows apps) lets you enter the stocks you own, with quantity,
average buy price and optionally the buy date. For each one it shows:

* a verdict: **Strong hold**, **Hold**, **Review** or **Consider selling**, with the reasons (chart strengths and concerns)
* its 3- and 5-month outlook, likely range and chance of beating Nifty
* your profit/loss, and short- vs long-term tax timing if you gave a buy date
* this week's picks as alternatives when something looks weak
* how each verdict actually performed in the backtest, so you can judge how much to trust it

Verdicts are forward-looking: the buy price is only used for P&L and tax, because what you paid doesn't change what the
stock does next. Every Nifty 500 stock is scored each week (`--coverage`), while picks still come from the Nifty 200.
Holdings are stored only on the device (with a copy/restore backup), never uploaded.

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
NSE_PICKS_FILE=../docs/data/latest.json npm start   # preview the Windows app with local data
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
  universe.py      index constituents (niftyindices.com, with built-in fallback)
  data.py          price download + cache, synthetic market for tests
  features.py      chart features and forward-return targets
  model.py         gradient-boosted ranker / classifier / quantile models, walk-forward backtest
  news.py          headlines, sentiment, red flags, optional Claude analyst
  fundamentals.py  quality overlay
  pipeline.py      weekly run -> docs/data/*.json
docs/index.html    the dashboard (static, no build step), also an installable app (manifest, sw.js, icons/)
desktop/           Windows app (Electron) that shows the same dashboard
.github/workflows/weekly-picks.yml   weekly model run
.github/workflows/windows-app.yml    builds the Windows installer into Releases
```
