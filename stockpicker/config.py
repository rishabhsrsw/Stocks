from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data" / "cache"
SITE_DATA_DIR = ROOT / "docs" / "data"

TRADING_DAYS_PER_MONTH = 21


@dataclass
class Config:
    # Prediction horizons in trading days. The picks are ranked on the 5-month horizon;
    # the 3- and 4-month forecasts are shown alongside.
    horizons: dict = field(default_factory=lambda: {"3m": 63, "4m": 84, "5m": 105})
    rank_horizon: str = "5m"
    top_n: int = 5
    history_years: int = 12
    benchmark: str = "^NSEI"  # Nifty 50

    # Liquidity / risk filters applied before ranking.
    min_median_traded_value_cr: float = 10.0  # ₹ crore per day, 60-day median
    max_annual_volatility: float = 0.75
    max_per_sector: int = 2

    # Weights of the final blend (the price model is the only backtested component).
    w_model: float = 0.70
    w_news: float = 0.15
    w_fundamentals: float = 0.15

    # News
    news_lookback_days: int = 21
    news_half_life_days: float = 5.0
    news_sleep_seconds: float = 0.4
    llm_shortlist: int = 20  # how many top candidates get the Claude news read (if key present)
    llm_model: str = "claude-opus-5-5"

    # Daily picks: a separate short-term model (next ~2 weeks), refreshed every trading day.
    daily_horizon_days: int = 10
    daily_top_n: int = 3
    daily_max_per_sector: int = 1
    daily_retrain_every_n_days: int = 126  # backtest retrains less often to keep the daily job quick
    daily_llm_shortlist: int = 8
    w_daily_model: float = 0.80
    w_daily_news: float = 0.20
    round_trip_cost: float = 0.004  # brokerage + STT + slippage for one buy and one sell, as a fraction

    # Holdings review beyond the Nifty 500: every NSE/BSE-listed company with enough history.
    extended_years: int = 2
    extended_min_rows: int = 252  # about a year of trading, needed for the chart features
    extended_yahoo_minutes: float = 40  # per run; stocks not fetched in time wait for the next run
    daily_yahoo_minutes: float = 15  # the daily run also fills in missing stocks, a little at a time
    stale_days: int = 10  # skip stocks that haven't traded in this many sessions

    # Model / backtest
    sample_every_n_days: int = 5  # weekly snapshots for the training panel
    retrain_every_n_days: int = 63
    backtest_years: int = 5
    random_state: int = 42
