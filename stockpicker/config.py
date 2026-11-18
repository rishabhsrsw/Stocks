from dataclasses import dataclass, field, replace
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

    # Market this run covers. Everything below that differs between India and the US lives here, so
    # the same pipeline serves both (see us_config()).
    market: str = "IN"
    benchmark_name: str = "Nifty"  # how the benchmark is named in the text shown to users
    currency: str = "₹"
    yahoo_suffix: str = ".NS"  # appended to a symbol to get its Yahoo Finance ticker
    default_exchange: str = "NSE"
    market_close_utc: str = "10:00"  # a run before this (UTC) on a trading day saw intraday prices
    thin_traded_value: float = 1e7  # below this median daily turnover, warn that a stock is thinly traded

    # Liquidity / risk filters applied before ranking.
    min_median_traded_value_cr: float = 10.0  # ₹ crore per day, 60-day median
    min_median_traded_value: float | None = None  # in the market's currency; overrides the crore setting
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
    extended_history_minutes: float = 45  # reading exchange files per weekly run; the rest waits for the next run
    daily_history_minutes: float = 15  # the daily run also fills in missing days, a little at a time
    stale_days: int = 10  # skip stocks that haven't traded in this many sessions

    # Model / backtest
    sample_every_n_days: int = 5  # weekly snapshots for the training panel
    retrain_every_n_days: int = 63
    backtest_years: int = 5
    random_state: int = 42

    @property
    def min_traded_value(self) -> float:
        """Minimum 60-day median daily turnover, in the market's currency."""
        if self.min_median_traded_value is not None:
            return self.min_median_traded_value
        return self.min_median_traded_value_cr * 1e7


def us_config(base: Config | None = None) -> Config:
    """US market: S&P 500 benchmark, dollars, no ".NS" suffix, US close is 16:00 New York (20:00-21:00 UTC)."""
    return replace(
        base or Config(),
        market="US", benchmark="^GSPC", benchmark_name="S&P 500", currency="$", yahoo_suffix="",
        default_exchange="US", market_close_utc="21:00",
        min_median_traded_value=20e6,  # $20M a day, 60-day median
        thin_traded_value=2e6,
        # Commission-free brokers, but spreads and slippage still cost something on every trade.
        round_trip_cost=0.002,
    )
