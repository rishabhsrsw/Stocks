"""News tracking: per-company headlines from Google News RSS, scored for sentiment.

Two scorers:
- a finance-tuned lexicon scorer (always on, free), and
- an optional Claude news analyst for the shortlist (when ANTHROPIC_API_KEY is set),
  which reads the headlines like an analyst would and flags material events.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import time
import urllib.parse
from dataclasses import dataclass, field

import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

log = logging.getLogger(__name__)

FINANCE_LEXICON = {
    "beat": 2.0, "beats": 2.0, "upgrade": 2.5, "upgraded": 2.5, "outperform": 2.0, "overweight": 1.5,
    "surge": 2.0, "surges": 2.0, "soars": 2.5, "rally": 1.8, "rallies": 1.8, "jumps": 1.8, "gains": 1.2,
    "record": 1.5, "bullish": 2.0, "buy": 1.2, "accumulate": 1.0, "order": 0.8, "orders": 0.8,
    "wins": 1.5, "bags": 1.5, "secures": 1.5, "expansion": 1.2, "acquisition": 0.6, "dividend": 1.0,
    "buyback": 1.8, "bonus": 1.0, "approval": 1.5, "approved": 1.5, "profit": 1.0, "growth": 1.2,
    "miss": -2.0, "misses": -2.0, "downgrade": -2.5, "downgraded": -2.5, "underperform": -2.0,
    "underweight": -1.5, "sell": -1.2, "plunge": -2.5, "plunges": -2.5, "slumps": -2.2, "tanks": -2.5,
    "falls": -1.2, "drops": -1.2, "slides": -1.2, "crash": -3.0, "bearish": -2.0, "loss": -1.5,
    "losses": -1.5, "decline": -1.2, "weak": -1.5, "probe": -2.0, "raid": -2.5, "fraud": -3.5,
    "penalty": -2.0, "fine": -1.0, "ban": -2.5, "default": -3.0, "pledge": -1.5, "resigns": -2.0,
    "lawsuit": -2.0, "warning": -1.5, "recall": -2.0, "insolvency": -3.5, "bankruptcy": -3.5,
}

RED_FLAG_PATTERNS = [
    r"\bfraud\b", r"\bforensic audit\b", r"\bauditor (resign|quits)", r"\bED (raid|searches|probe)\b",
    r"\bincome tax (raid|search)", r"\bSEBI (bans|ban|order against|barred)", r"\binsolvency\b",
    r"\bNCLT admits\b", r"\bdefault(s|ed)? on\b", r"\bCBI\b", r"\bpromoter.*pledge.*(rise|increase|invoked)",
    r"\bwilful defaulter\b", r"\bdelist", r"\bgovernance (lapse|issue|concern)",
    # US regulators and courts
    r"\bSEC (charges|investigation|probe|subpoena|sues)", r"\bDOJ (probe|investigation|charges)\b",
    r"\bFBI raid", r"\bchapter 11\b", r"\bgoing concern\b", r"\baccounting (fraud|irregularit)",
    r"\brestate(s|d|ment)?\b.*\b(financial|earnings|results)", r"\bshort[- ]seller report\b",
]

# Google News edition and wording per market.
MARKETS = {
    "IN": {"hl": "en-IN", "gl": "IN", "ceid": "IN:en", "noun": "share", "analyst": "Indian listed companies",
           "listing": "NSE-listed", "tag": "NSE"},
    "US": {"hl": "en-US", "gl": "US", "ceid": "US:en", "noun": "stock", "analyst": "US listed companies",
           "listing": "US-listed", "tag": "US"},
}


@dataclass
class NewsResult:
    symbol: str
    score: float = 0.0  # recency-weighted sentiment, -1 .. 1
    n_articles: int = 0
    red_flags: list[str] = field(default_factory=list)
    headlines: list[dict] = field(default_factory=list)
    llm: dict | None = None


_analyzer: SentimentIntensityAnalyzer | None = None


def _vader() -> SentimentIntensityAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
        _analyzer.lexicon.update(FINANCE_LEXICON)
    return _analyzer


def score_headline(text: str) -> float:
    return _vader().polarity_scores(text)["compound"]


def find_red_flags(text: str) -> list[str]:
    return [p for p in RED_FLAG_PATTERNS if re.search(p, text, flags=re.IGNORECASE)]


def _clean_name(name: str) -> str:
    return re.sub(r"\b(Ltd\.?|Limited|India|Industries|Corporation|Company|Co\.)\b", "", name).strip(" .&")


def fetch_headlines(symbol: str, name: str, lookback_days: int, market: str = "IN") -> list[dict]:
    m = MARKETS[market]
    query = f'"{_clean_name(name) or name}" OR "{symbol}" {m["noun"]} when:{lookback_days}d'
    url = "https://news.google.com/rss/search?" + urllib.parse.urlencode(
        {"q": query, "hl": m["hl"], "gl": m["gl"], "ceid": m["ceid"]})
    feed = feedparser.parse(url)
    out, seen = [], set()
    for e in feed.entries[:40]:
        title = re.sub(r"\s+-\s+[^-]+$", "", e.get("title", "")).strip()  # strip " - Publisher"
        key = title.lower()[:80]
        if not title or key in seen:
            continue
        seen.add(key)
        published = e.get("published_parsed")
        when = dt.datetime(*published[:6], tzinfo=dt.timezone.utc) if published else None
        out.append({
            "title": title,
            "source": (e.get("source") or {}).get("title", ""),
            "link": e.get("link", ""),
            "published": when.isoformat() if when else None,
        })
    return out


def aggregate(symbol: str, headlines: list[dict], half_life_days: float, now: dt.datetime | None = None) -> NewsResult:
    now = now or dt.datetime.now(dt.timezone.utc)
    res = NewsResult(symbol=symbol, n_articles=len(headlines))
    num = den = 0.0
    for h in headlines:
        s = score_headline(h["title"])
        h["sentiment"] = round(s, 3)
        age = (now - dt.datetime.fromisoformat(h["published"])).total_seconds() / 86400 if h.get("published") else 7.0
        w = 0.5 ** (max(age, 0) / half_life_days)
        num += w * s
        den += w
        res.red_flags += find_red_flags(h["title"])
    # Shrink toward 0 when there is little news, so 1 headline can't dominate.
    res.score = float(num / (den + 1.0)) if den else 0.0
    res.red_flags = sorted(set(res.red_flags))
    res.headlines = sorted(headlines, key=lambda h: h.get("published") or "", reverse=True)[:8]
    return res


def collect_news(universe, cfg) -> dict[str, NewsResult]:
    out = {}
    for i, row in enumerate(universe.itertuples(index=False)):
        try:
            hl = fetch_headlines(row.symbol, row.name, cfg.news_lookback_days, cfg.market)
        except Exception as exc:
            log.warning("News fetch failed for %s: %s", row.symbol, exc)
            hl = []
        out[row.symbol] = aggregate(row.symbol, hl, cfg.news_half_life_days)
        if (i + 1) % 25 == 0:
            log.info("News: %d/%d companies", i + 1, len(universe))
        time.sleep(cfg.news_sleep_seconds)
    return out


# ---------------------------------------------------------------- Claude news analyst

LLM_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "number", "description": "-1 (very negative) to 1 (very positive) for the stock's next 3-5 months"},
        "material_events": {"type": "array", "items": {"type": "string"}},
        "red_flag": {"type": "boolean", "description": "governance, fraud, regulatory or solvency risk"},
        "summary": {"type": "string", "description": "two sentences max"},
    },
    "required": ["sentiment", "material_events", "red_flag", "summary"],
    "additionalProperties": False,
}

LLM_SYSTEM = (
    "You are an equity analyst covering {analyst}. You are given recent news "
    "headlines about one {listing} company. Judge what they imply for the stock over the next "
    "3-5 months: earnings trajectory, orders and guidance, regulatory or governance risk, and "
    "sector tailwinds. Ignore headlines that are only about the stock price moving, generic market "
    "wraps, or that are about a different company with a similar name. If the headlines carry no "
    "real information, return sentiment near 0."
)


def llm_analyse(symbol: str, name: str, headlines: list[dict], model: str, market: str = "IN") -> dict | None:
    if not os.environ.get("ANTHROPIC_API_KEY") or not headlines:
        return None
    import anthropic

    client = anthropic.Anthropic()
    lines = "\n".join(f"- [{(h.get('published') or '')[:10]}] {h['title']} ({h.get('source', '')})" for h in headlines[:25])
    try:
        resp = client.beta.messages.create(
            model=model,
            max_tokens=4000,
            system=LLM_SYSTEM.format(**MARKETS[market]),
            thinking={"type": "adaptive"},
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": LLM_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[{"role": "user", "content": f"Company: {name} ({MARKETS[market]['tag']}: {symbol})\nHeadlines:\n{lines}"}],
        )
    except anthropic.APIError as exc:
        log.warning("Claude news analysis failed for %s: %s", symbol, exc)
        return None
    if resp.stop_reason in ("refusal", "max_tokens"):
        return None
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if not text:
        return None
    data = json.loads(text)
    data["sentiment"] = max(-1.0, min(1.0, float(data["sentiment"])))
    return data
