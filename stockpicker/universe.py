"""Stock universe: official NSE index constituents, with a built-in fallback list."""
from __future__ import annotations

import io
import logging
import time

import pandas as pd
import requests

log = logging.getLogger(__name__)

INDEX_URLS = {
    "nifty50": "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv",
    "nifty100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "nifty200": "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv",
    "nifty500": "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
}

# Fallback if niftyindices.com is unreachable. Symbols that have since been renamed or
# delisted are skipped automatically when no price data comes back.
FALLBACK = [
    ("RELIANCE", "Reliance Industries", "Oil Gas & Consumable Fuels"),
    ("TCS", "Tata Consultancy Services", "Information Technology"),
    ("HDFCBANK", "HDFC Bank", "Financial Services"),
    ("ICICIBANK", "ICICI Bank", "Financial Services"),
    ("INFY", "Infosys", "Information Technology"),
    ("BHARTIARTL", "Bharti Airtel", "Telecommunication"),
    ("ITC", "ITC", "Fast Moving Consumer Goods"),
    ("SBIN", "State Bank of India", "Financial Services"),
    ("LT", "Larsen & Toubro", "Construction"),
    ("HINDUNILVR", "Hindustan Unilever", "Fast Moving Consumer Goods"),
    ("BAJFINANCE", "Bajaj Finance", "Financial Services"),
    ("KOTAKBANK", "Kotak Mahindra Bank", "Financial Services"),
    ("AXISBANK", "Axis Bank", "Financial Services"),
    ("HCLTECH", "HCL Technologies", "Information Technology"),
    ("MARUTI", "Maruti Suzuki India", "Automobile and Auto Components"),
    ("SUNPHARMA", "Sun Pharmaceutical Industries", "Healthcare"),
    ("M&M", "Mahindra & Mahindra", "Automobile and Auto Components"),
    ("ASIANPAINT", "Asian Paints", "Consumer Durables"),
    ("TITAN", "Titan Company", "Consumer Durables"),
    ("ULTRACEMCO", "UltraTech Cement", "Construction Materials"),
    ("NTPC", "NTPC", "Power"),
    ("POWERGRID", "Power Grid Corporation of India", "Power"),
    ("ONGC", "Oil & Natural Gas Corporation", "Oil Gas & Consumable Fuels"),
    ("TATASTEEL", "Tata Steel", "Metals & Mining"),
    ("JSWSTEEL", "JSW Steel", "Metals & Mining"),
    ("HINDALCO", "Hindalco Industries", "Metals & Mining"),
    ("COALINDIA", "Coal India", "Oil Gas & Consumable Fuels"),
    ("ADANIENT", "Adani Enterprises", "Metals & Mining"),
    ("ADANIPORTS", "Adani Ports and SEZ", "Services"),
    ("BAJAJFINSV", "Bajaj Finserv", "Financial Services"),
    ("BAJAJ-AUTO", "Bajaj Auto", "Automobile and Auto Components"),
    ("EICHERMOT", "Eicher Motors", "Automobile and Auto Components"),
    ("HEROMOTOCO", "Hero MotoCorp", "Automobile and Auto Components"),
    ("NESTLEIND", "Nestle India", "Fast Moving Consumer Goods"),
    ("BRITANNIA", "Britannia Industries", "Fast Moving Consumer Goods"),
    ("TATACONSUM", "Tata Consumer Products", "Fast Moving Consumer Goods"),
    ("WIPRO", "Wipro", "Information Technology"),
    ("TECHM", "Tech Mahindra", "Information Technology"),
    ("LTIM", "LTIMindtree", "Information Technology"),
    ("DRREDDY", "Dr. Reddy's Laboratories", "Healthcare"),
    ("CIPLA", "Cipla", "Healthcare"),
    ("DIVISLAB", "Divi's Laboratories", "Healthcare"),
    ("APOLLOHOSP", "Apollo Hospitals", "Healthcare"),
    ("GRASIM", "Grasim Industries", "Construction Materials"),
    ("INDUSINDBK", "IndusInd Bank", "Financial Services"),
    ("SBILIFE", "SBI Life Insurance", "Financial Services"),
    ("HDFCLIFE", "HDFC Life Insurance", "Financial Services"),
    ("SHRIRAMFIN", "Shriram Finance", "Financial Services"),
    ("TRENT", "Trent", "Consumer Services"),
    ("BEL", "Bharat Electronics", "Capital Goods"),
    ("HAL", "Hindustan Aeronautics", "Capital Goods"),
    ("ZOMATO", "Zomato (Eternal)", "Consumer Services"),
    ("DMART", "Avenue Supermarts", "Consumer Services"),
    ("PIDILITIND", "Pidilite Industries", "Chemicals"),
    ("SIEMENS", "Siemens", "Capital Goods"),
    ("ABB", "ABB India", "Capital Goods"),
    ("DLF", "DLF", "Realty"),
    ("GODREJCP", "Godrej Consumer Products", "Fast Moving Consumer Goods"),
    ("DABUR", "Dabur India", "Fast Moving Consumer Goods"),
    ("HAVELLS", "Havells India", "Consumer Durables"),
    ("VEDL", "Vedanta", "Metals & Mining"),
    ("TATAPOWER", "Tata Power", "Power"),
    ("ADANIPOWER", "Adani Power", "Power"),
    ("IOC", "Indian Oil Corporation", "Oil Gas & Consumable Fuels"),
    ("BPCL", "Bharat Petroleum", "Oil Gas & Consumable Fuels"),
    ("GAIL", "GAIL (India)", "Oil Gas & Consumable Fuels"),
    ("PFC", "Power Finance Corporation", "Financial Services"),
    ("RECLTD", "REC", "Financial Services"),
    ("BANKBARODA", "Bank of Baroda", "Financial Services"),
    ("PNB", "Punjab National Bank", "Financial Services"),
    ("CANBK", "Canara Bank", "Financial Services"),
    ("CHOLAFIN", "Cholamandalam Investment and Finance", "Financial Services"),
    ("ICICIGI", "ICICI Lombard General Insurance", "Financial Services"),
    ("LICI", "Life Insurance Corporation of India", "Financial Services"),
    ("JIOFIN", "Jio Financial Services", "Financial Services"),
    ("SHREECEM", "Shree Cement", "Construction Materials"),
    ("AMBUJACEM", "Ambuja Cements", "Construction Materials"),
    ("TVSMOTOR", "TVS Motor Company", "Automobile and Auto Components"),
    ("BOSCHLTD", "Bosch", "Automobile and Auto Components"),
    ("MOTHERSON", "Samvardhana Motherson International", "Automobile and Auto Components"),
    ("TORNTPHARM", "Torrent Pharmaceuticals", "Healthcare"),
    ("LUPIN", "Lupin", "Healthcare"),
    ("ZYDUSLIFE", "Zydus Lifesciences", "Healthcare"),
    ("MAXHEALTH", "Max Healthcare Institute", "Healthcare"),
    ("INDIGO", "InterGlobe Aviation", "Services"),
    ("IRCTC", "Indian Railway Catering and Tourism", "Consumer Services"),
    ("NAUKRI", "Info Edge (India)", "Consumer Services"),
    ("PERSISTENT", "Persistent Systems", "Information Technology"),
    ("COFORGE", "Coforge", "Information Technology"),
    ("MPHASIS", "Mphasis", "Information Technology"),
    ("POLYCAB", "Polycab India", "Capital Goods"),
    ("CGPOWER", "CG Power and Industrial Solutions", "Capital Goods"),
    ("CUMMINSIND", "Cummins India", "Capital Goods"),
    ("BHEL", "Bharat Heavy Electricals", "Capital Goods"),
    ("SRF", "SRF", "Chemicals"),
    ("UPL", "UPL", "Chemicals"),
    ("PIIND", "PI Industries", "Chemicals"),
    ("DIXON", "Dixon Technologies", "Consumer Durables"),
    ("VOLTAS", "Voltas", "Consumer Durables"),
    ("GODREJPROP", "Godrej Properties", "Realty"),
    ("LODHA", "Macrotech Developers", "Realty"),
    ("INDHOTEL", "Indian Hotels Company", "Consumer Services"),
    ("MUTHOOTFIN", "Muthoot Finance", "Financial Services"),
    ("HDFCAMC", "HDFC Asset Management Company", "Financial Services"),
    ("BSE", "BSE", "Financial Services"),
    ("IDFCFIRSTB", "IDFC First Bank", "Financial Services"),
    ("FEDERALBNK", "Federal Bank", "Financial Services"),
    ("AUBANK", "AU Small Finance Bank", "Financial Services"),
    ("NMDC", "NMDC", "Metals & Mining"),
    ("SAIL", "Steel Authority of India", "Metals & Mining"),
    ("JINDALSTEL", "Jindal Steel & Power", "Metals & Mining"),
    ("HINDZINC", "Hindustan Zinc", "Metals & Mining"),
    ("MARICO", "Marico", "Fast Moving Consumer Goods"),
    ("COLPAL", "Colgate Palmolive (India)", "Fast Moving Consumer Goods"),
    ("UNITDSPR", "United Spirits", "Fast Moving Consumer Goods"),
    ("PAGEIND", "Page Industries", "Textiles"),
    ("ASHOKLEY", "Ashok Leyland", "Capital Goods"),
    ("BHARATFORG", "Bharat Forge", "Automobile and Auto Components"),
    ("MRF", "MRF", "Automobile and Auto Components"),
    ("BALKRISIND", "Balkrishna Industries", "Automobile and Auto Components"),
    ("CONCOR", "Container Corporation of India", "Services"),
    ("IRFC", "Indian Railway Finance Corporation", "Financial Services"),
    ("MAZDOCK", "Mazagon Dock Shipbuilders", "Capital Goods"),
    ("SUZLON", "Suzlon Energy", "Capital Goods"),
    ("PATANJALI", "Patanjali Foods", "Fast Moving Consumer Goods"),
    ("OBEROIRLTY", "Oberoi Realty", "Realty"),
    ("PRESTIGE", "Prestige Estates Projects", "Realty"),
    ("KPITTECH", "KPIT Technologies", "Information Technology"),
    ("TATAELXSI", "Tata Elxsi", "Information Technology"),
    ("AUROPHARMA", "Aurobindo Pharma", "Healthcare"),
    ("ALKEM", "Alkem Laboratories", "Healthcare"),
    ("SOLARINDS", "Solar Industries India", "Chemicals"),
    ("ASTRAL", "Astral", "Capital Goods"),
    ("SUPREMEIND", "Supreme Industries", "Capital Goods"),
    ("JUBLFOOD", "Jubilant FoodWorks", "Consumer Services"),
    ("TATACOMM", "Tata Communications", "Telecommunication"),
    ("INDUSTOWER", "Indus Towers", "Telecommunication"),
    ("PETRONET", "Petronet LNG", "Oil Gas & Consumable Fuels"),
    ("HINDPETRO", "Hindustan Petroleum", "Oil Gas & Consumable Fuels"),
    ("NHPC", "NHPC", "Power"),
    ("TORNTPOWER", "Torrent Power", "Power"),
    ("JSWENERGY", "JSW Energy", "Power"),
    ("ADANIGREEN", "Adani Green Energy", "Power"),
    ("LICHSGFIN", "LIC Housing Finance", "Financial Services"),
    ("MFSL", "Max Financial Services", "Financial Services"),
    ("SBICARD", "SBI Cards and Payment Services", "Financial Services"),
    ("ABCAPITAL", "Aditya Birla Capital", "Financial Services"),
    ("POLICYBZR", "PB Fintech", "Financial Services"),
    ("PAYTM", "One 97 Communications", "Financial Services"),
    ("NYKAA", "FSN E-Commerce Ventures", "Consumer Services"),
]


def fallback_universe() -> pd.DataFrame:
    return pd.DataFrame(FALLBACK, columns=["symbol", "name", "sector"])


def load_universe(index: str = "nifty200") -> pd.DataFrame:
    """Return DataFrame[symbol, name, sector] (NSE symbols without the .NS suffix).

    niftyindices.com sometimes answers with a broken page, so it is tried a few times, and the
    last good list is kept (docs/data/universe/) for when the site is down."""
    from .config import SITE_DATA_DIR
    saved = SITE_DATA_DIR / "universe" / f"{index}.csv"
    url = INDEX_URLS.get(index)
    if url:
        for attempt in range(3):
            try:
                resp = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
                resp.raise_for_status()
                df = pd.read_csv(io.StringIO(resp.text))
                df = df.rename(columns={"Symbol": "symbol", "Company Name": "name", "Industry": "sector"})
                df = df[["symbol", "name", "sector"]].dropna(subset=["symbol"]).reset_index(drop=True)
                if len(df) < 40:
                    raise ValueError(f"only {len(df)} rows")
                log.info("Loaded %d constituents of %s", len(df), index)
                try:
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(saved, index=False)
                except OSError:
                    pass
                return df
            except Exception as exc:  # network blocked, site changed, etc.
                log.warning("Could not fetch %s constituents (%s), attempt %d/3", index, exc, attempt + 1)
                time.sleep(5 * (attempt + 1))
    if saved.exists():
        df = pd.read_csv(saved)
        log.info("Using the last saved list of %s (%d constituents)", index, len(df))
        return df
    log.warning("Using the built-in list for %s", index)
    return fallback_universe()


def yahoo_symbol(symbol: str) -> str:
    return f"{symbol}.NS"


# ---------------------------------------------------------------- whole-market directory
# Every listed company, so the holdings review can take any Indian stock, not just index members.
NSE_LIST_URLS = [
    "https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv",
    "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
]
NSE_ETF_URLS = [
    "https://nsearchives.nseindia.com/content/equities/eq_etfseclist.csv",
    "https://archives.nseindia.com/content/equities/eq_etfseclist.csv",
]
NSE_SME_URL = "https://nsearchives.nseindia.com/emerge/corporates/content/SME_EQUITY_L.csv"
BSE_LIST_URL = ("https://api.bseindia.com/BseIndiaAPI/api/ListofScripData/w"
                "?Group=&Scripcode=&industry=&segment=Equity&status=Active")
# BSE's servers reject requests that don't look like a browser.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/130.0.0.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://www.bseindia.com",
    "Referer": "https://www.bseindia.com/",
}

DIRECTORY_COLUMNS = ["symbol", "name", "exchange", "yahoo", "isin", "sector", "kind"]
# kind: "stock" (scored by the model), "etf" (exchange-traded fund units, ISIN INF...) or
# "sgb" (Sovereign Gold Bonds). Funds and bonds get price, P&L and trend, not a stock verdict.


def etf_category(name: str, underlying: str = "", asset: str = "") -> str:
    """Gold / Silver / Debt / Equity / Fund, from NSE's ETF list or (for BSE) the fund's name."""
    text = f"{asset} {name}".upper()
    if "GOLD" in text:
        return "Gold"
    if "SILVER" in text:
        return "Silver"
    u = underlying.upper()
    if u == "DEBT" or any(w in text for w in ("LIQUID", "GILT", "BOND", "G-SEC", "GSEC", "SDL", "MONEY MARKET", "OVERNIGHT", "T-BILL")):
        return "Debt"
    if u == "EQUITY" or any(w in text for w in ("NIFTY", "SENSEX", "BSE", "NASDAQ", "BANK", "INDEX", "ETF", "BEES")):
        return "Equity"
    return "Fund"


def instrument_kind(isin: str, name: str = "") -> str | None:
    """'stock', 'etf', 'sgb', or None for instruments the app doesn't list (government bonds, T-bills)."""
    isin = (isin or "").upper()
    if isin.startswith("INF"):
        return "etf"
    if isin.startswith("IN00"):
        return "sgb" if "GOLD" in name.upper() else None
    return "stock"


def parse_nse_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().upper().replace("_", " ") for c in df.columns]  # the SME file uses NAME_OF_COMPANY
    df = df[~df["SYMBOL"].astype(str).str.upper().str.startswith("DUMMY")]  # NSE test securities
    if "SERIES" in df:  # main board: ordinary shares (EQ) and trade-for-trade (BE/BZ); SME file has SM/ST
        df = df[df["SERIES"].astype(str).str.strip().isin(["EQ", "BE", "BZ", "SM", "ST"])]
    out = pd.DataFrame({
        "symbol": df["SYMBOL"].astype(str).str.strip(),
        "name": df["NAME OF COMPANY"].astype(str).str.strip(),
        "exchange": "NSE",
        "isin": df.get("ISIN NUMBER", pd.Series("", index=df.index)).astype(str).str.strip(),
        "sector": "",
        "kind": "stock",
    })
    out["yahoo"] = out["symbol"] + ".NS"
    return out[DIRECTORY_COLUMNS]


def parse_nse_etf_csv(text: str, names: dict[str, str] | None = None) -> pd.DataFrame:
    """NSE's ETF list. `names` (ISIN -> name from the bhavcopy) gives fuller fund names."""
    df = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
    df.columns = [c.strip().lower().replace(" ", "") for c in df.columns]
    rows = []
    for d in df.to_dict("records"):
        sym, isin = d.get("symbol", "").strip(), d.get("isinnumber", "").strip()
        asset, under = d.get("underlyingasset", "").strip(), d.get("etfunderlying", "").strip()
        if not sym:
            continue
        cat = etf_category(d.get("securityname", ""), under, asset)
        fund = (names or {}).get(isin, "").strip()
        name = f"{fund} ({asset})" if fund else f"{asset or sym} ETF"
        rows.append({"symbol": sym, "name": name, "exchange": "NSE", "yahoo": f"{sym}.NS", "isin": isin,
                     "sector": f"ETF · {cat}", "kind": "etf"})
    return pd.DataFrame(rows, columns=DIRECTORY_COLUMNS)


def nse_sgb_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Sovereign Gold Bonds (series GB) from a parsed NSE bhavcopy."""
    if df.empty or "SctySrs" not in df:
        return pd.DataFrame(columns=DIRECTORY_COLUMNS)
    g = df[df["SctySrs"].astype(str).str.strip() == "GB"]
    return pd.DataFrame({
        "symbol": g["TckrSymb"].str.strip(), "name": "Sovereign Gold Bond " + g["FinInstrmNm"].str.strip(),
        "exchange": "NSE", "yahoo": g["TckrSymb"].str.strip() + ".NS", "isin": g["ISIN"].str.strip(),
        "sector": "Sovereign Gold Bond", "kind": "sgb",
    }, columns=DIRECTORY_COLUMNS)


def parse_bse_json(rows: list[dict], taken_isins: set[str], taken_symbols: set[str]) -> pd.DataFrame:
    """BSE companies that are not also on the NSE (matched by ISIN)."""
    out = []
    for r in rows:
        r = {k.lower(): v for k, v in r.items()}
        code, isin = str(r.get("scrip_cd") or "").strip(), str(r.get("isin_number") or "").strip()
        if not code or isin in taken_isins or str(r.get("status", "Active")).strip().lower() != "active":
            continue
        sym = str(r.get("scrip_id") or code).strip().upper()
        name = str(r.get("scrip_name") or r.get("issuer_name") or sym).strip()
        kind = instrument_kind(isin, name)
        if kind is None:
            continue
        if sym in taken_symbols:  # same ticker as a different NSE company
            sym = f"{sym}-BSE"
        taken_symbols.add(sym)
        sector = {"etf": f"ETF · {etf_category(name)}", "sgb": "Sovereign Gold Bond"}.get(
            kind, str(r.get("industry") or "").strip())
        out.append({"symbol": sym, "name": name, "exchange": "BSE", "yahoo": f"{code}.BO", "isin": isin,
                    "sector": sector, "kind": kind})
    return pd.DataFrame(out, columns=DIRECTORY_COLUMNS)


def bse_rows_from_bhavcopy(df: pd.DataFrame) -> list[dict]:
    """Rows in the same shape as the BSE scrip-list API, from a parsed bhavcopy (see bhavcopy.parse)."""
    return [{"scrip_cd": r.FinInstrmId, "scrip_name": r.FinInstrmNm, "isin_number": r.ISIN,
             "scrip_id": r.TckrSymb, "status": "Active"} for r in df.itertuples(index=False)]


def _bse_rows() -> list[dict]:
    try:
        resp = requests.get(BSE_LIST_URL, timeout=60, headers=HEADERS)
        resp.raise_for_status()
        rows = resp.json()
        return rows.get("Table", rows) if isinstance(rows, dict) else rows
    except Exception as exc:
        log.warning("BSE scrip list API failed (%s); using the bhavcopy", exc)
    from . import bhavcopy  # the latest end-of-day file lists every traded BSE stock

    return bse_rows_from_bhavcopy(bhavcopy.latest("BSE")[1])


def load_market_directory() -> pd.DataFrame:
    """All listed equities: NSE main board + SME, then BSE-only companies. Empty on failure."""
    parts = []
    for url in NSE_LIST_URLS:
        try:
            resp = requests.get(url, timeout=30, headers=HEADERS)
            resp.raise_for_status()
            parts.append(parse_nse_csv(resp.text))
            break
        except Exception as exc:
            log.warning("NSE equity list failed from %s: %s", url, exc)
    try:
        resp = requests.get(NSE_SME_URL, timeout=30, headers=HEADERS)
        resp.raise_for_status()
        parts.append(parse_nse_csv(resp.text))
    except Exception as exc:
        log.warning("NSE SME list failed: %s", exc)
    from . import bhavcopy

    nse_bhav = bhavcopy.latest("NSE")[1]
    names = dict(zip(nse_bhav.get("ISIN", pd.Series(dtype=str)).astype(str).str.strip(),
                     nse_bhav.get("FinInstrmNm", pd.Series(dtype=str)).astype(str)))
    for url in NSE_ETF_URLS:
        try:
            resp = requests.get(url, timeout=30, headers=HEADERS)
            resp.raise_for_status()
            parts.append(parse_nse_etf_csv(resp.text, names))
            break
        except Exception as exc:
            log.warning("NSE ETF list failed from %s: %s", url, exc)
    parts.append(nse_sgb_rows(nse_bhav))
    nse = pd.concat(parts).drop_duplicates("symbol") if parts else pd.DataFrame(columns=DIRECTORY_COLUMNS)
    bse = parse_bse_json(_bse_rows(), set(nse["isin"]), set(nse["symbol"]))
    directory = pd.concat([nse, bse], ignore_index=True)
    counts = directory["kind"].value_counts().to_dict()
    log.info("Market directory: %d NSE + %d BSE-only listings (%s)", len(nse), len(bse),
             ", ".join(f"{v} {k}" for k, v in counts.items()))
    return directory
