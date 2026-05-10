"""
data/ingestion.py
─────────────────────────────────────────────────────────────
Fetches OHLCV stock data from yFinance and Alpha Vantage.
Caches locally to avoid redundant API calls.
"""

import os
import json
import time
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "./data/cache"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")


# ── Main Data Loader ─────────────────────────────────────────────────────────

def fetch_stock_data(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    use_cache: bool = True,
    cache_hours: int = 4,
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a given ticker from yFinance.

    Args:
        ticker:      Stock symbol e.g. 'AAPL', 'TSLA'
        period:      History length: '1mo','3mo','6mo','1y','2y','5y','10y','max'
        interval:    Bar size: '1d','1wk','1mo'  (use '1h' for intraday)
        use_cache:   Read from disk cache if fresh enough
        cache_hours: Max age (hours) before re-fetching

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Returns, Log_Returns
    """
    cache_key = hashlib.md5(f"{ticker}_{period}_{interval}".encode()).hexdigest()[:8]
    cache_file = DATA_DIR / f"{ticker}_{cache_key}.parquet"

    # ── Cache hit ────────────────────────────────────────────
    if use_cache and cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < cache_hours:
            logger.info(f"[Cache] Loading {ticker} from {cache_file.name}")
            df = pd.read_parquet(cache_file)
            logger.info(f"[Cache] {len(df)} rows, {df.index[0].date()} → {df.index[-1].date()}")
            return df

    # ── Fresh fetch ──────────────────────────────────────────
    logger.info(f"[yFinance] Downloading {ticker} | period={period} | interval={interval}")
    raw = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(f"No data returned for ticker '{ticker}'. Check the symbol.")

    # Flatten MultiIndex columns if present (yfinance ≥0.2.31)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.dropna(inplace=True)
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # ── Derived columns ──────────────────────────────────────
    df["Returns"]     = df["Close"].pct_change()
    df["Log_Returns"] = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"]  = df["Returns"].rolling(20).std() * np.sqrt(252)

    df.dropna(inplace=True)
    df.to_parquet(cache_file)
    logger.info(f"[yFinance] Saved {len(df)} rows to cache → {cache_file.name}")
    return df


def fetch_multiple_tickers(
    tickers: list[str],
    period: str = "2y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch multiple tickers and return a dict of DataFrames."""
    result = {}
    for t in tickers:
        try:
            result[t] = fetch_stock_data(t, period=period, interval=interval)
            logger.info(f"✓ {t}: {len(result[t])} rows")
        except Exception as e:
            logger.error(f"✗ {t}: {e}")
    return result


# ── Alpha Vantage ─────────────────────────────────────────────────────────────

def fetch_intraday_alpha_vantage(
    ticker: str,
    interval: str = "60min",
) -> Optional[pd.DataFrame]:
    """
    Fetch intraday OHLCV from Alpha Vantage (free tier: 25 req/day).
    interval: '1min','5min','15min','30min','60min'
    """
    if ALPHA_VANTAGE_KEY == "demo":
        logger.warning("[AV] No API key set. Using yFinance fallback.")
        return None

    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY"
        f"&symbol={ticker}"
        f"&interval={interval}"
        f"&outputsize=full"
        f"&apikey={ALPHA_VANTAGE_KEY}"
    )
    logger.info(f"[AlphaVantage] Fetching {ticker} {interval} intraday…")
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    key = f"Time Series ({interval})"
    if key not in data:
        logger.error(f"[AV] Unexpected response: {list(data.keys())}")
        return None

    df = pd.DataFrame(data[key]).T
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df.columns = ["Open", "High", "Low", "Close", "Volume"]
    df = df.astype(float)
    return df


# ── Market Meta ──────────────────────────────────────────────────────────────

def get_ticker_info(ticker: str) -> dict:
    """Return company name, sector, market cap etc."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        return {
            "name":        info.get("longName", ticker),
            "sector":      info.get("sector", "Unknown"),
            "industry":    info.get("industry", "Unknown"),
            "market_cap":  info.get("marketCap", 0),
            "currency":    info.get("currency", "USD"),
            "exchange":    info.get("exchange", ""),
            "website":     info.get("website", ""),
            "description": info.get("longBusinessSummary", "")[:300],
        }
    except Exception as e:
        logger.warning(f"Could not fetch info for {ticker}: {e}")
        return {"name": ticker}


def get_current_price(ticker: str) -> float:
    """Return the latest closing price."""
    df = fetch_stock_data(ticker, period="5d", use_cache=False)
    return float(df["Close"].iloc[-1])


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = fetch_stock_data("AAPL", period="2y")
    print(df.tail(5))
    print(f"\nShape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    info = get_ticker_info("AAPL")
    print(f"\n{info['name']} | {info['sector']}")
