"""
features/sentiment.py
─────────────────────────────────────────────────────────────
Scrapes financial news (RSS feeds) and Reddit r/wallstreetbets,
runs FinBERT sentiment scoring, and returns a daily sentiment
time-series that can be merged into the feature matrix.
"""

import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import feedparser
import requests
import praw
from loguru import logger
from dotenv import load_dotenv
from transformers import pipeline

load_dotenv()

# ── FinBERT Model (lazy-loaded on first use) ──────────────────────────────────
_finbert = None

def get_finbert():
    global _finbert
    if _finbert is None:
        logger.info("Loading FinBERT model (first run downloads ~500MB)…")
        # Force TensorFlow backend so PyTorch/torch is NOT required
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        _finbert = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,
            framework="tf",       # use TensorFlow — no torch needed
        )
        logger.info("FinBERT loaded ✓")
    return _finbert


# ── Score a single text ───────────────────────────────────────────────────────

def score_text(text: str) -> dict:
    """
    Return FinBERT sentiment scores for a headline/snippet.

    Returns:
        {
          "positive": float,   # 0.0 – 1.0
          "negative": float,
          "neutral":  float,
          "compound": float,   # positive − negative  (−1 to +1)
          "label":    str      # dominant label
        }
    """
    if not text or len(text.strip()) < 5:
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0, "label": "neutral"}

    model = get_finbert()
    # FinBERT max 512 tokens — truncate to ~400 chars to be safe
    truncated = text[:500]
    try:
        raw = model(truncated)[0]   # list of {label, score}
        scores = {r["label"].lower(): r["score"] for r in raw}
        compound = scores.get("positive", 0) - scores.get("negative", 0)
        label = max(scores, key=scores.get)
        return {**scores, "compound": compound, "label": label}
    except Exception as e:
        logger.warning(f"FinBERT error: {e}")
        return {"positive": 0.33, "negative": 0.33, "neutral": 0.34, "compound": 0.0, "label": "neutral"}


# ── RSS News Fetcher ──────────────────────────────────────────────────────────

NEWS_FEEDS = {
    "yahoo_finance":  "https://finance.yahoo.com/rss/topfinstories",
    "reuters_business":"https://feeds.reuters.com/reuters/businessNews",
    "seeking_alpha":  "https://seekingalpha.com/market_currents.xml",
    "marketwatch":    "https://feeds.marketwatch.com/marketwatch/topstories",
    "cnbc":           "https://www.cnbc.com/id/100003114/device/rss/rss.html",
}

TICKER_NEWS_TEMPLATE = "https://finance.yahoo.com/rss/headline?s={ticker}"


def fetch_news_for_ticker(ticker: str, max_items: int = 50) -> list[dict]:
    """Fetch recent news headlines for a specific ticker via Yahoo Finance RSS."""
    url  = TICKER_NEWS_TEMPLATE.format(ticker=ticker)
    feed = feedparser.parse(url)
    items = []
    for entry in feed.entries[:max_items]:
        published = entry.get("published", "")
        try:
            pub_dt = datetime(*entry.published_parsed[:6])
        except Exception:
            pub_dt = datetime.utcnow()
        items.append({
            "source":    "yahoo_finance",
            "title":     entry.get("title", ""),
            "summary":   entry.get("summary", "")[:300],
            "published": pub_dt,
            "url":       entry.get("link", ""),
            "ticker":    ticker,
        })
    logger.info(f"[RSS] Fetched {len(items)} headlines for {ticker}")
    return items


def fetch_general_news(max_items: int = 30) -> list[dict]:
    """Fetch general financial news from multiple RSS sources."""
    all_items = []
    for source, url in NEWS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_items]:
                try:
                    pub_dt = datetime(*entry.published_parsed[:6])
                except Exception:
                    pub_dt = datetime.utcnow()
                all_items.append({
                    "source":    source,
                    "title":     entry.get("title", ""),
                    "summary":   entry.get("summary", "")[:300],
                    "published": pub_dt,
                    "url":       entry.get("link", ""),
                    "ticker":    None,
                })
        except Exception as e:
            logger.warning(f"[RSS] {source} failed: {e}")
    logger.info(f"[RSS] {len(all_items)} general news items")
    return all_items


# ── Reddit Scraper ────────────────────────────────────────────────────────────

def fetch_reddit_posts(
    ticker: str,
    subreddits: list[str] = None,
    limit: int = 50,
) -> list[dict]:
    """
    Fetch Reddit posts mentioning the ticker from financial subreddits.
    Requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env
    """
    client_id     = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent    = os.getenv("REDDIT_USER_AGENT", "StockSenseBot/1.0")

    if not client_id or client_id == "your_reddit_client_id":
        logger.warning("[Reddit] No API credentials — skipping Reddit fetch")
        return []

    if subreddits is None:
        subreddits = ["wallstreetbets", "stocks", "investing", "stockmarket"]

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )

    items = []
    for sub in subreddits:
        try:
            for post in reddit.subreddit(sub).search(ticker, limit=limit // len(subreddits), sort="new"):
                items.append({
                    "source":    f"reddit/{sub}",
                    "title":     post.title,
                    "summary":   (post.selftext or "")[:300],
                    "published": datetime.utcfromtimestamp(post.created_utc),
                    "url":       f"https://reddit.com{post.permalink}",
                    "ticker":    ticker,
                    "score":     post.score,
                    "comments":  post.num_comments,
                })
        except Exception as e:
            logger.warning(f"[Reddit] r/{sub} failed: {e}")

    logger.info(f"[Reddit] {len(items)} posts for {ticker}")
    return items


# ── Sentiment Aggregator ──────────────────────────────────────────────────────

def compute_sentiment_series(
    ticker: str,
    lookback_days: int = 365,
    use_reddit: bool = True,
) -> pd.DataFrame:
    """
    Fetch all news + Reddit for a ticker, run FinBERT, aggregate by date.

    Returns DataFrame indexed by Date with columns:
        sentiment_mean, sentiment_std, sentiment_volume, news_count,
        pct_positive, pct_negative, pct_neutral
    """
    # Gather raw items
    raw_items = fetch_news_for_ticker(ticker)
    if use_reddit:
        raw_items += fetch_reddit_posts(ticker)

    if not raw_items:
        logger.warning(f"No news/social items for {ticker} — returning zero sentiment")
        return pd.DataFrame()

    # Score each item
    records = []
    for item in raw_items:
        text  = f"{item['title']}. {item.get('summary', '')}".strip()
        score = score_text(text)
        records.append({
            "date":      item["published"].date(),
            "compound":  score["compound"],
            "positive":  score["positive"],
            "negative":  score["negative"],
            "neutral":   score["neutral"],
            "label":     score["label"],
            "source":    item["source"],
        })

    df_raw = pd.DataFrame(records)
    df_raw["date"] = pd.to_datetime(df_raw["date"])

    # Aggregate to daily
    agg = df_raw.groupby("date").agg(
        sentiment_mean   = ("compound",  "mean"),
        sentiment_std    = ("compound",  "std"),
        news_count       = ("compound",  "count"),
        pct_positive     = ("label",     lambda x: (x == "positive").mean()),
        pct_negative     = ("label",     lambda x: (x == "negative").mean()),
        pct_neutral      = ("label",     lambda x: (x == "neutral").mean()),
    ).reset_index().rename(columns={"date": "Date"})
    agg.set_index("Date", inplace=True)
    agg["sentiment_std"].fillna(0, inplace=True)

    # Rolling 3-day smoothed sentiment
    agg["sentiment_3d"] = agg["sentiment_mean"].rolling(3, min_periods=1).mean()
    agg["sentiment_7d"] = agg["sentiment_mean"].rolling(7, min_periods=1).mean()

    # Sentiment momentum (rate of change)
    agg["sentiment_delta"] = agg["sentiment_mean"].diff()

    logger.info(f"[Sentiment] {len(agg)} daily records for {ticker}")
    return agg


def merge_sentiment(df_features: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Merge daily sentiment scores into the main feature DataFrame.
    Missing sentiment days are forward-filled (no future leakage).
    """
    sent_df = compute_sentiment_series(ticker)
    if sent_df.empty:
        logger.warning("Sentiment unavailable — adding zero columns")
        for col in ["sentiment_mean", "sentiment_3d", "sentiment_7d",
                    "sentiment_delta", "news_count", "pct_positive", "pct_negative"]:
            df_features[col] = 0.0
        return df_features

    df_merged = df_features.join(sent_df, how="left")
    sent_cols = sent_df.columns.tolist()
    df_merged[sent_cols] = df_merged[sent_cols].fillna(method="ffill").fillna(0)
    logger.info(f"Sentiment merged — {len(sent_cols)} new columns")
    return df_merged


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_texts = [
        "Apple smashes earnings expectations, stock surges 8% after hours",
        "Fed raises interest rates again amid recession fears",
        "AAPL announces new iPhone 17 product line with AI features",
        "Supply chain disruptions may impact tech sector this quarter",
    ]
    print("FinBERT Sentiment Demo:\n")
    for text in sample_texts:
        s = score_text(text)
        bar = "+" * int(abs(s["compound"]) * 20)
        sign = "↑" if s["compound"] > 0 else "↓"
        print(f"  {sign} [{s['label']:<8}] {s['compound']:+.3f}  {bar}")
        print(f"    {text[:70]}\n")
