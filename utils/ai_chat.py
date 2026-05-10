"""
utils/ai_chat.py
─────────────────────────────────────────────────────────────
AI Chat Assistant powered by Claude API (claude-sonnet-4-20250514).
Answers stock questions with live price data + technical indicators
as context. Users can ask anything in plain English.

Setup: Add to .env:
  ANTHROPIC_API_KEY=sk-ant-...
  Get key: console.anthropic.com
"""

import os
import json
import httpx
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from utils.cloud_config import get_anthropic_key
ANTHROPIC_API_KEY = get_anthropic_key()
CLAUDE_MODEL      = "claude-sonnet-4-20250514"
API_URL           = "https://api.anthropic.com/v1/messages"


# ── Build stock context for Claude ────────────────────────────────────────────

def build_stock_context(ticker: str, df_price, prediction: dict = None) -> str:
    """Build a rich context string about the stock to send to Claude."""
    cur   = float(df_price["Close"].iloc[-1])
    prev  = float(df_price["Close"].iloc[-2])
    chg   = (cur / prev - 1) * 100
    high  = float(df_price["High"].iloc[-1])
    low   = float(df_price["Low"].iloc[-1])
    vol   = int(df_price["Volume"].iloc[-1])
    h52   = float(df_price["Close"].rolling(min(252, len(df_price))).max().iloc[-1])
    l52   = float(df_price["Close"].rolling(min(252, len(df_price))).min().iloc[-1])
    avg_vol = int(df_price["Volume"].rolling(20).mean().iloc[-1])

    # Technical indicators
    close = df_price["Close"]
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = float(100 - 100 / (1 + gain.iloc[-1] / (loss.iloc[-1] + 1e-9)))

    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd  = float((ema12 - ema26).iloc[-1])
    sig   = float((ema12 - ema26).ewm(span=9).mean().iloc[-1])

    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(df_price) >= 50 else None
    sma200= float(close.rolling(200).mean().iloc[-1]) if len(df_price) >= 200 else None

    std20 = float(close.rolling(20).std().iloc[-1])
    bb_up = sma20 + 2 * std20
    bb_dn = sma20 - 2 * std20
    bb_pos= (cur - bb_dn) / (bb_up - bb_dn + 1e-9) * 100

    volatility = float(close.pct_change().rolling(20).std().iloc[-1] * (252**0.5) * 100)

    # 5-day return
    ret_5d = (cur / float(close.iloc[-6]) - 1) * 100 if len(df_price) >= 6 else 0
    ret_1m = (cur / float(close.iloc[-22]) - 1) * 100 if len(df_price) >= 22 else 0

    ctx = f"""
REAL-TIME STOCK DATA FOR {ticker}
{'='*50}
Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}

PRICE DATA:
- Current Price: ${cur:,.2f}
- Day Change: {chg:+.2f}%
- Today's Range: ${low:.2f} – ${high:.2f}
- 52-Week High: ${h52:,.2f} (current is {cur/h52*100:.0f}% of peak)
- 52-Week Low:  ${l52:,.2f} (current is +{(cur/l52-1)*100:.0f}% above)
- 5-Day Return: {ret_5d:+.2f}%
- 1-Month Return: {ret_1m:+.2f}%

VOLUME:
- Today's Volume: {vol/1e6:.1f}M shares
- 20-Day Avg Volume: {avg_vol/1e6:.1f}M shares
- Volume Ratio: {vol/avg_vol:.1f}x average {'(HIGH)' if vol/avg_vol > 1.5 else '(normal)'}

TECHNICAL INDICATORS:
- RSI (14): {rsi:.1f} {'→ OVERBOUGHT' if rsi > 70 else ('→ OVERSOLD' if rsi < 30 else '→ neutral')}
- MACD: {macd:.3f} | Signal: {sig:.3f} | {'→ BULLISH crossover' if macd > sig else '→ BEARISH crossover'}
- SMA 20: ${sma20:.2f} (price is {'above' if cur > sma20 else 'below'})
{f'- SMA 50: ${sma50:.2f} (price is {"above" if cur > sma50 else "below"})' if sma50 else ''}
{f'- SMA 200: ${sma200:.2f} (price is {"above" if cur > sma200 else "below"})' if sma200 else ''}
- Bollinger Bands: Upper ${bb_up:.2f} | Lower ${bb_dn:.2f} | Position: {bb_pos:.0f}%
- Annualised Volatility: {volatility:.1f}%
"""

    if prediction:
        sig_label = prediction.get("signal", "N/A")
        pred_p    = prediction.get("pred_price", 0)
        pred_r    = prediction.get("pred_ret", 0)
        conf      = prediction.get("confidence", 0)
        horizon   = prediction.get("horizon", 7)
        ctx += f"""
ML PREDICTION (from StockSense AI ensemble model):
- Predicted Price ({horizon} days): ${pred_p:,.2f}
- Expected Return: {pred_r:+.2f}%
- Model Signal: {sig_label}
- Model Confidence: {conf*100:.1f}%
- Models used: XGBoost + LightGBM + Stacking Ensemble
"""
    return ctx


SYSTEM_PROMPT = """You are StockSense AI Assistant — an expert financial analyst embedded in a stock prediction application.

You have access to real-time stock data, technical indicators, and ML model predictions provided in each message.

Your role:
1. Answer questions about stocks clearly and concisely
2. Explain technical indicators in plain English (not jargon)
3. Interpret the ML prediction signals
4. Provide balanced analysis — not just hype
5. Always remind users this is educational content, not financial advice

Your personality:
- Expert but approachable — like a knowledgeable friend who works in finance
- Data-driven — always reference the actual numbers provided
- Honest about uncertainty — ML models are not perfect
- Concise — give clear answers, not essays

Formatting:
- Use bullet points for lists
- Bold key numbers and signals
- Keep responses under 200 words unless asked for detail
- End with one actionable insight when relevant

Always end with: "⚠️ Not financial advice — always do your own research."
"""


# ── Main chat function ────────────────────────────────────────────────────────

def chat_with_ai(
    user_message:   str,
    conversation_history: list,
    stock_context:  str,
    max_tokens:     int = 600,
) -> tuple[str, list]:
    """
    Send a message to Claude API with stock context.

    Args:
        user_message:         The user's question
        conversation_history: List of {"role": "user/assistant", "content": "..."} dicts
        stock_context:        Real-time stock data string
        max_tokens:           Max response length

    Returns:
        (assistant_reply, updated_history)
    """
    if not ANTHROPIC_API_KEY:
        reply = "⚠️ AI Chat not configured. Add ANTHROPIC_API_KEY to your .env file.\n\nGet a free API key at: console.anthropic.com"
        return reply, conversation_history

    # Build messages with context injected into first user message
    messages = []
    for i, msg in enumerate(conversation_history):
        messages.append(msg)

    # Add context to the current message
    full_message = f"{stock_context}\n\nUSER QUESTION: {user_message}"
    messages.append({"role": "user", "content": full_message})

    headers = {
        "x-api-key":         ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    payload = {
        "model":      CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system":     SYSTEM_PROMPT,
        "messages":   messages,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(API_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data  = resp.json()
            reply = data["content"][0]["text"]

        # Update history (store original message, not the context-injected one)
        updated_history = conversation_history + [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": reply},
        ]
        # Keep last 10 turns to avoid token overflow
        if len(updated_history) > 20:
            updated_history = updated_history[-20:]

        return reply, updated_history

    except httpx.HTTPStatusError as e:
        logger.error(f"Claude API HTTP error: {e.response.status_code} — {e.response.text}")
        if e.response.status_code == 401:
            return "❌ Invalid API key. Check ANTHROPIC_API_KEY in .env", conversation_history
        return f"❌ API error {e.response.status_code}. Try again.", conversation_history
    except Exception as e:
        logger.error(f"Claude API error: {e}")
        return f"❌ Connection error: {e}", conversation_history


# ── Suggested questions ───────────────────────────────────────────────────────

SUGGESTED_QUESTIONS = [
    "What does the current RSI tell me about this stock?",
    "Should I be worried about the current price trend?",
    "Explain the MACD signal in simple terms",
    "What's the risk level of this stock right now?",
    "What could cause this stock to go up this week?",
    "Is the current volume significant?",
    "Explain the Bollinger Band position",
    "What does the ML prediction mean for me?",
    "Compare this stock to its 52-week performance",
    "What technical signals are bullish right now?",
]


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not ANTHROPIC_API_KEY:
        print("Add ANTHROPIC_API_KEY to .env first")
        print("Get key: console.anthropic.com")
    else:
        import sys
        sys.path.insert(0, ".")
        from data.ingestion import fetch_stock_data
        df = fetch_stock_data("AAPL", period="6mo")
        ctx = build_stock_context("AAPL", df)
        print("Context built. Testing chat…\n")
        reply, hist = chat_with_ai("What does the RSI tell me right now?", [], ctx)
        print(f"Assistant: {reply}")
