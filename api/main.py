"""
api/main.py
─────────────────────────────────────────────────────────────
FastAPI REST API for StockSense AI.
Endpoints: predict, batch predict, backtest, model info,
           ticker search, health check.

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""

import os
import sys
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

# ── Lazy imports (avoid slow TF at startup) ───────────────────────────────────
_data_cache: dict = {}
_model_cache: dict = {}


def get_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    key = f"{ticker}_{period}"
    if key not in _data_cache:
        from data.ingestion import fetch_stock_data
        from features.engineering import build_features
        df_raw  = fetch_stock_data(ticker, period=period)
        df_feat = build_features(df_raw)
        _data_cache[key] = df_feat
    return _data_cache[key]


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "StockSense AI",
    description = "ML-powered stock price prediction API with LSTM + XGBoost ensemble",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],   # lock down in production
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    ticker:          str   = Field("AAPL", description="Stock ticker symbol")
    horizon_days:    int   = Field(7,      ge=1, le=30, description="Prediction horizon (days)")
    include_sentiment: bool= Field(True,   description="Include NLP sentiment features")
    model_type:      str   = Field("ensemble", description="lstm | xgboost | ensemble")

class PredictionResponse(BaseModel):
    ticker:           str
    current_price:    float
    predicted_price:  float
    predicted_return: float
    confidence:       float
    direction:        str          # "BULLISH" | "BEARISH" | "NEUTRAL"
    signal:           str          # "BUY" | "SELL" | "HOLD"
    horizon_days:     int
    sentiment_score:  Optional[float]
    model_used:       str
    timestamp:        str
    feature_count:    int

class BacktestRequest(BaseModel):
    ticker:        str   = Field("AAPL")
    period:        str   = Field("2y")
    initial_capital: float = Field(100_000, ge=1_000)
    stop_loss_pct: float = Field(0.05)
    commission_pct:float = Field(0.001)

class TickerInfoResponse(BaseModel):
    ticker: str
    name:   str
    sector: str
    industry: str
    market_cap: int
    current_price: float

class HealthResponse(BaseModel):
    status:    str
    version:   str
    timestamp: str
    models_loaded: list[str]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "StockSense AI API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status       = "healthy",
        version      = "1.0.0",
        timestamp    = datetime.utcnow().isoformat(),
        models_loaded= list(_model_cache.keys()),
    )


@app.get("/ticker/{ticker}", response_model=TickerInfoResponse, tags=["Data"])
async def get_ticker_info(ticker: str):
    """Return company info and current price for a ticker."""
    try:
        from data.ingestion import get_ticker_info, get_current_price
        info  = get_ticker_info(ticker.upper())
        price = get_current_price(ticker.upper())
        return TickerInfoResponse(
            ticker       = ticker.upper(),
            name         = info.get("name", ticker),
            sector       = info.get("sector", "Unknown"),
            industry     = info.get("industry", "Unknown"),
            market_cap   = info.get("market_cap", 0),
            current_price= round(price, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Ticker not found: {e}")


@app.get("/history/{ticker}", tags=["Data"])
async def get_price_history(
    ticker: str,
    period: str = Query("6mo", description="1mo|3mo|6mo|1y|2y|5y"),
):
    """Return OHLCV price history as JSON."""
    try:
        from data.ingestion import fetch_stock_data
        df = fetch_stock_data(ticker.upper(), period=period)
        df = df.tail(252)   # max 1 year of data in response
        return {
            "ticker": ticker.upper(),
            "period": period,
            "count":  len(df),
            "data":   df[["Open","High","Low","Close","Volume"]].round(2).to_dict("index"),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(req: PredictionRequest):
    """
    Generate a price prediction for a given ticker.
    Uses the stacking ensemble by default (LSTM + XGBoost + LightGBM).
    """
    ticker = req.ticker.upper()
    t0     = time.time()

    try:
        from data.ingestion import fetch_stock_data, get_current_price
        from features.engineering import build_features, get_feature_columns, scale_features
        from models.xgboost_model import StackingEnsemble

        # ── Get data ──────────────────────────────────────────
        df_raw  = fetch_stock_data(ticker, period="3y")
        df_feat = build_features(df_raw, horizon=req.horizon_days)
        feat_cols = get_feature_columns(df_feat, req.horizon_days)

        current_price = float(df_feat["Close"].iloc[-1])
        target_price_col = f"Target_Price_{req.horizon_days}d"
        target_ret_col   = f"Target_Return_{req.horizon_days}d"

        X = df_feat[feat_cols].values
        y = df_feat[target_price_col].values if target_price_col in df_feat.columns else df_feat["Close"].values

        # ── Train/inference split ─────────────────────────────
        split = int(0.85 * len(X))
        X_train, X_val, X_test = X[:int(0.75*len(X))], X[int(0.75*len(X)):split], X[split:]
        y_train, y_val, y_test = y[:int(0.75*len(y))], y[int(0.75*len(y)):split], y[split:]

        # ── Use cached model or train fresh ───────────────────
        cache_key = f"{ticker}_{req.model_type}_{req.horizon_days}"
        if cache_key not in _model_cache:
            logger.info(f"Training model for {ticker} (this takes 30–60s first time)…")
            ensemble = StackingEnsemble(task="regression", ticker=ticker)
            ensemble.fit(X_train, y_train, X_val, y_val)
            _model_cache[cache_key] = ensemble
            logger.info(f"Model cached as '{cache_key}'")

        model = _model_cache[cache_key]

        # ── Predict on latest data point ──────────────────────
        X_latest = X[-1:].reshape(1, -1)
        pred_price = float(model.predict(X_latest)[0])
        pred_return = (pred_price / current_price - 1) * 100

        # ── Confidence (based on recent accuracy) ─────────────
        recent_preds = model.predict(X_test[-30:])
        recent_true  = y_test[-30:]
        mape = np.mean(np.abs((recent_true - recent_preds) / (recent_true + 1e-9))) * 100
        confidence   = max(0.3, min(0.99, 1.0 - mape / 50))

        # ── Signal ────────────────────────────────────────────
        if pred_return > 1.5:
            direction, signal = "BULLISH", "BUY"
        elif pred_return < -1.5:
            direction, signal = "BEARISH", "SELL"
        else:
            direction, signal = "NEUTRAL", "HOLD"

        elapsed = round((time.time() - t0) * 1000)
        logger.info(f"[Predict] {ticker} → ${pred_price:.2f} ({pred_return:+.1f}%) in {elapsed}ms")

        return PredictionResponse(
            ticker           = ticker,
            current_price    = round(current_price, 2),
            predicted_price  = round(pred_price, 2),
            predicted_return = round(pred_return, 3),
            confidence       = round(confidence, 3),
            direction        = direction,
            signal           = signal,
            horizon_days     = req.horizon_days,
            sentiment_score  = None,
            model_used       = req.model_type,
            timestamp        = datetime.utcnow().isoformat(),
            feature_count    = len(feat_cols),
        )

    except Exception as e:
        logger.error(f"Prediction error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/backtest", tags=["Backtest"])
async def run_backtest(req: BacktestRequest):
    """
    Run a walk-forward backtest for a ticker and return performance metrics.
    """
    ticker = req.ticker.upper()
    try:
        from data.ingestion import fetch_stock_data
        from features.engineering import build_features, get_feature_columns
        from models.xgboost_model import XGBoostStockModel
        from backtest.engine import BacktestEngine, SignalGenerator

        df_raw  = fetch_stock_data(ticker, period=req.period)
        df_feat = build_features(df_raw, horizon=7)
        feat_cols = get_feature_columns(df_feat)
        target_col = "Target_Return_7d"

        X = df_feat[feat_cols].values
        y = df_feat[target_col].values

        split = int(0.75 * len(X))
        val_split = int(0.85 * len(X))

        model = XGBoostStockModel(task="regression", ticker=ticker)
        model.fit(X[:split], y[:split], X[split:val_split], y[split:val_split])

        preds   = model.predict(X[val_split:])
        signals = SignalGenerator.threshold_signal(preds)
        prices  = df_feat["Close"].iloc[val_split:]

        engine = BacktestEngine(
            initial_capital=req.initial_capital,
            commission_pct=req.commission_pct,
            stop_loss_pct=req.stop_loss_pct,
        )
        result = engine.run(prices, signals)
        summary= result.summary().to_dict()

        return {
            "ticker":       ticker,
            "period":       req.period,
            "metrics":      summary,
            "num_trades":   len(result.trades),
            "equity_curve": result.equity_curve.round(2).to_dict(),
        }

    except Exception as e:
        logger.error(f"Backtest error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features/{ticker}", tags=["Data"])
async def get_feature_importance(ticker: str):
    """Return top feature importances from trained XGBoost model."""
    ticker = ticker.upper()
    cache_key = f"{ticker}_ensemble_7"
    if cache_key not in _model_cache:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model for {ticker}. Call /predict first."
        )
    model  = _model_cache[cache_key]
    try:
        from data.ingestion import fetch_stock_data
        from features.engineering import build_features, get_feature_columns
        df = build_features(fetch_stock_data(ticker, period="1y"))
        feat_cols = get_feature_columns(df)
        imp_df = model.xgb_model.feature_importance(feat_cols).head(20)
        return {"ticker": ticker, "top_features": imp_df.to_dict("records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/batch-predict", tags=["Prediction"])
async def batch_predict(
    tickers: str = Query("AAPL,MSFT,TSLA,NVDA,AMZN",
                         description="Comma-separated ticker list"),
    horizon: int = Query(7, ge=1, le=30),
):
    """Predict for multiple tickers in one request."""
    ticker_list = [t.strip().upper() for t in tickers.split(",")][:10]
    results = []
    for ticker in ticker_list:
        try:
            req = PredictionRequest(ticker=ticker, horizon_days=horizon)
            res = await predict(req)
            results.append(res.dict())
        except Exception as e:
            results.append({"ticker": ticker, "error": str(e)})
    return {"predictions": results, "count": len(results)}


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
