"""
train.py
─────────────────────────────────────────────────────────────
End-to-end training script for StockSense AI.
Trains LSTM + XGBoost ensemble, runs backtest, saves all models.

Usage:
    python train.py                      # train AAPL with defaults
    python train.py --ticker TSLA        # train single ticker
    python train.py --ticker AAPL MSFT NVDA   # train multiple
    python train.py --tune               # enable Optuna tuning (~10 min)
    python train.py --no-lstm            # skip LSTM (faster)
"""

import argparse
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent))


def parse_args():
    p = argparse.ArgumentParser(description="StockSense AI — Training Pipeline")
    p.add_argument("--ticker",   nargs="+", default=["AAPL"], help="Ticker(s) to train")
    p.add_argument("--period",   default="5y",   help="Data history period")
    p.add_argument("--horizon",  type=int, default=7, help="Prediction horizon (days)")
    p.add_argument("--seq-len",  type=int, default=60, help="LSTM sequence length")
    p.add_argument("--epochs",   type=int, default=100, help="Max LSTM epochs")
    p.add_argument("--tune",     action="store_true", help="Run Optuna hyperparameter tuning")
    p.add_argument("--no-lstm",  action="store_true", help="Skip LSTM training")
    p.add_argument("--no-backtest", action="store_true", help="Skip backtest")
    return p.parse_args()


def train_single_ticker(
    ticker: str,
    period: str,
    horizon: int,
    seq_len: int,
    epochs: int,
    tune: bool,
    skip_lstm: bool,
    skip_backtest: bool,
):
    logger.info(f"\n{'='*60}")
    logger.info(f"  Training pipeline for {ticker}")
    logger.info(f"{'='*60}\n")

    # ── 1. Data Ingestion ──────────────────────────────────────
    logger.info("Step 1/6 — Data Ingestion")
    from data.ingestion import fetch_stock_data, get_ticker_info
    df_raw = fetch_stock_data(ticker, period=period)
    info   = get_ticker_info(ticker)
    logger.info(f"  {info.get('name', ticker)} | {info.get('sector', '')} | {len(df_raw)} rows")

    # ── 2. Feature Engineering ─────────────────────────────────
    logger.info("Step 2/6 — Feature Engineering")
    from features.engineering import (
        build_features, get_feature_columns, scale_features, make_sequences
    )
    df_feat   = build_features(df_raw, horizon=horizon, drop_na=True)
    feat_cols = get_feature_columns(df_feat, horizon)
    logger.info(f"  {len(feat_cols)} features × {len(df_feat)} rows")

    # ── 3. Train / Val / Test Split (time-ordered) ─────────────
    logger.info("Step 3/6 — Data Splitting")
    X = df_feat[feat_cols].values.astype(np.float32)
    y_price  = df_feat[f"Target_Price_{horizon}d"].values.astype(np.float32)
    y_return = df_feat[f"Target_Return_{horizon}d"].values.astype(np.float32)
    y_dir    = df_feat[f"Target_Direction_{horizon}d"].values.astype(np.float32)

    n       = len(X)
    tr_end  = int(0.70 * n)
    vl_end  = int(0.85 * n)

    X_train, X_val, X_test = X[:tr_end], X[tr_end:vl_end], X[vl_end:]
    yp_tr,   yp_v,  yp_te  = y_price[:tr_end], y_price[tr_end:vl_end], y_price[vl_end:]
    yr_tr,   yr_v,  yr_te  = y_return[:tr_end], y_return[tr_end:vl_end], y_return[vl_end:]
    yd_tr,   yd_v,  yd_te  = y_dir[:tr_end], y_dir[tr_end:vl_end], y_dir[vl_end:]

    logger.info(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # Scale
    X_tr_s, X_vl_s, X_te_s = scale_features(
        pd.DataFrame(X_train, columns=feat_cols),
        pd.DataFrame(X_val,   columns=feat_cols),
        pd.DataFrame(X_test,  columns=feat_cols),
    )

    # ── 4. XGBoost + Ensemble ──────────────────────────────────
    logger.info("Step 4/6 — XGBoost + Stacking Ensemble")
    from models.xgboost_model import StackingEnsemble, evaluate_model

    ensemble_reg = StackingEnsemble(task="regression", ticker=ticker)
    ensemble_reg.fit(X_tr_s, yp_tr, X_vl_s, yp_v, tune=tune)

    ensemble_cls = StackingEnsemble(task="classification", ticker=ticker)
    ensemble_cls.fit(X_tr_s, yd_tr, X_vl_s, yd_v)

    reg_metrics = evaluate_model(yp_te, ensemble_reg.predict(X_te_s), "regression")
    cls_metrics = evaluate_model(yd_te, ensemble_cls.predict_proba(X_te_s)[:, 1], "classification")

    logger.info(f"  Regression  → MAE=${reg_metrics['MAE']:.2f}  RMSE=${reg_metrics['RMSE']:.2f}  "
                f"R²={reg_metrics['R2']:.3f}  DirAcc={reg_metrics['DirAcc']:.1%}")
    logger.info(f"  Classification → Acc={cls_metrics['Accuracy']:.1%}  AUC={cls_metrics['AUC']:.3f}")

    ensemble_reg.save()
    ensemble_cls.save()

    lstm_reg_preds = None

    # ── 5. LSTM ────────────────────────────────────────────────
    if not skip_lstm:
        logger.info("Step 5/6 — LSTM with Attention")
        from models.lstm_model import train_lstm, evaluate_lstm, save_lstm

        X_seq_tr, y_seq_tr = make_sequences(X_tr_s, yp_tr, seq_len)
        X_seq_vl, y_seq_vl = make_sequences(
            np.vstack([X_tr_s[-seq_len:], X_vl_s]), yp_v, seq_len
        )
        X_seq_te, y_seq_te = make_sequences(
            np.vstack([X_vl_s[-seq_len:], X_te_s]), yp_te, seq_len
        )

        lstm_model, history = train_lstm(
            X_seq_tr, y_seq_tr,
            X_seq_vl, y_seq_vl,
            task="regression", ticker=ticker, epochs=epochs,
        )
        lstm_metrics = evaluate_lstm(lstm_model, X_seq_te, y_seq_te, task="regression")
        logger.info(f"  LSTM → MAE=${lstm_metrics['MAE']:.2f}  DirAcc={lstm_metrics['Directional_Accuracy']:.1%}")
        save_lstm(lstm_model, ticker, "regression")
    else:
        logger.info("Step 5/6 — LSTM skipped (--no-lstm)")

    # ── 6. Backtest ────────────────────────────────────────────
    if not skip_backtest:
        logger.info("Step 6/6 — Backtesting")
        from backtest.engine import BacktestEngine, SignalGenerator

        preds   = ensemble_reg.predict(X_te_s)
        signals = SignalGenerator.threshold_signal(preds)
        prices  = df_feat["Close"].iloc[vl_end:]

        engine = BacktestEngine(initial_capital=100_000, stop_loss_pct=0.05)
        result = engine.run(prices, signals)
        summary = result.summary()

        logger.info("  ── Backtest Summary ──────────────────────────────")
        for k, v in summary.items():
            logger.info(f"    {k:30s}: {v}")
    else:
        logger.info("Step 6/6 — Backtest skipped (--no-backtest)")

    logger.info(f"\n✅ Pipeline complete for {ticker}\n")


def main():
    args = parse_args()
    os.makedirs("./models/saved", exist_ok=True)
    os.makedirs("./data/cache",   exist_ok=True)

    for ticker in args.ticker:
        train_single_ticker(
            ticker        = ticker.upper(),
            period        = args.period,
            horizon       = args.horizon,
            seq_len       = args.seq_len,
            epochs        = args.epochs,
            tune          = args.tune,
            skip_lstm     = args.no_lstm,
            skip_backtest = args.no_backtest,
        )

    logger.info("🎉 All tickers trained successfully!")


if __name__ == "__main__":
    main()
