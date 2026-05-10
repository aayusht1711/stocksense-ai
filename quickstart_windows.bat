@echo off
REM ─────────────────────────────────────────────────────────────
REM  StockSense AI — Quick Start (Windows)
REM  Trains AAPL and launches Streamlit dashboard
REM ─────────────────────────────────────────────────────────────
echo.
echo  Starting StockSense AI...
echo.

REM Copy .env if not exists
if not exist .env (
    copy .env.example .env
    echo  Created .env from template
)

REM Create output dirs
if not exist data\cache mkdir data\cache
if not exist models\saved mkdir models\saved

REM Train model (fast mode - XGBoost only)
echo  Training model for AAPL (XGBoost, ~2 min)...
python train.py --ticker AAPL --no-lstm --no-backtest

REM Launch dashboard
echo.
echo  Launching dashboard at http://localhost:8501
echo  Press Ctrl+C to stop
echo.
streamlit run dashboard/app.py
