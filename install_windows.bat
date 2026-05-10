@echo off
REM ─────────────────────────────────────────────────────────────
REM  StockSense AI — Windows Install Helper
REM  Run this script instead of: pip install -r requirements.txt
REM  Usage: Double-click install_windows.bat  OR  run in CMD
REM ─────────────────────────────────────────────────────────────

echo.
echo  ====================================================
echo   StockSense AI — Windows Setup
echo  ====================================================
echo.

REM Upgrade pip first
echo [1/6] Upgrading pip...
python -m pip install --upgrade pip

REM Core data stack
echo [2/6] Installing data + ML packages...
pip install "yfinance>=0.2.36" "pandas>=2.0.0" "numpy>=1.24.0,<2.0.0" "ta>=0.11.0"
pip install "scikit-learn>=1.3.0" "xgboost>=2.0.0" "lightgbm>=4.0.0"
pip install "joblib>=1.3.0" "loguru>=0.7.0" "tqdm>=4.65.0" "pyarrow>=12.0.0"

REM TensorFlow (Windows wheel available — no build needed)
echo [4/6] Installing TensorFlow...
pip install "tensorflow>=2.13.0" "keras>=2.13.0"

REM NLP — transformers with TF backend (NO torch/pytorch needed)
echo [5/6] Installing NLP packages...
pip install "transformers>=4.35.0" "sentencepiece>=0.1.99"
pip install "praw>=7.7.0" "beautifulsoup4>=4.12.0" "feedparser>=6.0.10" "requests>=2.28.0"

REM API + Dashboard
echo [6/6] Installing API + Dashboard...
pip install "fastapi>=0.100.0" "uvicorn[standard]>=0.22.0" "pydantic>=2.0.0"
pip install "python-dotenv>=1.0.0" "httpx>=0.24.0" "sqlalchemy>=2.0.0"
pip install "streamlit>=1.28.0" "plotly>=5.18.0" "schedule>=1.2.0"

echo.
echo  ====================================================
echo   Installation complete!
echo.
echo   Next steps:
echo     1. copy .env.example .env
echo     2. python train.py --ticker AAPL --no-lstm
echo     3. streamlit run dashboard/app.py
echo  ====================================================
echo.
pause

REM ── New additions: Claude API + PyTorch ──
echo [7/7] Installing Claude API + PyTorch...
pip install "anthropic>=0.25.0"
echo.
echo  PyTorch must be installed separately:
echo  CPU version (recommended):
echo    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo.
echo  For GPU (if you have NVIDIA):
echo    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.
