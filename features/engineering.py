"""
features/engineering.py
─────────────────────────────────────────────────────────────
Computes 50+ technical indicators and prepares ML feature
matrices. Uses pandas-ta for robust TA computations.
"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import RobustScaler
import joblib
from pathlib import Path

# ta library provides all indicators we need (installs cleanly on Windows/Python 3.11)
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta library not found — using manual indicator calculations only")


# ── Trend Indicators ─────────────────────────────────────────────────────────

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"]  = close.rolling(w).mean()
        df[f"EMA_{w}"]  = close.ewm(span=w, adjust=False).mean()
    # Golden cross / death cross signals
    df["Golden_Cross"] = (df["SMA_50"] > df["SMA_200"]).astype(int)
    df["Price_vs_SMA20"] = (close - df["SMA_20"]) / df["SMA_20"]
    df["Price_vs_SMA50"] = (close - df["SMA_50"]) / df["SMA_50"]
    return df


def add_macd(df: pd.DataFrame) -> pd.DataFrame:
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    df["MACD_Cross"]  = np.where(df["MACD"] > df["MACD_Signal"], 1, -1)
    return df


# ── Momentum Indicators ──────────────────────────────────────────────────────

def add_rsi(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta  = df["Close"].diff()
    gain   = delta.clip(lower=0).rolling(window).mean()
    loss   = (-delta.clip(upper=0)).rolling(window).mean()
    rs     = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI_Oversold"]   = (df["RSI"] < 30).astype(int)
    df["RSI_Overbought"] = (df["RSI"] > 70).astype(int)
    return df


def add_stochastic(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    low_min  = df["Low"].rolling(window).min()
    high_max = df["High"].rolling(window).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    return df


def add_williams_r(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    high_max = df["High"].rolling(window).max()
    low_min  = df["Low"].rolling(window).min()
    df["Williams_R"] = -100 * (high_max - df["Close"]) / (high_max - low_min + 1e-9)
    return df


def add_roc(df: pd.DataFrame) -> pd.DataFrame:
    for p in [5, 10, 20, 60]:
        df[f"ROC_{p}"] = df["Close"].pct_change(p) * 100
    return df


def add_cci(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    tp  = (df["High"] + df["Low"] + df["Close"]) / 3
    mad = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
    df["CCI"] = (tp - tp.rolling(window).mean()) / (0.015 * mad + 1e-9)
    return df


# ── Volatility Indicators ─────────────────────────────────────────────────────

def add_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    sma  = df["Close"].rolling(window).mean()
    std  = df["Close"].rolling(window).std()
    df["BB_Upper"]  = sma + 2 * std
    df["BB_Lower"]  = sma - 2 * std
    df["BB_Middle"] = sma
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / (sma + 1e-9)
    df["BB_Pos"]    = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-9)
    df["BB_Squeeze"]= (df["BB_Width"] < df["BB_Width"].rolling(126).quantile(0.2)).astype(int)
    return df


def add_atr(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    hl  = df["High"] - df["Low"]
    hc  = (df["High"] - df["Close"].shift()).abs()
    lc  = (df["Low"]  - df["Close"].shift()).abs()
    tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    df["ATR"]          = tr.rolling(window).mean()
    df["ATR_Pct"]      = df["ATR"] / df["Close"]
    df["Volatility_20"]= df["Returns"].rolling(20).std() * np.sqrt(252)
    return df


def add_keltner_channels(df: pd.DataFrame) -> pd.DataFrame:
    ema20 = df["Close"].ewm(span=20, adjust=False).mean()
    df["KC_Upper"] = ema20 + 2 * df.get("ATR", df["High"] - df["Low"])
    df["KC_Lower"] = ema20 - 2 * df.get("ATR", df["High"] - df["Low"])
    return df


# ── Volume Indicators ─────────────────────────────────────────────────────────

def add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df["Volume_SMA_20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"]  = df["Volume"] / (df["Volume_SMA_20"] + 1e-9)

    # On-Balance Volume
    obv = [0]
    for i in range(1, len(df)):
        if df["Close"].iloc[i] > df["Close"].iloc[i - 1]:
            obv.append(obv[-1] + df["Volume"].iloc[i])
        elif df["Close"].iloc[i] < df["Close"].iloc[i - 1]:
            obv.append(obv[-1] - df["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    df["OBV"] = obv
    df["OBV_EMA"] = pd.Series(obv, index=df.index).ewm(span=20, adjust=False).mean()

    # VWAP (rolling)
    typical = (df["High"] + df["Low"] + df["Close"]) / 3
    df["VWAP"] = (typical * df["Volume"]).rolling(20).sum() / df["Volume"].rolling(20).sum()

    # Chaikin Money Flow
    mf_mult  = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"] + 1e-9)
    mf_vol   = mf_mult * df["Volume"]
    df["CMF"] = mf_vol.rolling(20).sum() / (df["Volume"].rolling(20).sum() + 1e-9)

    return df


# ── Candlestick Patterns ──────────────────────────────────────────────────────

def add_candle_patterns(df: pd.DataFrame) -> pd.DataFrame:
    body   = (df["Close"] - df["Open"]).abs()
    range_ = df["High"] - df["Low"] + 1e-9
    df["Body_Ratio"]  = body / range_
    df["Bullish_Bar"] = (df["Close"] > df["Open"]).astype(int)
    df["Upper_Wick"]  = (df["High"] - df[["Open", "Close"]].max(axis=1)) / range_
    df["Lower_Wick"]  = (df[["Open", "Close"]].min(axis=1) - df["Low"])  / range_
    # Doji
    df["Doji"] = (df["Body_Ratio"] < 0.1).astype(int)
    # Hammer
    df["Hammer"] = ((df["Lower_Wick"] > 2 * df["Body_Ratio"]) & (df["Upper_Wick"] < 0.1)).astype(int)
    return df


# ── Lag & Rolling Stats ───────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    for lag in [1, 2, 3, 5, 10, 21]:
        df[f"Close_Lag_{lag}"]   = df["Close"].shift(lag)
        df[f"Returns_Lag_{lag}"] = df["Returns"].shift(lag)
        df[f"Volume_Lag_{lag}"]  = df["Volume"].shift(lag)

    for w in [5, 10, 20, 60]:
        df[f"Close_Roll_Mean_{w}"] = df["Close"].rolling(w).mean()
        df[f"Close_Roll_Std_{w}"]  = df["Close"].rolling(w).std()
        df[f"Returns_Roll_Max_{w}"]= df["Returns"].rolling(w).max()
        df[f"Returns_Roll_Min_{w}"]= df["Returns"].rolling(w).min()
    return df


# ── Calendar Features ─────────────────────────────────────────────────────────

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Day_of_Week"]  = df.index.dayofweek          # 0=Mon, 4=Fri
    df["Month"]        = df.index.month
    df["Quarter"]      = df.index.quarter
    df["Week_of_Year"] = df.index.isocalendar().week.astype(int)
    df["Is_Month_End"] = df.index.is_month_end.astype(int)
    df["Is_Quarter_End"]= df.index.is_quarter_end.astype(int)
    # Cyclical encoding (avoids discontinuity at Dec→Jan etc.)
    df["Month_Sin"]    = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_Cos"]    = np.cos(2 * np.pi * df["Month"] / 12)
    df["DOW_Sin"]      = np.sin(2 * np.pi * df["Day_of_Week"] / 5)
    df["DOW_Cos"]      = np.cos(2 * np.pi * df["Day_of_Week"] / 5)
    return df


# ── Target Variable ───────────────────────────────────────────────────────────

def add_target(df: pd.DataFrame, horizon: int = 7) -> pd.DataFrame:
    """
    Multi-target columns:
    - Target_Return_N : forward N-day percentage return
    - Target_Direction : 1 if price higher in N days, 0 otherwise
    - Target_Price    : actual future closing price
    """
    df[f"Target_Return_{horizon}d"]    = df["Close"].shift(-horizon) / df["Close"] - 1
    df[f"Target_Direction_{horizon}d"] = (df[f"Target_Return_{horizon}d"] > 0).astype(int)
    df[f"Target_Price_{horizon}d"]     = df["Close"].shift(-horizon)
    return df


# ── Master Pipeline ───────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
    horizon: int = 7,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Apply all feature engineering steps and return enriched DataFrame.
    """
    logger.info(f"Building features for {len(df)} rows…")
    df = df.copy()

    # Make sure Returns column exists
    if "Returns" not in df.columns:
        df["Returns"] = df["Close"].pct_change()

    df = add_moving_averages(df)
    df = add_macd(df)
    df = add_rsi(df)
    df = add_stochastic(df)
    df = add_williams_r(df)
    df = add_roc(df)
    df = add_cci(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_keltner_channels(df)
    df = add_volume_indicators(df)
    df = add_candle_patterns(df)
    df = add_lag_features(df)
    df = add_calendar_features(df)
    df = add_target(df, horizon=horizon)

    if drop_na:
        before = len(df)
        df.dropna(inplace=True)
        logger.info(f"Dropped {before - len(df)} NaN rows → {len(df)} remaining")

    logger.info(f"Feature matrix: {df.shape[1]} columns")
    return df


# ── Feature Selection & Scaling ───────────────────────────────────────────────

EXCLUDED_COLS = {
    "Open", "High", "Low", "Close", "Volume",
    "Log_Returns",
}

def get_feature_columns(df: pd.DataFrame, horizon: int = 7) -> list[str]:
    """Return list of feature column names (excludes OHLCV and targets)."""
    target_cols = [c for c in df.columns if c.startswith("Target_")]
    return [
        c for c in df.columns
        if c not in EXCLUDED_COLS and c not in target_cols
    ]


def scale_features(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_path: str = "./models/saved/scaler.pkl",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit RobustScaler on train, transform all splits."""
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v  = scaler.transform(X_val)
    X_te = scaler.transform(X_test)
    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved → {scaler_path}")
    return X_tr, X_v, X_te


def make_sequences(
    X: np.ndarray,
    y: np.ndarray,
    seq_len: int = 60,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert flat feature matrix to (samples, timesteps, features) for LSTM.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.ingestion import fetch_stock_data

    df_raw = fetch_stock_data("AAPL", period="3y")
    df_feat = build_features(df_raw, horizon=7)
    feat_cols = get_feature_columns(df_feat)
    print(f"\nTotal features: {len(feat_cols)}")
    print("Sample features:", feat_cols[:10])
    print(df_feat[feat_cols].tail(3))
