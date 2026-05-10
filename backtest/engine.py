"""
backtest/engine.py
─────────────────────────────────────────────────────────────
Walk-forward backtesting engine with Sharpe ratio, max drawdown,
win rate, and comparison to buy-and-hold benchmark.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional
from loguru import logger


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_date:  pd.Timestamp
    exit_date:   pd.Timestamp
    entry_price: float
    exit_price:  float
    direction:   str        # "long" | "short"
    shares:      float
    pnl:         float = 0.0
    pnl_pct:     float = 0.0

    def __post_init__(self):
        if self.direction == "long":
            self.pnl     = (self.exit_price - self.entry_price) * self.shares
            self.pnl_pct = (self.exit_price / self.entry_price - 1) * 100
        else:
            self.pnl     = (self.entry_price - self.exit_price) * self.shares
            self.pnl_pct = (self.entry_price / self.exit_price - 1) * 100


# ── Performance Metrics ───────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    trades:           list[Trade]
    equity_curve:     pd.Series
    returns:          pd.Series
    benchmark_returns:pd.Series
    initial_capital:  float
    final_capital:    float

    # Computed metrics (filled by compute_metrics)
    total_return_pct: float = 0.0
    benchmark_return: float = 0.0
    annualized_return:float = 0.0
    sharpe_ratio:     float = 0.0
    sortino_ratio:    float = 0.0
    max_drawdown:     float = 0.0
    calmar_ratio:     float = 0.0
    win_rate:         float = 0.0
    profit_factor:    float = 0.0
    avg_trade_pnl:    float = 0.0
    num_trades:       int   = 0
    alpha:            float = 0.0
    beta:             float = 0.0

    def compute_metrics(self) -> "BacktestResult":
        r  = self.returns
        bm = self.benchmark_returns
        ann = 252

        # Total return
        self.total_return_pct = (self.final_capital / self.initial_capital - 1) * 100
        self.benchmark_return = (1 + bm).prod() - 1

        # Annualised return
        n_days = len(r)
        self.annualized_return = ((1 + self.total_return_pct / 100) ** (ann / max(n_days, 1)) - 1) * 100

        # Sharpe (risk-free rate ≈ 5% annual = 0.0198% daily)
        rf_daily = 0.05 / ann
        excess   = r - rf_daily
        self.sharpe_ratio = (excess.mean() / (r.std() + 1e-9)) * np.sqrt(ann)

        # Sortino (downside std only)
        downside = r[r < rf_daily]
        self.sortino_ratio = (excess.mean() / (downside.std() + 1e-9)) * np.sqrt(ann)

        # Max drawdown
        cum     = (1 + r).cumprod()
        rolling_max = cum.cummax()
        drawdown    = (cum - rolling_max) / (rolling_max + 1e-9)
        self.max_drawdown = drawdown.min() * 100

        # Calmar
        self.calmar_ratio = self.annualized_return / (abs(self.max_drawdown) + 1e-9)

        # Trade stats
        if self.trades:
            wins  = [t for t in self.trades if t.pnl > 0]
            losses= [t for t in self.trades if t.pnl <= 0]
            self.num_trades  = len(self.trades)
            self.win_rate    = len(wins) / self.num_trades * 100
            gross_profit = sum(t.pnl for t in wins)
            gross_loss   = abs(sum(t.pnl for t in losses))
            self.profit_factor = gross_profit / (gross_loss + 1e-9)
            self.avg_trade_pnl = sum(t.pnl for t in self.trades) / self.num_trades

        # Alpha / Beta vs benchmark
        if len(r) == len(bm) and bm.std() > 1e-9:
            cov  = np.cov(r, bm)
            self.beta  = cov[0, 1] / (bm.var() + 1e-9)
            self.alpha = (r.mean() - rf_daily - self.beta * (bm.mean() - rf_daily)) * ann * 100

        return self

    def summary(self) -> pd.Series:
        return pd.Series({
            "Total Return (%)":    round(self.total_return_pct, 2),
            "Benchmark Return (%)":round(self.benchmark_return * 100, 2),
            "Annualised Return (%)":round(self.annualized_return, 2),
            "Sharpe Ratio":        round(self.sharpe_ratio, 3),
            "Sortino Ratio":       round(self.sortino_ratio, 3),
            "Max Drawdown (%)":    round(self.max_drawdown, 2),
            "Calmar Ratio":        round(self.calmar_ratio, 3),
            "Win Rate (%)":        round(self.win_rate, 2),
            "Profit Factor":       round(self.profit_factor, 3),
            "Num Trades":          self.num_trades,
            "Avg Trade PnL ($)":   round(self.avg_trade_pnl, 2),
            "Alpha (%)":           round(self.alpha, 3),
            "Beta":                round(self.beta, 3),
        })


# ── Strategy Signals ──────────────────────────────────────────────────────────

class SignalGenerator:
    """Converts model predictions to trading signals."""

    @staticmethod
    def threshold_signal(
        pred_returns: np.ndarray,
        buy_threshold:  float = 0.005,
        sell_threshold: float = -0.005,
    ) -> np.ndarray:
        """
        +1 = BUY, -1 = SELL, 0 = HOLD
        """
        signals = np.zeros(len(pred_returns))
        signals[pred_returns >  buy_threshold ] =  1
        signals[pred_returns <  sell_threshold] = -1
        return signals

    @staticmethod
    def probability_signal(
        buy_proba: np.ndarray,
        buy_confidence:  float = 0.60,
        sell_confidence: float = 0.40,
    ) -> np.ndarray:
        signals = np.zeros(len(buy_proba))
        signals[buy_proba >  buy_confidence ] =  1
        signals[buy_proba <  sell_confidence] = -1
        return signals

    @staticmethod
    def combined_signal(
        regression_signal: np.ndarray,
        classification_signal: np.ndarray,
    ) -> np.ndarray:
        """Only enter when both models agree."""
        combined = np.zeros(len(regression_signal))
        agree_buy  = (regression_signal == 1) & (classification_signal == 1)
        agree_sell = (regression_signal == -1) & (classification_signal == -1)
        combined[agree_buy]  =  1
        combined[agree_sell] = -1
        return combined


# ── Backtesting Engine ────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Walk-forward backtester with position sizing and risk management.
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission_pct:  float = 0.001,    # 0.1% per trade
        slippage_pct:    float = 0.0005,   # 0.05% slippage
        position_size:   float = 0.95,     # fraction of capital per trade
        stop_loss_pct:   float = 0.05,     # 5% stop-loss
        take_profit_pct: float = 0.15,     # 15% take-profit
        max_hold_days:   int   = 10,
    ):
        self.initial_capital = initial_capital
        self.commission_pct  = commission_pct
        self.slippage_pct    = slippage_pct
        self.position_size   = position_size
        self.stop_loss_pct   = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_hold_days   = max_hold_days

    def run(
        self,
        prices:  pd.Series,         # daily close prices, DatetimeIndex
        signals: np.ndarray,        # +1 buy, -1 sell, 0 hold (same length as prices)
    ) -> BacktestResult:
        """Execute the backtest and return a BacktestResult."""
        assert len(prices) == len(signals), "prices and signals must have same length"

        capital   = self.initial_capital
        position  = 0.0     # shares held (+ve long, −ve short)
        entry_px  = 0.0
        entry_date= None
        hold_days = 0

        equity     = []
        daily_ret  = []
        trades     = []

        px = prices.values
        dt = prices.index

        for i in range(len(px)):
            current_price = px[i]

            # ── Check exit conditions if in position ──────────
            if position != 0 and entry_px > 0:
                pnl_pct = (current_price - entry_px) / entry_px
                if position < 0:
                    pnl_pct = -pnl_pct

                hit_stop   = pnl_pct <= -self.stop_loss_pct
                hit_target = pnl_pct >=  self.take_profit_pct
                hit_max    = hold_days >= self.max_hold_days
                exit_sig   = (signals[i] == -1 and position > 0) or \
                             (signals[i] ==  1 and position < 0)

                if hit_stop or hit_target or hit_max or exit_sig:
                    # Exit trade
                    exit_px = current_price * (1 - self.slippage_pct if position > 0 else
                                               1 + self.slippage_pct)
                    direction = "long" if position > 0 else "short"
                    trade = Trade(
                        entry_date=entry_date, exit_date=dt[i],
                        entry_price=entry_px, exit_price=exit_px,
                        direction=direction, shares=abs(position),
                    )
                    capital += trade.pnl - abs(trade.pnl) * self.commission_pct
                    trades.append(trade)
                    position = 0.0; entry_px = 0.0; hold_days = 0

            # ── Enter new position ─────────────────────────────
            if position == 0 and signals[i] != 0:
                direction = "long" if signals[i] == 1 else "short"
                slip      = self.slippage_pct if direction == "long" else -self.slippage_pct
                entry_px  = current_price * (1 + slip)
                shares    = (capital * self.position_size) / entry_px
                position  = shares if direction == "long" else -shares
                capital  -= (entry_px * shares) * self.commission_pct
                entry_date= dt[i]
                hold_days = 0

            # ── Mark to market ─────────────────────────────────
            if position != 0:
                mtm = position * (current_price - (entry_px or current_price))
                equity_val = capital + mtm
                hold_days += 1
            else:
                equity_val = capital

            equity.append(equity_val)

        # Close any open position at end
        if position != 0 and entry_px > 0:
            exit_px = px[-1]
            direction = "long" if position > 0 else "short"
            trade = Trade(
                entry_date=entry_date, exit_date=dt[-1],
                entry_price=entry_px, exit_price=exit_px,
                direction=direction, shares=abs(position),
            )
            capital += trade.pnl
            trades.append(trade)

        equity_series = pd.Series(equity, index=dt, name="equity")
        ret_series    = equity_series.pct_change().fillna(0)
        bm_returns    = prices.pct_change().fillna(0)

        result = BacktestResult(
            trades            = trades,
            equity_curve      = equity_series,
            returns           = ret_series,
            benchmark_returns = bm_returns,
            initial_capital   = self.initial_capital,
            final_capital     = equity[-1],
        )
        return result.compute_metrics()


# ── Walk-Forward Validation ───────────────────────────────────────────────────

def walk_forward_validation(
    df_features: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    price_col: str,
    model_factory: Callable,        # callable() → model with .fit() and .predict()
    n_splits: int = 5,
    train_window: int = 252,        # 1 year
    test_window:  int = 63,         # 1 quarter
) -> list[BacktestResult]:
    """
    Run n walk-forward folds and return list of BacktestResults.
    """
    results = []
    tscv    = TimeSeriesWalkForward(n_splits=n_splits,
                                    train_size=train_window,
                                    test_size=test_window)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df_features)):
        logger.info(f"[WalkForward] Fold {fold+1}/{n_splits} | train={len(train_idx)} test={len(test_idx)}")

        train_df = df_features.iloc[train_idx]
        test_df  = df_features.iloc[test_idx]

        X_tr = train_df[feature_cols].values
        y_tr = train_df[target_col].values
        X_te = test_df[feature_cols].values
        y_te = test_df[target_col].values

        model = model_factory()
        split = int(0.85 * len(X_tr))
        model.fit(X_tr[:split], y_tr[:split], X_tr[split:], y_tr[split:])

        preds   = model.predict(X_te)
        signals = SignalGenerator.threshold_signal(preds)
        prices  = test_df[price_col]

        engine = BacktestEngine()
        result = engine.run(prices, signals)
        results.append(result)

        s = result.summary()
        logger.info(f"  Sharpe={s['Sharpe Ratio']:.2f}  Return={s['Total Return (%)']:.1f}%  "
                    f"WinRate={s['Win Rate (%)']:.1f}%  MaxDD={s['Max Drawdown (%)']:.1f}%")

    return results


class TimeSeriesWalkForward:
    def __init__(self, n_splits, train_size, test_size):
        self.n_splits   = n_splits
        self.train_size = train_size
        self.test_size  = test_size

    def split(self, X):
        n = len(X)
        step = (n - self.train_size - self.test_size) // self.n_splits
        for i in range(self.n_splits):
            start = i * step
            mid   = start + self.train_size
            end   = mid   + self.test_size
            if end > n:
                break
            yield np.arange(start, mid), np.arange(mid, end)


# ── CLI quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data.ingestion import fetch_stock_data

    df = fetch_stock_data("AAPL", period="2y")
    prices = df["Close"]

    # Simulate random model signals for demo
    np.random.seed(42)
    signals = np.random.choice([-1, 0, 0, 1], size=len(prices))

    engine = BacktestEngine(initial_capital=100_000, stop_loss_pct=0.05)
    result = engine.run(prices, signals)

    print("\n── Backtest Result ──────────────────────────────")
    print(result.summary().to_string())
    print(f"\nFirst 5 trades:")
    for t in result.trades[:5]:
        print(f"  {t.entry_date.date()} → {t.exit_date.date()} "
              f"{t.direction:5s} ${t.entry_price:.2f}→${t.exit_price:.2f}  "
              f"PnL: ${t.pnl:+.2f} ({t.pnl_pct:+.1f}%)")
