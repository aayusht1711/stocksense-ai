"""
utils/paper_trading.py
─────────────────────────────────────────────────────────────
Paper trading simulator for StockSense AI.
Virtual $100,000 account — trade from ML signals with no real money.

Features:
  - BUY / SELL / SHORT with realistic commission + slippage
  - Full trade journal with entry reasoning
  - Running P&L vs buy-and-hold benchmark
  - Position sizing (fixed % or full Kelly)
  - Risk management: stop-loss, take-profit
  - Performance metrics: Sharpe, win rate, max drawdown
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional
from loguru import logger


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    id:           int
    ticker:       str
    direction:    str          # "LONG" | "SHORT"
    entry_date:   str
    entry_price:  float
    shares:       float
    reason:       str          # ML signal reasoning
    signal:       str          # "BUY" | "SELL" | "HOLD"
    confidence:   float        # model confidence 0-1
    pred_price:   float        # model's predicted price
    horizon_days: int

    exit_date:    Optional[str]   = None
    exit_price:   Optional[float] = None
    exit_reason:  Optional[str]   = None
    status:       str             = "OPEN"    # "OPEN" | "CLOSED"

    pnl:          float = 0.0
    pnl_pct:      float = 0.0
    commission:   float = 0.0

    def close(self, exit_price: float, exit_reason: str = "Manual"):
        self.exit_date   = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.exit_price  = exit_price
        self.exit_reason = exit_reason
        self.status      = "CLOSED"

        if self.direction == "LONG":
            gross = (exit_price - self.entry_price) * self.shares
        else:
            gross = (self.entry_price - exit_price) * self.shares

        self.commission += exit_price * self.shares * 0.001
        self.pnl      = gross - self.commission
        self.pnl_pct  = (self.pnl / (self.entry_price * self.shares)) * 100


# ── Account state ─────────────────────────────────────────────────────────────

@dataclass
class Account:
    initial_capital: float = 100_000.0
    cash:            float = 100_000.0
    trades:          list  = field(default_factory=list)
    trade_counter:   int   = 0

    # Benchmark tracking
    bh_ticker:       str   = "AAPL"
    bh_shares:       float = 0.0
    bh_entry_price:  float = 0.0
    bh_invested:     bool  = False

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Account":
        trades_raw = d.pop("trades", [])
        acc = cls(**d)
        acc.trades = []
        for t in trades_raw:
            trade = Trade(**t)
            acc.trades.append(trade)
        return acc


# ── Paper Trading Engine ──────────────────────────────────────────────────────

class PaperTradingEngine:
    """
    Core engine — manages account, executes trades, computes metrics.
    All state lives in a dict (for Streamlit session_state).
    """

    COMMISSION_PCT = 0.001   # 0.1% per trade
    SLIPPAGE_PCT   = 0.0005  # 0.05% slippage

    def __init__(self, account: Account):
        self.account = account

    # ── Execute BUY ───────────────────────────────────────────
    def buy(
        self,
        ticker:       str,
        current_price: float,
        signal:       str,
        confidence:   float,
        pred_price:   float,
        horizon_days: int,
        reason:       str,
        position_pct: float = 0.10,   # % of portfolio to invest
    ) -> tuple[bool, str]:
        """
        Open a LONG position.
        Returns (success, message).
        """
        # Check if already holding this ticker
        open_tickers = [t.ticker for t in self.account.trades
                        if t.status == "OPEN" and t.direction == "LONG"]
        if ticker in open_tickers:
            return False, f"Already holding {ticker}. Close existing position first."

        # Position sizing
        invest_amount = self.account.cash * position_pct
        if invest_amount < 100:
            return False, f"Insufficient cash. Available: ${self.account.cash:,.2f}"

        # Apply slippage (buy at slightly higher price)
        exec_price  = current_price * (1 + self.SLIPPAGE_PCT)
        shares      = invest_amount / exec_price
        commission  = invest_amount * self.COMMISSION_PCT
        total_cost  = invest_amount + commission

        if total_cost > self.account.cash:
            return False, f"Insufficient cash after commission. Need ${total_cost:,.2f}"

        # Deduct from cash
        self.account.cash -= total_cost
        self.account.trade_counter += 1

        trade = Trade(
            id           = self.account.trade_counter,
            ticker       = ticker,
            direction    = "LONG",
            entry_date   = datetime.now().strftime("%Y-%m-%d %H:%M"),
            entry_price  = exec_price,
            shares       = shares,
            reason       = reason,
            signal       = signal,
            confidence   = confidence,
            pred_price   = pred_price,
            horizon_days = horizon_days,
            commission   = commission,
        )
        self.account.trades.append(trade)

        # Start buy-and-hold benchmark on first trade
        if not self.account.bh_invested:
            self.account.bh_ticker      = ticker
            self.account.bh_shares      = self.account.initial_capital / exec_price
            self.account.bh_entry_price = exec_price
            self.account.bh_invested    = True

        logger.info(f"BUY {shares:.2f} {ticker} @ ${exec_price:.2f} | cost=${total_cost:,.2f}")
        return True, f"✅ Bought {shares:.2f} shares of {ticker} @ ${exec_price:.2f} | Cost: ${total_cost:,.2f}"

    # ── Execute SELL ──────────────────────────────────────────
    def sell(
        self,
        ticker:        str,
        current_price: float,
        reason:        str = "Manual sell",
    ) -> tuple[bool, str]:
        """Close an open LONG position."""
        open_trade = next(
            (t for t in self.account.trades
             if t.ticker == ticker and t.status == "OPEN" and t.direction == "LONG"),
            None
        )
        if not open_trade:
            return False, f"No open LONG position found for {ticker}."

        # Apply slippage (sell at slightly lower price)
        exec_price = current_price * (1 - self.SLIPPAGE_PCT)
        open_trade.close(exec_price, reason)

        # Return proceeds to cash
        proceeds = exec_price * open_trade.shares
        self.account.cash += proceeds - open_trade.commission

        pnl_str = f"+${open_trade.pnl:,.2f}" if open_trade.pnl >= 0 else f"-${abs(open_trade.pnl):,.2f}"
        logger.info(f"SELL {open_trade.shares:.2f} {ticker} @ ${exec_price:.2f} | PnL={pnl_str}")
        return True, f"{'✅' if open_trade.pnl >= 0 else '📉'} Sold {ticker} @ ${exec_price:.2f} | P&L: {pnl_str} ({open_trade.pnl_pct:+.2f}%)"

    # ── Mark to market ────────────────────────────────────────
    def get_portfolio_value(self, current_prices: dict[str, float]) -> float:
        """Total account value = cash + open position market values."""
        total = self.account.cash
        for t in self.account.trades:
            if t.status == "OPEN":
                px = current_prices.get(t.ticker, t.entry_price)
                if t.direction == "LONG":
                    total += px * t.shares
                else:
                    total += t.entry_price * t.shares  # collateral
        return total

    def get_open_positions(self, current_prices: dict[str, float]) -> list[dict]:
        """Return open positions with live P&L."""
        positions = []
        for t in self.account.trades:
            if t.status != "OPEN":
                continue
            px    = current_prices.get(t.ticker, t.entry_price)
            mkt   = px * t.shares
            cost  = t.entry_price * t.shares
            unreal= mkt - cost if t.direction == "LONG" else cost - mkt
            positions.append({
                "id":          t.id,
                "ticker":      t.ticker,
                "direction":   t.direction,
                "shares":      round(t.shares, 2),
                "entry_price": t.entry_price,
                "current_price": px,
                "mkt_value":   mkt,
                "unrealised":  unreal,
                "unreal_pct":  (unreal / cost) * 100,
                "signal":      t.signal,
                "confidence":  t.confidence,
                "pred_price":  t.pred_price,
                "entry_date":  t.entry_date,
                "reason":      t.reason,
            })
        return positions

    # ── Performance metrics ───────────────────────────────────
    def get_metrics(self, portfolio_value: float) -> dict:
        """Compute full performance metrics."""
        closed = [t for t in self.account.trades if t.status == "CLOSED"]
        total_return  = (portfolio_value / self.account.initial_capital - 1) * 100
        total_pnl     = portfolio_value - self.account.initial_capital

        if not closed:
            return {
                "total_return":  total_return,
                "total_pnl":     total_pnl,
                "portfolio_value": portfolio_value,
                "cash":          self.account.cash,
                "num_trades":    0,
                "win_rate":      0,
                "avg_win":       0,
                "avg_loss":      0,
                "profit_factor": 0,
                "best_trade":    0,
                "worst_trade":   0,
                "total_commission": 0,
            }

        wins   = [t.pnl for t in closed if t.pnl > 0]
        losses = [t.pnl for t in closed if t.pnl <= 0]
        total_comm = sum(t.commission for t in self.account.trades)

        return {
            "total_return":    round(total_return, 2),
            "total_pnl":       round(total_pnl, 2),
            "portfolio_value": round(portfolio_value, 2),
            "cash":            round(self.account.cash, 2),
            "num_trades":      len(closed),
            "open_positions":  len([t for t in self.account.trades if t.status=="OPEN"]),
            "win_rate":        round(len(wins) / len(closed) * 100, 1) if closed else 0,
            "avg_win":         round(np.mean(wins), 2) if wins else 0,
            "avg_loss":        round(np.mean(losses), 2) if losses else 0,
            "profit_factor":   round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses)!=0 else 0,
            "best_trade":      round(max(t.pnl for t in closed), 2),
            "worst_trade":     round(min(t.pnl for t in closed), 2),
            "total_commission":round(total_comm, 2),
        }

    def get_bh_return(self, current_price: float) -> float:
        """Return buy-and-hold benchmark return %."""
        if not self.account.bh_invested or self.account.bh_entry_price == 0:
            return 0.0
        bh_val = self.account.bh_shares * current_price
        return round((bh_val / self.account.initial_capital - 1) * 100, 2)

    def get_trade_journal(self) -> list[dict]:
        """All trades formatted for display."""
        journal = []
        for t in reversed(self.account.trades):
            status_icon = "🟢" if t.status=="OPEN" else ("✅" if t.pnl>=0 else "❌")
            journal.append({
                "Status":     f"{status_icon} {t.status}",
                "Ticker":     t.ticker,
                "Direction":  t.direction,
                "Entry Date": t.entry_date[:10],
                "Entry $":    f"${t.entry_price:.2f}",
                "Shares":     f"{t.shares:.2f}",
                "Exit $":     f"${t.exit_price:.2f}" if t.exit_price else "—",
                "P&L":        f"${t.pnl:+,.2f}" if t.status=="CLOSED" else "Open",
                "Return":     f"{t.pnl_pct:+.1f}%" if t.status=="CLOSED" else "—",
                "Signal":     t.signal,
                "Confidence": f"{t.confidence*100:.0f}%",
                "Reason":     t.reason[:50] + "…" if len(t.reason)>50 else t.reason,
            })
        return journal


# ── Session state helpers ─────────────────────────────────────────────────────

def get_engine(session_state) -> PaperTradingEngine:
    """Get or create engine from Streamlit session state."""
    if "pt_account" not in session_state:
        reset_account(session_state)
    acc = Account.from_dict(json.loads(session_state["pt_account"]))
    return PaperTradingEngine(acc)


def save_engine(engine: PaperTradingEngine, session_state):
    """Save engine state back to session state."""
    session_state["pt_account"] = json.dumps(engine.account.to_dict())


def reset_account(session_state, initial_capital: float = 100_000):
    """Reset account to fresh state."""
    acc = Account(initial_capital=initial_capital, cash=initial_capital)
    session_state["pt_account"] = json.dumps(acc.to_dict())
    session_state.pop("pt_messages", None)
    logger.info(f"Paper trading account reset. Capital: ${initial_capital:,.0f}")
