"""
api/trading.py
─────────────────────────────────────────────────────────────
Alpaca Trading Integration for StockSense AI.
Handles account details, market orders, and position management.
"""

import os
from typing import Optional, Dict, Any
from loguru import logger
import alpaca_trade_api as tradeapi
from dotenv import load_dotenv

load_dotenv()

class AlpacaTradingService:
    def __init__(self):
        self.api_key = os.getenv("ALPACA_API_KEY", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca API keys are missing. Trading capabilities will be disabled.")
            self.api = None
        else:
            try:
                self.api = tradeapi.REST(
                    key_id=self.api_key,
                    secret_key=self.secret_key,
                    base_url=self.base_url,
                    api_version='v2'
                )
                logger.info(f"Connected to Alpaca REST API at {self.base_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca API: {e}")
                self.api = None

    def get_account_summary(self) -> Dict[str, Any]:
        """Fetch account equity, buying power, and status."""
        if not self.api:
            return {"error": "Alpaca API not initialized"}
        try:
            account = self.api.get_account()
            return {
                "status": account.status,
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error fetching account info: {e}")
            return {"error": str(e)}

    def get_positions(self):
        """Fetch all current positions."""
        if not self.api:
            return {"error": "Alpaca API not initialized"}
        try:
            positions = self.api.list_positions()
            return [
                {
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "current_price": float(p.current_price),
                    "avg_entry_price": float(p.avg_entry_price),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc) * 100
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {"error": str(e)}

    def submit_order(self, ticker: str, qty: float, side: str, order_type: str = "market") -> Dict[str, Any]:
        """
        Submit a new order.
        side: 'buy' or 'sell'
        order_type: 'market', 'limit', etc.
        """
        if not self.api:
            return {"error": "Alpaca API not initialized"}
        try:
            order = self.api.submit_order(
                symbol=ticker.upper(),
                qty=qty,
                side=side.lower(),
                type=order_type.lower(),
                time_in_force='gtc'
            )
            logger.info(f"Order submitted: {side.upper()} {qty} {ticker.upper()}")
            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": order.qty,
                "side": order.side,
                "type": order.type,
                "status": order.status
            }
        except Exception as e:
            logger.error(f"Error submitting order for {ticker}: {e}")
            return {"error": str(e)}

trading_service = AlpacaTradingService()
