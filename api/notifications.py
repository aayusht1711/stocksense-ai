import os
import requests
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class NotificationService:
    def __init__(self):
        self.discord_webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    def send_trade_alert(self, ticker: str, side: str, qty: float, price: float = None):
        """
        Sends an embedded alert to Discord about a trade execution.
        """
        if not self.discord_webhook_url:
            logger.info("DISCORD_WEBHOOK_URL not set. Skipping trade alert.")
            return False

        color = 5814783 if side.lower() == "buy" else 16734296 # Green vs Red
        action_text = "Bought" if side.lower() == "buy" else "Sold"
        
        description = f"**{action_text} {qty} shares/units of {ticker}**"
        if price:
            description += f" at approx ${price:.2f}"

        payload = {
            "username": "StockSense AI",
            "avatar_url": "https://cdn-icons-png.flaticon.com/512/864/864685.png",
            "embeds": [
                {
                    "title": f"🚨 Automated Trade Executed",
                    "description": description,
                    "color": color,
                    "footer": {
                        "text": "Powered by StockSense AI"
                    }
                }
            ]
        }

        try:
            response = requests.post(self.discord_webhook_url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully sent Discord alert for {ticker}")
            return True
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")
            return False

    def send_test_alert(self):
        """Sends a simple test message."""
        if not self.discord_webhook_url:
            logger.info("DISCORD_WEBHOOK_URL not set. Cannot send test alert.")
            return False
            
        payload = {
            "username": "StockSense AI",
            "content": "✅ **StockSense AI Notification System is Online!** Webhook integration is successful."
        }
        
        try:
            response = requests.post(self.discord_webhook_url, json=payload)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send test alert: {e}")
            return False

notification_service = NotificationService()
