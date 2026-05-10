"""
utils/email_alerts.py
─────────────────────────────────────────────────────────────
Real email alerts via Gmail SMTP.
Sends: price alerts, BUY/SELL signals, daily summary, weekly report.

Setup (one-time):
  1. Go to myaccount.google.com → Security → App Passwords
  2. Create App Password for "Mail"
  3. Add to .env:
       ALERT_EMAIL_FROM=youremail@gmail.com
       ALERT_EMAIL_PASSWORD=your_16_char_app_password
       ALERT_EMAIL_TO=recipient@gmail.com
"""

import os
import smtplib
import ssl
import json
import threading
import time
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

from utils.cloud_config import get_email_from, get_email_password, get_email_to
EMAIL_FROM     = get_email_from()
EMAIL_PASSWORD = get_email_password()
EMAIL_TO       = get_email_to()
SMTP_HOST      = "smtp.gmail.com"
SMTP_PORT      = 587

ALERTS_FILE    = Path("./data/cache/alerts.json")
ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)


# ── Core email sender ─────────────────────────────────────────────────────────

def send_email(subject: str, html_body: str, to: str = None) -> bool:
    """
    Send an HTML email via Gmail SMTP.
    Returns True on success, False on failure.
    """
    recipient = to or EMAIL_TO
    if not all([EMAIL_FROM, EMAIL_PASSWORD, recipient]):
        logger.warning("Email not configured. Add ALERT_EMAIL_FROM/PASSWORD/TO to .env")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"StockSense AI <{EMAIL_FROM}>"
    msg["To"]      = recipient
    msg.attach(MIMEText(html_body, "html"))

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls(context=ctx)
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, recipient, msg.as_string())
        logger.info(f"Email sent: {subject} → {recipient}")
        return True
    except Exception as e:
        logger.error(f"Email failed: {e}")
        return False


# ── Email templates ───────────────────────────────────────────────────────────

def _base_template(content: str) -> str:
    return f"""
    <html><body style="margin:0;padding:0;background:#0A0C10;font-family:'Segoe UI',Arial,sans-serif">
    <div style="max-width:600px;margin:0 auto;padding:32px 20px">
      <div style="background:#111520;border:1px solid rgba(196,160,80,0.25);border-radius:16px;overflow:hidden">
        <div style="background:linear-gradient(135deg,#0C0F14,#111520);padding:24px 28px;border-bottom:1px solid rgba(196,160,80,0.15)">
          <div style="font-size:20px;font-weight:700;color:#F0D888;letter-spacing:-0.3px">📈 StockSense AI</div>
          <div style="font-size:12px;color:#555;margin-top:2px">Automated Market Intelligence</div>
        </div>
        <div style="padding:24px 28px">{content}</div>
        <div style="padding:16px 28px;border-top:1px solid rgba(255,255,255,0.05);font-size:11px;color:#444;text-align:center">
          StockSense AI · Not financial advice · Educational purposes only<br>
          {datetime.now().strftime('%B %d, %Y at %H:%M')}
        </div>
      </div>
    </div>
    </body></html>"""


def price_alert_email(ticker: str, alert_type: str, target: float, current: float) -> str:
    direction = "above" if "Above" in alert_type else "below"
    chg = (current / target - 1) * 100
    color = "#4ADE80" if direction == "above" else "#F87171"
    return _base_template(f"""
      <div style="text-align:center;padding:16px 0 24px">
        <div style="font-size:48px;margin-bottom:8px">🔔</div>
        <div style="font-size:28px;font-weight:700;color:{color}">{ticker}</div>
        <div style="font-size:15px;color:#C8C0B0;margin-top:6px">Price Alert Triggered</div>
      </div>
      <div style="background:rgba(196,160,80,0.06);border:1px solid rgba(196,160,80,0.2);border-radius:12px;padding:20px 24px;margin-bottom:20px">
        <table style="width:100%;border-collapse:collapse">
          <tr>
            <td style="color:#666;font-size:13px;padding:8px 0">Alert Type</td>
            <td style="color:#E8E0D0;font-size:13px;text-align:right;font-weight:500">{alert_type}</td>
          </tr>
          <tr>
            <td style="color:#666;font-size:13px;padding:8px 0">Your Target</td>
            <td style="color:#C4A050;font-size:13px;text-align:right;font-family:monospace">${target:,.2f}</td>
          </tr>
          <tr>
            <td style="color:#666;font-size:13px;padding:8px 0">Current Price</td>
            <td style="color:{color};font-size:16px;font-weight:700;text-align:right;font-family:monospace">${current:,.2f}</td>
          </tr>
          <tr>
            <td style="color:#666;font-size:13px;padding:8px 0">Change from Target</td>
            <td style="color:{color};font-size:13px;text-align:right;font-family:monospace">{chg:+.2f}%</td>
          </tr>
        </table>
      </div>
      <div style="font-size:12px;color:#555;text-align:center">
        Open StockSense AI to view full analysis and run a new prediction for {ticker}.
      </div>""")


def signal_alert_email(ticker: str, signal: str, pred_price: float,
                        current: float, confidence: float, horizon: int) -> str:
    sig_color = "#4ADE80" if signal == "BUY" else ("#F87171" if signal == "SELL" else "#C4A050")
    sig_icon  = "🟢" if signal == "BUY" else ("🔴" if signal == "SELL" else "🟡")
    exp_ret   = (pred_price / current - 1) * 100
    return _base_template(f"""
      <div style="text-align:center;padding:16px 0 24px">
        <div style="font-size:48px;margin-bottom:8px">{sig_icon}</div>
        <div style="display:inline-block;background:{'rgba(74,222,128,0.1)' if signal=='BUY' else 'rgba(248,113,113,0.1)'};
             border:1px solid {sig_color}44;border-radius:10px;padding:10px 28px;margin-top:4px">
          <span style="font-size:22px;font-weight:800;color:{sig_color};letter-spacing:0.05em">{signal}</span>
        </div>
        <div style="font-size:15px;color:#C8C0B0;margin-top:12px">New ML Signal — <strong style="color:#C4A050">{ticker}</strong></div>
      </div>
      <div style="background:rgba(196,160,80,0.06);border:1px solid rgba(196,160,80,0.2);border-radius:12px;padding:20px 24px;margin-bottom:20px">
        <table style="width:100%;border-collapse:collapse">
          <tr><td style="color:#666;font-size:13px;padding:7px 0">Current Price</td>
              <td style="color:#E8E0D0;font-size:13px;text-align:right;font-family:monospace">${current:,.2f}</td></tr>
          <tr><td style="color:#666;font-size:13px;padding:7px 0">Predicted Price ({horizon}d)</td>
              <td style="color:{sig_color};font-size:15px;font-weight:700;text-align:right;font-family:monospace">${pred_price:,.2f}</td></tr>
          <tr><td style="color:#666;font-size:13px;padding:7px 0">Expected Return</td>
              <td style="color:{sig_color};font-size:13px;text-align:right;font-family:monospace">{exp_ret:+.2f}%</td></tr>
          <tr><td style="color:#666;font-size:13px;padding:7px 0">Model Confidence</td>
              <td style="color:#C4A050;font-size:13px;text-align:right">
                <div style="display:inline-block;background:rgba(196,160,80,0.1);border-radius:20px;padding:2px 12px">{confidence*100:.1f}%</div>
              </td></tr>
        </table>
      </div>
      <div style="background:rgba(248,113,113,0.05);border:1px solid rgba(248,113,113,0.15);border-radius:8px;padding:12px 16px">
        <div style="font-size:11px;color:#666;line-height:1.6">
          ⚠️ <strong style="color:#C8C0B0">Not financial advice.</strong> This signal is generated by an ML model based on historical patterns.
          Always do your own research before making investment decisions.
        </div>
      </div>""")


def daily_summary_email(portfolio_data: list, market_data: dict) -> str:
    total_val  = sum(p.get("market_value", 0) for p in portfolio_data)
    total_pnl  = sum(p.get("pnl", 0) for p in portfolio_data)
    pnl_color  = "#4ADE80" if total_pnl >= 0 else "#F87171"
    rows = ""
    for p in portfolio_data:
        clr = "#4ADE80" if p.get("pnl", 0) >= 0 else "#F87171"
        rows += f"""
        <tr>
          <td style="color:#C4A050;font-family:monospace;padding:8px 0;font-weight:600">{p.get('ticker','')}</td>
          <td style="color:#E8E0D0;text-align:right;font-family:monospace">${p.get('price', 0):,.2f}</td>
          <td style="color:{clr};text-align:right;font-family:monospace">{p.get('day_chg', 0):+.2f}%</td>
          <td style="color:{clr};text-align:right;font-family:monospace">${p.get('pnl', 0):+,.2f}</td>
        </tr>"""

    return _base_template(f"""
      <div style="margin-bottom:20px">
        <div style="font-size:16px;font-weight:600;color:#F0D888;margin-bottom:4px">
          Good morning! Here's your daily market summary ☀️
        </div>
        <div style="font-size:12px;color:#555">{datetime.now().strftime('%A, %B %d, %Y')}</div>
      </div>
      <div style="display:flex;gap:12px;margin-bottom:20px">
        <div style="flex:1;background:rgba(196,160,80,0.06);border:1px solid rgba(196,160,80,0.2);border-radius:10px;padding:16px;text-align:center">
          <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Portfolio Value</div>
          <div style="font-size:20px;font-weight:700;color:#E8E0D0;font-family:monospace">${total_val:,.2f}</div>
        </div>
        <div style="flex:1;background:{'rgba(74,222,128,0.06)' if total_pnl>=0 else 'rgba(248,113,113,0.06)'};border:1px solid {pnl_color}33;border-radius:10px;padding:16px;text-align:center">
          <div style="font-size:11px;color:#555;text-transform:uppercase;letter-spacing:0.08em;margin-bottom:6px">Total P&L</div>
          <div style="font-size:20px;font-weight:700;color:{pnl_color};font-family:monospace">${total_pnl:+,.2f}</div>
        </div>
      </div>
      <table style="width:100%;border-collapse:collapse">
        <tr style="border-bottom:1px solid rgba(196,160,80,0.2)">
          <th style="color:#555;font-size:11px;text-align:left;padding:6px 0;font-weight:600;letter-spacing:0.08em">TICKER</th>
          <th style="color:#555;font-size:11px;text-align:right;padding:6px 0;font-weight:600;letter-spacing:0.08em">PRICE</th>
          <th style="color:#555;font-size:11px;text-align:right;padding:6px 0;font-weight:600;letter-spacing:0.08em">TODAY</th>
          <th style="color:#555;font-size:11px;text-align:right;padding:6px 0;font-weight:600;letter-spacing:0.08em">P&L</th>
        </tr>
        {rows}
      </table>""")


# ── Alert manager ─────────────────────────────────────────────────────────────

class AlertManager:
    """
    Persistent alert storage + checking engine.
    Alerts are saved to JSON so they survive app restarts.
    """

    def __init__(self, alerts_file: Path = ALERTS_FILE):
        self.file = alerts_file
        self.alerts = self._load()

    def _load(self) -> list:
        if self.file.exists():
            try:
                return json.loads(self.file.read_text())
            except Exception:
                pass
        return []

    def _save(self):
        self.file.write_text(json.dumps(self.alerts, indent=2))

    def add_alert(self, ticker: str, alert_type: str, value: float, email: str = None) -> dict:
        alert = {
            "id":        len(self.alerts),
            "ticker":    ticker.upper(),
            "type":      alert_type,
            "value":     value,
            "email":     email or EMAIL_TO,
            "active":    True,
            "triggered": False,
            "created":   datetime.now().isoformat(),
        }
        self.alerts.append(alert)
        self._save()
        logger.info(f"Alert added: {ticker} {alert_type} ${value:.2f}")
        return alert

    def check_alerts(self, prices: dict[str, float]) -> list:
        """
        Check all active alerts against current prices.
        `prices` = {"AAPL": 218.47, "TSLA": 245.10, ...}
        Returns list of triggered alerts.
        """
        triggered = []
        for alert in self.alerts:
            if not alert["active"] or alert["triggered"]:
                continue
            ticker  = alert["ticker"]
            current = prices.get(ticker)
            if current is None:
                continue
            fire = False
            if alert["type"] == "Price Above"   and current > alert["value"]: fire = True
            if alert["type"] == "Price Below"   and current < alert["value"]: fire = True

            if fire:
                alert["triggered"]  = True
                alert["active"]     = False
                alert["fired_at"]   = datetime.now().isoformat()
                alert["fired_price"]= current
                triggered.append(alert)
                # Send email
                html = price_alert_email(ticker, alert["type"], alert["value"], current)
                send_email(f"🔔 StockSense Alert: {ticker} hit ${current:.2f}", html, alert["email"])

        if triggered:
            self._save()
        return triggered

    def remove_alert(self, alert_id: int):
        self.alerts = [a for a in self.alerts if a["id"] != alert_id]
        self._save()

    def get_active(self) -> list:
        return [a for a in self.alerts if a["active"]]

    def send_signal_alert(self, ticker: str, signal: str, pred_price: float,
                           current: float, confidence: float, horizon: int):
        """Send a trading signal email immediately."""
        if signal not in ("BUY", "SELL"):
            return False
        html = signal_alert_email(ticker, signal, pred_price, current, confidence, horizon)
        return send_email(f"{signal} Signal 🚨 {ticker} — StockSense AI", html)

    def send_daily_summary(self, portfolio_data: list, market_data: dict = None):
        """Send daily portfolio summary email."""
        html = daily_summary_email(portfolio_data, market_data or {})
        return send_email(f"📊 Daily Summary — {datetime.now().strftime('%b %d')} — StockSense AI", html)


# ── Background scheduler ──────────────────────────────────────────────────────

class AlertScheduler:
    """
    Runs alert checking every N minutes in a background thread.
    """
    def __init__(self, manager: AlertManager, check_interval_min: int = 5):
        self.manager  = manager
        self.interval = check_interval_min * 60
        self._thread  = None
        self._stop    = threading.Event()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Alert scheduler started — checking every {self.interval//60} min")

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            try:
                active = self.manager.get_active()
                if active:
                    from data.ingestion import get_current_price
                    tickers = list({a["ticker"] for a in active})
                    prices  = {}
                    for t in tickers:
                        try: prices[t] = get_current_price(t)
                        except: pass
                    triggered = self.manager.check_alerts(prices)
                    if triggered:
                        logger.info(f"Alerts triggered: {[a['ticker'] for a in triggered]}")
            except Exception as e:
                logger.error(f"Alert check error: {e}")
            time.sleep(self.interval)


# ── Test / demo ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing email configuration…")
    if not all([EMAIL_FROM, EMAIL_PASSWORD, EMAIL_TO]):
        print("\n⚠️  Email not configured. Add to .env:")
        print("   ALERT_EMAIL_FROM=your@gmail.com")
        print("   ALERT_EMAIL_PASSWORD=your_app_password")
        print("   ALERT_EMAIL_TO=recipient@gmail.com")
        print("\nGet App Password: myaccount.google.com → Security → App Passwords")
    else:
        result = send_email(
            "🧪 StockSense AI — Test Email",
            _base_template("<h2 style='color:#F0D888'>Test successful!</h2><p style='color:#888'>Your email alerts are configured correctly.</p>")
        )
        print("✅ Test email sent!" if result else "❌ Email failed — check credentials")
