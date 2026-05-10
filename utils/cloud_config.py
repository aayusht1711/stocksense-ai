"""
utils/cloud_config.py
─────────────────────────────────────────────────────────────
Handles config for both local (.env) and Streamlit Cloud (st.secrets).
Import this instead of os.getenv() for API keys.
"""

import os
import streamlit as st


def get_secret(key: str, default: str = "") -> str:
    """
    Get a secret — works both locally (from .env) and on Streamlit Cloud (from st.secrets).
    Priority: st.secrets → environment variable → default
    """
    # Try Streamlit secrets first (Streamlit Cloud)
    try:
        val = st.secrets.get(key, None)
        if val:
            return str(val)
    except Exception:
        pass

    # Fall back to environment variable (local .env)
    return os.getenv(key, default)


def has_secret(key: str) -> bool:
    """Check if a secret is configured."""
    return bool(get_secret(key))


def get_anthropic_key() -> str:
    return get_secret("ANTHROPIC_API_KEY")

def get_email_from() -> str:
    return get_secret("ALERT_EMAIL_FROM")

def get_email_password() -> str:
    return get_secret("ALERT_EMAIL_PASSWORD")

def get_email_to() -> str:
    return get_secret("ALERT_EMAIL_TO")

def get_alpha_vantage_key() -> str:
    return get_secret("ALPHA_VANTAGE_API_KEY")


def show_setup_banner():
    """Show a friendly setup banner if API keys are missing."""
    missing = []
    if not has_secret("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY — needed for AI Chat (get free at console.anthropic.com)")
    if not has_secret("ALERT_EMAIL_FROM"):
        missing.append("ALERT_EMAIL_FROM/PASSWORD/TO — needed for Email Alerts")

    if missing:
        with st.expander("⚙️ Optional features not configured", expanded=False):
            st.markdown("Add these to enable all features:")
            for m in missing:
                st.markdown(f"- `{m}`")
            st.markdown("**Local:** Add to `.env` file  |  **Streamlit Cloud:** Settings → Secrets")
