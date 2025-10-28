"""
Configuration and settings management for WhatsApp Analyzer.

Handles API keys, environment variables, and application settings
in a centralized, secure manner.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("whatsapp_analyzer")


@dataclass
class Settings:
    """
    Application settings dataclass.
    
    Attributes:
        jina_api_key: API key for Jina AI classification service
        textrazor_api_key: API key for TextRazor NLP service
        debug_mode: Enable debug logging
    """
    jina_api_key: Optional[str] = None
    textrazor_api_key: Optional[str] = None
    debug_mode: bool = False


def get_settings() -> Settings:
    """
    Get application settings from environment variables or Streamlit secrets.
    
    Priority order:
    1. Environment variables (JINA_API_KEY, TEXTRAZOR_API_KEY)
    2. Streamlit secrets (if available)
    3. None (requires manual configuration)
    
    Returns:
        Settings object with configured values
        
    Note:
        API keys should NEVER be hard-coded. Always use environment
        variables or Streamlit secrets for production deployments.
    """
    settings = Settings()
    
    # Try to get from environment first
    settings.jina_api_key = os.environ.get("JINA_API_KEY")
    settings.textrazor_api_key = os.environ.get("TEXTRAZOR_API_KEY")
    settings.debug_mode = os.environ.get("DEBUG_MODE", "").lower() in ("true", "1", "yes")
    
    # Fallback to Streamlit secrets if available (only in Streamlit context)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            if not settings.jina_api_key and "JINA_API_KEY" in st.secrets:
                settings.jina_api_key = st.secrets["JINA_API_KEY"]
            if not settings.textrazor_api_key and "TEXTRAZOR_API_KEY" in st.secrets:
                settings.textrazor_api_key = st.secrets["TEXTRAZOR_API_KEY"]
    except ImportError:
        # Streamlit not available (e.g., in tests)
        pass
    
    # Log warnings if keys are missing (but don't fail - allow degraded functionality)
    if not settings.jina_api_key:
        logger.warning(
            "JINA_API_KEY not configured. Topic classification will not work. "
            "Set JINA_API_KEY environment variable or add to Streamlit secrets."
        )
    
    if not settings.textrazor_api_key:
        logger.warning(
            "TEXTRAZOR_API_KEY not configured. Advanced NLP features may not work. "
            "Set TEXTRAZOR_API_KEY environment variable or add to Streamlit secrets."
        )
    
    return settings


def mask_key(key: Optional[str]) -> str:
    """
    Mask an API key for safe display in logs or UI.
    
    Args:
        key: API key to mask
        
    Returns:
        Masked version of the key (e.g., "abc...xyz")
    """
    if not key:
        return "<not configured>"
    if len(key) <= 8:
        return key[:2] + "..." + key[-2:]
    return key[:4] + "..." + key[-4:]
