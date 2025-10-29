"""
TextRazor NLP client.

Provides wrapper for TextRazor API with configuration and error handling.
"""

import logging

import textrazor

from app.cache import cached_resource

logger = logging.getLogger("whatsapp_analyzer")


@cached_resource
def get_textrazor_client(api_key: str, language: str = "ger") -> textrazor.TextRazor:
    """
    Get configured TextRazor client instance (cached).

    Args:
        api_key: TextRazor API key
        language: Language code for analysis (default: "ger" for German)

    Returns:
        Configured TextRazor client
    """
    logger.debug(f"Initializing TextRazor client (language={language})")

    # Set global API key
    textrazor.api_key = api_key

    # Create client with entity extraction
    client = textrazor.TextRazor(extractors=["entities"])
    client.set_language_override(language)

    logger.debug("TextRazor client initialized")
    return client


def configure_textrazor(api_key: str | None = None, language: str = "ger"):
    """
    Configure TextRazor API with given settings.

    This function sets up the global TextRazor configuration.
    Should be called once during application initialization.

    Args:
        api_key: TextRazor API key (if None, must be set in environment)
        language: Language code for analysis (default: "ger")
    """
    if api_key:
        textrazor.api_key = api_key
        logger.debug(f"TextRazor configured with API key (language={language})")
    else:
        logger.warning("TextRazor API key not provided. Some features may not work.")
