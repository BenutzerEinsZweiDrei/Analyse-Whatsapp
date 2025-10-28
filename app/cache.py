"""
Caching abstraction for WhatsApp Analyzer.

Provides decorators that use Streamlit caching when available,
and fall back to functools.lru_cache for testing/non-Streamlit contexts.
"""

import functools
import logging
from typing import Callable, Any

logger = logging.getLogger("whatsapp_analyzer")


def cached_resource(func: Callable) -> Callable:
    """
    Decorator for caching expensive resources (models, lexicons, etc.).
    
    Uses streamlit.cache_resource when available, otherwise functools.lru_cache.
    Resources are loaded once and reused across all sessions.
    
    Args:
        func: Function to cache
        
    Returns:
        Cached function
    """
    try:
        import streamlit as st
        # Use Streamlit's cache_resource decorator
        logger.debug(f"Using streamlit.cache_resource for {func.__name__}")
        return st.cache_resource(func)
    except ImportError:
        # Streamlit not available, use lru_cache as fallback
        logger.debug(f"Using functools.lru_cache for {func.__name__} (Streamlit not available)")
        return functools.lru_cache(maxsize=None)(func)


def cached_data(show_spinner: bool = False, ttl: int = None) -> Callable:
    """
    Decorator factory for caching data computations.
    
    Uses streamlit.cache_data when available, otherwise functools.lru_cache.
    Data is cached per unique input arguments.
    
    Args:
        show_spinner: Show spinner during computation (Streamlit only)
        ttl: Time to live in seconds (Streamlit only, None = infinite)
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        try:
            import streamlit as st
            # Use Streamlit's cache_data decorator
            logger.debug(f"Using streamlit.cache_data for {func.__name__}")
            return st.cache_data(show_spinner=show_spinner, ttl=ttl)(func)
        except ImportError:
            # Streamlit not available, use lru_cache as fallback
            logger.debug(f"Using functools.lru_cache for {func.__name__} (Streamlit not available)")
            return functools.lru_cache(maxsize=128)(func)
    
    return decorator
