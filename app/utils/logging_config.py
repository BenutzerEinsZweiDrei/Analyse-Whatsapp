"""
Logging configuration for WhatsApp Analyzer.

Provides centralized logging setup with in-memory buffer for Streamlit UI display.
"""

import io
import logging

# Global memory buffer for logs (accessible by Streamlit UI)
_log_stream: io.StringIO | None = None
_logger_initialized = False


def configure_logging(
    debug: bool = False, logger_name: str = "whatsapp_analyzer"
) -> logging.Logger:
    """
    Configure application logging with console and memory stream handlers.

    Args:
        debug: If True, set logging level to DEBUG, otherwise INFO
        logger_name: Name of the logger to configure

    Returns:
        Configured logger instance
    """
    global _log_stream, _logger_initialized

    # Create logger
    logger = logging.getLogger(logger_name)

    # Set level
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)

    # Only add handlers once to avoid duplicates on reruns
    if not _logger_initialized:
        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Memory stream handler for Streamlit UI
        _log_stream = io.StringIO()
        stream_handler = logging.StreamHandler(_log_stream)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        _logger_initialized = True
    else:
        # Update level on existing logger
        logger.setLevel(level)

    if debug:
        logger.debug(f"Logging configured in {'DEBUG' if debug else 'INFO'} mode")

    return logger


def set_debug_mode(enabled: bool):
    """
    Enable or disable debug mode dynamically.

    Args:
        enabled: If True, enable DEBUG logging, otherwise INFO
    """
    logger = logging.getLogger("whatsapp_analyzer")
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)

    # Clear log buffer on mode change
    if _log_stream:
        _log_stream.truncate(0)
        _log_stream.seek(0)

    logger.debug(f"Debug mode set to {enabled}")


def get_logs() -> str:
    """
    Retrieve accumulated logs from memory buffer.

    Returns:
        String containing all logged messages
    """
    if _log_stream is None:
        return ""

    # Flush all handlers to ensure logs are written
    logger = logging.getLogger("whatsapp_analyzer")
    for handler in logger.handlers:
        handler.flush()

    return _log_stream.getvalue()


def clear_logs():
    """Clear the memory log buffer."""
    if _log_stream:
        _log_stream.truncate(0)
        _log_stream.seek(0)
