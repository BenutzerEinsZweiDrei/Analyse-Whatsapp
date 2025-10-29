"""
File I/O utilities for WhatsApp Analyzer.

Provides helpers for reading files with encoding fallback and safe decoding.
"""

import logging
from typing import Tuple

logger = logging.getLogger("whatsapp_analyzer")


def decode_file_content(file_bytes: bytes, filename: str = "unknown") -> Tuple[str, str]:
    """
    Decode file content with fallback encoding strategies.
    
    Attempts UTF-8 first, then falls back to latin-1 if UTF-8 fails.
    
    Args:
        file_bytes: Raw bytes to decode
        filename: Name of the file (for logging purposes)
        
    Returns:
        Tuple of (decoded_content, encoding_used)
        
    Raises:
        ValueError: If content cannot be decoded with any supported encoding
    """
    # Try UTF-8 first (most common)
    try:
        content = file_bytes.decode("utf-8")
        logger.debug(f"File '{filename}' decoded with UTF-8")
        return content, "utf-8"
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 decoding failed for '{filename}', trying latin-1")
    
    # Fallback to latin-1 (supports all byte values)
    try:
        content = file_bytes.decode("latin-1")
        logger.warning(f"File '{filename}' decoded with latin-1 fallback")
        return content, "latin-1"
    except Exception as e:
        logger.exception(f"Failed to decode file '{filename}': {e}")
        raise ValueError(f"Could not decode file '{filename}' with UTF-8 or latin-1") from e


def read_text_file(filepath: str, encoding: str = "utf-8") -> str:
    """
    Read a text file with error handling.
    
    Args:
        filepath: Path to the file to read
        encoding: Character encoding to use (default: utf-8)
        
    Returns:
        File content as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        IOError: If file cannot be read
    """
    try:
        with open(filepath, "r", encoding=encoding, errors="replace") as f:
            content = f.read()
        logger.debug(f"Successfully read file '{filepath}' ({len(content)} characters)")
        return content
    except FileNotFoundError:
        logger.error(f"File not found: '{filepath}'")
        raise
    except Exception as e:
        logger.exception(f"Error reading file '{filepath}': {e}")
        raise IOError(f"Failed to read file '{filepath}'") from e
