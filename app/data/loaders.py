"""
Data loaders for WhatsApp Analyzer.

Provides functions to load JSON assets (stopwords, emoji mappings, sentiment ratings)
with explicit error handling.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("whatsapp_analyzer")


def load_json_asset(filepath: str, description: str = "asset") -> Any:
    """
    Load a JSON asset file with error handling.

    Args:
        filepath: Path to JSON file (relative to project root or absolute)
        description: Human-readable description for error messages

    Returns:
        Loaded JSON data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    try:
        # Try relative path first
        if not os.path.isabs(filepath):
            # Look in data/ directory relative to project root
            possible_paths = [
                filepath,
                os.path.join("data", filepath),
                os.path.join(os.path.dirname(__file__), "..", "..", "data", filepath),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    filepath = path
                    break

        logger.debug(f"Loading {description} from {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        logger.debug(f"Successfully loaded {description} ({len(str(data))} bytes)")
        return data

    except FileNotFoundError:
        logger.error(f"{description.capitalize()} file not found: {filepath}")
        raise FileNotFoundError(
            f"Required data file not found: {filepath}. "
            f"Please ensure the file exists in the data/ directory."
        )
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {description} file: {filepath} - {e}")
        raise json.JSONDecodeError(f"Invalid JSON in {description} file: {filepath}", e.doc, e.pos)
    except Exception as e:
        logger.exception(f"Unexpected error loading {description} from {filepath}: {e}")
        raise


def load_stopwords(filepath: str = "stwd.json") -> List[str]:
    """
    Load stopwords from JSON file.

    Args:
        filepath: Path to stopwords JSON file

    Returns:
        List of stopwords

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        data = load_json_asset(filepath, "stopwords")

        # Handle different possible formats
        if isinstance(data, list):
            stopwords = data
        elif isinstance(data, dict) and "stopwords" in data:
            stopwords = data["stopwords"]
        else:
            raise ValueError(
                f"Unexpected stopwords file format. Expected list or dict with 'stopwords' key."
            )

        logger.debug(f"Loaded {len(stopwords)} stopwords")
        return stopwords

    except FileNotFoundError:
        logger.warning(f"Stopwords file {filepath} not found. Returning empty list.")
        return []
    except Exception as e:
        logger.warning(f"Error loading stopwords: {e}. Returning empty list.")
        return []


def load_emoji_mappings(filepath: str = "emos.json") -> Dict[str, str]:
    """
    Load emoji to meaning mappings from JSON file.

    Expected format:
    {
        "emojis": [
            {"emoji": "😊", "meaning": "positiv"},
            {"emoji": "😢", "meaning": "traurig"},
            ...
        ]
    }

    Args:
        filepath: Path to emoji mappings JSON file

    Returns:
        Dictionary mapping emoji to meaning (e.g., "😊" -> "positiv")

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    try:
        data = load_json_asset(filepath, "emoji mappings")

        # Parse the expected format
        if not isinstance(data, dict) or "emojis" not in data:
            raise ValueError("Expected dict with 'emojis' key in emoji mappings file")

        emoji_dict = {}
        for entry in data["emojis"]:
            if not isinstance(entry, dict) or "emoji" not in entry or "meaning" not in entry:
                logger.warning(f"Skipping invalid emoji entry: {entry}")
                continue
            emoji_dict[entry["emoji"]] = entry["meaning"]

        logger.debug(f"Loaded {len(emoji_dict)} emoji mappings")
        return emoji_dict

    except FileNotFoundError:
        logger.warning(f"Emoji mappings file {filepath} not found. Returning empty dict.")
        return {}
    except Exception as e:
        logger.warning(f"Error loading emoji mappings: {e}. Returning empty dict.")
        return {}


def load_sentiment_ratings(filepath: str = "sent_rating.json") -> Dict[str, Any]:
    """
    Load sentiment ratings from JSON file.

    Args:
        filepath: Path to sentiment ratings JSON file

    Returns:
        Dictionary of word to sentiment rating

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    try:
        data = load_json_asset(filepath, "sentiment ratings")

        if not isinstance(data, dict):
            raise ValueError("Expected dict in sentiment ratings file")

        logger.debug(f"Loaded {len(data)} sentiment ratings")
        return data

    except FileNotFoundError:
        logger.warning(f"Sentiment ratings file {filepath} not found. Returning empty dict.")
        return {}
    except Exception as e:
        logger.warning(f"Error loading sentiment ratings: {e}. Returning empty dict.")
        return {}
