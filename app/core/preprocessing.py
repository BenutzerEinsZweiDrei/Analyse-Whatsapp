"""
Text preprocessing for WhatsApp Analyzer.

Provides tokenization, stopword removal, and lemmatization for text analysis.
"""

import logging
import string

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from app.data.loaders import load_stopwords

logger = logging.getLogger("whatsapp_analyzer")

# Track which NLTK resources have been downloaded
_nltk_resources_initialized = False


def init_nltk(resources: list[str] = None):
    """
    Initialize NLTK resources by downloading required data.

    This function should be called once at startup to ensure NLTK resources
    are available. It avoids heavy downloads at import time and only downloads
    what's necessary.

    Args:
        resources: List of NLTK resource names to download.
                  If None, downloads commonly used resources.
    """
    global _nltk_resources_initialized

    if _nltk_resources_initialized:
        logger.debug("NLTK resources already initialized")
        return

    if resources is None:
        resources = [
            "punkt",
            "punkt_tab",
            "averaged_perceptron_tagger",
            "wordnet",
            "stopwords",
            "omw-1.4",
            "averaged_perceptron_tagger_eng",
        ]

    for res in resources:
        try:
            nltk.data.find(res)
            logger.debug(f"NLTK resource already present: {res}")
        except LookupError:
            logger.info(f"NLTK resource {res} not found. Downloading...")
            try:
                nltk.download(res, quiet=True)
                logger.info(f"Downloaded NLTK resource: {res}")
            except Exception as e:
                logger.warning(f"Failed to download NLTK resource {res}: {e}")

    _nltk_resources_initialized = True
    logger.debug("NLTK resources initialized")


def preprocess_text(
    text: str, lang: str = "german", extra_stopwords: list[str] = None
) -> list[str]:
    """
    Preprocess text for analysis.

    Steps:
    1. Convert to lowercase
    2. Tokenize (word-level)
    3. Remove punctuation
    4. Remove stopwords (NLTK + custom)
    5. Lemmatize tokens

    Args:
        text: Input text to preprocess
        lang: Language for stopwords (default: 'german')
        extra_stopwords: Additional stopwords to filter (optional)

    Returns:
        List of processed tokens

    Example:
        >>> preprocess_text("Hello, this is a test!", lang='english')
        ['hello', 'test']
    """
    logger.debug(f"Preprocessing text: length={len(text or '')}")

    if not text or not text.strip():
        logger.debug("Empty text provided to preprocess_text")
        return []

    # Ensure NLTK resources are available
    init_nltk()

    # Convert to lowercase
    text = text.lower()

    # Tokenize
    try:
        tokens = word_tokenize(text)
    except Exception as e:
        logger.warning(f"word_tokenize failed: {e}. Falling back to simple split.")
        tokens = text.split()

    # Remove punctuation
    tokens = [t for t in tokens if t not in string.punctuation]

    # Load stopwords
    stop_words = set()

    # Try to load NLTK stopwords
    try:
        from nltk.corpus import stopwords as nltk_stopwords

        stop_words = set(nltk_stopwords.words(lang))
        logger.debug(f"Loaded {len(stop_words)} NLTK stopwords for {lang}")
    except Exception as e:
        logger.warning(f"Could not load NLTK {lang} stopwords: {e}. Continuing without them.")

    # Add extra stopwords from file
    try:
        file_stopwords = load_stopwords("stwd.json")
        if file_stopwords:
            stop_words.update(file_stopwords)
            logger.debug(f"Added {len(file_stopwords)} extra stopwords from file")
    except Exception as e:
        logger.debug(f"Could not load extra stopwords: {e}")

    # Add custom extra stopwords if provided
    if extra_stopwords:
        stop_words.update(extra_stopwords)

    # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]

    # Lemmatize
    try:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    except Exception as e:
        logger.warning(f"Lemmatization failed: {e}. Skipping lemmatization.")

    logger.debug(f"Preprocessing result: {len(tokens)} tokens")
    return tokens


def preprocess(text: str) -> list[str]:
    """
    Legacy function for backward compatibility.

    Wrapper around preprocess_text with default parameters.

    Args:
        text: Input text to preprocess

    Returns:
        List of processed tokens
    """
    return preprocess_text(text, lang="german")
