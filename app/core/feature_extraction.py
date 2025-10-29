"""
Robust feature extraction for WhatsApp messages.

Extracts comprehensive per-message metadata including:
- Timestamps and time-based features
- Text statistics and linguistic features
- Emoji and emoticon features
- URL and media detection
- Language detection
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from collections import Counter

logger = logging.getLogger("whatsapp_analyzer.feature_extraction")

# Optional dependencies with fallbacks
try:
    import emoji

    HAS_EMOJI = True
except ImportError:
    HAS_EMOJI = False
    logger.warning("emoji library not available - emoji normalization will be limited")

try:
    from langdetect import detect, LangDetectException

    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False
    logger.warning("langdetect not available - language detection will be skipped")


@dataclass
class MessageFeatures:
    """Comprehensive feature set for a single message."""

    # Identity
    message_id: int
    author: Optional[str]
    is_system: bool
    is_media: bool

    # Temporal
    timestamp: Optional[datetime]
    timestamp_utc: Optional[str]  # ISO format
    time_of_day_hour: Optional[int]
    day_of_week: Optional[int]

    # Text content
    text: str
    text_normalized: str
    message_length_chars: int
    token_count: int
    word_count: int
    sentence_count: int

    # Linguistic features
    avg_word_length: float
    lexical_diversity: float  # type-token ratio
    stopword_ratio: float
    uppercase_ratio: float
    punctuation_density: float

    # Special characters
    emoji_count: int
    emoticon_count: int
    url_count: int
    mention_count: int
    hashtag_count: int
    question_marks: int
    exclamation_marks: int

    # Parsed elements
    emojis: List[str] = field(default_factory=list)
    emoji_descriptors: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)

    # Language
    detected_language: Optional[str] = None
    language_confidence: Optional[float] = None

    # Reply indicators
    has_reply_indicator: bool = False
    reply_to: Optional[str] = None


def extract_emojis_from_text(text: str) -> Tuple[List[str], List[str]]:
    """
    Extract emojis and their descriptive names from text.

    Args:
        text: Input text

    Returns:
        Tuple of (emoji_list, descriptor_list)
    """
    if not HAS_EMOJI or not text:
        return [], []

    emoji_list = []
    descriptors = []

    try:
        # Extract emojis using emoji library
        for char in text:
            if emoji.is_emoji(char):
                emoji_list.append(char)
                # Get descriptive name
                descriptor = emoji.demojize(char)
                descriptors.append(descriptor)
    except Exception as e:
        logger.debug(f"Error extracting emojis: {e}")

    return emoji_list, descriptors


def detect_language(text: str) -> Tuple[Optional[str], Optional[float]]:
    """
    Detect language of text.

    Args:
        text: Input text

    Returns:
        Tuple of (language_code, confidence)
    """
    if not HAS_LANGDETECT or not text or len(text.strip()) < 10:
        return None, None

    try:
        lang = detect(text)
        # langdetect doesn't provide confidence, so we use a fixed value
        return lang, 0.9
    except LangDetectException:
        return None, None
    except Exception as e:
        logger.debug(f"Language detection error: {e}")
        return None, None


def count_emoticons(text: str) -> int:
    """
    Count text-based emoticons in message.

    Args:
        text: Input text

    Returns:
        Number of emoticons found
    """
    emoticon_patterns = [
        r":-?\)",  # :) :-)
        r":-?\(",  # :( :-(
        r":-?D",  # :D :-D
        r":-?P",  # :P :-P
        r";-?\)",  # ;) ;-)
        r"<3",  # <3
        r":-?O",  # :O :-O
        r":-?\|",  # :| :-|
        r"xD",  # xD
        r"XD",  # XD
    ]

    count = 0
    for pattern in emoticon_patterns:
        count += len(re.findall(pattern, text))

    return count


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.

    Args:
        text: Input text

    Returns:
        List of URLs found
    """
    url_pattern = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return url_pattern.findall(text)


def calculate_lexical_diversity(words: List[str]) -> float:
    """
    Calculate lexical diversity (type-token ratio).

    Args:
        words: List of words

    Returns:
        Lexical diversity score (0-1)
    """
    if not words:
        return 0.0

    unique_words = len(set(words))
    total_words = len(words)

    return unique_words / total_words if total_words > 0 else 0.0


def calculate_stopword_ratio(words: List[str]) -> float:
    """
    Calculate ratio of stopwords to total words.

    Args:
        words: List of words

    Returns:
        Stopword ratio (0-1)
    """
    # Common stopwords across multiple languages
    stopwords = {
        # English
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "can",
        "may",
        "might",
        "must",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "it",
        "this",
        "that",
        "these",
        "those",
        # German
        "der",
        "die",
        "das",
        "und",
        "oder",
        "aber",
        "ist",
        "sind",
        "war",
        "waren",
        "haben",
        "hat",
        "hatte",
        "werden",
        "wird",
        "wurde",
        "in",
        "auf",
        "an",
        "zu",
        "von",
        "mit",
        "bei",
        "aus",
        "für",
        "als",
        "ich",
        "du",
        "er",
        "sie",
        "es",
        "wir",
        "ihr",
    }

    if not words:
        return 0.0

    stopword_count = sum(1 for word in words if word.lower() in stopwords)
    return stopword_count / len(words) if words else 0.0


def extract_message_features(
    message_id: int,
    author: Optional[str],
    text: str,
    timestamp: Optional[datetime] = None,
    is_system: bool = False,
    is_media: bool = False,
) -> MessageFeatures:
    """
    Extract comprehensive features from a single message.

    Args:
        message_id: Unique message identifier
        author: Message author (None for system messages)
        text: Message text content
        timestamp: Message timestamp
        is_system: Whether this is a system message
        is_media: Whether this is a media message

    Returns:
        MessageFeatures object with all extracted features
    """
    # Normalize text
    text_normalized = text.strip()

    # Basic counts
    message_length = len(text)
    words = text_normalized.split()
    word_count = len(words)
    token_count = word_count  # Simplified tokenization

    # Sentence count (approximate)
    sentence_count = max(1, len(re.split(r"[.!?]+", text_normalized)))

    # Average word length
    avg_word_len = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

    # Extract emojis
    emoji_list, emoji_descriptors = extract_emojis_from_text(text)
    emoji_count = len(emoji_list)

    # Count emoticons
    emoticon_count = count_emoticons(text)

    # Extract URLs
    urls = extract_urls(text)
    url_count = len(urls)

    # Count mentions (@username)
    mentions = re.findall(r"@\w+", text)
    mention_count = len(mentions)

    # Count hashtags
    hashtags = re.findall(r"#\w+", text)
    hashtag_count = len(hashtags)

    # Count special punctuation (excluding emojis)
    question_marks = text.count("?")
    exclamation_marks = text.count("!")

    # Uppercase ratio
    if text:
        uppercase_chars = sum(1 for c in text if c.isupper())
        uppercase_ratio = uppercase_chars / len(text)
    else:
        uppercase_ratio = 0.0

    # Punctuation density (excluding emojis)
    punctuation_chars = sum(1 for c in text if c in ".,;:!?-()[]{}\"'")
    punctuation_density = punctuation_chars / len(text) if text else 0.0

    # Lexical diversity
    lexical_diversity = calculate_lexical_diversity(words)

    # Stopword ratio
    stopword_ratio = calculate_stopword_ratio(words)

    # Detect language
    detected_lang, lang_conf = detect_language(text_normalized)

    # Temporal features
    timestamp_utc = timestamp.isoformat() if timestamp else None
    time_of_day = timestamp.hour if timestamp else None
    day_of_week = timestamp.weekday() if timestamp else None

    # Reply indicators
    has_reply = "@" in text and mention_count > 0
    reply_to = mentions[0][1:] if mentions else None  # Remove @ sign

    return MessageFeatures(
        message_id=message_id,
        author=author,
        is_system=is_system,
        is_media=is_media,
        timestamp=timestamp,
        timestamp_utc=timestamp_utc,
        time_of_day_hour=time_of_day,
        day_of_week=day_of_week,
        text=text,
        text_normalized=text_normalized,
        message_length_chars=message_length,
        token_count=token_count,
        word_count=word_count,
        sentence_count=sentence_count,
        avg_word_length=avg_word_len,
        lexical_diversity=lexical_diversity,
        stopword_ratio=stopword_ratio,
        uppercase_ratio=uppercase_ratio,
        punctuation_density=punctuation_density,
        emoji_count=emoji_count,
        emoticon_count=emoticon_count,
        url_count=url_count,
        mention_count=mention_count,
        hashtag_count=hashtag_count,
        question_marks=question_marks,
        exclamation_marks=exclamation_marks,
        emojis=emoji_list,
        emoji_descriptors=emoji_descriptors,
        urls=urls,
        mentions=mentions,
        detected_language=detected_lang,
        language_confidence=lang_conf,
        has_reply_indicator=has_reply,
        reply_to=reply_to,
    )


def aggregate_language_stats(features: List[MessageFeatures]) -> Dict[str, int]:
    """
    Aggregate language statistics across messages.

    Args:
        features: List of message features

    Returns:
        Dictionary mapping language codes to message counts
    """
    lang_counter = Counter()

    for feat in features:
        if feat.detected_language:
            lang_counter[feat.detected_language] += 1

    return dict(lang_counter)


def get_dominant_language(features: List[MessageFeatures]) -> Optional[str]:
    """
    Get the dominant language across all messages.

    Args:
        features: List of message features

    Returns:
        Most common language code or None
    """
    lang_stats = aggregate_language_stats(features)

    if not lang_stats:
        return None

    return max(lang_stats.items(), key=lambda x: x[1])[0]
