"""
Sentiment analysis for WhatsApp Analyzer.

Wraps VADER sentiment analyzer and provides sentiment label conversion.
"""

import logging

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from app.cache import cached_resource

logger = logging.getLogger("whatsapp_analyzer")


@cached_resource
def get_vader_analyzer() -> SentimentIntensityAnalyzer:
    """
    Get cached VADER sentiment analyzer instance.

    Returns:
        SentimentIntensityAnalyzer instance
    """
    logger.debug("Initializing VADER analyzer (cached)")
    return SentimentIntensityAnalyzer()


def sentiment_label_from_compound(compound_score: float) -> str:
    """
    Convert VADER compound score to sentiment label.

    Args:
        compound_score: VADER compound score (-1 to 1)

    Returns:
        Sentiment label: "positive", "negative", or "neutral"
    """
    if compound_score >= 0.05:
        return "positive"
    elif compound_score <= -0.05:
        return "negative"
    else:
        return "neutral"


def compound_to_rating(compound_score: float) -> float:
    """
    Convert VADER compound score (-1 to 1) to 0-10 rating scale.

    Args:
        compound_score: VADER compound score (-1 to 1)

    Returns:
        Rating on 0-10 scale
    """
    # Linear mapping: -1 -> 0, 0 -> 5, 1 -> 10
    rating = (compound_score + 1) * 5
    return round(rating, 1)


def analyze_sentiment(text: str) -> dict:
    """
    Analyze sentiment of text using VADER.

    Returns detailed scores, compound score, label, and scaled rating.

    Args:
        text: Input text to analyze

    Returns:
        Dictionary with:
        - vader: Full VADER scores (neg, neu, pos, compound)
        - compound: Compound score (-1 to 1)
        - label: Sentiment label (positive/negative/neutral)
        - scaled_rating: Rating on 0-10 scale

    Example:
        >>> result = analyze_sentiment("I love this!")
        >>> result['label']
        'positive'
        >>> result['scaled_rating']
        8.5
    """
    logger.debug(f"Analyzing sentiment for text (length={len(text)})")

    if not text or not text.strip():
        return {
            "vader": {"neg": 0.0, "neu": 1.0, "pos": 0.0, "compound": 0.0},
            "compound": 0.0,
            "label": "neutral",
            "scaled_rating": 5.0,
        }

    # Get VADER analyzer
    analyzer = get_vader_analyzer()

    # Get sentiment scores
    vader_scores = analyzer.polarity_scores(text)
    compound_score = vader_scores["compound"]

    # Convert to label
    label = sentiment_label_from_compound(compound_score)

    # Convert to 0-10 rating
    rating = compound_to_rating(compound_score)

    result = {
        "vader": vader_scores,
        "compound": compound_score,
        "label": label,
        "scaled_rating": rating,
    }

    logger.debug(
        f"Sentiment analysis: label={label}, rating={rating}, compound={compound_score:.3f}"
    )

    return result
