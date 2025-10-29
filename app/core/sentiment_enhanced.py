"""
Enhanced ensemble sentiment analysis.

Combines multiple approaches:
- VADER lexicon-based sentiment
- Transformer-based models (optional)
- Emoji sentiment mapping
- Calibrated confidence scores
"""

import logging
from dataclasses import dataclass

from app.cache import cached_resource
from app.core.sentiment import analyze_sentiment as vader_sentiment

logger = logging.getLogger("whatsapp_analyzer.sentiment_enhanced")

# Optional transformer models
try:
    from transformers import pipeline

    HAS_TRANSFORMERS = True
    logger.info("Transformers library available - enhanced sentiment analysis enabled")
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers not available - falling back to VADER-only sentiment")


@dataclass
class SentimentResult:
    """Comprehensive sentiment analysis result."""

    # Overall sentiment
    polarity: str  # "positive", "negative", "neutral"
    compound_score: float  # -1 to 1
    scaled_rating: float  # 0 to 10

    # Component scores
    positive_score: float  # 0 to 1
    negative_score: float  # 0 to 1
    neutral_score: float  # 0 to 1

    # Confidence and sources
    confidence: float  # 0 to 1
    method_used: str  # "vader", "ensemble", "transformer"

    # Additional details
    vader_scores: dict | None = None
    transformer_label: str | None = None
    transformer_score: float | None = None


@cached_resource
def get_sentiment_pipeline():
    """
    Load and cache transformer-based sentiment model.

    Returns:
        Sentiment analysis pipeline or None if not available
    """
    if not HAS_TRANSFORMERS:
        return None

    try:
        # Use Cardiff NLP's Twitter RoBERTa model (good for short text)
        logger.info("Loading sentiment transformer model (cached)")
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        return pipeline("sentiment-analysis", model=model_name, max_length=512, truncation=True)
    except Exception as e:
        logger.warning(f"Failed to load sentiment transformer: {e}")
        return None


def emoji_sentiment_score(emojis: list[str]) -> tuple[float, int]:
    """
    Calculate sentiment from emojis.

    Args:
        emojis: List of emoji characters

    Returns:
        Tuple of (average_sentiment, count)
        Sentiment ranges from -1 (negative) to 1 (positive)
    """
    if not emojis:
        return 0.0, 0

    # Simple emoji sentiment mapping
    positive_emojis = {"😊", "😃", "😄", "😁", "🙂", "😍", "🥰", "❤️", "💕", "👍", "🎉", "✨", "🌟"}
    negative_emojis = {"😢", "😭", "😞", "😔", "😩", "😤", "😡", "😠", "💔", "👎", "😰", "😨"}
    very_positive = {"🤩", "😻", "💖", "🎊", "🏆", "🥳"}
    very_negative = {"😡", "🤬", "💀", "👿", "😈"}

    scores = []
    for emoji in emojis:
        if emoji in very_positive:
            scores.append(1.0)
        elif emoji in positive_emojis:
            scores.append(0.7)
        elif emoji in very_negative:
            scores.append(-1.0)
        elif emoji in negative_emojis:
            scores.append(-0.7)
        else:
            scores.append(0.0)  # Neutral

    avg_score = sum(scores) / len(scores) if scores else 0.0
    return avg_score, len(emojis)


def analyze_sentiment_ensemble(
    text: str,
    emojis: list[str] | None = None,
    use_transformer: bool = True,
) -> SentimentResult:
    """
    Analyze sentiment using ensemble approach.

    Combines VADER, transformer models (if available), and emoji sentiment.

    Args:
        text: Input text to analyze
        emojis: Optional list of emojis in the text
        use_transformer: Whether to use transformer model if available

    Returns:
        SentimentResult with comprehensive analysis
    """
    if not text or not text.strip():
        return SentimentResult(
            polarity="neutral",
            compound_score=0.0,
            scaled_rating=5.0,
            positive_score=0.0,
            negative_score=0.0,
            neutral_score=1.0,
            confidence=0.0,
            method_used="empty",
        )

    # Step 1: VADER analysis (baseline)
    vader_result = vader_sentiment(text)
    vader_compound = vader_result.get("compound", 0.0)
    vader_label = vader_result.get("label", "neutral")
    vader_scores = vader_result.get("vader", {})

    # Step 2: Emoji sentiment
    emoji_score, emoji_count = emoji_sentiment_score(emojis or [])

    # Step 3: Transformer analysis (if available and requested)
    transformer_label = None
    transformer_score = None
    transformer_sentiment = 0.0

    if use_transformer and HAS_TRANSFORMERS:
        sentiment_pipeline = get_sentiment_pipeline()
        if sentiment_pipeline:
            try:
                # Truncate text if too long
                text_truncated = text[:500]
                result = sentiment_pipeline(text_truncated)
                if result:
                    transformer_label = result[0]["label"].lower()
                    transformer_score = result[0]["score"]

                    # Convert label to compound score
                    if "positive" in transformer_label:
                        transformer_sentiment = transformer_score
                    elif "negative" in transformer_label:
                        transformer_sentiment = -transformer_score
                    else:  # neutral
                        transformer_sentiment = 0.0
            except Exception as e:
                logger.debug(f"Transformer sentiment failed: {e}")

    # Step 4: Ensemble combination
    if transformer_label is not None:
        # Weighted average: VADER (40%), Transformer (50%), Emoji (10%)
        weights = [0.4, 0.5, 0.1]
        scores = [vader_compound, transformer_sentiment, emoji_score]
        compound = sum(w * s for w, s in zip(weights, scores))
        method = "ensemble"

        # Confidence based on agreement
        # Check if methods agree on polarity
        polarities = []
        if vader_compound > 0.05:
            polarities.append("positive")
        elif vader_compound < -0.05:
            polarities.append("negative")
        else:
            polarities.append("neutral")

        if transformer_sentiment > 0.05:
            polarities.append("positive")
        elif transformer_sentiment < -0.05:
            polarities.append("negative")
        else:
            polarities.append("neutral")

        # Confidence is higher when methods agree
        agreement = len(set(polarities)) == 1
        confidence = 0.85 if agreement else 0.65
        if transformer_score:
            confidence = min(0.95, confidence * transformer_score)

    elif emoji_count > 0:
        # VADER + Emoji (without transformer)
        weights = [0.7, 0.3]
        scores = [vader_compound, emoji_score]
        compound = sum(w * s for w, s in zip(weights, scores))
        method = "vader_emoji"
        confidence = 0.70  # Medium confidence without transformer
    else:
        # VADER only
        compound = vader_compound
        method = "vader"
        confidence = 0.60  # Lower confidence with single method

    # Determine final polarity
    if compound >= 0.05:
        polarity = "positive"
    elif compound <= -0.05:
        polarity = "negative"
    else:
        polarity = "neutral"

    # Scale to 0-10 rating
    scaled_rating = (compound + 1) * 5

    # Component scores (normalized)
    pos_score = vader_scores.get("pos", 0.0)
    neg_score = vader_scores.get("neg", 0.0)
    neu_score = vader_scores.get("neu", 0.0)

    return SentimentResult(
        polarity=polarity,
        compound_score=compound,
        scaled_rating=scaled_rating,
        positive_score=pos_score,
        negative_score=neg_score,
        neutral_score=neu_score,
        confidence=confidence,
        method_used=method,
        vader_scores=vader_scores,
        transformer_label=transformer_label,
        transformer_score=transformer_score,
    )


def batch_analyze_sentiment(
    texts: list[str],
    emojis_list: list[list[str]] | None = None,
    use_transformer: bool = True,
) -> list[SentimentResult]:
    """
    Analyze sentiment for multiple texts efficiently.

    Args:
        texts: List of texts to analyze
        emojis_list: Optional list of emoji lists for each text
        use_transformer: Whether to use transformer models

    Returns:
        List of SentimentResult objects
    """
    if emojis_list is None:
        emojis_list = [None] * len(texts)

    results = []
    for text, emojis in zip(texts, emojis_list):
        result = analyze_sentiment_ensemble(text, emojis, use_transformer)
        results.append(result)

    return results
