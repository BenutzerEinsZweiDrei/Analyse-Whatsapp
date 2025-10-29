"""
Multi-method emotion detection.

Provides emotion classification using:
- Lexicon-based approaches (NRC, emoji mapping)
- Transformer-based models (optional)
- Empath category mapping
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger("whatsapp_analyzer.emotion_detection")

# Optional dependencies
try:
    from transformers import pipeline

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("Transformers not available - emotion detection will use lexicon only")

try:
    from app.cache import cached_resource
except ImportError:
    # Fallback decorator if cache not available
    def cached_resource(func):
        return func


# Emotion categories
EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]


@dataclass
class EmotionResult:
    """Emotion detection result."""

    # Primary emotion
    primary_emotion: str
    primary_score: float  # 0 to 1

    # All emotion scores
    emotion_scores: dict[str, float]

    # Confidence
    confidence: float  # 0 to 1
    method_used: str  # "lexicon", "transformer", "ensemble"

    # Additional details
    transformer_details: dict | None = None


@cached_resource
def get_emotion_pipeline():
    """
    Load and cache emotion classification model.

    Returns:
        Emotion classification pipeline or None
    """
    if not HAS_TRANSFORMERS:
        return None

    try:
        logger.info("Loading emotion transformer model (cached)")
        # Using j-hartmann's emotion model (good for general text)
        model_name = "j-hartmann/emotion-english-distilroberta-base"
        return pipeline(
            "text-classification",
            model=model_name,
            top_k=None,  # Return all emotion scores
            max_length=512,
            truncation=True,
        )
    except Exception as e:
        logger.warning(f"Failed to load emotion transformer: {e}")
        return None


def emoji_emotion_mapping(emojis: list[str]) -> dict[str, float]:
    """
    Map emojis to emotion scores.

    Args:
        emojis: List of emoji characters

    Returns:
        Dictionary of emotion scores
    """
    # Emoji to emotion mapping
    emoji_map = {
        # Joy
        "😊": "joy",
        "😃": "joy",
        "😄": "joy",
        "😁": "joy",
        "🙂": "joy",
        "😍": "joy",
        "🥰": "joy",
        "❤️": "joy",
        "🎉": "joy",
        "🥳": "joy",
        # Sadness
        "😢": "sadness",
        "😭": "sadness",
        "😞": "sadness",
        "😔": "sadness",
        "💔": "sadness",
        "😿": "sadness",
        # Anger
        "😡": "anger",
        "😠": "anger",
        "🤬": "anger",
        "😤": "anger",
        "💢": "anger",
        # Fear
        "😨": "fear",
        "😰": "fear",
        "😱": "fear",
        "😖": "fear",
        # Disgust
        "🤢": "disgust",
        "🤮": "disgust",
        "😬": "disgust",
        # Surprise
        "😮": "surprise",
        "😯": "surprise",
        "😲": "surprise",
        "🤯": "surprise",
    }

    emotion_counts = dict.fromkeys(EMOTION_LABELS, 0)

    for emoji in emojis:
        if emoji in emoji_map:
            emotion = emoji_map[emoji]
            emotion_counts[emotion] += 1

    # Normalize to scores
    total = sum(emotion_counts.values())
    if total > 0:
        emotion_scores = {k: v / total for k, v in emotion_counts.items()}
    else:
        emotion_scores = dict.fromkeys(EMOTION_LABELS, 0.0)

    return emotion_scores


def lexicon_based_emotion(text: str, emojis: list[str] | None = None) -> dict[str, float]:
    """
    Detect emotions using lexicon-based approach.

    Args:
        text: Input text
        emojis: Optional list of emojis

    Returns:
        Dictionary of emotion scores
    """
    # Simple keyword-based emotion detection
    emotion_keywords = {
        "joy": [
            "happy",
            "glad",
            "joy",
            "pleased",
            "delighted",
            "excited",
            "wonderful",
            "great",
            "love",
            "amazing",
            "fantastic",
            "glücklich",
            "froh",
            "freude",
        ],
        "sadness": [
            "sad",
            "unhappy",
            "depressed",
            "down",
            "miserable",
            "sorry",
            "disappointed",
            "crying",
            "traurig",
            "unglücklich",
        ],
        "anger": [
            "angry",
            "mad",
            "furious",
            "annoyed",
            "irritated",
            "hate",
            "rage",
            "frustrated",
            "wütend",
            "ärgerlich",
        ],
        "fear": [
            "afraid",
            "scared",
            "fearful",
            "anxious",
            "worried",
            "nervous",
            "terrified",
            "angst",
            "ängstlich",
        ],
        "disgust": [
            "disgusted",
            "revolted",
            "gross",
            "yuck",
            "awful",
            "terrible",
            "horrible",
            "ekelhaft",
        ],
        "surprise": ["surprised", "amazed", "shocked", "astonished", "wow", "überrascht"],
    }

    text_lower = text.lower()
    emotion_scores = dict.fromkeys(EMOTION_LABELS, 0)

    # Count keyword matches
    for emotion, keywords in emotion_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                emotion_scores[emotion] += 1

    # Add emoji-based scores
    if emojis:
        emoji_scores = emoji_emotion_mapping(emojis)
        for emotion, score in emoji_scores.items():
            emotion_scores[emotion] += score * 2  # Weight emojis higher

    # Normalize
    total = sum(emotion_scores.values())
    if total > 0:
        emotion_scores = {k: v / total for k, v in emotion_scores.items()}
    else:
        # Default to neutral
        emotion_scores["neutral"] = 1.0

    return emotion_scores


def detect_emotion(
    text: str,
    emojis: list[str] | None = None,
    use_transformer: bool = True,
) -> EmotionResult:
    """
    Detect emotions in text using ensemble approach.

    Args:
        text: Input text
        emojis: Optional list of emojis
        use_transformer: Whether to use transformer model

    Returns:
        EmotionResult with detected emotions
    """
    if not text or not text.strip():
        return EmotionResult(
            primary_emotion="neutral",
            primary_score=1.0,
            emotion_scores={"neutral": 1.0},
            confidence=0.0,
            method_used="empty",
        )

    # Lexicon-based detection
    lexicon_scores = lexicon_based_emotion(text, emojis)

    # Transformer-based detection (if available)
    transformer_scores = None
    transformer_details = None

    if use_transformer and HAS_TRANSFORMERS:
        emotion_pipeline = get_emotion_pipeline()
        if emotion_pipeline:
            try:
                # Truncate text if too long
                text_truncated = text[:500]
                result = emotion_pipeline(text_truncated)

                if result and isinstance(result, list) and len(result) > 0:
                    # Convert to our format
                    transformer_scores = {}
                    for item in result[0]:
                        label = item["label"].lower()
                        score = item["score"]
                        # Map model labels to our labels
                        if label in EMOTION_LABELS:
                            transformer_scores[label] = score

                    transformer_details = {"raw_result": result[0]}
            except Exception as e:
                logger.debug(f"Transformer emotion detection failed: {e}")

    # Ensemble combination
    if transformer_scores:
        # Weighted average: transformer (70%), lexicon (30%)
        emotion_scores = {}
        for emotion in EMOTION_LABELS:
            trans_score = transformer_scores.get(emotion, 0.0)
            lex_score = lexicon_scores.get(emotion, 0.0)
            emotion_scores[emotion] = 0.7 * trans_score + 0.3 * lex_score

        method = "ensemble"
        confidence = 0.85
    else:
        # Lexicon only
        emotion_scores = lexicon_scores
        method = "lexicon"
        confidence = 0.60

    # Find primary emotion
    if emotion_scores:
        primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        primary_emotion_label = primary_emotion[0]
        primary_emotion_score = primary_emotion[1]
    else:
        primary_emotion_label = "neutral"
        primary_emotion_score = 1.0

    return EmotionResult(
        primary_emotion=primary_emotion_label,
        primary_score=primary_emotion_score,
        emotion_scores=emotion_scores,
        confidence=confidence,
        method_used=method,
        transformer_details=transformer_details,
    )


def batch_detect_emotion(
    texts: list[str],
    emojis_list: list[list[str]] | None = None,
    use_transformer: bool = True,
) -> list[EmotionResult]:
    """
    Detect emotions for multiple texts efficiently.

    Args:
        texts: List of texts
        emojis_list: Optional list of emoji lists
        use_transformer: Whether to use transformer models

    Returns:
        List of EmotionResult objects
    """
    if emojis_list is None:
        emojis_list = [None] * len(texts)

    results = []
    for text, emojis in zip(texts, emojis_list):
        result = detect_emotion(text, emojis, use_transformer)
        results.append(result)

    return results
