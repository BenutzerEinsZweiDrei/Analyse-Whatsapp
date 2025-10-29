"""
Personality and emotion analysis module for WhatsApp conversations.
Provides rule-based Big Five (OCEAN) trait estimation, MBTI mapping,
and enhanced emotion analysis without ML/AI models.
"""

import logging
from collections import Counter

logger = logging.getLogger("whatsapp_analyzer")


# ---------------------------
# Big Five (OCEAN) Indicators
# ---------------------------

# Lexical indicators for each Big Five trait
# These are simple keyword-based heuristics
OPENNESS_INDICATORS = {
    "positive": [
        "creative",
        "creative",
        "idea",
        "ideas",
        "imagine",
        "art",
        "artistic",
        "novel",
        "curious",
        "explore",
        "adventure",
        "new",
        "different",
        "unique",
        "philosophical",
        "abstract",
        "theory",
        "concept",
        "innovative",
    ],
    "negative": ["traditional", "conventional", "routine", "practical", "concrete"],
}

CONSCIENTIOUSNESS_INDICATORS = {
    "positive": [
        "organized",
        "plan",
        "planning",
        "schedule",
        "prepared",
        "finish",
        "complete",
        "goal",
        "goals",
        "discipline",
        "careful",
        "detail",
        "details",
        "responsibility",
        "duty",
        "achievement",
        "efficient",
        "systematic",
    ],
    "negative": ["messy", "disorganized", "chaotic", "spontaneous", "careless", "lazy"],
}

EXTRAVERSION_INDICATORS = {
    "positive": [
        "party",
        "social",
        "friends",
        "meet",
        "meeting",
        "talk",
        "chat",
        "energy",
        "energetic",
        "excited",
        "exciting",
        "fun",
        "active",
        "group",
        "people",
        "together",
        "outgoing",
    ],
    "negative": ["alone", "quiet", "shy", "tired", "exhausted", "introvert", "solitude"],
}

AGREEABLENESS_INDICATORS = {
    "positive": [
        "help",
        "helping",
        "kind",
        "kindness",
        "care",
        "caring",
        "support",
        "supportive",
        "friendly",
        "nice",
        "compassion",
        "understanding",
        "cooperative",
        "trust",
        "generous",
        "sympathetic",
    ],
    "negative": ["conflict", "argue", "argument", "fight", "competitive", "critical", "harsh"],
}

NEUROTICISM_INDICATORS = {
    "positive": [
        "stress",
        "stressed",
        "worry",
        "worried",
        "anxious",
        "anxiety",
        "nervous",
        "fear",
        "afraid",
        "sad",
        "depressed",
        "overwhelmed",
        "tense",
        "panic",
        "upset",
        "frustrated",
    ],
    "negative": ["calm", "relaxed", "stable", "confident", "secure", "peaceful"],
}


# Emoji indicators for personality traits
EMOJI_PERSONALITY_MAP = {
    # Openness - creative, curious emojis
    "openness": ["🎨", "🎭", "🎪", "🌟", "✨", "💡", "🔮", "🌈", "🦄", "🎨"],
    # Conscientiousness - organized, goal-oriented emojis
    "conscientiousness": ["✅", "📝", "📋", "📊", "🎯", "💪", "📅", "⏰", "🏆"],
    # Extraversion - social, energetic emojis
    "extraversion": ["🎉", "🎊", "🥳", "😄", "😃", "😁", "🤗", "👋", "🙌", "💃", "🕺"],
    # Agreeableness - kind, supportive emojis
    "agreeableness": ["❤️", "💕", "💖", "🤗", "🙏", "😊", "😇", "🤝", "👍"],
    # Neuroticism - anxious, worried emojis
    "neuroticism": ["😰", "😥", "😟", "😞", "😢", "😭", "😖", "😣", "😩", "😫"],
}


def calculate_big_five_scores(text: str, emojis: list[str]) -> dict[str, float]:
    """
    Calculate Big Five personality trait scores based on lexical and emoji indicators.

    Uses simple rule-based heuristics:
    - Count positive and negative indicators for each trait
    - Factor in emoji usage patterns
    - Normalize scores to 0-10 scale

    Args:
        text: The conversation text to analyze
        emojis: List of emojis used in the conversation

    Returns:
        Dictionary with OCEAN trait scores (0-10 scale)
    """
    logger.debug("Calculating Big Five scores")

    text_lower = text.lower()
    words = text_lower.split()

    scores = {
        "openness": 5.0,
        "conscientiousness": 5.0,
        "extraversion": 5.0,
        "agreeableness": 5.0,
        "neuroticism": 5.0,
    }

    # Calculate Openness
    openness_pos = sum(1 for word in words if word in OPENNESS_INDICATORS["positive"])
    openness_neg = sum(1 for word in words if word in OPENNESS_INDICATORS["negative"])
    emoji_openness = sum(1 for emoji in emojis if emoji in EMOJI_PERSONALITY_MAP["openness"])
    scores["openness"] = (
        5.0
        + min(2.5, openness_pos * 0.5)
        - min(2.5, openness_neg * 0.5)
        + min(1.0, emoji_openness * 0.3)
    )

    # Calculate Conscientiousness
    consc_pos = sum(1 for word in words if word in CONSCIENTIOUSNESS_INDICATORS["positive"])
    consc_neg = sum(1 for word in words if word in CONSCIENTIOUSNESS_INDICATORS["negative"])
    emoji_consc = sum(1 for emoji in emojis if emoji in EMOJI_PERSONALITY_MAP["conscientiousness"])
    scores["conscientiousness"] = (
        5.0 + min(2.5, consc_pos * 0.5) - min(2.5, consc_neg * 0.5) + min(1.0, emoji_consc * 0.3)
    )

    # Calculate Extraversion
    extra_pos = sum(1 for word in words if word in EXTRAVERSION_INDICATORS["positive"])
    extra_neg = sum(1 for word in words if word in EXTRAVERSION_INDICATORS["negative"])
    emoji_extra = sum(1 for emoji in emojis if emoji in EMOJI_PERSONALITY_MAP["extraversion"])
    scores["extraversion"] = (
        5.0 + min(2.5, extra_pos * 0.5) - min(2.5, extra_neg * 0.5) + min(1.0, emoji_extra * 0.3)
    )

    # Calculate Agreeableness
    agree_pos = sum(1 for word in words if word in AGREEABLENESS_INDICATORS["positive"])
    agree_neg = sum(1 for word in words if word in AGREEABLENESS_INDICATORS["negative"])
    emoji_agree = sum(1 for emoji in emojis if emoji in EMOJI_PERSONALITY_MAP["agreeableness"])
    scores["agreeableness"] = (
        5.0 + min(2.5, agree_pos * 0.5) - min(2.5, agree_neg * 0.5) + min(1.0, emoji_agree * 0.3)
    )

    # Calculate Neuroticism
    neuro_pos = sum(1 for word in words if word in NEUROTICISM_INDICATORS["positive"])
    neuro_neg = sum(1 for word in words if word in NEUROTICISM_INDICATORS["negative"])
    emoji_neuro = sum(1 for emoji in emojis if emoji in EMOJI_PERSONALITY_MAP["neuroticism"])
    scores["neuroticism"] = (
        5.0 + min(2.5, neuro_pos * 0.5) - min(2.5, neuro_neg * 0.5) + min(1.0, emoji_neuro * 0.3)
    )

    # Clamp scores to 0-10 range
    for trait in scores:
        scores[trait] = max(0.0, min(10.0, scores[trait]))
        scores[trait] = round(scores[trait], 2)

    logger.debug(f"Big Five scores: {scores}")
    return scores


def map_big_five_to_mbti(big_five: dict[str, float]) -> str:
    """
    Map Big Five personality scores to MBTI type using rule-based heuristics.

    MBTI dimensions mapped from Big Five:
    - E/I (Extraversion/Introversion): Based on Extraversion score
    - S/N (Sensing/Intuition): Based on Openness score
    - T/F (Thinking/Feeling): Based on Agreeableness score
    - J/P (Judging/Perceiving): Based on Conscientiousness score

    Args:
        big_five: Dictionary with OCEAN scores (0-10)

    Returns:
        4-letter MBTI type string (e.g., "ENTJ", "ISFP")
    """
    logger.debug("Mapping Big Five to MBTI")

    # Extraversion vs Introversion (threshold at 5.0)
    e_or_i = "E" if big_five["extraversion"] >= 5.0 else "I"

    # Sensing vs Intuition (based on Openness - higher openness = Intuition)
    s_or_n = "N" if big_five["openness"] >= 5.5 else "S"

    # Thinking vs Feeling (based on Agreeableness - higher agreeableness = Feeling)
    t_or_f = "F" if big_five["agreeableness"] >= 5.5 else "T"

    # Judging vs Perceiving (based on Conscientiousness - higher = Judging)
    j_or_p = "J" if big_five["conscientiousness"] >= 5.5 else "P"

    mbti_type = e_or_i + s_or_n + t_or_f + j_or_p

    logger.debug(f"MBTI type: {mbti_type}")
    return mbti_type


# ---------------------------
# Enhanced Emotion Analysis
# ---------------------------

# Emotion categories mapped to emojis
EMOJI_EMOTION_MAP = {
    "joy": [
        "😀",
        "😃",
        "😄",
        "😁",
        "😆",
        "😅",
        "😂",
        "🤣",
        "😊",
        "😇",
        "🙂",
        "🙃",
        "😉",
        "😌",
        "😍",
        "🥰",
        "😘",
        "😗",
        "😙",
        "😚",
        "😋",
        "😛",
        "😝",
        "😜",
        "🤪",
        "🤗",
        "🥳",
        "😺",
        "😸",
        "😹",
        "💖",
        "💕",
        "💗",
        "💓",
        "💞",
        "✨",
        "🌟",
    ],
    "sadness": [
        "😢",
        "😭",
        "😿",
        "💔",
        "😞",
        "😔",
        "😟",
        "😕",
        "🙁",
        "☹️",
        "😣",
        "😖",
        "😫",
        "😩",
        "🥺",
        "😥",
    ],
    "anger": ["😠", "😡", "🤬", "😤", "💢", "👿", "😾"],
    "fear": ["😨", "😰", "😱", "🥶", "😵", "🤯"],
    "surprise": ["😲", "😮", "😯", "😳", "🤯", "🙀"],
    "disgust": ["🤢", "🤮", "🤧", "😷", "🤒", "🤕"],
    "love": ["❤️", "💕", "💖", "💗", "💓", "💞", "💘", "💝", "😍", "🥰", "😘", "😻", "💑", "💏"],
    "excitement": ["🎉", "🎊", "🥳", "🎈", "🎆", "🎇", "✨", "💫", "⚡", "🔥"],
    "gratitude": ["🙏", "💝", "🎁", "👍", "🤝", "💐"],
    "neutral": ["🤔", "😐", "😑", "🤨", "🧐", "😶"],
}


def classify_emotion_from_emojis(emojis: list[str]) -> dict[str, float]:
    """
    Classify emotions based on emoji usage.

    Args:
        emojis: List of emojis from the conversation

    Returns:
        Dictionary with emotion labels and their ratios (0-1)
    """
    logger.debug(f"Classifying emotions from {len(emojis)} emojis")

    if not emojis:
        return {"neutral": 1.0}

    emotion_counts = Counter()

    for emoji in emojis:
        for emotion, emoji_list in EMOJI_EMOTION_MAP.items():
            if emoji in emoji_list:
                emotion_counts[emotion] += 1

    # If no emotions detected, mark as neutral
    if not emotion_counts:
        return {"neutral": 1.0}

    # Calculate ratios
    total = sum(emotion_counts.values())
    emotion_ratios = {emotion: count / total for emotion, count in emotion_counts.items()}

    logger.debug(f"Emotion ratios: {emotion_ratios}")
    return emotion_ratios


def get_dominant_emotion(emotion_ratios: dict[str, float]) -> str:
    """
    Get the dominant emotion from emotion ratios.

    Args:
        emotion_ratios: Dictionary with emotion labels and their ratios

    Returns:
        The emotion with the highest ratio
    """
    if not emotion_ratios:
        return "neutral"

    dominant = max(emotion_ratios.items(), key=lambda x: x[1])
    return dominant[0]


def classify_emotion_from_sentiment(sentiment_score: float) -> str:
    """
    Map sentiment score to emotion category.

    Args:
        sentiment_score: VADER compound score (-1 to 1)

    Returns:
        Emotion label based on sentiment
    """
    if sentiment_score >= 0.5:
        return "joy"
    elif sentiment_score >= 0.1:
        return "contentment"
    elif sentiment_score > -0.1:
        return "neutral"
    elif sentiment_score > -0.5:
        return "sadness"
    else:
        return "distress"


def calculate_emotion_analysis(emojis: list[str], sentiment_score: float) -> dict:
    """
    Comprehensive emotion analysis combining emoji and sentiment data.

    Args:
        emojis: List of emojis used
        sentiment_score: VADER compound sentiment score

    Returns:
        Dictionary with emotion analysis results
    """
    # Get emoji-based emotions
    emoji_emotions = classify_emotion_from_emojis(emojis)

    # Get sentiment-based emotion
    sentiment_emotion = classify_emotion_from_sentiment(sentiment_score)

    # Combine: emoji emotions take priority if available
    if emoji_emotions and emoji_emotions != {"neutral": 1.0}:
        dominant_emotion = get_dominant_emotion(emoji_emotions)
        emotion_ratios = emoji_emotions
    else:
        dominant_emotion = sentiment_emotion
        emotion_ratios = {sentiment_emotion: 1.0}

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_ratios": emotion_ratios,
        "sentiment_emotion": sentiment_emotion,
    }
