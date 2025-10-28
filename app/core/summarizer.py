"""
Matrix summarization for WhatsApp Analyzer.

Aggregates analysis results and generates summary statistics.
"""

import logging
import re
from typing import Dict, List

import regex

from app.core.nouns import extract_nouns

logger = logging.getLogger("whatsapp_analyzer")


def summarize_matrix(matrix: Dict) -> Dict:
    """
    Summarize analysis matrix to extract key insights.

    Processes conversation analysis results to identify positive/negative topics
    and calculate emotion variability.

    Args:
        matrix: Analysis matrix from run_analysis

    Returns:
        Dictionary with:
        - positive_topics: List of topics associated with positive sentiment
        - negative_topics: List of topics associated with negative sentiment
        - emotion_variability: Standard deviation of sentiment ratings
        - matrix: Original matrix (preserved)
        - analysis: Simplified analysis dict per conversation
    """
    logger.debug(f"summarize_matrix called with matrix_size={len(matrix)}")

    addtopic = []  # Positive topics
    negtopic = []  # Negative topics
    emo_vars = []  # Emotion variability scores
    analysis = {}

    for idx, entry in matrix.items():
        emo_bew = entry.get("emo_bew", [])
        topic = entry.get("topic", [])
        sentiment = entry.get("sentiment", [])
        sent_rating = entry.get("sent_rating", [])
        words = entry.get("words", [])

        # Collect sentiment ratings for variability calculation
        if sent_rating:
            emo_vars.append(sent_rating)

        # Classify topics as positive or negative
        if topic and topic != ["no topic"]:
            if sent_rating:
                # Use sentiment rating if available
                if sent_rating[0] >= 5.0:
                    addtopic.append(topic)
                else:
                    negtopic.append(topic)
            elif emo_bew:
                # Fall back to emoji evaluation
                if emo_bew[0] in ["sehr positiv", "eher positiv"]:
                    addtopic.append(topic)
                else:
                    negtopic.append(topic)

        # Build simplified analysis entry
        analysis[idx] = {
            "topic": topic,
            "emojies": emo_bew,
            "sentiment": sentiment,
            "wordcloud": words,
            "big_five": entry.get("big_five", {}),
            "mbti": entry.get("mbti", ""),
            "emotion_analysis": entry.get("emotion_analysis", {}),
            "response_times": entry.get("response_times", {}),
            "emotional_reciprocity": entry.get("emotional_reciprocity", 0.5),
        }

    # Calculate emotion variability (standard deviation)
    emo_vars = [x for x in emo_vars if x]  # Filter empty values
    flat_emo_vars = [x[0] if isinstance(x, list) else x for x in emo_vars]

    if flat_emo_vars:
        mittelwert = sum(flat_emo_vars) / len(flat_emo_vars)
        varianz = sum((x - mittelwert) ** 2 for x in flat_emo_vars) / len(flat_emo_vars)
        std_abweichung = varianz**0.5
    else:
        std_abweichung = 0.0

    # Flatten topic lists
    flat_addtopic = [item for sublist in addtopic for item in sublist]
    flat_negtopic = [item for sublist in negtopic for item in sublist]

    # Extract nouns from topics
    addtopic_nouns = extract_nouns(" ".join(flat_addtopic))
    negtopic_nouns = extract_nouns(" ".join(flat_negtopic))

    # Clean topics: remove URLs
    url_pattern = re.compile(r"\b(?:https?://|www\.)?\S+\.\S+\b", re.IGNORECASE)
    addtopic_clean = url_pattern.sub("", " ".join(addtopic_nouns))
    negtopic_clean = url_pattern.sub("", " ".join(negtopic_nouns))

    # Clean topics: remove symbols and emojis
    symbol_pattern = regex.compile(r"[\p{S}\p{P}\p{Emoji}]", regex.UNICODE)
    addtopic_clean = symbol_pattern.sub("", addtopic_clean)
    negtopic_clean = symbol_pattern.sub("", negtopic_clean)

    # Deduplicate
    addtopic_set = set(addtopic_clean.split())
    negtopic_set = set(negtopic_clean.split())

    # Final extraction to ensure quality nouns
    addtopic_final = extract_nouns(" ".join(addtopic_set))
    negtopic_final = extract_nouns(" ".join(negtopic_set))

    logger.debug(
        f"summarize_matrix: positive_topics={len(addtopic_final)}, "
        f"negative_topics={len(negtopic_final)}"
    )

    return {
        "positive_topics": addtopic_final,
        "negative_topics": negtopic_final,
        "emotion_variability": std_abweichung,
        "matrix": matrix,
        "analysis": analysis,
    }
