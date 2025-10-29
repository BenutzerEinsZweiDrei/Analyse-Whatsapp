"""
Matrix summarization for WhatsApp Analyzer.

Aggregates analysis results and generates summary statistics.
"""

import logging
import re
from collections import Counter

import regex

from app.core.nouns import extract_nouns

logger = logging.getLogger("whatsapp_analyzer")


def summarize_matrix(matrix: dict) -> dict:
    """
    Summarize analysis matrix to extract key insights.

    Processes conversation analysis results to identify positive/negative topics
    and calculate emotion variability.

    Args:
        matrix: Analysis matrix from run_analysis

    Returns:
        Dictionary with (BACKWARD COMPATIBLE - existing keys preserved):
        - positive_topics: List of topics associated with positive sentiment
        - negative_topics: List of topics associated with negative sentiment
        - emotion_variability: Standard deviation of sentiment ratings
        - matrix: Original matrix (preserved)
        - analysis: Simplified analysis dict per conversation

        NEW FIELDS (optional, enrichment):
        - top_keywords: List of dicts with {keyword, score, support}
        - topic_coherence_scores: Dict mapping topic->coherence score
        - per_author_stats: Dict of author->{message_count, avg_sentiment, etc}
        - feature_summary: Dict with aggregate linguistic features
        - summary_confidence: Overall confidence in analysis (0-1)
        - analysis_text: Brief human-readable summary for AI generation
    """
    logger.debug(f"summarize_matrix called with matrix_size={len(matrix)}")

    addtopic = []  # Positive topics
    negtopic = []  # Negative topics
    emo_vars = []  # Emotion variability scores
    analysis = {}

    # New aggregations
    all_keywords = Counter()
    per_author_data = {}
    sentiment_scores = []

    for idx, entry in matrix.items():
        emo_bew = entry.get("emo_bew", [])
        topic = entry.get("topic", [])
        sentiment = entry.get("sentiment", [])
        sent_rating = entry.get("sent_rating", [])
        words = entry.get("words", [])

        # Collect sentiment ratings for variability calculation
        if sent_rating:
            emo_vars.append(sent_rating)
            # Track sentiment scores for confidence
            rating = sent_rating[0] if isinstance(sent_rating, list) else sent_rating
            if isinstance(rating, (int, float)):
                sentiment_scores.append(rating)

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

        # Aggregate keywords from wordcloud
        if words:
            word_list = words if isinstance(words, list) else [words]
            for word in word_list:
                if isinstance(word, str) and len(word) > 2:
                    all_keywords[word.lower()] += 1

        # Per-author statistics (if available)
        # Note: Matrix may not always have author info, so we handle missing data gracefully
        if "messages" in entry:
            for msg in entry.get("messages", []):
                author = msg.get("user") or msg.get("author")
                if author:
                    if author not in per_author_data:
                        per_author_data[author] = {
                            "message_count": 0,
                            "total_sentiment": 0.0,
                            "total_length": 0,
                            "emoji_count": 0,
                            "question_count": 0,
                        }

                    per_author_data[author]["message_count"] += 1

                    # Add sentiment if available
                    if sent_rating and len(sentiment_scores) > 0:
                        per_author_data[author]["total_sentiment"] += sentiment_scores[-1]

                    # Add length
                    msg_text = msg.get("message", "")
                    per_author_data[author]["total_length"] += len(msg_text)

                    # Count emojis and questions
                    per_author_data[author]["emoji_count"] += msg_text.count("😊") + msg_text.count(
                        "🎉"
                    )  # Simplified
                    per_author_data[author]["question_count"] += msg_text.count("?")

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

    # NEW ENRICHMENT FIELDS

    # Top keywords with scores
    top_keywords = []
    for keyword, count in all_keywords.most_common(15):
        top_keywords.append(
            {
                "keyword": str(keyword),
                "score": float(count / len(matrix)) if matrix else 0.0,
                "support": int(count),
            }
        )

    # Topic coherence (simplified - percentage of conversations with clear topics)
    topics_with_content = sum(
        1 for entry in matrix.values() if entry.get("topic") and entry.get("topic") != ["no topic"]
    )
    topic_coherence = float(topics_with_content / len(matrix)) if matrix else 0.0

    # Per-author statistics (normalized)
    per_author_stats = {}
    for author, data in per_author_data.items():
        msg_count = data["message_count"]
        per_author_stats[str(author)] = {
            "message_count": int(msg_count),
            "avg_sentiment": float(data["total_sentiment"] / msg_count) if msg_count > 0 else 5.0,
            "avg_length": float(data["total_length"] / msg_count) if msg_count > 0 else 0.0,
            "emoji_ratio": float(data["emoji_count"] / msg_count) if msg_count > 0 else 0.0,
            "question_ratio": float(data["question_count"] / msg_count) if msg_count > 0 else 0.0,
        }

    # Feature summary
    feature_summary = {
        "total_conversations": int(len(matrix)),
        "avg_sentiment": (
            float(sum(sentiment_scores) / len(sentiment_scores)) if sentiment_scores else 5.0
        ),
        "sentiment_variability": float(std_abweichung),
        "keyword_diversity": int(len(all_keywords)),
        "topic_coverage": float(topic_coherence),
    }

    # Summary confidence (based on data availability and quality)
    confidence_factors = []

    # Factor 1: Data completeness
    if len(matrix) > 5:
        confidence_factors.append(0.9)
    elif len(matrix) > 2:
        confidence_factors.append(0.7)
    else:
        confidence_factors.append(0.5)

    # Factor 2: Topic clarity
    confidence_factors.append(float(topic_coherence))

    # Factor 3: Sentiment data availability
    if sentiment_scores:
        confidence_factors.append(0.9)
    else:
        confidence_factors.append(0.5)

    summary_confidence = (
        float(sum(confidence_factors) / len(confidence_factors)) if confidence_factors else 0.5
    )

    # Generate analysis text for AI
    analysis_text = _generate_analysis_text(
        addtopic_final,
        negtopic_final,
        std_abweichung,
        top_keywords,
        per_author_stats,
    )

    return {
        # EXISTING KEYS (backward compatible)
        "positive_topics": addtopic_final,
        "negative_topics": negtopic_final,
        "emotion_variability": std_abweichung,
        "matrix": matrix,
        "analysis": analysis,
        # NEW KEYS (enrichment)
        "top_keywords": top_keywords,
        "topic_coherence_scores": {"overall": topic_coherence},
        "per_author_stats": per_author_stats,
        "feature_summary": feature_summary,
        "summary_confidence": summary_confidence,
        "analysis_text": analysis_text,
    }


def _generate_analysis_text(
    positive_topics: list[str],
    negative_topics: list[str],
    emotion_variability: float,
    top_keywords: list[dict],
    per_author_stats: dict,
) -> str:
    """
    Generate human-readable analysis text for AI generation.

    Args:
        positive_topics: List of positive topic nouns
        negative_topics: List of negative topic nouns
        emotion_variability: Emotion variability score
        top_keywords: Top keywords with scores
        per_author_stats: Per-author statistics

    Returns:
        Brief analysis summary text
    """
    parts = []

    # Topics
    if positive_topics:
        parts.append(f"Positive topics: {', '.join(positive_topics[:5])}")
    if negative_topics:
        parts.append(f"Negative topics: {', '.join(negative_topics[:5])}")

    # Emotion variability
    if emotion_variability > 2.0:
        parts.append(f"High emotion variability ({emotion_variability:.2f})")
    elif emotion_variability < 0.5:
        parts.append(f"Low emotion variability ({emotion_variability:.2f})")
    else:
        parts.append(f"Moderate emotion variability ({emotion_variability:.2f})")

    # Keywords
    if top_keywords:
        kw_str = ", ".join(kw["keyword"] for kw in top_keywords[:5])
        parts.append(f"Key terms: {kw_str}")

    # Authors
    if per_author_stats:
        n_authors = len(per_author_stats)
        parts.append(f"{n_authors} participant(s)")

    return ". ".join(parts) + "."
