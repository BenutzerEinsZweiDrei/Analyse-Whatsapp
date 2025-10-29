"""
Topic and keyword extraction.

Provides multiple methods for extracting topics and keywords:
- KeyBERT (embedding-based)
- YAKE (statistical)
- Simple TF-IDF baseline
- Topic modeling with coherence scores
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re

logger = logging.getLogger("whatsapp_analyzer.topic_extraction")

# Optional dependencies
try:
    import yake

    HAS_YAKE = True
except ImportError:
    HAS_YAKE = False
    logger.warning("YAKE not available - using simpler keyword extraction")

try:
    from keybert import KeyBERT

    HAS_KEYBERT = True
except ImportError:
    HAS_KEYBERT = False
    logger.warning("KeyBERT not available - using simpler keyword extraction")

try:
    from app.cache import cached_resource
except ImportError:

    def cached_resource(func):
        return func


@dataclass
class TopicResult:
    """Topic extraction result."""

    # Top keywords
    keywords: List[Tuple[str, float]]  # (keyword, score)

    # Topic coherence
    coherence_score: Optional[float]

    # Representative messages
    representative_messages: List[Dict]  # [{message_id, snippet}]

    # Method used
    method: str  # "tfidf", "yake", "keybert", "ensemble"


@cached_resource
def get_keybert_model():
    """
    Load and cache KeyBERT model.

    Returns:
        KeyBERT model or None
    """
    if not HAS_KEYBERT:
        return None

    try:
        logger.info("Loading KeyBERT model (cached)")
        return KeyBERT()
    except Exception as e:
        logger.warning(f"Failed to load KeyBERT: {e}")
        return None


def simple_tfidf_keywords(texts: List[str], top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract keywords using simple TF-IDF approach.

    Args:
        texts: List of text documents
        top_n: Number of top keywords to return

    Returns:
        List of (keyword, score) tuples
    """
    if not texts:
        return []

    # Tokenize and count
    word_freq = Counter()
    doc_freq = Counter()

    for text in texts:
        words = re.findall(r"\b\w+\b", text.lower())
        # Remove very short words and common stopwords
        words = [w for w in words if len(w) > 3]

        # Term frequency
        word_freq.update(words)

        # Document frequency
        unique_words = set(words)
        doc_freq.update(unique_words)

    # Calculate TF-IDF
    n_docs = len(texts)
    tfidf_scores = {}

    for word, tf in word_freq.items():
        df = doc_freq[word]
        idf = n_docs / df if df > 0 else 0
        tfidf_scores[word] = tf * idf

    # Sort and return top N
    sorted_keywords = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_keywords[:top_n]


def extract_keywords_yake(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract keywords using YAKE.

    Args:
        text: Input text
        top_n: Number of keywords to extract

    Returns:
        List of (keyword, score) tuples
    """
    if not HAS_YAKE or not text:
        return []

    try:
        # Initialize YAKE
        kw_extractor = yake.KeywordExtractor(
            lan="en",  # Language
            n=2,  # Max n-gram size
            dedupLim=0.7,  # Deduplication threshold
            top=top_n,
        )

        keywords = kw_extractor.extract_keywords(text)
        # YAKE returns (keyword, score) where lower score is better
        # Invert scores for consistency
        return [(kw, 1.0 - min(score, 1.0)) for kw, score in keywords]
    except Exception as e:
        logger.debug(f"YAKE extraction failed: {e}")
        return []


def extract_keywords_keybert(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Extract keywords using KeyBERT.

    Args:
        text: Input text
        top_n: Number of keywords to extract

    Returns:
        List of (keyword, score) tuples
    """
    if not HAS_KEYBERT or not text:
        return []

    model = get_keybert_model()
    if not model:
        return []

    try:
        keywords = model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=top_n,
            use_mmr=True,  # Maximal Marginal Relevance for diversity
            diversity=0.7,
        )
        return keywords
    except Exception as e:
        logger.debug(f"KeyBERT extraction failed: {e}")
        return []


def extract_topics(
    texts: List[str],
    message_ids: Optional[List[int]] = None,
    top_n: int = 10,
    use_advanced: bool = True,
) -> TopicResult:
    """
    Extract topics and keywords from texts.

    Args:
        texts: List of text documents
        message_ids: Optional list of message IDs
        top_n: Number of top keywords to return
        use_advanced: Whether to use advanced methods (KeyBERT, YAKE)

    Returns:
        TopicResult with extracted topics
    """
    if not texts:
        return TopicResult(
            keywords=[],
            coherence_score=0.0,
            representative_messages=[],
            method="empty",
        )

    # Combine all texts for keyword extraction
    combined_text = " ".join(texts)

    keywords = []
    method = "tfidf"

    # Try advanced methods first
    if use_advanced:
        # Try KeyBERT
        if HAS_KEYBERT:
            keywords = extract_keywords_keybert(combined_text, top_n)
            if keywords:
                method = "keybert"

        # Fallback to YAKE if KeyBERT failed
        if not keywords and HAS_YAKE:
            keywords = extract_keywords_yake(combined_text, top_n)
            if keywords:
                method = "yake"

    # Fallback to simple TF-IDF
    if not keywords:
        keywords = simple_tfidf_keywords(texts, top_n)
        method = "tfidf"

    # Calculate coherence (simplified)
    # Coherence measures how often keywords co-occur
    coherence = calculate_keyword_coherence(keywords, texts)

    # Find representative messages (messages containing most keywords)
    representative = find_representative_messages(keywords, texts, message_ids, top_k=3)

    return TopicResult(
        keywords=keywords,
        coherence_score=coherence,
        representative_messages=representative,
        method=method,
    )


def calculate_keyword_coherence(
    keywords: List[Tuple[str, float]], texts: List[str]
) -> float:
    """
    Calculate coherence score for keywords.

    Measures how often keywords co-occur in the same documents.

    Args:
        keywords: List of (keyword, score) tuples
        texts: List of text documents

    Returns:
        Coherence score (0 to 1)
    """
    if not keywords or not texts:
        return 0.0

    # Extract just the keywords
    kw_list = [kw for kw, score in keywords[:5]]  # Top 5 keywords

    # Count co-occurrences
    co_occurrence_count = 0
    total_pairs = 0

    for i in range(len(kw_list)):
        for j in range(i + 1, len(kw_list)):
            kw1 = kw_list[i].lower()
            kw2 = kw_list[j].lower()
            total_pairs += 1

            # Count documents where both keywords appear
            for text in texts:
                text_lower = text.lower()
                if kw1 in text_lower and kw2 in text_lower:
                    co_occurrence_count += 1
                    break  # Count each pair once per document set

    coherence = co_occurrence_count / total_pairs if total_pairs > 0 else 0.0
    return min(coherence, 1.0)


def find_representative_messages(
    keywords: List[Tuple[str, float]],
    texts: List[str],
    message_ids: Optional[List[int]],
    top_k: int = 3,
) -> List[Dict]:
    """
    Find messages that best represent the topics.

    Args:
        keywords: List of (keyword, score) tuples
        texts: List of text documents
        message_ids: Optional list of message IDs
        top_k: Number of representative messages to return

    Returns:
        List of representative message dicts
    """
    if not keywords or not texts:
        return []

    # Score each message by keyword coverage
    kw_list = [kw.lower() for kw, score in keywords[:10]]
    message_scores = []

    for idx, text in enumerate(texts):
        text_lower = text.lower()
        score = sum(1 for kw in kw_list if kw in text_lower)

        if score > 0:
            # Create snippet
            snippet = text[:100] + "..." if len(text) > 100 else text
            msg_id = message_ids[idx] if message_ids and idx < len(message_ids) else idx

            message_scores.append({"message_id": msg_id, "snippet": snippet, "score": score})

    # Sort by score and return top K
    message_scores.sort(key=lambda x: x["score"], reverse=True)
    return [{"message_id": m["message_id"], "snippet": m["snippet"]} for m in message_scores[:top_k]]


def aggregate_topics_by_sentiment(
    texts: List[str],
    sentiments: List[str],
    message_ids: Optional[List[int]] = None,
    top_n: int = 5,
) -> Dict[str, TopicResult]:
    """
    Extract topics separately for each sentiment category.

    Args:
        texts: List of texts
        sentiments: List of sentiment labels for each text
        message_ids: Optional list of message IDs
        top_n: Number of keywords per sentiment

    Returns:
        Dictionary mapping sentiment to TopicResult
    """
    # Group by sentiment
    sentiment_groups = {}
    for i, (text, sentiment) in enumerate(zip(texts, sentiments)):
        if sentiment not in sentiment_groups:
            sentiment_groups[sentiment] = {"texts": [], "ids": []}

        sentiment_groups[sentiment]["texts"].append(text)
        sentiment_groups[sentiment]["ids"].append(message_ids[i] if message_ids else i)

    # Extract topics for each sentiment
    results = {}
    for sentiment, data in sentiment_groups.items():
        result = extract_topics(data["texts"], data["ids"], top_n=top_n)
        results[sentiment] = result

    return results
