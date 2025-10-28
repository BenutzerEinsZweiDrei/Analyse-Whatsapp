"""
Keyword extraction for WhatsApp Analyzer.

Provides gensim LDA-based topic modeling and simple TF-based fallback.
"""

import logging
from collections import Counter
from typing import List

from app.core.preprocessing import preprocess_text

logger = logging.getLogger("whatsapp_analyzer")

# Maximum weight for term frequency in keyword scoring
MAX_FREQ_WEIGHT = 0.5


def get_keywords_gensim(text: str, num_topics: int = 3, num_keywords: int = 5) -> List[str]:
    """
    Extract keywords using gensim LDA topic modeling with frequency weighting.

    Args:
        text: Input text to analyze
        num_topics: Number of topics to extract (default: 3)
        num_keywords: Number of keywords per topic to return (default: 5)

    Returns:
        List of keywords extracted from the most relevant topics
    """
    logger.debug("get_keywords_gensim called with gensim topic analysis")

    if not text or not text.strip():
        logger.debug("Empty text provided to get_keywords_gensim")
        return []

    try:
        from gensim import corpora
        from gensim.models import LdaModel
    except ImportError:
        logger.warning("gensim not available, falling back to simple TF")
        return get_keywords_simple_tf(text, num_keywords=num_keywords)

    # Preprocess the text to get tokens
    tokens = preprocess_text(text)

    if not tokens or len(tokens) < 3:
        logger.debug(f"Insufficient tokens after preprocessing: {len(tokens)}")
        return []

    # Calculate term frequencies for importance weighting
    token_freq = Counter(tokens)

    try:
        # Split tokens into smaller chunks to give LDA some structure
        chunk_size = max(10, len(tokens) // 5)
        chunks = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]

        # Filter out very small chunks
        chunks = [chunk for chunk in chunks if len(chunk) >= 3]

        if not chunks:
            logger.debug("No valid chunks created from tokens")
            return []

        # Create dictionary and corpus for gensim
        dictionary = corpora.Dictionary(chunks)

        # Filter extremes: keep terms that appear at least once but not everywhere
        dictionary.filter_extremes(no_below=1, no_above=0.7, keep_n=100)

        if len(dictionary) == 0:
            logger.debug("Dictionary is empty after filtering")
            return []

        # Create bag-of-words corpus
        corpus = [dictionary.doc2bow(chunk) for chunk in chunks]

        # Build LDA model with fewer topics for short texts
        actual_num_topics = min(num_topics, len(chunks), len(dictionary))
        if actual_num_topics < 1:
            logger.debug("Cannot create topics with current parameters")
            return []

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=actual_num_topics,
            random_state=42,
            passes=10,
            alpha="auto",
            per_word_topics=True,
        )

        # Extract keywords from all topics with probability weighting
        keyword_scores = {}
        for topic_id in range(actual_num_topics):
            topic_words = lda_model.show_topic(topic_id, topn=num_keywords)
            for word, prob in topic_words:
                if len(word) > 2:  # Avoid very short words
                    # Weight by both LDA probability and term frequency
                    freq_weight = min(token_freq.get(word, 1) / len(tokens), MAX_FREQ_WEIGHT)
                    combined_score = prob * (1 + freq_weight)
                    if word not in keyword_scores or combined_score > keyword_scores[word]:
                        keyword_scores[word] = combined_score

        # Sort by combined score and return top keywords
        sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, score in sorted_keywords]

        logger.debug(f"Extracted {len(keywords)} keywords using gensim LDA")
        return keywords[:15]  # Return top 15 keywords maximum

    except Exception as e:
        logger.exception(f"Error in gensim topic analysis: {e}")
        return []


def get_keywords_simple_tf(text: str, num_keywords: int = 15) -> List[str]:
    """
    Extract keywords using simple term frequency.

    Fallback method when gensim is not available or fails.

    Args:
        text: Input text to analyze
        num_keywords: Number of top keywords to return

    Returns:
        List of keywords sorted by frequency
    """
    logger.debug("get_keywords_simple_tf called")

    if not text or not text.strip():
        return []

    # Preprocess text
    tokens = preprocess_text(text)

    if not tokens:
        return []

    # Count frequencies
    freq_counter = Counter(tokens)

    # Filter short words and get top keywords
    keywords = []
    for word, count in freq_counter.most_common():
        if len(word) > 2:  # Skip very short words
            keywords.append(word)
        if len(keywords) >= num_keywords:
            break

    logger.debug(f"Extracted {len(keywords)} keywords using simple TF")
    return keywords


def get_keywords(text: str, num_topics: int = 3, num_keywords: int = 5) -> List[str]:
    """
    Extract keywords from text (wrapper function for backward compatibility).

    Tries gensim LDA first, falls back to simple TF if it fails.

    Args:
        text: Input text to analyze
        num_topics: Number of topics to extract (default: 3)
        num_keywords: Number of keywords per topic to return (default: 5)

    Returns:
        List of keywords
    """
    keywords = get_keywords_gensim(text, num_topics, num_keywords)
    if not keywords:
        # Fallback to simple TF
        keywords = get_keywords_simple_tf(text, num_keywords=15)
    return keywords
