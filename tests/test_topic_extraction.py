"""
Unit tests for topic extraction.
"""

import pytest

from app.core.topic_extraction import (
    aggregate_topics_by_sentiment,
    calculate_keyword_coherence,
    extract_topics,
    find_representative_messages,
    simple_tfidf_keywords,
)


class TestSimpleTFIDF:
    """Test simple TF-IDF keyword extraction."""

    def test_basic_extraction(self):
        """Test extracting keywords from texts."""
        texts = [
            "python programming language is great",
            "python is a powerful programming tool",
            "learn python programming today",
        ]

        keywords = simple_tfidf_keywords(texts, top_n=5)

        # Should return list of tuples
        assert isinstance(keywords, list)
        assert all(isinstance(kw, tuple) and len(kw) == 2 for kw in keywords)

        # "python" and "programming" should be top keywords
        kw_words = [kw[0] for kw in keywords]
        assert "python" in kw_words or "programming" in kw_words

    def test_empty_texts(self):
        """Test with empty text list."""
        keywords = simple_tfidf_keywords([], top_n=5)
        assert keywords == []

    def test_single_text(self):
        """Test with single text."""
        texts = ["python programming language"]
        keywords = simple_tfidf_keywords(texts, top_n=5)
        assert len(keywords) > 0


class TestKeywordCoherence:
    """Test keyword coherence calculation."""

    def test_coherent_keywords(self):
        """Test with highly coherent keywords."""
        keywords = [("python", 1.0), ("programming", 0.9), ("code", 0.8)]
        texts = [
            "python programming code",
            "python code programming",
            "programming python code",
        ]

        coherence = calculate_keyword_coherence(keywords, texts)
        # Keywords co-occur frequently, should have high coherence
        assert coherence > 0

    def test_incoherent_keywords(self):
        """Test with incoherent keywords."""
        keywords = [("apple", 1.0), ("car", 0.9), ("book", 0.8)]
        texts = [
            "I like apples",
            "Nice car today",
            "Reading a book",
        ]

        coherence = calculate_keyword_coherence(keywords, texts)
        # Keywords don't co-occur, should have low coherence
        assert coherence >= 0

    def test_empty_keywords(self):
        """Test with empty keywords."""
        coherence = calculate_keyword_coherence([], ["text"])
        assert coherence == 0.0


class TestRepresentativeMessages:
    """Test finding representative messages."""

    def test_find_representatives(self):
        """Test finding messages with most keywords."""
        keywords = [("python", 1.0), ("programming", 0.9), ("language", 0.8)]
        texts = [
            "python programming language tutorial",
            "hello world",
            "python is a programming language",
        ]
        message_ids = [1, 2, 3]

        representatives = find_representative_messages(keywords, texts, message_ids, top_k=2)

        # Should return list of dicts
        assert isinstance(representatives, list)
        assert len(representatives) <= 2

        # Check structure
        for rep in representatives:
            assert "message_id" in rep
            assert "snippet" in rep

    def test_no_matches(self):
        """Test when no messages match keywords."""
        keywords = [("python", 1.0)]
        texts = ["hello world", "goodbye"]

        representatives = find_representative_messages(keywords, texts, None, top_k=2)

        # Should return empty or very low scoring matches
        assert isinstance(representatives, list)

    def test_empty_keywords(self):
        """Test with empty keywords."""
        representatives = find_representative_messages([], ["text"], None, top_k=2)
        assert representatives == []


class TestTopicExtraction:
    """Test complete topic extraction."""

    def test_basic_extraction(self):
        """Test basic topic extraction."""
        texts = [
            "python programming is fun",
            "learn python coding today",
            "python development tutorial",
        ]

        result = extract_topics(texts, top_n=5, use_advanced=False)

        # Check structure
        assert result.keywords is not None
        assert isinstance(result.keywords, list)
        assert result.coherence_score is not None
        assert isinstance(result.representative_messages, list)
        assert result.method in ["tfidf", "yake", "keybert", "empty"]

    def test_with_message_ids(self):
        """Test with message IDs."""
        texts = ["hello world", "python programming"]
        message_ids = [100, 200]

        result = extract_topics(texts, message_ids=message_ids, top_n=3, use_advanced=False)

        # Representative messages should have correct IDs
        for rep in result.representative_messages:
            assert rep["message_id"] in [100, 200]

    def test_empty_texts(self):
        """Test with empty text list."""
        result = extract_topics([], top_n=5)

        assert result.keywords == []
        assert result.method == "empty"

    def test_single_text(self):
        """Test with single text."""
        texts = ["python programming language tutorial"]

        result = extract_topics(texts, top_n=5, use_advanced=False)

        assert len(result.keywords) > 0
        assert result.coherence_score is not None


class TestTopicsBySentiment:
    """Test topic extraction by sentiment."""

    def test_aggregate_by_sentiment(self):
        """Test aggregating topics by sentiment."""
        texts = [
            "I love python programming",
            "This is terrible code",
            "python is great for data science",
        ]
        sentiments = ["positive", "negative", "positive"]
        message_ids = [1, 2, 3]

        results = aggregate_topics_by_sentiment(texts, sentiments, message_ids, top_n=3)

        # Should return dict with sentiment keys
        assert isinstance(results, dict)
        assert "positive" in results
        assert "negative" in results

        # Each result should be a TopicResult
        for sentiment, topic_result in results.items():
            assert hasattr(topic_result, "keywords")
            assert hasattr(topic_result, "coherence_score")

    def test_single_sentiment(self):
        """Test with all same sentiment."""
        texts = ["good", "great", "excellent"]
        sentiments = ["positive", "positive", "positive"]

        results = aggregate_topics_by_sentiment(texts, sentiments, top_n=3)

        assert "positive" in results
        assert len(results) == 1

    def test_empty_texts(self):
        """Test with empty lists."""
        results = aggregate_topics_by_sentiment([], [], top_n=3)

        assert isinstance(results, dict)
        assert len(results) == 0
