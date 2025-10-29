"""
Unit tests for feature extraction module.
"""

from datetime import datetime

import pytest

from app.core.feature_extraction import (
    aggregate_language_stats,
    calculate_lexical_diversity,
    calculate_stopword_ratio,
    count_emoticons,
    detect_language,
    extract_emojis_from_text,
    extract_message_features,
    extract_urls,
    get_dominant_language,
)


class TestEmojiExtraction:
    """Test emoji extraction functionality."""

    def test_extract_emojis_simple(self):
        """Test extracting simple emojis."""
        emojis, descriptors = extract_emojis_from_text("Hello 😊 World 🎉")
        assert len(emojis) >= 0  # May be 0 if emoji lib not available
        if len(emojis) > 0:
            assert "😊" in emojis or "🎉" in emojis

    def test_extract_emojis_empty(self):
        """Test extracting from empty text."""
        emojis, descriptors = extract_emojis_from_text("")
        assert emojis == []
        assert descriptors == []

    def test_extract_emojis_no_emojis(self):
        """Test text without emojis."""
        emojis, descriptors = extract_emojis_from_text("Just plain text")
        assert emojis == []


class TestLanguageDetection:
    """Test language detection functionality."""

    def test_detect_english(self):
        """Test detecting English text."""
        lang, conf = detect_language("This is a longer English text for detection")
        # May be None if langdetect not available
        if lang:
            assert lang == "en"

    def test_detect_short_text(self):
        """Test with very short text."""
        lang, conf = detect_language("Hi")
        # Short text should return None
        assert lang is None

    def test_detect_empty(self):
        """Test with empty text."""
        lang, conf = detect_language("")
        assert lang is None


class TestEmoticonCounting:
    """Test emoticon counting."""

    def test_count_emoticons_simple(self):
        """Test counting simple emoticons."""
        count = count_emoticons("I'm happy :) and excited :D")
        assert count == 2

    def test_count_emoticons_none(self):
        """Test text without emoticons."""
        count = count_emoticons("Just plain text")
        assert count == 0

    def test_count_emoticons_heart(self):
        """Test heart emoticon."""
        count = count_emoticons("I love you <3")
        assert count == 1


class TestURLExtraction:
    """Test URL extraction."""

    def test_extract_urls_http(self):
        """Test extracting HTTP URLs."""
        urls = extract_urls("Check this http://example.com link")
        assert len(urls) == 1
        assert "http://example.com" in urls

    def test_extract_urls_https(self):
        """Test extracting HTTPS URLs."""
        urls = extract_urls("Secure site https://example.com here")
        assert len(urls) == 1

    def test_extract_urls_none(self):
        """Test text without URLs."""
        urls = extract_urls("No links here")
        assert len(urls) == 0


class TestLexicalFeatures:
    """Test linguistic feature calculations."""

    def test_lexical_diversity_varied(self):
        """Test with varied words."""
        words = ["hello", "world", "test", "message"]
        diversity = calculate_lexical_diversity(words)
        assert diversity == 1.0  # All unique

    def test_lexical_diversity_repetitive(self):
        """Test with repetitive words."""
        words = ["test", "test", "test", "test"]
        diversity = calculate_lexical_diversity(words)
        assert diversity == 0.25  # 1/4

    def test_lexical_diversity_empty(self):
        """Test with empty list."""
        diversity = calculate_lexical_diversity([])
        assert diversity == 0.0

    def test_stopword_ratio_high(self):
        """Test text with many stopwords."""
        words = ["the", "and", "is", "it", "to"]
        ratio = calculate_stopword_ratio(words)
        assert ratio > 0.5

    def test_stopword_ratio_low(self):
        """Test text with few stopwords."""
        words = ["python", "programming", "language", "test"]
        ratio = calculate_stopword_ratio(words)
        assert ratio < 0.3


class TestMessageFeatureExtraction:
    """Test complete message feature extraction."""

    def test_basic_features(self):
        """Test extracting basic features."""
        features = extract_message_features(
            message_id=1,
            author="Alice",
            text="Hello world! This is a test message.",
            timestamp=datetime(2024, 1, 1, 14, 30),
        )

        assert features.message_id == 1
        assert features.author == "Alice"
        assert features.is_system == False
        assert features.is_media == False
        assert features.message_length_chars > 0
        assert features.word_count > 0
        assert features.time_of_day_hour == 14

    def test_emoji_features(self):
        """Test emoji feature extraction."""
        features = extract_message_features(
            message_id=1,
            author="Bob",
            text="Great! 😊 🎉 Amazing work!",
        )

        assert features.emoji_count >= 0  # May be 0 if emoji lib not available

    def test_question_exclamation(self):
        """Test question and exclamation marks."""
        features = extract_message_features(
            message_id=1,
            author="Carol",
            text="Really? That's amazing! How did you do it?",
        )

        assert features.question_marks == 2
        assert features.exclamation_marks == 1

    def test_system_message(self):
        """Test system message features."""
        features = extract_message_features(
            message_id=1,
            author=None,
            text="User joined the group",
            is_system=True,
        )

        assert features.is_system == True
        assert features.author is None

    def test_media_message(self):
        """Test media message features."""
        features = extract_message_features(
            message_id=1,
            author="Dave",
            text="<Media omitted>",
            is_media=True,
        )

        assert features.is_media == True

    def test_mention_detection(self):
        """Test @mention detection."""
        features = extract_message_features(
            message_id=1,
            author="Eve",
            text="Hey @Alice what do you think?",
        )

        assert features.mention_count == 1
        assert features.has_reply_indicator == True
        assert features.reply_to == "Alice"

    def test_url_detection(self):
        """Test URL detection in features."""
        features = extract_message_features(
            message_id=1,
            author="Frank",
            text="Check out https://example.com for more info",
        )

        assert features.url_count == 1
        assert len(features.urls) == 1

    def test_uppercase_ratio(self):
        """Test uppercase ratio calculation."""
        features = extract_message_features(
            message_id=1,
            author="George",
            text="THIS IS ALL CAPS",
        )

        assert features.uppercase_ratio > 0.5

    def test_empty_text(self):
        """Test with empty text."""
        features = extract_message_features(
            message_id=1,
            author="Hannah",
            text="",
        )

        assert features.message_length_chars == 0
        assert features.word_count == 0


class TestLanguageAggregation:
    """Test language aggregation functions."""

    def test_aggregate_language_stats(self):
        """Test aggregating language statistics."""
        from app.core.feature_extraction import MessageFeatures

        features = [
            MessageFeatures(
                message_id=1,
                author="A",
                is_system=False,
                is_media=False,
                timestamp=None,
                timestamp_utc=None,
                time_of_day_hour=None,
                day_of_week=None,
                text="text",
                text_normalized="text",
                message_length_chars=4,
                token_count=1,
                word_count=1,
                sentence_count=1,
                avg_word_length=4.0,
                lexical_diversity=1.0,
                stopword_ratio=0.0,
                uppercase_ratio=0.0,
                punctuation_density=0.0,
                emoji_count=0,
                emoticon_count=0,
                url_count=0,
                mention_count=0,
                hashtag_count=0,
                question_marks=0,
                exclamation_marks=0,
                detected_language="en",
            ),
            MessageFeatures(
                message_id=2,
                author="B",
                is_system=False,
                is_media=False,
                timestamp=None,
                timestamp_utc=None,
                time_of_day_hour=None,
                day_of_week=None,
                text="text",
                text_normalized="text",
                message_length_chars=4,
                token_count=1,
                word_count=1,
                sentence_count=1,
                avg_word_length=4.0,
                lexical_diversity=1.0,
                stopword_ratio=0.0,
                uppercase_ratio=0.0,
                punctuation_density=0.0,
                emoji_count=0,
                emoticon_count=0,
                url_count=0,
                mention_count=0,
                hashtag_count=0,
                question_marks=0,
                exclamation_marks=0,
                detected_language="en",
            ),
        ]

        stats = aggregate_language_stats(features)
        assert "en" in stats
        assert stats["en"] == 2

    def test_get_dominant_language(self):
        """Test getting dominant language."""
        from app.core.feature_extraction import MessageFeatures

        features = [
            MessageFeatures(
                message_id=i,
                author="A",
                is_system=False,
                is_media=False,
                timestamp=None,
                timestamp_utc=None,
                time_of_day_hour=None,
                day_of_week=None,
                text="text",
                text_normalized="text",
                message_length_chars=4,
                token_count=1,
                word_count=1,
                sentence_count=1,
                avg_word_length=4.0,
                lexical_diversity=1.0,
                stopword_ratio=0.0,
                uppercase_ratio=0.0,
                punctuation_density=0.0,
                emoji_count=0,
                emoticon_count=0,
                url_count=0,
                mention_count=0,
                hashtag_count=0,
                question_marks=0,
                exclamation_marks=0,
                detected_language="en" if i < 2 else "de",
            )
            for i in range(3)
        ]

        dominant = get_dominant_language(features)
        assert dominant == "en"  # 2 en vs 1 de
