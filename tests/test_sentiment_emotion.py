"""
Unit tests for enhanced sentiment and emotion detection.
"""

import pytest

from app.core.emotion_detection import (
    batch_detect_emotion,
    detect_emotion,
    emoji_emotion_mapping,
    lexicon_based_emotion,
)
from app.core.sentiment_enhanced import (
    analyze_sentiment_ensemble,
    batch_analyze_sentiment,
    emoji_sentiment_score,
)


class TestEmojiSentiment:
    """Test emoji sentiment scoring."""

    def test_positive_emojis(self):
        """Test positive emoji sentiment."""
        emojis = ["😊", "😃", "🎉"]
        score, count = emoji_sentiment_score(emojis)
        assert score > 0  # Should be positive
        assert count == 3

    def test_negative_emojis(self):
        """Test negative emoji sentiment."""
        emojis = ["😢", "😭", "😞"]
        score, count = emoji_sentiment_score(emojis)
        assert score < 0  # Should be negative
        assert count == 3

    def test_mixed_emojis(self):
        """Test mixed emoji sentiment."""
        emojis = ["😊", "😢"]
        score, count = emoji_sentiment_score(emojis)
        assert count == 2
        # Score should be somewhere in between

    def test_no_emojis(self):
        """Test with no emojis."""
        score, count = emoji_sentiment_score([])
        assert score == 0.0
        assert count == 0


class TestSentimentEnsemble:
    """Test ensemble sentiment analysis."""

    def test_positive_text(self):
        """Test positive sentiment detection."""
        result = analyze_sentiment_ensemble("I love this! It's amazing and wonderful!")
        assert result.polarity == "positive"
        assert result.compound_score > 0
        assert result.confidence > 0

    def test_negative_text(self):
        """Test negative sentiment detection."""
        result = analyze_sentiment_ensemble("This is terrible. I hate it so much.")
        assert result.polarity == "negative"
        assert result.compound_score < 0
        assert result.confidence > 0

    def test_neutral_text(self):
        """Test neutral sentiment detection."""
        result = analyze_sentiment_ensemble("The meeting is at 3 PM.")
        assert result.polarity == "neutral"
        assert result.confidence > 0

    def test_empty_text(self):
        """Test with empty text."""
        result = analyze_sentiment_ensemble("")
        assert result.polarity == "neutral"
        assert result.confidence == 0.0
        assert result.method_used == "empty"

    def test_with_emojis(self):
        """Test sentiment with emojis."""
        result = analyze_sentiment_ensemble(
            "This is okay", emojis=["😊", "🎉"], use_transformer=False
        )
        # Emojis should boost positive sentiment
        assert result.compound_score > 0

    def test_confidence_scoring(self):
        """Test that confidence is calculated."""
        result = analyze_sentiment_ensemble("Great product!", use_transformer=False)
        assert 0 <= result.confidence <= 1
        assert result.method_used in ["vader", "vader_emoji", "ensemble"]


class TestBatchSentiment:
    """Test batch sentiment analysis."""

    def test_batch_analysis(self):
        """Test analyzing multiple texts."""
        texts = ["I love this!", "This is terrible.", "The sky is blue."]
        results = batch_analyze_sentiment(texts, use_transformer=False)

        assert len(results) == 3
        assert results[0].polarity == "positive"
        assert results[1].polarity == "negative"
        # Third should be neutral or slightly positive/negative

    def test_empty_batch(self):
        """Test with empty list."""
        results = batch_analyze_sentiment([])
        assert len(results) == 0


class TestEmojiEmotion:
    """Test emoji emotion mapping."""

    def test_joy_emojis(self):
        """Test joy emoji mapping."""
        emojis = ["😊", "😃", "🎉"]
        scores = emoji_emotion_mapping(emojis)
        assert scores["joy"] > 0

    def test_sadness_emojis(self):
        """Test sadness emoji mapping."""
        emojis = ["😢", "😭"]
        scores = emoji_emotion_mapping(emojis)
        assert scores["sadness"] > 0

    def test_anger_emojis(self):
        """Test anger emoji mapping."""
        emojis = ["😡", "😠"]
        scores = emoji_emotion_mapping(emojis)
        assert scores["anger"] > 0

    def test_no_emojis(self):
        """Test with no emojis."""
        scores = emoji_emotion_mapping([])
        # All scores should be 0
        for emotion, score in scores.items():
            assert score == 0.0


class TestLexiconEmotion:
    """Test lexicon-based emotion detection."""

    def test_joy_keywords(self):
        """Test joy emotion from keywords."""
        scores = lexicon_based_emotion("I am so happy and delighted!")
        assert scores["joy"] > 0

    def test_sadness_keywords(self):
        """Test sadness emotion from keywords."""
        scores = lexicon_based_emotion("I feel so sad and depressed.")
        assert scores["sadness"] > 0

    def test_anger_keywords(self):
        """Test anger emotion from keywords."""
        scores = lexicon_based_emotion("I am so angry and furious!")
        assert scores["anger"] > 0

    def test_fear_keywords(self):
        """Test fear emotion from keywords."""
        scores = lexicon_based_emotion("I am scared and anxious.")
        assert scores["fear"] > 0

    def test_neutral_text(self):
        """Test neutral text."""
        scores = lexicon_based_emotion("The meeting is scheduled for Monday.")
        # Should default to neutral
        assert scores["neutral"] > 0

    def test_with_emojis(self):
        """Test with both text and emojis."""
        scores = lexicon_based_emotion("I'm okay", emojis=["😊"])
        # Should have some joy from emoji
        assert scores["joy"] > 0


class TestEmotionDetection:
    """Test complete emotion detection."""

    def test_joy_detection(self):
        """Test detecting joy."""
        result = detect_emotion("I am so happy and excited!", use_transformer=False)
        assert result.primary_emotion in ["joy", "neutral"]
        assert result.confidence > 0

    def test_sadness_detection(self):
        """Test detecting sadness."""
        result = detect_emotion("I feel terrible and sad.", use_transformer=False)
        assert result.primary_emotion in ["sadness", "neutral"]
        assert result.confidence > 0

    def test_anger_detection(self):
        """Test detecting anger."""
        result = detect_emotion("This makes me so angry!", use_transformer=False)
        assert result.primary_emotion in ["anger", "neutral"]
        assert result.confidence > 0

    def test_empty_text(self):
        """Test with empty text."""
        result = detect_emotion("")
        assert result.primary_emotion == "neutral"
        assert result.primary_score == 1.0
        assert result.method_used == "empty"

    def test_emotion_scores(self):
        """Test that emotion scores are provided."""
        result = detect_emotion("I love this!", use_transformer=False)
        assert isinstance(result.emotion_scores, dict)
        assert "joy" in result.emotion_scores
        assert all(0 <= score <= 1 for score in result.emotion_scores.values())

    def test_confidence_range(self):
        """Test confidence is in valid range."""
        result = detect_emotion("Happy day!", use_transformer=False)
        assert 0 <= result.confidence <= 1


class TestBatchEmotion:
    """Test batch emotion detection."""

    def test_batch_detection(self):
        """Test detecting emotions in multiple texts."""
        texts = ["I'm so happy!", "This is terrible.", "Normal day."]
        results = batch_detect_emotion(texts, use_transformer=False)

        assert len(results) == 3
        assert all(r.confidence > 0 for r in results)

    def test_empty_batch(self):
        """Test with empty list."""
        results = batch_detect_emotion([])
        assert len(results) == 0

    def test_with_emojis_batch(self):
        """Test batch with emojis."""
        texts = ["Hello", "World"]
        emojis_list = [["😊"], ["😢"]]
        results = batch_detect_emotion(texts, emojis_list, use_transformer=False)

        assert len(results) == 2
