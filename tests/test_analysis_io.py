"""
Tests for analysis.io module.
"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.io import (
    normalize_data,
    normalize_emotion_counts,
    normalize_mbti_data,
    normalize_response_time,
    normalize_trait_value,
)


class TestNormalizeTraitValue:
    """Test normalize_trait_value function."""

    def test_direct_float(self):
        """Test normalization of direct float value."""
        assert normalize_trait_value(0.75) == 0.75

    def test_direct_int(self):
        """Test normalization of direct int value."""
        assert normalize_trait_value(1) == 1.0

    def test_dict_with_mean(self):
        """Test normalization of dict with 'mean' key."""
        assert normalize_trait_value({"mean": 0.82}) == 0.82

    def test_dict_with_value(self):
        """Test normalization of dict with 'value' key."""
        assert normalize_trait_value({"value": 0.65}) == 0.65

    def test_dict_with_score(self):
        """Test normalization of dict with 'score' key."""
        assert normalize_trait_value({"score": 0.90}) == 0.90

    def test_none_value(self):
        """Test normalization of None."""
        assert normalize_trait_value(None) is None

    def test_invalid_value(self):
        """Test normalization of invalid value."""
        assert normalize_trait_value("invalid") is None
        assert normalize_trait_value({}) is None


class TestNormalizeEmotionCounts:
    """Test normalize_emotion_counts function."""

    def test_simple_counts(self):
        """Test normalization of simple count dict."""
        counts = {"joy": 10, "sadness": 5}
        result = normalize_emotion_counts(counts)
        assert result == {"joy": 10, "sadness": 5}

    def test_nested_counts(self):
        """Test normalization of nested count dict."""
        counts = {"joy": {"count": 15}, "sadness": {"count": 8}}
        result = normalize_emotion_counts(counts)
        assert result == {"joy": 15, "sadness": 8}

    def test_invalid_counts(self):
        """Test handling of invalid count values."""
        counts = {"joy": "invalid", "sadness": None, "anger": 5}
        result = normalize_emotion_counts(counts)
        assert result == {"joy": 0, "sadness": 0, "anger": 5}


class TestNormalizeMBTIData:
    """Test normalize_mbti_data function."""

    def test_simple_format(self):
        """Test normalization of simple MBTI format."""
        mbti_data = {"INTJ": 10, "ENFP": 5}
        result = normalize_mbti_data(mbti_data)
        assert result["INTJ"]["count"] == 10
        assert result["ENFP"]["count"] == 5

    def test_nested_format(self):
        """Test normalization of nested MBTI format."""
        mbti_data = {
            "INTJ": {"count": 10, "mean_reciprocity": 0.75, "mean_response_time": 120}
        }
        result = normalize_mbti_data(mbti_data)
        assert result["INTJ"]["count"] == 10
        assert result["INTJ"]["mean_reciprocity"] == 0.75
        assert result["INTJ"]["mean_response_time"] == 120


class TestNormalizeResponseTime:
    """Test normalize_response_time function."""

    def test_seconds(self):
        """Test normalization of time in seconds."""
        assert normalize_response_time(120.5) == 120.5

    def test_milliseconds(self):
        """Test normalization of time in milliseconds."""
        assert normalize_response_time(120500) == 120.5

    def test_none_value(self):
        """Test normalization of None."""
        assert normalize_response_time(None) is None


class TestNormalizeData:
    """Test normalize_data function."""

    def test_full_normalization(self):
        """Test normalization of complete profile data."""
        data = {
            "personality_aggregation": {"openness": 0.75, "conscientiousness": {"mean": 0.65}},
            "basic_metrics": {
                "dominant_emotion_counts": {"joy": 10, "sadness": {"count": 5}},
                "emotional_reciprocity": 0.68,
                "mean_response_time": 145.5,
            },
            "mbti_summary": {"INTJ": 15, "ENFP": {"count": 8}},
        }

        result = normalize_data(data)

        # Check personality traits
        assert result["personality_aggregation"]["openness"] == 0.75
        assert result["personality_aggregation"]["conscientiousness"] == 0.65

        # Check emotions
        assert result["basic_metrics"]["dominant_emotion_counts"]["joy"] == 10
        assert result["basic_metrics"]["dominant_emotion_counts"]["sadness"] == 5

        # Check reciprocity and response time
        assert result["basic_metrics"]["emotional_reciprocity"] == 0.68
        assert result["basic_metrics"]["mean_response_time"] == 145.5

        # Check MBTI
        assert result["mbti_summary"]["INTJ"]["count"] == 15
        assert result["mbti_summary"]["ENFP"]["count"] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
