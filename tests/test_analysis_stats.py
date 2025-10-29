"""
Tests for analysis.stats module.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.stats import compute_aggregated_statistics


class TestComputeAggregatedStatistics:
    """Test compute_aggregated_statistics function."""

    def test_basic_statistics(self):
        """Test basic statistical computation."""
        data_list = [
            {
                "personality_aggregation": {
                    "openness": 0.75,
                    "conscientiousness": 0.65,
                    "extraversion": 0.82,
                },
                "basic_metrics": {
                    "dominant_emotion_counts": {"joy": 10, "sadness": 5},
                    "emotional_reciprocity": 0.68,
                    "mean_response_time": 145.5,
                },
                "mbti_summary": {"INTJ": {"count": 15}},
            },
            {
                "personality_aggregation": {
                    "openness": 0.68,
                    "conscientiousness": 0.78,
                    "extraversion": 0.55,
                },
                "basic_metrics": {
                    "dominant_emotion_counts": {"joy": 20, "anger": 8},
                    "emotional_reciprocity": 0.75,
                    "mean_response_time": 98.3,
                },
                "mbti_summary": {"INTJ": {"count": 10}, "ENFP": {"count": 5}},
            },
        ]

        result = compute_aggregated_statistics(data_list)

        # Check reciprocity stats
        assert result["reciprocity"]["mean"] is not None
        assert result["reciprocity"]["std"] is not None

        # Check response time stats
        assert result["response_time"]["mean"] is not None
        assert result["response_time"]["std"] is not None

        # Check top emotions
        assert len(result["top_emotions"]) > 0
        assert result["top_emotions"][0]["emotion"] == "joy"  # Most common

        # Check MBTI percentages
        assert "INTJ" in result["mbti_percentages"]

    def test_trait_rankings(self):
        """Test trait ranking computation."""
        data_list = [
            {
                "personality_aggregation": {
                    "openness": 0.90,  # Highest
                    "conscientiousness": 0.65,
                    "extraversion": 0.50,  # Lowest
                }
            },
            {
                "personality_aggregation": {
                    "openness": 0.85,  # Highest
                    "conscientiousness": 0.70,
                    "extraversion": 0.55,  # Lowest
                }
            },
        ]

        result = compute_aggregated_statistics(data_list)

        # Most common top trait should be openness
        assert result["trait_rankings"]["most_common_top"] == "openness"
        # Most common bottom trait should be extraversion
        assert result["trait_rankings"]["most_common_bottom"] == "extraversion"

    def test_empty_data(self):
        """Test handling of empty data list."""
        result = compute_aggregated_statistics([])

        assert result["reciprocity"]["mean"] is None
        assert result["response_time"]["mean"] is None
        assert len(result["top_emotions"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
