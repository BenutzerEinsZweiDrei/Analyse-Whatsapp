"""
Integration tests for Profile Fusion v2.0.
"""

import json
import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import (
    compute_aggregated_statistics,
    compute_correlations,
    create_profile_matrix,
    generate_emotional_insights,
    generate_topic_insights,
    merge_all_data,
)


class TestProfileFusionIntegration:
    """Integration tests using fixture data."""

    @pytest.fixture
    def sample_profiles(self):
        """Load sample profile fixtures."""
        fixtures_dir = Path(__file__).parent / "fixtures"
        profiles = []

        for i in [1, 2]:
            with open(fixtures_dir / f"profile{i}.json") as f:
                profiles.append(json.load(f))

        return profiles

    def test_full_merge_workflow(self, sample_profiles):
        """Test complete merge workflow with fixtures."""
        # Merge data
        merged = merge_all_data(sample_profiles)

        # Verify Big Five merged
        assert not merged["big_five"].empty
        assert "trait" in merged["big_five"].columns
        assert "mean" in merged["big_five"].columns
        assert "std" in merged["big_five"].columns

        # Verify emotions merged
        assert not merged["emotions"].empty
        assert "joy" in merged["emotions"].index

        # Verify MBTI merged
        assert not merged["mbti"].empty
        assert "INTJ" in merged["mbti"]["mbti_type"].values

    def test_aggregated_statistics(self, sample_profiles):
        """Test aggregated statistics computation."""
        stats = compute_aggregated_statistics(sample_profiles)

        # Check required keys
        assert "reciprocity" in stats
        assert "response_time" in stats
        assert "top_emotions" in stats
        assert "mbti_percentages" in stats
        assert "trait_rankings" in stats

        # Check top emotions detection
        assert len(stats["top_emotions"]) > 0
        top_emotion = stats["top_emotions"][0]
        assert "emotion" in top_emotion
        assert "count" in top_emotion

    def test_correlations(self, sample_profiles):
        """Test correlation analysis."""
        correlations = compute_correlations(sample_profiles, p_threshold=0.05)

        # Check required keys
        assert "all_correlations" in correlations
        assert "significant_correlations" in correlations
        assert "trait_behavior_summary" in correlations

        # Should have some correlations from fixture data
        assert len(correlations["all_correlations"]) > 0

    def test_profile_matrix(self, sample_profiles):
        """Test profile matrix creation."""
        filenames = ["Profile1.json", "Profile2.json"]
        matrix = create_profile_matrix(sample_profiles, filenames)

        # Check matrix structure
        assert not matrix.empty
        assert "Profile1.json" in matrix.columns
        assert "Profile2.json" in matrix.columns

        # Check features present
        assert "openness" in matrix.index
        assert "reciprocity" in matrix.index

    def test_emotional_insights(self, sample_profiles):
        """Test emotional insights generation."""
        insights = generate_emotional_insights(sample_profiles)

        # Check required keys
        assert "top_emotions" in insights
        assert "emotion_details" in insights

        # Should detect joy as top emotion
        assert len(insights["top_emotions"]) > 0
        assert insights["top_emotions"][0]["emotion"] == "joy"

    def test_topic_insights(self, sample_profiles):
        """Test topic-level insights generation."""
        insights = generate_topic_insights(sample_profiles)

        # Check required keys
        assert "highest_reciprocity_topics" in insights
        assert "lowest_reciprocity_topics" in insights
        assert "most_positive_topics" in insights

        # Should detect work topic
        if insights["highest_reciprocity_topics"]:
            topics = [t["topic"] for t in insights["highest_reciprocity_topics"]]
            assert "work" in topics or "family" in topics or "hobbies" in topics

    def test_backward_compatibility(self, sample_profiles):
        """Test that merged output is backward compatible with v1.0."""
        merged = merge_all_data(sample_profiles)

        # Create a simplified v1.0-style merged dict
        merged_v1 = {
            "personality_aggregation": {},
            "basic_metrics": {"dominant_emotion_counts": {}},
            "mbti_summary": {},
        }

        # Populate from merged data
        for _, row in merged["big_five"].iterrows():
            merged_v1["personality_aggregation"][row["trait"]] = row["mean"]

        for emotion, count in merged["emotions"]["count"].to_dict().items():
            merged_v1["basic_metrics"]["dominant_emotion_counts"][emotion] = int(count)

        for _, row in merged["mbti"].iterrows():
            merged_v1["mbti_summary"][row["mbti_type"]] = int(row["count"])

        # Verify v1.0 structure
        assert "personality_aggregation" in merged_v1
        assert "basic_metrics" in merged_v1
        assert "mbti_summary" in merged_v1
        assert "dominant_emotion_counts" in merged_v1["basic_metrics"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
