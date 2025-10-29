"""
Tests for Profile_Fusion.py merge functions.
"""

import sys
import unittest
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import merge_big_five, merge_emotions, merge_mbti


class TestProfileFusion(unittest.TestCase):
    """Test Profile_Fusion merge functions."""

    def test_merge_mbti_nested_format(self):
        """Test merge_mbti with nested dictionary format (actual format from local_profile.py)."""
        test_data = [
            {
                "mbti_summary": {
                    "INTJ": {"count": 5, "mean_reciprocity": 0.75, "mean_response_time": 120.5},
                    "ENFP": {"count": 3, "mean_reciprocity": 0.80, "mean_response_time": 90.2},
                }
            },
            {
                "mbti_summary": {
                    "INTJ": {"count": 2, "mean_reciprocity": 0.70, "mean_response_time": 130.0},
                    "ISFJ": {"count": 4, "mean_reciprocity": 0.85, "mean_response_time": 95.0},
                }
            },
        ]

        result = merge_mbti(test_data)

        # Verify DataFrame structure
        self.assertEqual(len(result), 3)
        self.assertIn("mbti_type", result.columns)
        self.assertIn("count", result.columns)

        # Verify counts are summed correctly
        intj_count = result[result["mbti_type"] == "INTJ"]["count"].values[0]
        enfp_count = result[result["mbti_type"] == "ENFP"]["count"].values[0]
        isfj_count = result[result["mbti_type"] == "ISFJ"]["count"].values[0]

        self.assertEqual(intj_count, 7)
        self.assertEqual(enfp_count, 3)
        self.assertEqual(isfj_count, 4)

        # Verify sorting by count descending
        self.assertEqual(result.iloc[0]["mbti_type"], "INTJ")
        self.assertEqual(result.iloc[1]["mbti_type"], "ISFJ")
        self.assertEqual(result.iloc[2]["mbti_type"], "ENFP")

    def test_merge_mbti_simple_format(self):
        """Test merge_mbti with simple count format (legacy format)."""
        test_data = [
            {"mbti_summary": {"INTJ": 5, "ENFP": 3}},
            {"mbti_summary": {"INTJ": 2, "ISFJ": 4}},
        ]

        result = merge_mbti(test_data)

        # Verify counts are summed correctly
        intj_count = result[result["mbti_type"] == "INTJ"]["count"].values[0]
        enfp_count = result[result["mbti_type"] == "ENFP"]["count"].values[0]
        isfj_count = result[result["mbti_type"] == "ISFJ"]["count"].values[0]

        self.assertEqual(intj_count, 7)
        self.assertEqual(enfp_count, 3)
        self.assertEqual(isfj_count, 4)

    def test_merge_mbti_mixed_formats(self):
        """Test merge_mbti with mixed formats (some simple, some nested)."""
        test_data = [
            {"mbti_summary": {"INTJ": 5, "ENFP": 3}},  # Simple format
            {
                "mbti_summary": {
                    "INTJ": {"count": 2, "mean_reciprocity": 0.70},  # Nested format
                    "ISFJ": {"count": 4, "mean_reciprocity": 0.85},
                }
            },
        ]

        result = merge_mbti(test_data)

        # Verify counts are summed correctly from mixed formats
        intj_count = result[result["mbti_type"] == "INTJ"]["count"].values[0]
        self.assertEqual(intj_count, 7)

    def test_merge_mbti_empty_data(self):
        """Test merge_mbti with empty or missing mbti_summary."""
        test_data = [
            {"mbti_summary": {}},
            {"basic_metrics": {}},  # Missing mbti_summary
        ]

        result = merge_mbti(test_data)

        # Should return empty DataFrame
        self.assertEqual(len(result), 0)

    def test_merge_mbti_invalid_values(self):
        """Test merge_mbti with invalid values (should skip them)."""
        test_data = [
            {
                "mbti_summary": {
                    "INTJ": 5,
                    "ENFP": "invalid",  # Invalid value
                    "ISFJ": None,  # Invalid value
                    "ESTP": {"count": 0},  # Zero count (should be skipped)
                    "ENTP": {"count": 2},  # Valid
                }
            }
        ]

        result = merge_mbti(test_data)

        # Should only include valid entries
        self.assertEqual(len(result), 2)  # INTJ and ENTP
        mbti_types = result["mbti_type"].tolist()
        self.assertIn("INTJ", mbti_types)
        self.assertIn("ENTP", mbti_types)
        self.assertNotIn("ENFP", mbti_types)
        self.assertNotIn("ISFJ", mbti_types)
        self.assertNotIn("ESTP", mbti_types)

    def test_merge_big_five_basic(self):
        """Test merge_big_five function."""
        test_data = [
            {
                "personality_aggregation": {
                    "openness": 7.5,
                    "conscientiousness": 6.0,
                    "extraversion": 8.2,
                }
            },
            {
                "personality_aggregation": {
                    "openness": 6.5,
                    "conscientiousness": 7.0,
                    "agreeableness": 8.0,
                }
            },
        ]

        result = merge_big_five(test_data)

        # Verify structure (v2.0 format with mean, std, count)
        self.assertIn("trait", result.columns)
        self.assertIn("mean", result.columns)
        self.assertIn("std", result.columns)
        self.assertIn("count", result.columns)

        # Verify averaging
        openness_score = result[result["trait"] == "openness"]["mean"].values[0]
        self.assertAlmostEqual(openness_score, 7.0, places=2)

    def test_merge_emotions_basic(self):
        """Test merge_emotions function."""
        test_data = [
            {"basic_metrics": {"dominant_emotion_counts": {"joy": 10, "sadness": 5}}},
            {"basic_metrics": {"dominant_emotion_counts": {"joy": 8, "anger": 3}}},
        ]

        result = merge_emotions(test_data)

        # Verify counts are summed
        self.assertEqual(result.loc["joy", "count"], 18)
        self.assertEqual(result.loc["sadness", "count"], 5)
        self.assertEqual(result.loc["anger", "count"], 3)


if __name__ == "__main__":
    unittest.main()
