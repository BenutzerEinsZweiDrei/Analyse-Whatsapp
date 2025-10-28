"""
Tests for local_profile module (canonical implementation).
"""

import json
import os
import sys
import unittest
from unittest.mock import patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.local_profile import (
    aggregate_personality_data,
    clean_data,
    compute_basic_metrics,
    correlation_analysis,
    emotion_insights,
    export_results,
    filter_and_segment,
    highlights_and_rankings,
    load_and_validate,
    normalize_structure,
    run_local_analysis,
    safe_float,
)


class TestLocalProfile(unittest.TestCase):
    """Test cases for local profile generator functions."""

    def setUp(self):
        """Set up test data."""
        # Create synthetic test matrix
        self.test_matrix = {
            "1": {
                "idx": 1,
                "conversation_id": "1",
                "topic": ["friendship"],
                "emojies": ["😊", "😄"],
                "sentiment": ["positive"],
                "sentiment_compound": 0.7,
                "big_five": {
                    "openness": 6.5,
                    "conscientiousness": 7.0,
                    "extraversion": 8.0,
                    "agreeableness": 7.5,
                    "neuroticism": 3.0,
                },
                "mbti": "ENFJ",
                "emotion_analysis": {
                    "dominant_emotion": "joy",
                    "emotion_ratios": {"joy": 0.8, "neutral": 0.2},
                },
                "emotional_reciprocity": 0.75,
                "response_times": {
                    "per_user": {"user1": 15.5, "user2": 20.0},
                    "topic_average": 17.75,
                },
                "words": ["friend", "happy", "good"],
            },
            "2": {
                "idx": 2,
                "conversation_id": "2",
                "topic": ["work"],
                "emojies": ["😐"],
                "sentiment": ["neutral"],
                "sentiment_compound": 0.1,
                "big_five": {
                    "openness": 5.0,
                    "conscientiousness": 8.5,
                    "extraversion": 4.0,
                    "agreeableness": 6.0,
                    "neuroticism": 5.5,
                },
                "mbti": "ISTJ",
                "emotion_analysis": {
                    "dominant_emotion": "neutral",
                    "emotion_ratios": {"neutral": 1.0},
                },
                "emotional_reciprocity": 0.45,
                "response_times": {"per_user": {"user1": 30.0}, "topic_average": 30.0},
                "words": ["task", "deadline", "project"],
            },
            "3": {
                "idx": 3,
                "conversation_id": "3",
                "topic": ["family"],
                "emojies": ["❤️", "😢"],
                "sentiment": ["positive"],
                "sentiment_compound": 0.3,
                "big_five": {
                    "openness": 6.0,
                    "conscientiousness": 6.5,
                    "extraversion": 5.5,
                    "agreeableness": 8.0,
                    "neuroticism": 6.0,
                },
                "mbti": "ESFJ",
                "emotion_analysis": {
                    "dominant_emotion": "love",
                    "emotion_ratios": {"love": 0.5, "sadness": 0.3, "joy": 0.2},
                },
                "emotional_reciprocity": 0.85,
                "response_times": {
                    "per_user": {"user1": 10.0, "user2": 12.0},
                    "topic_average": 11.0,
                },
                "words": ["family", "mom", "love"],
            },
        }

        # Create synthetic test summary
        self.test_summary = {
            "positive_topics": ["friendship", "family"],
            "negative_topics": [],
            "emotion_variability": 0.15,
            "analysis": {
                "1": {
                    "topic": ["friendship"],
                    "emojies": ["sehr positiv"],
                    "sentiment": ["positive"],
                    "wordcloud": ["friend", "happy", "good"],
                    "big_five": self.test_matrix["1"]["big_five"],
                    "mbti": "ENFJ",
                    "emotion_analysis": self.test_matrix["1"]["emotion_analysis"],
                    "response_times": self.test_matrix["1"]["response_times"],
                    "emotional_reciprocity": 0.75,
                },
                "2": {
                    "topic": ["work"],
                    "emojies": ["neutral"],
                    "sentiment": ["neutral"],
                    "wordcloud": ["task", "deadline", "project"],
                    "big_five": self.test_matrix["2"]["big_five"],
                    "mbti": "ISTJ",
                    "emotion_analysis": self.test_matrix["2"]["emotion_analysis"],
                    "response_times": self.test_matrix["2"]["response_times"],
                    "emotional_reciprocity": 0.45,
                },
                "3": {
                    "topic": ["family"],
                    "emojies": ["eher positiv"],
                    "sentiment": ["positive"],
                    "wordcloud": ["family", "mom", "love"],
                    "big_five": self.test_matrix["3"]["big_five"],
                    "mbti": "ESFJ",
                    "emotion_analysis": self.test_matrix["3"]["emotion_analysis"],
                    "response_times": self.test_matrix["3"]["response_times"],
                    "emotional_reciprocity": 0.85,
                },
            },
            "matrix": self.test_matrix,
        }

    def _is_dataframe(self, obj):
        """Helper to check if object is a pandas DataFrame."""
        try:
            import pandas as pd

            return isinstance(obj, pd.DataFrame)
        except ImportError:
            return False

    def test_load_and_validate(self):
        """Test step 1: load and validate."""
        summary, matrix = load_and_validate(self.test_summary, self.test_matrix)

        # Check that matrix keys are strings
        self.assertIsInstance(summary, dict)
        self.assertIsInstance(matrix, dict)
        for key in matrix.keys():
            self.assertIsInstance(key, str)

        # Check that conversation_id is set
        for conv_id, entry in matrix.items():
            self.assertIn("conversation_id", entry)

    def test_normalize_structure(self):
        """Test step 2: normalize structure."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)

        # Check that we get records (can be DataFrame or list)
        if self._is_dataframe(records):
            self.assertEqual(len(records), 3)

            # Check columns exist
            self.assertIn("conversation_id", records.columns)
            self.assertIn("topic", records.columns)
            self.assertIn("dominant_emotion", records.columns)
            self.assertIn("emotional_reciprocity", records.columns)
            self.assertIn("mbti", records.columns)

            # Check Big Five flattening
            self.assertIn("big_five_openness", records.columns)
            self.assertIn("big_five_conscientiousness", records.columns)
            self.assertIn("big_five_extraversion", records.columns)
            self.assertIn("big_five_agreeableness", records.columns)
            self.assertIn("big_five_neuroticism", records.columns)
        elif isinstance(records, list):
            self.assertEqual(len(records), 3)

            # Check first record structure
            record = records[0]
            self.assertIn("conversation_id", record)
            self.assertIn("topic", record)
            self.assertIn("dominant_emotion", record)
            self.assertIn("emotional_reciprocity", record)
            self.assertIn("mbti", record)

            # Check Big Five flattening
            self.assertIn("big_five_openness", record)
            self.assertIn("big_five_conscientiousness", record)
            self.assertIn("big_five_extraversion", record)
            self.assertIn("big_five_agreeableness", record)
            self.assertIn("big_five_neuroticism", record)

    def test_clean_data(self):
        """Test step 3: clean data."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)

        # Check that we still have records (can be DataFrame or list)
        if self._is_dataframe(cleaned):
            self.assertEqual(len(cleaned), 3)

            # Check that numeric columns exist and have proper types
            self.assertIn("emotional_reciprocity", cleaned.columns)
            self.assertIn("big_five_openness", cleaned.columns)
        elif isinstance(cleaned, list):
            self.assertEqual(len(cleaned), 3)

            # Check that numeric fields are properly typed
            for record in cleaned:
                self.assertIsInstance(record.get("emotional_reciprocity"), (int, float))
                self.assertIsInstance(record.get("big_five_openness"), (int, float))

    def test_compute_basic_metrics(self):
        """Test step 4: compute basic metrics."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)
        metrics = compute_basic_metrics(cleaned)

        # Check that metrics are computed
        self.assertIsInstance(metrics, dict)
        self.assertIn("average_emotional_reciprocity", metrics)

        recip = metrics["average_emotional_reciprocity"]
        self.assertIn("mean", recip)
        self.assertIn("std", recip)
        self.assertIn("n", recip)

        # Check that mean is reasonable (should be average of 0.75, 0.45, 0.85)
        expected_mean = (0.75 + 0.45 + 0.85) / 3
        self.assertAlmostEqual(recip["mean"], expected_mean, places=2)

    def test_aggregate_personality_data(self):
        """Test step 5: aggregate personality data."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)
        personality = aggregate_personality_data(cleaned)

        # Check that personality data is aggregated
        self.assertIsInstance(personality, dict)

        # Check that we have stats for each trait
        for trait in [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]:
            self.assertIn(trait, personality)
            self.assertIn("mean", personality[trait])
            self.assertIn("std", personality[trait])
            self.assertIn("n", personality[trait])

        # Check that top and bottom traits are identified
        self.assertIn("top_trait", personality)
        self.assertIn("bottom_trait", personality)

    def test_correlation_analysis(self):
        """Test step 6: correlation analysis."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)
        correlations = correlation_analysis(cleaned)

        # Check that correlations are computed
        self.assertIsInstance(correlations, dict)

        # Check structure of correlation results
        for label, result in correlations.items():
            self.assertIn("pearson_r", result)
            self.assertIn("spearman_r", result)
            self.assertIn("n", result)

            # Correlation values should be between -1 and 1
            if result["pearson_r"] is not None:
                self.assertGreaterEqual(result["pearson_r"], -1.0)
                self.assertLessEqual(result["pearson_r"], 1.0)

    def test_filter_and_segment(self):
        """Test step 7: filter and segment."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)
        segments = filter_and_segment(cleaned)

        # Check that segments are created
        self.assertIsInstance(segments, dict)
        self.assertIn("by_topic", segments)
        self.assertIn("by_mbti", segments)

        # Check topic segmentation
        topics = segments["by_topic"]
        self.assertGreater(len(topics), 0)

        # Check MBTI segmentation
        mbti_segments = segments["by_mbti"]
        self.assertGreater(len(mbti_segments), 0)

    def test_emotion_insights(self):
        """Test step 8: emotion insights."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)
        insights = emotion_insights(cleaned)

        # Check that insights are generated
        self.assertIsInstance(insights, dict)
        self.assertIn("most_common_emotion", insights)
        self.assertIn("flagged_conversations", insights)

        # Check that flagged conversations is a list
        self.assertIsInstance(insights["flagged_conversations"], list)

    def test_export_results(self):
        """Test step 11: export results."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)

        # Create minimal results
        results = {
            "basic_metrics": compute_basic_metrics(cleaned),
            "big_five_aggregation": aggregate_personality_data(cleaned),
            "correlations": {},
            "segments": {},
            "emotion_insights": emotion_insights(cleaned),
            "advanced_analysis": {},
        }

        exports = export_results(results, cleaned)

        # Check that exports are created
        self.assertIsInstance(exports, dict)
        self.assertIn("metrics_json", exports)
        self.assertIn("per_conversation_csv", exports)
        self.assertIn("flagged_json", exports)

        # Check that JSON is valid
        metrics_data = json.loads(exports["metrics_json"])
        self.assertIsInstance(metrics_data, dict)

        # Check that CSV has content
        self.assertGreater(len(exports["per_conversation_csv"]), 0)

    def test_run_local_analysis(self):
        """Test the complete pipeline."""
        results, profile_text = run_local_analysis(self.test_summary, self.test_matrix)

        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIsInstance(profile_text, str)

        # Check required keys in results
        required_keys = [
            "basic_metrics",
            "big_five_aggregation",
            "correlations",
            "topics_summary",
            "mbti_summary",
            "emotion_insights",
            "per_conversation_table",
        ]
        for key in required_keys:
            self.assertIn(key, results)

        # Check that profile text is not empty
        self.assertGreater(len(profile_text), 0)

        # Check that exports are included
        self.assertIn("exports", results)

        # Check that highlights_and_rankings is included
        self.assertIn("highlights_and_rankings", results)

    def test_highlights_and_rankings_basic(self):
        """Test highlights_and_rankings function with basic data."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)

        # Run highlights and rankings
        highlights = highlights_and_rankings(cleaned)

        # Check that all required keys are present
        self.assertIsInstance(highlights, dict)
        self.assertIn("topics_aggregated", highlights)
        self.assertIn("reciprocity_ranking", highlights)
        self.assertIn("response_time_ranking", highlights)
        self.assertIn("emotional_highlights", highlights)
        self.assertIn("summary_text", highlights)
        self.assertIn("final_insight", highlights)

        # Check topics_aggregated structure
        topics = highlights["topics_aggregated"]
        self.assertGreater(len(topics), 0)

        # Check that each topic has required fields
        for topic, stats in topics.items():
            self.assertIn("n", stats)
            self.assertIn("mean_emotional_reciprocity", stats)
            self.assertIn("mean_response_time_minutes", stats)
            self.assertIn("median_response_time_minutes", stats)
            self.assertIn("dominant_emotion", stats)

        # Check reciprocity ranking structure
        recip_rank = highlights["reciprocity_ranking"]
        self.assertIn("top_topics", recip_rank)
        self.assertIn("lowest_topics", recip_rank)

        # Check response time ranking structure
        rt_rank = highlights["response_time_ranking"]
        self.assertIn("fastest_topics", rt_rank)
        self.assertIn("slowest_topics", rt_rank)

        # Check emotional highlights structure
        emo_hl = highlights["emotional_highlights"]
        self.assertIn("dominant_emotion_percentages", emo_hl)
        self.assertIn("high_gratitude_topics", emo_hl)
        self.assertIn("high_sadness_topics", emo_hl)

        # Check summary text
        summary = highlights["summary_text"]
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertIn("Emotional Reciprocity Ranking", summary)
        self.assertIn("Response Speed Ranking", summary)
        self.assertIn("Emotional Highlights", summary)

    def test_highlights_and_rankings_with_list(self):
        """Test highlights_and_rankings with list of dicts instead of DataFrame."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)

        # Convert to list if it's a DataFrame
        try:
            import pandas as pd
            if isinstance(cleaned, pd.DataFrame):
                cleaned = cleaned.to_dict("records")
        except ImportError:
            pass

        # Run highlights and rankings
        highlights = highlights_and_rankings(cleaned)

        # Basic checks
        self.assertIsInstance(highlights, dict)
        self.assertIn("topics_aggregated", highlights)
        self.assertIn("summary_text", highlights)

    def test_highlights_and_rankings_with_thresholds(self):
        """Test highlights_and_rankings with custom thresholds."""
        _, matrix = load_and_validate(self.test_summary, self.test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)

        # Run with custom thresholds
        highlights = highlights_and_rankings(
            cleaned, min_topic_n=1, reciprocity_thresholds=(0.8, 0.6), include_final_insight=False
        )

        # Check that final insight is None when disabled
        self.assertIsNone(highlights["final_insight"])

        # Check that min_topic_n is respected
        topics = highlights["topics_aggregated"]
        for topic, stats in topics.items():
            self.assertGreaterEqual(stats["n"], 1)

    def test_highlights_and_rankings_topic_aggregation(self):
        """Test that topics are properly aggregated and metrics computed."""
        # Create test data with specific topics
        test_matrix = {
            "1": {
                "conversation_id": "1",
                "topic": ["danke"],
                "emojies": ["😊"],
                "sentiment": ["positive"],
                "big_five": {
                    "openness": 7.0,
                    "conscientiousness": 7.0,
                    "extraversion": 8.0,
                    "agreeableness": 7.5,
                    "neuroticism": 3.0,
                },
                "mbti": "ENFJ",
                "emotion_analysis": {
                    "dominant_emotion": "gratitude",
                    "emotion_ratios": {"gratitude": 0.9, "joy": 0.1},
                },
                "emotional_reciprocity": 0.95,
                "response_times": {"per_user": {"u1": 10}, "topic_average": 10.0},
                "words": [],
            },
            "2": {
                "conversation_id": "2",
                "topic": ["danke"],
                "emojies": ["😊"],
                "sentiment": ["positive"],
                "big_five": {
                    "openness": 7.0,
                    "conscientiousness": 7.0,
                    "extraversion": 8.0,
                    "agreeableness": 7.5,
                    "neuroticism": 3.0,
                },
                "mbti": "ENFJ",
                "emotion_analysis": {
                    "dominant_emotion": "gratitude",
                    "emotion_ratios": {"gratitude": 0.8, "joy": 0.2},
                },
                "emotional_reciprocity": 0.92,
                "response_times": {"per_user": {"u1": 15}, "topic_average": 15.0},
                "words": [],
            },
            "3": {
                "conversation_id": "3",
                "topic": ["error"],
                "emojies": [],
                "sentiment": ["negative"],
                "big_five": {
                    "openness": 5.0,
                    "conscientiousness": 5.0,
                    "extraversion": 4.0,
                    "agreeableness": 5.0,
                    "neuroticism": 6.0,
                },
                "mbti": "ISTJ",
                "emotion_analysis": {
                    "dominant_emotion": "sadness",
                    "emotion_ratios": {"sadness": 0.7, "neutral": 0.3},
                },
                "emotional_reciprocity": 0.70,
                "response_times": {"per_user": {"u1": 120}, "topic_average": 120.0},
                "words": [],
            },
        }

        test_summary = {
            "positive_topics": ["danke"],
            "negative_topics": ["error"],
            "emotion_variability": 0.2,
            "analysis": {},
        }

        summary, matrix = load_and_validate(test_summary, test_matrix)
        records = normalize_structure(matrix)
        cleaned = clean_data(records)

        highlights = highlights_and_rankings(cleaned, min_topic_n=1)

        # Check topic aggregation
        topics = highlights["topics_aggregated"]
        self.assertIn("danke", topics)
        self.assertIn("error", topics)

        # Check danke topic stats (should have 2 conversations)
        danke_stats = topics["danke"]
        self.assertEqual(danke_stats["n"], 2)
        # Mean reciprocity should be (0.95 + 0.92) / 2 = 0.935
        self.assertAlmostEqual(danke_stats["mean_emotional_reciprocity"], 0.935, places=2)
        # Mean response time should be (10 + 15) / 2 = 12.5
        self.assertAlmostEqual(danke_stats["mean_response_time_minutes"], 12.5, places=1)

        # Check error topic stats (should have 1 conversation)
        error_stats = topics["error"]
        self.assertEqual(error_stats["n"], 1)
        self.assertEqual(error_stats["mean_emotional_reciprocity"], 0.70)
        self.assertEqual(error_stats["mean_response_time_minutes"], 120.0)

        # Check rankings
        recip_rank = highlights["reciprocity_ranking"]
        top_topics = recip_rank["top_topics"]
        # Danke should be first
        self.assertEqual(top_topics[0][0], "danke")

    def test_safe_float(self):
        """Test safe_float helper function."""
        # Test valid conversions
        self.assertEqual(safe_float(5), 5.0)
        self.assertEqual(safe_float("3.14"), 3.14)
        self.assertEqual(safe_float(2.5), 2.5)

        # Test invalid conversions
        self.assertEqual(safe_float(None), 0.0)
        self.assertEqual(safe_float("invalid"), 0.0)
        self.assertEqual(safe_float("invalid", 1.0), 1.0)

    def test_empty_matrix(self):
        """Test handling of empty matrix."""
        empty_matrix = {}
        empty_summary = {
            "positive_topics": [],
            "negative_topics": [],
            "emotion_variability": 0.0,
            "analysis": {},
        }

        results, profile_text = run_local_analysis(empty_summary, empty_matrix)

        # Check that we get an error result
        self.assertIn("error", results)
        self.assertIn("No conversations", profile_text)

    def test_backward_compatibility_adapter(self):
        """Test that the adapter wrapper maintains backward compatibility."""
        # Import from old location
        from local_profile_generator import run_local_analysis as old_run_local_analysis

        # Run analysis from adapter
        results_adapter, profile_adapter = old_run_local_analysis(
            self.test_summary, self.test_matrix
        )

        # Run analysis from canonical module
        from app.core.local_profile import run_local_analysis as new_run_local_analysis

        results_canonical, profile_canonical = new_run_local_analysis(
            self.test_summary, self.test_matrix
        )

        # Check that both return the same structure
        self.assertEqual(set(results_adapter.keys()), set(results_canonical.keys()))
        self.assertIsInstance(profile_adapter, str)
        self.assertIsInstance(profile_canonical, str)

    def test_edge_case_missing_big_five(self):
        """Test handling of conversations with missing big five traits."""
        matrix_missing_big_five = {
            "1": {
                "conversation_id": "1",
                "topic": ["test"],
                "emojies": [],
                "sentiment": ["neutral"],
                "big_five": {},  # Empty big five
                "mbti": "XXXX",
                "emotion_analysis": {"dominant_emotion": "neutral", "emotion_ratios": {}},
                "emotional_reciprocity": 0.5,
                "response_times": {"per_user": {}, "topic_average": 0.0},
                "words": [],
            }
        }

        summary = {
            "positive_topics": [],
            "negative_topics": [],
            "emotion_variability": 0.0,
            "analysis": {},
        }

        # Should not raise an exception
        results, profile_text = run_local_analysis(summary, matrix_missing_big_five)

        # Check that results are returned
        self.assertIsInstance(results, dict)
        self.assertIn("big_five_aggregation", results)

    def test_edge_case_list_vs_string_topic(self):
        """Test handling of topics as both lists and strings."""
        matrix_mixed_topics = {
            "1": {
                "conversation_id": "1",
                "topic": ["test_list"],  # Topic as list
                "emojies": [],
                "sentiment": ["neutral"],
                "big_five": {"openness": 5.0},
                "mbti": "XXXX",
                "emotion_analysis": {"dominant_emotion": "neutral", "emotion_ratios": {}},
                "emotional_reciprocity": 0.5,
                "response_times": {"per_user": {}, "topic_average": 0.0},
                "words": [],
            },
            "2": {
                "conversation_id": "2",
                "topic": "test_string",  # Topic as string
                "emojies": [],
                "sentiment": ["neutral"],
                "big_five": {"openness": 5.0},
                "mbti": "XXXX",
                "emotion_analysis": {"dominant_emotion": "neutral", "emotion_ratios": {}},
                "emotional_reciprocity": 0.5,
                "response_times": {"per_user": {}, "topic_average": 0.0},
                "words": [],
            },
        }

        summary = {
            "positive_topics": [],
            "negative_topics": [],
            "emotion_variability": 0.0,
            "analysis": {},
        }

        # Should handle both formats
        results, profile_text = run_local_analysis(summary, matrix_mixed_topics)

        # Check that both conversations are processed
        self.assertEqual(results["basic_metrics"]["per_conversation_count"], 2)

    def test_highlights_included_in_results(self):
        """Test that highlights_and_rankings is included in results."""
        results, _ = run_local_analysis(self.test_summary, self.test_matrix)

        # Check that highlights_and_rankings is in results
        self.assertIn("highlights_and_rankings", results)

        highlights = results["highlights_and_rankings"]
        self.assertIsInstance(highlights, dict)

        # Check required keys
        self.assertIn("topics_aggregated", highlights)
        self.assertIn("reciprocity_ranking", highlights)
        self.assertIn("response_time_ranking", highlights)
        self.assertIn("emotional_highlights", highlights)
        self.assertIn("summary_text", highlights)

    def test_exports_include_highlights(self):
        """Test that exports include highlights_and_rankings in JSON."""
        results, _ = run_local_analysis(self.test_summary, self.test_matrix)

        # Check that exports exist
        self.assertIn("exports", results)
        exports = results["exports"]

        # Check that metrics_json includes highlights
        metrics_json = json.loads(exports["metrics_json"])
        self.assertIn("highlights_and_rankings", metrics_json)


if __name__ == "__main__":
    unittest.main()
