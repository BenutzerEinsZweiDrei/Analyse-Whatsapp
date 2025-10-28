#!/usr/bin/env python
"""
Integration test to verify local_profile_generator works with streamlit_app data structures.
This simulates the workflow without actually running Streamlit.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from local_profile_generator import run_local_analysis


def test_integration():
    """Test the integration with realistic data structure from streamlit_app."""

    # Simulate data from streamlit_app.run_analysis
    matrix = {
        1: {
            "idx": 1,
            "conversation_id": "1",
            "topic": ["work"],
            "emojies": ["😊", "👍"],
            "emo_bew": ["eher positiv"],
            "sentiment": ["positive"],
            "sent_rating": [6.5],
            "sentiment_compound": 0.6,
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
            "response_times": {"per_user": {"Alice": 15.5, "Bob": 20.0}, "topic_average": 17.75},
            "words": ["project", "deadline", "meeting"],
            "keywords": ["project", "deadline"],
            "nouns": ["project", "deadline", "meeting"],
        },
        2: {
            "idx": 2,
            "conversation_id": "2",
            "topic": ["family"],
            "emojies": ["❤️", "😢"],
            "emo_bew": ["neutral"],
            "sentiment": ["positive"],
            "sent_rating": [7.0],
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
            "response_times": {"per_user": {"Alice": 10.0, "Charlie": 12.0}, "topic_average": 11.0},
            "words": ["mom", "family", "visit"],
            "keywords": ["family", "visit"],
            "nouns": ["mom", "family"],
        },
    }

    # Simulate data from streamlit_app.summarize_matrix
    summary = {
        "positive_topics": ["work", "family"],
        "negative_topics": [],
        "emotion_variability": 0.15,
        "matrix": matrix,
        "analysis": {
            1: {
                "topic": ["work"],
                "emojies": ["eher positiv"],
                "sentiment": ["positive"],
                "wordcloud": ["project", "deadline", "meeting"],
                "big_five": matrix[1]["big_five"],
                "mbti": "ENFJ",
                "emotion_analysis": matrix[1]["emotion_analysis"],
                "response_times": matrix[1]["response_times"],
                "emotional_reciprocity": 0.75,
            },
            2: {
                "topic": ["family"],
                "emojies": ["neutral"],
                "sentiment": ["positive"],
                "wordcloud": ["mom", "family", "visit"],
                "big_five": matrix[2]["big_five"],
                "mbti": "ESFJ",
                "emotion_analysis": matrix[2]["emotion_analysis"],
                "response_times": matrix[2]["response_times"],
                "emotional_reciprocity": 0.85,
            },
        },
    }

    print("Running local analysis integration test...")
    print("=" * 70)

    # Call the main function
    results, profile_text = run_local_analysis(summary, matrix)

    # Verify results structure
    assert isinstance(results, dict), "Results should be a dictionary"
    assert isinstance(profile_text, str), "Profile text should be a string"

    # Check required keys
    required_keys = [
        "basic_metrics",
        "big_five_aggregation",
        "correlations",
        "topics_summary",
        "mbti_summary",
        "emotion_insights",
        "per_conversation_table",
        "exports",
    ]

    for key in required_keys:
        assert key in results, f"Missing required key: {key}"

    # Verify exports structure
    assert "metrics_json" in results["exports"], "Missing metrics_json in exports"
    assert "per_conversation_csv" in results["exports"], "Missing per_conversation_csv in exports"
    assert "flagged_json" in results["exports"], "Missing flagged_json in exports"

    # Display results
    print("\n✓ Analysis completed successfully!")
    print("\n" + "=" * 70)
    print("PROFILE TEXT:")
    print("=" * 70)
    print(profile_text)
    print("\n" + "=" * 70)
    print("BASIC METRICS:")
    print("=" * 70)
    print(f"Conversations analyzed: {results['basic_metrics'].get('per_conversation_count', 0)}")

    recip = results["basic_metrics"].get("average_emotional_reciprocity", {})
    if recip:
        print(f"Average emotional reciprocity: {recip.get('mean', 0):.3f} (n={recip.get('n', 0)})")

    rt = results["basic_metrics"].get("response_time_stats", {})
    if rt and rt.get("n", 0) > 0:
        print(f"Average response time: {rt.get('mean', 0):.1f} minutes (n={rt.get('n', 0)})")

    print("\n" + "=" * 70)
    print("PERSONALITY AGGREGATION:")
    print("=" * 70)
    personality = results.get("big_five_aggregation", {})
    if personality:
        print(f"Top trait: {personality.get('top_trait', 'unknown')}")
        print(f"Bottom trait: {personality.get('bottom_trait', 'unknown')}")

        for trait in [
            "openness",
            "conscientiousness",
            "extraversion",
            "agreeableness",
            "neuroticism",
        ]:
            if trait in personality:
                trait_data = personality[trait]
                print(
                    f"  {trait.title()}: {trait_data.get('mean', 0):.2f} (±{trait_data.get('std', 0):.2f})"
                )

    print("\n" + "=" * 70)
    print("EMOTION INSIGHTS:")
    print("=" * 70)
    emotion_insights = results.get("emotion_insights", {})
    print(f"Most common emotion: {emotion_insights.get('most_common_emotion', 'unknown')}")

    flagged = emotion_insights.get("flagged_conversations", [])
    print(f"Flagged conversations: {len(flagged)}")
    for flag in flagged[:3]:  # Show first 3
        print(
            f"  - Conv {flag.get('conversation_id')}: {flag.get('reason')} (value={flag.get('value', 0):.3f})"
        )

    print("\n" + "=" * 70)
    print("EXPORT SIZES:")
    print("=" * 70)
    print(f"Metrics JSON: {len(results['exports']['metrics_json'])} bytes")
    print(f"Per-conversation CSV: {len(results['exports']['per_conversation_csv'])} bytes")
    print(f"Flagged JSON: {len(results['exports']['flagged_json'])} bytes")

    print("\n" + "=" * 70)
    print("✓ ALL INTEGRATION TESTS PASSED")
    print("=" * 70)

    return True


if __name__ == "__main__":
    try:
        test_integration()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Integration test failed: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
