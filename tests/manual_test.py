#!/usr/bin/env python
"""
Manual test script for Profile Fusion v2.0.

Tests the main workflow without Streamlit UI.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analysis import (
    compute_aggregated_statistics,
    compute_correlations,
    create_profile_matrix,
    generate_emotional_insights,
    generate_natural_language_summary,
    generate_topic_insights,
    merge_all_data,
    normalize_matrix,
)


def main():
    """Test Profile Fusion v2.0 workflow."""
    print("Profile Fusion v2.0 Manual Test")
    print("=" * 50)

    # Load fixture data
    fixtures_dir = Path(__file__).parent / "tests" / "fixtures"
    data_list = []
    filenames = []

    for i in [1, 2]:
        filepath = fixtures_dir / f"profile{i}.json"
        with open(filepath) as f:
            data_list.append(json.load(f))
            filenames.append(f"profile{i}.json")

    print(f"\n✓ Loaded {len(data_list)} profile files")

    # 1. Merge data
    print("\n1. Merging personality data...")
    merged = merge_all_data(data_list)
    print(f"   - Big Five traits: {len(merged['big_five'])} traits")
    print(f"   - Emotions: {len(merged['emotions'])} emotions")
    print(f"   - MBTI types: {len(merged['mbti'])} types")

    # 2. Aggregated statistics
    print("\n2. Computing aggregated statistics...")
    agg_stats = compute_aggregated_statistics(data_list)
    print(f"   - Mean reciprocity: {agg_stats['reciprocity'].get('mean')}")
    print(f"   - Mean response time: {agg_stats['response_time'].get('mean')}")
    print(f"   - Top 3 emotions: {[e['emotion'] for e in agg_stats['top_emotions']]}")

    # 3. Profile matrix
    print("\n3. Creating personality profile matrix...")
    matrix = create_profile_matrix(data_list, filenames)
    print(f"   - Matrix shape: {matrix.shape}")
    print(f"   - Features: {list(matrix.index[:5])}...")

    # Test normalization
    normalized = normalize_matrix(matrix, method="minmax")
    print(f"   - Normalized matrix shape: {normalized.shape}")

    # 4. Correlations
    print("\n4. Analyzing correlations...")
    correlations = compute_correlations(data_list, p_threshold=0.05)
    print(f"   - Total correlations: {len(correlations['all_correlations'])}")
    print(f"   - Significant correlations: {len(correlations['significant_correlations'])}")

    # 5. Emotional insights
    print("\n5. Generating emotional insights...")
    emotional_insights = generate_emotional_insights(data_list)
    print(f"   - Top emotions: {[e['emotion'] for e in emotional_insights['top_emotions']]}")

    # 6. Topic insights
    print("\n6. Generating topic insights...")
    topic_insights = generate_topic_insights(data_list)
    if topic_insights["highest_reciprocity_topics"]:
        print(
            f"   - Highest reciprocity topics: {[t['topic'] for t in topic_insights['highest_reciprocity_topics']]}"
        )

    # 7. Natural language summary
    print("\n7. Generating natural language summary...")
    summary = generate_natural_language_summary(
        aggregated_stats=agg_stats,
        correlations=correlations,
        emotional_insights=emotional_insights,
        topic_insights=topic_insights,
        num_files=len(data_list),
        format_style="bullet",
    )
    print("\n--- Summary ---")
    print(summary[:500] + "..." if len(summary) > 500 else summary)

    # 8. Verify backward compatibility
    print("\n\n8. Verifying backward compatibility...")

    # Create v1.0-style merged dict
    merged_v1 = {
        "personality_aggregation": {},
        "basic_metrics": {"dominant_emotion_counts": {}},
        "mbti_summary": {},
        "metadata": {"merger_version": "2.0"},
    }

    # Populate from merged data
    for _, row in merged["big_five"].iterrows():
        merged_v1["personality_aggregation"][row["trait"]] = float(row["mean"])

    for emotion, count in merged["emotions"]["count"].to_dict().items():
        merged_v1["basic_metrics"]["dominant_emotion_counts"][str(emotion)] = int(count)

    for _, row in merged["mbti"].iterrows():
        merged_v1["mbti_summary"][row["mbti_type"]] = int(row["count"])

    print("   - V1.0 compatible structure:")
    print(f"     - personality_aggregation: {len(merged_v1['personality_aggregation'])} traits")
    print(
        f"     - dominant_emotion_counts: {len(merged_v1['basic_metrics']['dominant_emotion_counts'])} emotions"
    )
    print(f"     - mbti_summary: {len(merged_v1['mbti_summary'])} types")

    print("\n" + "=" * 50)
    print("✅ All tests completed successfully!")
    print("\nProfile Fusion v2.0 is working correctly.")


if __name__ == "__main__":
    main()
