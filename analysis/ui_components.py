"""
UI components module for Profile Fusion.

Reusable Streamlit UI components for the personality fusion tool.
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import streamlit as st


def create_merged_json_export(
    big_five_df: pd.DataFrame,
    emotions_df: pd.DataFrame,
    mbti_df: pd.DataFrame,
    profile_matrix: pd.DataFrame,
    aggregated_stats: Dict[str, Any],
    correlations: Dict[str, Any],
    emotional_insights: Dict[str, Any],
    topic_insights: Dict[str, Any],
) -> str:
    """
    Create comprehensive JSON export with backward compatibility.

    Preserves v1.0 structure and adds v2.0 enhancements.

    Args:
        big_five_df: Big Five traits DataFrame
        emotions_df: Emotions DataFrame
        mbti_df: MBTI DataFrame
        profile_matrix: Profile matrix DataFrame
        aggregated_stats: Aggregated statistics
        correlations: Correlation results
        emotional_insights: Emotional insights
        topic_insights: Topic insights

    Returns:
        JSON string
    """
    merged_data = {
        # V1.0 backward-compatible structure
        "personality_aggregation": {},
        "basic_metrics": {"dominant_emotion_counts": {}},
        "mbti_summary": {},
        "metadata": {
            "merged_at": datetime.now(timezone.utc).isoformat(),
            "merger_version": "2.0",
        },
        # V2.0 enhancements
        "profile_matrix": {},
        "aggregated_statistics": aggregated_stats,
        "correlations": {
            "all": correlations.get("all_correlations", []),
            "significant": correlations.get("significant_correlations", []),
            "trait_behavior_summary": correlations.get("trait_behavior_summary", {}),
        },
        "emotional_insights": emotional_insights,
        "topic_insights": topic_insights,
        "analysis_metadata": {
            "version": "2.0",
            "features": [
                "profile_matrix",
                "correlation_analysis",
                "emotional_insights",
                "topic_insights",
                "natural_language_summary",
            ],
        },
    }

    # Add Big Five data (v1.0 format)
    if not big_five_df.empty:
        for _, row in big_five_df.iterrows():
            merged_data["personality_aggregation"][row["trait"]] = float(row["mean"])

    # Add emotion data (v1.0 format)
    if not emotions_df.empty:
        merged_data["basic_metrics"]["dominant_emotion_counts"] = {
            str(emotion): int(count) for emotion, count in emotions_df["count"].to_dict().items()
        }

    # Add MBTI data (v1.0 format)
    if not mbti_df.empty:
        for _, row in mbti_df.iterrows():
            merged_data["mbti_summary"][row["mbti_type"]] = int(row["count"])

    # Add profile matrix (v2.0)
    if not profile_matrix.empty:
        merged_data["profile_matrix"] = profile_matrix.to_dict()

    return json.dumps(merged_data, indent=2, ensure_ascii=False)


def display_correlation_table(correlations: Dict[str, Any], p_threshold: float) -> None:
    """
    Display correlation results as an interactive table.

    Args:
        correlations: Correlation results dictionary
        p_threshold: P-value threshold used for filtering
    """
    st.subheader("🔗 Correlation Analysis")

    significant = correlations.get("significant_correlations", [])

    if not significant:
        st.info(
            f"No statistically significant correlations found at p < {p_threshold}. "
            "Try adjusting the p-value threshold."
        )
        return

    # Convert to DataFrame for display
    corr_df = pd.DataFrame(significant)

    # Select relevant columns
    display_columns = [
        "variable1",
        "variable2",
        "pearson_r",
        "pearson_p",
        "spearman_r",
        "spearman_p",
    ]
    available_columns = [col for col in display_columns if col in corr_df.columns]

    if available_columns:
        display_df = corr_df[available_columns]

        # Round numeric columns
        for col in ["pearson_r", "pearson_p", "spearman_r", "spearman_p"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].round(4)

        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Showing {len(significant)} statistically significant correlation(s) "
            f"with p < {p_threshold}"
        )


def display_profile_matrix_heatmap(matrix: pd.DataFrame) -> None:
    """
    Display profile matrix as a heatmap.

    Args:
        matrix: Profile matrix DataFrame
    """
    st.subheader("🎨 Personality Profile Matrix")

    if matrix.empty:
        st.info("No data available for profile matrix visualization.")
        return

    # Display numeric data as heatmap
    numeric_matrix = matrix.select_dtypes(include=["number"])

    if not numeric_matrix.empty:
        # Use Streamlit's built-in heatmap via plotly or just show as styled dataframe
        styled = numeric_matrix.style.background_gradient(cmap="RdYlGn", axis=1)
        st.dataframe(styled, use_container_width=True)

        st.caption(
            "Personality features across files. Colors indicate relative values "
            "(red=low, yellow=medium, green=high) normalized per row."
        )
    else:
        st.dataframe(matrix, use_container_width=True)


def display_trait_behavior_summary(trait_behavior_summary: Dict[str, Any]) -> None:
    """
    Display trait-behavior association summary.

    Args:
        trait_behavior_summary: Trait behavior summary from correlations
    """
    st.subheader("🧩 Trait-Behavior Associations")

    if not trait_behavior_summary:
        st.info("No trait-behavior associations available.")
        return

    summary_data = []

    for trait, behaviors in trait_behavior_summary.items():
        for behavior, data in behaviors.items():
            if data.get("avg_pearson_r") is not None:
                summary_data.append(
                    {
                        "Trait": trait.capitalize(),
                        "Behavior": behavior.replace("_", " ").capitalize(),
                        "Avg Correlation": round(data["avg_pearson_r"], 4),
                        "Consistency": data.get("consistency", "N/A"),
                    }
                )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.caption(
            "Average correlation coefficients between personality traits and behavioral metrics. "
            "Consistency shows how many files exhibit the same direction of association."
        )
    else:
        st.info("No significant trait-behavior associations found.")


def display_emotional_insights(emotional_insights: Dict[str, Any]) -> None:
    """
    Display emotional pattern insights.

    Args:
        emotional_insights: Emotional insights dictionary
    """
    st.subheader("😊 Emotional Patterns")

    top_emotions = emotional_insights.get("top_emotions", [])

    if not top_emotions:
        st.info("No emotional data available.")
        return

    # Display top emotions
    st.write("**Top 3 Emotions:**")

    for i, emotion_info in enumerate(top_emotions, 1):
        emotion = emotion_info["emotion"]
        count = emotion_info["count"]
        percentage = emotion_info["percentage"]

        st.write(f"{i}. **{emotion.capitalize()}**: {count} occurrences ({percentage}%)")

        # Show details for this emotion
        emotion_details = emotional_insights.get("emotion_details", {}).get(emotion, {})

        if emotion_details.get("mean_reciprocity") is not None:
            st.write(f"   - Average reciprocity: {emotion_details['mean_reciprocity']:.4f}")

        if emotion_details.get("mean_response_time") is not None:
            st.write(
                f"   - Average response time: {emotion_details['mean_response_time']:.2f}s"
            )


def display_topic_insights(topic_insights: Dict[str, Any]) -> None:
    """
    Display topic-level insights.

    Args:
        topic_insights: Topic insights dictionary
    """
    st.subheader("📊 Topic-Level Insights")

    # Highest reciprocity topics
    highest_r = topic_insights.get("highest_reciprocity_topics", [])
    if highest_r:
        st.write("**Topics with Highest Reciprocity:**")
        for i, topic_info in enumerate(highest_r, 1):
            st.write(f"{i}. {topic_info['topic']}: {topic_info['reciprocity']:.4f}")

    # Most positive topics
    most_positive = topic_insights.get("most_positive_topics", [])
    if most_positive:
        st.write("**Most Positive Topics:**")
        for i, topic_info in enumerate(most_positive, 1):
            st.write(
                f"{i}. {topic_info['topic']}: {topic_info['positive_count']} positive expressions"
            )

    # Outliers
    outliers = topic_insights.get("outlier_topics", [])
    if outliers:
        st.write("**Outlier Topics (>2 SD from mean):**")
        for outlier in outliers:
            st.write(
                f"- {outlier['topic']}: {outlier['metric']} = {outlier['value']:.4f} "
                f"(z-score: {outlier['z_score']})"
            )

    if not highest_r and not most_positive and not outliers:
        st.info("No topic-level data available.")
