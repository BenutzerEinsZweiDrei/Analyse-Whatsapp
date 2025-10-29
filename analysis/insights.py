"""
Insights module for Profile Fusion.

Generates emotional patterns and topic-level insights.
"""

from typing import Any

import streamlit as st

from .io import normalize_data
from .stats import compute_per_topic_statistics


@st.cache_data
def generate_emotional_insights(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate insights about emotional patterns across profiles.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        Dictionary with emotional insights
    """
    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    insights = {
        "top_emotions": [],
        "emotion_details": {},
        "outlier_emotions": [],
    }

    # Aggregate emotion counts
    emotion_totals = {}
    for data in normalized_list:
        basic_metrics = data.get("basic_metrics", {})
        emotion_counts = basic_metrics.get("dominant_emotion_counts", {})
        for emotion, count in emotion_counts.items():
            emotion_totals[emotion] = emotion_totals.get(emotion, 0) + count

    # Get top 3 emotions
    sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
    insights["top_emotions"] = [
        {"emotion": emotion, "count": count, "percentage": 0}
        for emotion, count in sorted_emotions[:3]
    ]

    # Calculate percentages
    total_emotions = sum(emotion_totals.values())
    if total_emotions > 0:
        for emotion_info in insights["top_emotions"]:
            emotion_info["percentage"] = round((emotion_info["count"] / total_emotions) * 100, 2)

    # For each top emotion, gather details
    for emotion_info in insights["top_emotions"]:
        emotion = emotion_info["emotion"]
        insights["emotion_details"][emotion] = {
            "topics_where_dominant": [],
            "mean_reciprocity": None,
            "mean_response_time": None,
        }

        reciprocities = []
        response_times = []

        # Check topics
        for data in normalized_list:
            per_topic = data.get("per_topic_analysis", {})
            for topic, topic_data in per_topic.items():
                dominant_emotions = topic_data.get("dominant_emotions", {})
                if emotion in dominant_emotions and dominant_emotions[emotion] > 0:
                    insights["emotion_details"][emotion]["topics_where_dominant"].append(
                        {
                            "topic": topic,
                            "count": dominant_emotions[emotion],
                        }
                    )

            # Collect metrics where this emotion dominates
            basic_metrics = data.get("basic_metrics", {})
            emotion_counts = basic_metrics.get("dominant_emotion_counts", {})

            # Simple heuristic: if this emotion is in top 3 for this file
            sorted_file_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            top_3_emotions = [e[0] for e in sorted_file_emotions[:3]]

            if emotion in top_3_emotions:
                reciprocity = basic_metrics.get("emotional_reciprocity")
                if reciprocity is not None:
                    reciprocities.append(reciprocity)

                response_time = basic_metrics.get("mean_response_time")
                if response_time is not None:
                    response_times.append(response_time)

        # Compute means
        if reciprocities:
            insights["emotion_details"][emotion]["mean_reciprocity"] = round(
                sum(reciprocities) / len(reciprocities), 4
            )

        if response_times:
            insights["emotion_details"][emotion]["mean_response_time"] = round(
                sum(response_times) / len(response_times), 2
            )

    return insights


@st.cache_data
def generate_topic_insights(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Generate topic-level insights across profiles.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        Dictionary with topic-level insights
    """
    topic_stats = compute_per_topic_statistics(data_list)

    insights = {
        "highest_reciprocity_topics": [],
        "lowest_reciprocity_topics": [],
        "highest_response_time_topics": [],
        "lowest_response_time_topics": [],
        "most_positive_topics": [],
        "outlier_topics": [],
    }

    if not topic_stats:
        return insights

    # Sort topics by reciprocity
    topics_with_reciprocity = [
        (topic, data["mean_reciprocity"])
        for topic, data in topic_stats.items()
        if data["mean_reciprocity"] is not None
    ]
    if topics_with_reciprocity:
        topics_with_reciprocity.sort(key=lambda x: x[1], reverse=True)
        insights["highest_reciprocity_topics"] = [
            {"topic": topic, "reciprocity": reciprocity}
            for topic, reciprocity in topics_with_reciprocity[:3]
        ]
        insights["lowest_reciprocity_topics"] = [
            {"topic": topic, "reciprocity": reciprocity}
            for topic, reciprocity in topics_with_reciprocity[-3:]
        ]

    # Sort topics by response time
    topics_with_response_time = [
        (topic, data["mean_response_time"])
        for topic, data in topic_stats.items()
        if data["mean_response_time"] is not None
    ]
    if topics_with_response_time:
        topics_with_response_time.sort(key=lambda x: x[1], reverse=True)
        insights["highest_response_time_topics"] = [
            {"topic": topic, "response_time": response_time}
            for topic, response_time in topics_with_response_time[:3]
        ]
        insights["lowest_response_time_topics"] = [
            {"topic": topic, "response_time": response_time}
            for topic, response_time in topics_with_response_time[-3:]
        ]

    # Find most positive topics (high joy/positive emotion counts)
    positive_emotions = ["joy", "happiness", "gratitude", "love", "positive"]
    topics_with_positivity = []

    for topic, data in topic_stats.items():
        positive_count = sum(
            count for emotion, count in data["top_emotions"] if emotion in positive_emotions
        )
        if positive_count > 0:
            topics_with_positivity.append((topic, positive_count))

    if topics_with_positivity:
        topics_with_positivity.sort(key=lambda x: x[1], reverse=True)
        insights["most_positive_topics"] = [
            {"topic": topic, "positive_count": count} for topic, count in topics_with_positivity[:3]
        ]

    # Identify outliers (>2 std dev from mean)
    if topics_with_reciprocity and len(topics_with_reciprocity) >= 3:
        reciprocity_values = [r for _, r in topics_with_reciprocity]
        mean_r = sum(reciprocity_values) / len(reciprocity_values)
        std_r = (
            sum((x - mean_r) ** 2 for x in reciprocity_values) / len(reciprocity_values)
        ) ** 0.5

        if std_r > 0:
            for topic, reciprocity in topics_with_reciprocity:
                z_score = abs((reciprocity - mean_r) / std_r)
                if z_score > 2:
                    insights["outlier_topics"].append(
                        {
                            "topic": topic,
                            "metric": "reciprocity",
                            "value": reciprocity,
                            "z_score": round(z_score, 2),
                        }
                    )

    return insights
