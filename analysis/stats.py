"""
Statistics module for Profile Fusion.

Computes aggregated statistics across personality profiles.
"""

from typing import Any

import streamlit as st

from .io import normalize_data


@st.cache_data
def compute_aggregated_statistics(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregated statistics across all personality profiles.

    Computes:
    - Mean and std for emotional reciprocity
    - Mean and std for response time
    - Top 3 emotions overall
    - MBTI distribution percentages
    - Most frequent top/bottom Big Five traits

    Args:
        data_list: List of personality result dictionaries

    Returns:
        Dictionary with aggregated statistics
    """
    stats = {
        "reciprocity": {"values": [], "mean": None, "std": None},
        "response_time": {"values": [], "mean": None, "std": None},
        "top_emotions": [],
        "mbti_percentages": {},
        "trait_rankings": {"most_common_top": None, "most_common_bottom": None},
    }

    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    # Collect reciprocity values
    for data in normalized_list:
        basic_metrics = data.get("basic_metrics", {})
        reciprocity = basic_metrics.get("emotional_reciprocity")
        if reciprocity is not None:
            stats["reciprocity"]["values"].append(reciprocity)

    # Compute reciprocity statistics
    if stats["reciprocity"]["values"]:
        values = stats["reciprocity"]["values"]
        mean_val = sum(values) / len(values)
        stats["reciprocity"]["mean"] = round(mean_val, 4)

        if len(values) > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
            stats["reciprocity"]["std"] = round(variance**0.5, 4)

    # Collect response time values
    for data in normalized_list:
        basic_metrics = data.get("basic_metrics", {})
        response_time = basic_metrics.get("mean_response_time")
        if response_time is not None:
            stats["response_time"]["values"].append(response_time)

    # Compute response time statistics
    if stats["response_time"]["values"]:
        values = stats["response_time"]["values"]
        mean_val = sum(values) / len(values)
        stats["response_time"]["mean"] = round(mean_val, 2)

        if len(values) > 1:
            variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
            stats["response_time"]["std"] = round(variance**0.5, 2)

    # Aggregate emotion counts for top 3
    emotion_totals = {}
    for data in normalized_list:
        basic_metrics = data.get("basic_metrics", {})
        emotion_counts = basic_metrics.get("dominant_emotion_counts", {})
        for emotion, count in emotion_counts.items():
            emotion_totals[emotion] = emotion_totals.get(emotion, 0) + count

    # Get top 3 emotions
    sorted_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)
    stats["top_emotions"] = [
        {"emotion": emotion, "count": count} for emotion, count in sorted_emotions[:3]
    ]

    # Compute MBTI percentages
    mbti_totals = {}
    total_mbti_count = 0
    for data in normalized_list:
        mbti_summary = data.get("mbti_summary", {})
        for mbti_type, type_data in mbti_summary.items():
            count = type_data.get("count", 0)
            if count and count > 0:
                mbti_totals[mbti_type] = mbti_totals.get(mbti_type, 0) + count
                total_mbti_count += count

    if total_mbti_count > 0:
        stats["mbti_percentages"] = {
            mbti_type: round((count / total_mbti_count) * 100, 2)
            for mbti_type, count in mbti_totals.items()
        }

    # Identify most frequent top and bottom traits
    top_traits = []
    bottom_traits = []

    for data in normalized_list:
        personality_agg = data.get("personality_aggregation", {})
        if personality_agg:
            # Sort traits by value
            sorted_traits = sorted(personality_agg.items(), key=lambda x: x[1] or 0, reverse=True)
            if sorted_traits:
                top_traits.append(sorted_traits[0][0])  # Highest trait
                bottom_traits.append(sorted_traits[-1][0])  # Lowest trait

    # Find most common
    if top_traits:
        from collections import Counter

        top_counter = Counter(top_traits)
        stats["trait_rankings"]["most_common_top"] = top_counter.most_common(1)[0][0]

    if bottom_traits:
        from collections import Counter

        bottom_counter = Counter(bottom_traits)
        stats["trait_rankings"]["most_common_bottom"] = bottom_counter.most_common(1)[0][0]

    return stats


@st.cache_data
def compute_per_topic_statistics(data_list: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute statistics at the topic level across profiles.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        Dictionary with per-topic statistics
    """
    topic_stats = {}

    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    for data in normalized_list:
        per_topic = data.get("per_topic_analysis", {})

        for topic, topic_data in per_topic.items():
            if topic not in topic_stats:
                topic_stats[topic] = {
                    "reciprocities": [],
                    "response_times": [],
                    "emotion_counts": {},
                }

            # Collect reciprocity
            if "mean_reciprocity" in topic_data:
                reciprocity = topic_data["mean_reciprocity"]
                if reciprocity is not None:
                    topic_stats[topic]["reciprocities"].append(reciprocity)

            # Collect response time
            if "mean_response_time" in topic_data:
                response_time = topic_data["mean_response_time"]
                if response_time is not None:
                    topic_stats[topic]["response_times"].append(response_time)

            # Collect emotion counts
            if "dominant_emotions" in topic_data:
                emotions = topic_data["dominant_emotions"]
                for emotion, count in emotions.items():
                    topic_stats[topic]["emotion_counts"][emotion] = (
                        topic_stats[topic]["emotion_counts"].get(emotion, 0) + count
                    )

    # Compute aggregates for each topic
    results = {}
    for topic, data in topic_stats.items():
        results[topic] = {
            "mean_reciprocity": (
                round(sum(data["reciprocities"]) / len(data["reciprocities"]), 4)
                if data["reciprocities"]
                else None
            ),
            "mean_response_time": (
                round(sum(data["response_times"]) / len(data["response_times"]), 2)
                if data["response_times"]
                else None
            ),
            "top_emotions": sorted(
                data["emotion_counts"].items(), key=lambda x: x[1], reverse=True
            )[:3],
        }

    return results
