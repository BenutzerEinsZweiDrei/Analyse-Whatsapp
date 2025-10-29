"""
Merge module for Profile Fusion.

Handles merging of personality traits, emotions, and MBTI data across multiple profiles.
"""

from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from .io import normalize_data, normalize_trait_value


@st.cache_data
def merge_big_five(data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge Big Five personality traits across multiple profiles.

    Computes the average and standard deviation for each trait across all files
    that contain that trait.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        pandas DataFrame with columns ["trait", "mean", "std", "count"]
        sorted by mean score descending
    """
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    trait_values = {trait: [] for trait in traits}

    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    # Collect trait values from all files
    for data in normalized_list:
        personality_agg = data.get("personality_aggregation", {})
        for trait in traits:
            if trait in personality_agg:
                value = personality_agg[trait]
                if value is not None:
                    trait_values[trait].append(float(value))

    # Compute statistics
    results = []
    for trait, values in trait_values.items():
        if values:  # Only include traits that have at least one value
            mean_score = sum(values) / len(values)
            std_score = 0.0
            if len(values) > 1:
                variance = sum((x - mean_score) ** 2 for x in values) / (len(values) - 1)
                std_score = variance**0.5

            results.append(
                {
                    "trait": trait,
                    "mean": round(mean_score, 4),
                    "std": round(std_score, 4),
                    "count": len(values),
                }
            )

    # Sort by mean score descending
    results.sort(key=lambda x: x["mean"], reverse=True)

    return pd.DataFrame(results)


@st.cache_data
def merge_emotions(data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge emotion counts across multiple profiles.

    Sums up the counts for each emotion type across all files.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        pandas DataFrame with emotion names as index and 'count' as column,
        sorted by count descending
    """
    emotion_totals = {}

    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    # Combine emotion counts from all files
    for data in normalized_list:
        basic_metrics = data.get("basic_metrics", {})
        emotion_counts = basic_metrics.get("dominant_emotion_counts", {})

        for emotion, count in emotion_counts.items():
            if isinstance(count, (int, float)) and count > 0:
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + count

    # Sort by count descending
    sorted_emotions = dict(sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True))

    # Convert to DataFrame
    if sorted_emotions:
        return pd.DataFrame.from_dict(sorted_emotions, orient="index", columns=["count"])
    else:
        return pd.DataFrame()


@st.cache_data
def merge_mbti(data_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Merge MBTI type counts across multiple profiles.

    Sums up the counts and averages reciprocity/response_time for each MBTI type.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        pandas DataFrame with columns ["mbti_type", "count", "mean_reciprocity",
        "mean_response_time"] sorted by count descending
    """
    mbti_data = {}

    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    # Combine MBTI data from all files
    for data in normalized_list:
        mbti_summary = data.get("mbti_summary", {})

        for mbti_type, type_data in mbti_summary.items():
            if mbti_type not in mbti_data:
                mbti_data[mbti_type] = {"counts": [], "reciprocities": [], "response_times": []}

            count = type_data.get("count", 0)
            if count and count > 0:
                mbti_data[mbti_type]["counts"].append(count)

                reciprocity = type_data.get("mean_reciprocity")
                if reciprocity is not None:
                    mbti_data[mbti_type]["reciprocities"].append(reciprocity)

                response_time = type_data.get("mean_response_time")
                if response_time is not None:
                    mbti_data[mbti_type]["response_times"].append(response_time)

    # Compute aggregates
    results = []
    for mbti_type, values in mbti_data.items():
        if values["counts"]:
            total_count = sum(values["counts"])

            mean_reciprocity = None
            if values["reciprocities"]:
                mean_reciprocity = sum(values["reciprocities"]) / len(values["reciprocities"])

            mean_response_time = None
            if values["response_times"]:
                mean_response_time = sum(values["response_times"]) / len(values["response_times"])

            results.append(
                {
                    "mbti_type": mbti_type,
                    "count": int(total_count),
                    "mean_reciprocity": (
                        round(mean_reciprocity, 4) if mean_reciprocity is not None else None
                    ),
                    "mean_response_time": (
                        round(mean_response_time, 2) if mean_response_time is not None else None
                    ),
                }
            )

    # Sort by count descending
    results.sort(key=lambda x: x["count"], reverse=True)

    return pd.DataFrame(results)


def merge_all_data(
    data_list: List[Dict[str, Any]]
) -> Dict[str, pd.DataFrame]:
    """
    Merge all personality data across multiple profiles.

    Convenience function that calls all merge functions.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        Dictionary with keys 'big_five', 'emotions', 'mbti' mapping to DataFrames
    """
    return {
        "big_five": merge_big_five(data_list),
        "emotions": merge_emotions(data_list),
        "mbti": merge_mbti(data_list),
    }
