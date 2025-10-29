"""
Profile matrix module for Profile Fusion.

Creates personality profile matrices that fuse features across files.
"""

from typing import Any, Literal

import pandas as pd
import streamlit as st

from .io import normalize_data


@st.cache_data
def create_profile_matrix(
    data_list: list[dict[str, Any]], filenames: list[str] | None = None
) -> pd.DataFrame:
    """
    Create personality profile matrix across files.

    Matrix rows: personality dimensions (Big Five, emotions, MBTI, metrics)
    Matrix columns: file sources

    Args:
        data_list: List of personality result dictionaries
        filenames: Optional list of filenames for column labels

    Returns:
        pandas DataFrame with features as rows and files as columns
    """
    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    # Create column names
    if filenames is None:
        columns = [f"File_{i+1}" for i in range(len(data_list))]
    else:
        columns = filenames

    # Initialize matrix data structure
    matrix_data = {}

    # Big Five traits
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    for trait in traits:
        row_data = []
        for data in normalized_list:
            personality_agg = data.get("personality_aggregation", {})
            value = personality_agg.get(trait)
            row_data.append(value)
        matrix_data[trait] = row_data

    # Basic metrics
    for metric_name, metric_key in [
        ("reciprocity", "emotional_reciprocity"),
        ("response_time", "mean_response_time"),
    ]:
        row_data = []
        for data in normalized_list:
            basic_metrics = data.get("basic_metrics", {})
            value = basic_metrics.get(metric_key)
            row_data.append(value)
        matrix_data[metric_name] = row_data

    # Top emotions (get top 3 across all files)
    all_emotions = set()
    for data in normalized_list:
        basic_metrics = data.get("basic_metrics", {})
        emotion_counts = basic_metrics.get("dominant_emotion_counts", {})
        all_emotions.update(emotion_counts.keys())

    # Get emotion counts for each file
    emotion_totals = dict.fromkeys(all_emotions, 0)
    for data in normalized_list:
        basic_metrics = data.get("basic_metrics", {})
        emotion_counts = basic_metrics.get("dominant_emotion_counts", {})
        for emotion, count in emotion_counts.items():
            emotion_totals[emotion] += count

    # Top 3 emotions
    top_emotions = sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True)[:3]

    for emotion, _ in top_emotions:
        row_data = []
        for data in normalized_list:
            basic_metrics = data.get("basic_metrics", {})
            emotion_counts = basic_metrics.get("dominant_emotion_counts", {})
            count = emotion_counts.get(emotion, 0)
            row_data.append(count)
        matrix_data[f"emotion_{emotion}"] = row_data

    # MBTI - most common type per file
    for i, data in enumerate(normalized_list):
        mbti_summary = data.get("mbti_summary", {})
        if mbti_summary:
            # Find most common type
            max_type = max(mbti_summary.items(), key=lambda x: x[1].get("count", 0), default=None)
            if max_type:
                matrix_data.setdefault("mbti_primary", [None] * len(data_list))[i] = max_type[0]

    # Create DataFrame
    df = pd.DataFrame(matrix_data, index=columns).T

    return df


@st.cache_data
def normalize_matrix(
    matrix: pd.DataFrame, method: Literal["none", "minmax", "zscore"] = "none"
) -> pd.DataFrame:
    """
    Normalize matrix values.

    Args:
        matrix: Input DataFrame
        method: Normalization method ('none', 'minmax', or 'zscore')

    Returns:
        Normalized DataFrame
    """
    if method == "none":
        return matrix

    # Work with numeric columns only
    numeric_matrix = matrix.select_dtypes(include=["number"])

    if method == "minmax":
        # Min-max normalization per row
        normalized = numeric_matrix.apply(
            lambda row: (
                (row - row.min()) / (row.max() - row.min()) if row.max() != row.min() else row
            ),
            axis=1,
        )
    elif method == "zscore":
        # Z-score normalization per row
        normalized = numeric_matrix.apply(
            lambda row: (row - row.mean()) / row.std() if row.std() != 0 else row, axis=1
        )
    else:
        normalized = numeric_matrix

    # Fill NaN values with 0
    normalized = normalized.fillna(0)

    # Merge back with non-numeric columns
    result = matrix.copy()
    result[normalized.columns] = normalized

    return result


@st.cache_data
def create_aggregated_profile_matrix(data_list: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Create aggregated profile matrix with mean, std, min, max across files.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        pandas DataFrame with features as rows and statistics as columns
    """
    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    matrix_data = {
        "feature": [],
        "mean": [],
        "std": [],
        "min": [],
        "max": [],
        "count": [],
    }

    # Big Five traits
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    for trait in traits:
        values = []
        for data in normalized_list:
            personality_agg = data.get("personality_aggregation", {})
            value = personality_agg.get(trait)
            if value is not None:
                values.append(value)

        if values:
            matrix_data["feature"].append(trait)
            matrix_data["mean"].append(round(sum(values) / len(values), 4))
            matrix_data["std"].append(
                round(
                    (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values))
                    ** 0.5,
                    4,
                )
                if len(values) > 1
                else 0.0
            )
            matrix_data["min"].append(round(min(values), 4))
            matrix_data["max"].append(round(max(values), 4))
            matrix_data["count"].append(len(values))

    # Basic metrics
    for metric_name, metric_key in [
        ("reciprocity", "emotional_reciprocity"),
        ("response_time", "mean_response_time"),
    ]:
        values = []
        for data in normalized_list:
            basic_metrics = data.get("basic_metrics", {})
            value = basic_metrics.get(metric_key)
            if value is not None:
                values.append(value)

        if values:
            matrix_data["feature"].append(metric_name)
            matrix_data["mean"].append(round(sum(values) / len(values), 4))
            matrix_data["std"].append(
                round(
                    (sum((x - sum(values) / len(values)) ** 2 for x in values) / len(values))
                    ** 0.5,
                    4,
                )
                if len(values) > 1
                else 0.0
            )
            matrix_data["min"].append(round(min(values), 4))
            matrix_data["max"].append(round(max(values), 4))
            matrix_data["count"].append(len(values))

    df = pd.DataFrame(matrix_data)
    return df
