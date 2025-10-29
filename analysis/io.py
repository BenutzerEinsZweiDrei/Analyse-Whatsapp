"""
IO and validation module for Profile Fusion.

Handles loading, validation, and normalization of JSON personality profile files.
"""

import json
from typing import Any

import streamlit as st


def load_and_validate_json_files(
    uploaded_files: list[Any],
) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Load and validate JSON files from uploaded file objects.

    Args:
        uploaded_files: List of uploaded file objects from st.file_uploader

    Returns:
        Tuple of (list of parsed JSON dictionaries, list of filenames)
        Skips files that fail to parse with error messages
    """
    data_list = []
    filenames = []

    for file in uploaded_files:
        try:
            # Reset file pointer to beginning
            file.seek(0)
            # Read and parse JSON
            content = json.load(file)

            # Basic validation - ensure it's a dictionary
            if not isinstance(content, dict):
                st.error(
                    f"❌ {file.name}: Expected JSON object (dictionary), got {type(content).__name__}"
                )
                continue

            data_list.append(content)
            filenames.append(file.name)

        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse {file.name}: Invalid JSON format - {str(e)}")
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {str(e)}")

    return data_list, filenames


@st.cache_data
def normalize_trait_value(value: Any) -> float | None:
    """
    Normalize trait values that might be in different formats.

    Handles:
    - Direct float/int values
    - Dict with 'mean' key
    - Dict with 'value' key
    - Invalid/missing values

    Args:
        value: Value to normalize (float, int, dict, or other)

    Returns:
        Normalized float value or None if invalid
    """
    if value is None:
        return None

    # Direct numeric value
    if isinstance(value, (int, float)):
        return float(value)

    # Dictionary with nested value
    if isinstance(value, dict):
        # Try 'mean' first, then 'value'
        if "mean" in value:
            return normalize_trait_value(value["mean"])
        elif "value" in value:
            return normalize_trait_value(value["value"])
        elif "score" in value:
            return normalize_trait_value(value["score"])

    # Unable to normalize
    return None


@st.cache_data
def normalize_emotion_counts(emotion_data: dict[str, Any]) -> dict[str, int]:
    """
    Normalize emotion count dictionaries to ensure integer counts.

    Args:
        emotion_data: Dictionary mapping emotion names to counts (various formats)

    Returns:
        Dictionary mapping emotion names to integer counts
    """
    normalized = {}

    for emotion, count in emotion_data.items():
        # Handle nested dictionaries
        if isinstance(count, dict):
            count = count.get("count", count.get("value", 0))

        # Convert to int, default to 0 if invalid
        try:
            normalized[emotion] = int(count) if count is not None else 0
        except (ValueError, TypeError):
            normalized[emotion] = 0

    return normalized


@st.cache_data
def normalize_mbti_data(mbti_data: dict[str, Any]) -> dict[str, dict[str, float]]:
    """
    Normalize MBTI data to consistent format.

    Handles both simple count format and nested dictionary format.

    Args:
        mbti_data: Dictionary mapping MBTI types to counts or nested dicts

    Returns:
        Dictionary mapping MBTI types to dicts with 'count', 'mean_reciprocity', etc.
    """
    normalized = {}

    for mbti_type, value in mbti_data.items():
        if isinstance(value, dict):
            # Already in nested format
            normalized[mbti_type] = {
                "count": float(value.get("count", 0)),
                "mean_reciprocity": normalize_trait_value(value.get("mean_reciprocity")),
                "mean_response_time": normalize_trait_value(value.get("mean_response_time")),
            }
        elif isinstance(value, (int, float)):
            # Simple count format
            normalized[mbti_type] = {
                "count": float(value),
                "mean_reciprocity": None,
                "mean_response_time": None,
            }

    return normalized


@st.cache_data
def normalize_response_time(time_value: Any) -> float | None:
    """
    Normalize response time to seconds.

    Args:
        time_value: Time value (seconds, milliseconds, or nested dict)

    Returns:
        Time in seconds as float, or None if invalid
    """
    value = normalize_trait_value(time_value)

    if value is None:
        return None

    # Assume if value > 10000, it's in milliseconds
    if value > 10000:
        return value / 1000.0

    return value


def normalize_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a single personality profile data dictionary.

    Ensures all fields are in consistent formats for downstream processing.

    Args:
        data: Raw personality profile dictionary

    Returns:
        Normalized dictionary with consistent field formats
    """
    normalized = {}

    # Normalize personality aggregation (Big Five)
    if "personality_aggregation" in data:
        normalized["personality_aggregation"] = {
            trait: normalize_trait_value(value)
            for trait, value in data["personality_aggregation"].items()
        }

    # Normalize basic metrics
    if "basic_metrics" in data:
        basic_metrics = data["basic_metrics"]
        normalized["basic_metrics"] = {}

        # Emotion counts
        if "dominant_emotion_counts" in basic_metrics:
            normalized["basic_metrics"]["dominant_emotion_counts"] = normalize_emotion_counts(
                basic_metrics["dominant_emotion_counts"]
            )

        # Reciprocity
        if "emotional_reciprocity" in basic_metrics:
            normalized["basic_metrics"]["emotional_reciprocity"] = normalize_trait_value(
                basic_metrics["emotional_reciprocity"]
            )

        # Response time
        if "mean_response_time" in basic_metrics:
            normalized["basic_metrics"]["mean_response_time"] = normalize_response_time(
                basic_metrics["mean_response_time"]
            )

    # Normalize MBTI summary
    if "mbti_summary" in data:
        normalized["mbti_summary"] = normalize_mbti_data(data["mbti_summary"])

    # Copy over topics if present
    if "per_topic_analysis" in data:
        normalized["per_topic_analysis"] = data["per_topic_analysis"]

    # Copy over correlations if present
    if "correlations" in data:
        normalized["correlations"] = data["correlations"]

    # Copy over metadata
    if "metadata" in data:
        normalized["metadata"] = data["metadata"]

    return normalized
