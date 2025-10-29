"""
Local psychological profile generator module.

Provides deterministic analysis pipeline (steps 1-11) to analyze WhatsApp conversations
without external AI calls. Uses in-app data structures from streamlit_app.py.
"""

import json
import logging
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional imports with fallbacks
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_stats = None

try:
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


logger = logging.getLogger("whatsapp_analyzer.local_profile")


# ---------------------------
# JSON Encoder for numpy types
# ---------------------------


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types."""

    def default(self, obj):
        if HAS_NUMPY:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
        return super().default(obj)


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert numpy types to Python native types.
    
    Handles both keys and values in dicts, and elements in lists.
    
    Args:
        obj: Object to convert
    
    Returns:
        Converted object with Python native types
    """
    if HAS_NUMPY:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
    
    if isinstance(obj, dict):
        # Convert both keys and values
        return {str(k) if not isinstance(k, (str, int, float, bool, type(None))) else k: _convert_numpy_types(v) 
                for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def safe_json_dumps(data: Any, **kwargs) -> str:
    """
    Safely dump data to JSON, handling numpy types in keys and values.

    Args:
        data: Data to serialize
        **kwargs: Additional arguments for json.dumps

    Returns:
        JSON string
    """
    # First convert numpy types (including dict keys)
    converted_data = _convert_numpy_types(data)
    # Then use JSON encoder for any remaining special types
    return json.dumps(converted_data, cls=NumpyEncoder, **kwargs)


# ---------------------------
# Step 1: Load and Validate
# ---------------------------


def load_and_validate(summary: Dict, matrix: Dict) -> Tuple[Dict, Dict]:
    """
    Validate types and keys in summary and matrix.
    Coerce conversation IDs to strings for consistency.

    Args:
        summary: Summary dictionary from summarize_matrix
        matrix: Matrix dictionary from run_analysis

    Returns:
        Tuple of (normalized_summary, normalized_matrix)
    """
    logger.debug("Step 1: load_and_validate")

    if not isinstance(summary, dict):
        raise TypeError(f"summary must be a dict, got {type(summary)}")
    if not isinstance(matrix, dict):
        raise TypeError(f"matrix must be a dict, got {type(matrix)}")

    # Normalize matrix keys to strings
    normalized_matrix = {}
    for key, value in matrix.items():
        str_key = str(key)
        if isinstance(value, dict):
            normalized_matrix[str_key] = value.copy()
            # Ensure conversation_id is set
            if "conversation_id" not in normalized_matrix[str_key]:
                normalized_matrix[str_key]["conversation_id"] = str_key
        else:
            normalized_matrix[str_key] = value

    # Validate required keys in summary
    required_summary_keys = [
        "positive_topics",
        "negative_topics",
        "emotion_variability",
        "analysis",
    ]
    for key in required_summary_keys:
        if key not in summary:
            logger.warning(f"Missing key in summary: {key}")

    normalized_summary = summary.copy()

    logger.debug(f"Validated {len(normalized_matrix)} conversations")
    return normalized_summary, normalized_matrix


# ---------------------------
# Step 2: Normalize Structure
# ---------------------------


def normalize_structure(matrix: Dict) -> Union[List[Dict], "pd.DataFrame"]:
    """
    Flatten each conversation entry into a single dict with consistent fields.

    Args:
        matrix: Normalized matrix from step 1

    Returns:
        List of flattened records (or DataFrame if pandas available)
    """
    logger.debug("Step 2: normalize_structure")

    records = []

    for conv_id, entry in matrix.items():
        if not isinstance(entry, dict):
            logger.warning(f"Skipping non-dict entry for conversation {conv_id}")
            continue

        record = {"conversation_id": str(conv_id)}

        # Handle topic (can be list or string)
        topic = entry.get("topic", ["no topic"])
        if isinstance(topic, list):
            record["topic"] = topic[0] if topic else "no topic"
            record["topic_full"] = "|".join(topic) if len(topic) > 1 else record["topic"]
        else:
            record["topic"] = str(topic)
            record["topic_full"] = record["topic"]

        # Emojis (list)
        record["emojies"] = entry.get("emojies", [])

        # Sentiment (extract from list if needed)
        sentiment = entry.get("sentiment", ["neutral"])
        record["sentiment"] = sentiment[0] if isinstance(sentiment, list) else str(sentiment)

        # Emotion analysis
        emotion_analysis = entry.get("emotion_analysis", {})
        record["dominant_emotion"] = emotion_analysis.get("dominant_emotion", "neutral")

        # Flatten emotion ratios
        emotion_ratios = emotion_analysis.get("emotion_ratios", {})
        for emotion, ratio in emotion_ratios.items():
            record[f"emotion_ratio_{emotion}"] = float(ratio)

        # Big Five traits
        big_five = entry.get("big_five", {})
        for trait, score in big_five.items():
            record[f"big_five_{trait}"] = float(score) if score is not None else None

        # MBTI
        record["mbti"] = entry.get("mbti", "XXXX")

        # Emotional reciprocity
        record["emotional_reciprocity"] = float(entry.get("emotional_reciprocity", 0.5))

        # Response times
        response_times = entry.get("response_times", {})
        record["response_time_topic_average"] = float(response_times.get("topic_average", 0.0))

        # Average per-user response times if available
        per_user = response_times.get("per_user", {})
        if per_user:
            avg_per_user = sum(per_user.values()) / len(per_user)
            record["response_time_per_user_avg"] = float(avg_per_user)
        else:
            record["response_time_per_user_avg"] = 0.0

        # Words/keywords
        record["words"] = entry.get("words", [])

        records.append(record)

    logger.debug(f"Normalized {len(records)} conversation records")

    # Convert to DataFrame if pandas available
    if HAS_PANDAS:
        df = pd.DataFrame(records)
        logger.debug(f"Created DataFrame with shape {df.shape}")
        return df
    else:
        logger.debug("Pandas not available, returning list of dicts")
        return records


# ---------------------------
# Step 3: Clean Data
# ---------------------------


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with fallback.

    Args:
        value: Value to convert
        default: Default value if conversion fails

    Returns:
        Float value or default
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_data(records: Union["pd.DataFrame", List[Dict]]) -> Union["pd.DataFrame", List[Dict]]:
    """
    Replace missing fields with sensible defaults.
    Convert numeric-like strings to floats.
    Ensure categorical fields are strings.

    Args:
        records: DataFrame or list of dicts from normalize_structure

    Returns:
        Cleaned records (same type as input)
    """
    logger.debug("Step 3: clean_data")

    if HAS_PANDAS and isinstance(records, pd.DataFrame):
        df = records.copy()

        # Fill missing Big Five traits with neutral value (5.0)
        big_five_cols = [col for col in df.columns if col.startswith("big_five_")]
        for col in big_five_cols:
            df[col] = df[col].fillna(5.0)

        # Fill missing emotion ratios with 0
        emotion_cols = [col for col in df.columns if col.startswith("emotion_ratio_")]
        for col in emotion_cols:
            df[col] = df[col].fillna(0.0)

        # Fill missing reciprocity with neutral value (0.5)
        if "emotional_reciprocity" in df.columns:
            df["emotional_reciprocity"] = df["emotional_reciprocity"].fillna(0.5)

        # Fill missing response times with 0
        if "response_time_topic_average" in df.columns:
            df["response_time_topic_average"] = df["response_time_topic_average"].fillna(0.0)
        if "response_time_per_user_avg" in df.columns:
            df["response_time_per_user_avg"] = df["response_time_per_user_avg"].fillna(0.0)

        # Ensure categorical fields are strings
        for col in ["topic", "sentiment", "dominant_emotion", "mbti"]:
            if col in df.columns:
                df[col] = df[col].fillna("unknown").astype(str)

        logger.debug(f"Cleaned DataFrame with shape {df.shape}")
        return df
    else:
        # Clean list of dicts
        cleaned = []
        for record in records:
            cleaned_record = record.copy()

            # Clean Big Five traits
            for trait in [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]:
                key = f"big_five_{trait}"
                if key in cleaned_record:
                    cleaned_record[key] = safe_float(cleaned_record[key], 5.0)

            # Clean emotion ratios
            for key in list(cleaned_record.keys()):
                if key.startswith("emotion_ratio_"):
                    cleaned_record[key] = safe_float(cleaned_record[key], 0.0)

            # Clean reciprocity
            cleaned_record["emotional_reciprocity"] = safe_float(
                cleaned_record.get("emotional_reciprocity"), 0.5
            )

            # Clean response times
            cleaned_record["response_time_topic_average"] = safe_float(
                cleaned_record.get("response_time_topic_average"), 0.0
            )
            cleaned_record["response_time_per_user_avg"] = safe_float(
                cleaned_record.get("response_time_per_user_avg"), 0.0
            )

            # Ensure categorical fields are strings
            for field in ["topic", "sentiment", "dominant_emotion", "mbti"]:
                if field in cleaned_record:
                    cleaned_record[field] = str(cleaned_record[field] or "unknown")

            cleaned.append(cleaned_record)

        logger.debug(f"Cleaned {len(cleaned)} records")
        return cleaned


# ---------------------------
# Step 4: Compute Basic Metrics
# ---------------------------


def compute_basic_metrics(df: Union["pd.DataFrame", List[Dict]]) -> Dict:
    """
    Compute basic statistical metrics across all conversations.

    Args:
        df: Cleaned records from step 3

    Returns:
        Dictionary of computed metrics
    """
    logger.debug("Step 4: compute_basic_metrics")

    metrics = {}

    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # Use pandas for efficient computation
        if "emotional_reciprocity" in df.columns:
            recip_vals = df["emotional_reciprocity"].dropna()
            metrics["average_emotional_reciprocity"] = {
                "mean": float(recip_vals.mean()) if len(recip_vals) > 0 else 0.5,
                "std": float(recip_vals.std()) if len(recip_vals) > 1 else 0.0,
                "n": int(len(recip_vals)),
            }

        if "dominant_emotion" in df.columns:
            metrics["dominant_emotion_counts"] = dict(df["dominant_emotion"].value_counts())

        if "mbti" in df.columns:
            metrics["mbti_distribution"] = dict(df["mbti"].value_counts())

        if "response_time_topic_average" in df.columns:
            rt_vals = df["response_time_topic_average"].dropna()
            rt_vals = rt_vals[rt_vals > 0]  # Filter out zeros
            if len(rt_vals) > 0:
                metrics["response_time_stats"] = {
                    "mean": float(rt_vals.mean()),
                    "std": float(rt_vals.std()) if len(rt_vals) > 1 else 0.0,
                    "n": int(len(rt_vals)),
                }
            else:
                metrics["response_time_stats"] = {"mean": 0.0, "std": 0.0, "n": 0}

        # Per-conversation summary
        metrics["per_conversation_count"] = len(df)

    else:
        # Manual computation for list of dicts
        recip_values = [
            r.get("emotional_reciprocity", 0.5)
            for r in df
            if r.get("emotional_reciprocity") is not None
        ]
        if recip_values:
            mean_recip = sum(recip_values) / len(recip_values)
            if len(recip_values) > 1:
                variance = sum((x - mean_recip) ** 2 for x in recip_values) / (
                    len(recip_values) - 1
                )
                std_recip = variance**0.5
            else:
                std_recip = 0.0
            metrics["average_emotional_reciprocity"] = {
                "mean": mean_recip,
                "std": std_recip,
                "n": len(recip_values),
            }

        # Count dominant emotions
        emotions = [r.get("dominant_emotion", "neutral") for r in df]
        metrics["dominant_emotion_counts"] = dict(Counter(emotions))

        # MBTI distribution
        mbtis = [r.get("mbti", "XXXX") for r in df]
        metrics["mbti_distribution"] = dict(Counter(mbtis))

        # Response time stats
        rt_values = [r.get("response_time_topic_average", 0.0) for r in df]
        rt_values = [rt for rt in rt_values if rt > 0]
        if rt_values:
            mean_rt = sum(rt_values) / len(rt_values)
            if len(rt_values) > 1:
                variance_rt = sum((x - mean_rt) ** 2 for x in rt_values) / (len(rt_values) - 1)
                std_rt = variance_rt**0.5
            else:
                std_rt = 0.0
            metrics["response_time_stats"] = {"mean": mean_rt, "std": std_rt, "n": len(rt_values)}
        else:
            metrics["response_time_stats"] = {"mean": 0.0, "std": 0.0, "n": 0}

        metrics["per_conversation_count"] = len(df)

    logger.debug(f"Computed basic metrics: {len(metrics)} categories")
    return metrics


# ---------------------------
# Step 5: Aggregate Personality Data
# ---------------------------


def aggregate_personality_data(df: Union["pd.DataFrame", List[Dict]]) -> Dict:
    """
    Aggregate personality trait statistics (Big Five).

    Args:
        df: Cleaned records from step 3

    Returns:
        Dictionary with personality aggregations
    """
    logger.debug("Step 5: aggregate_personality_data")

    personality = {}
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        for trait in traits:
            col = f"big_five_{trait}"
            if col in df.columns:
                vals = df[col].dropna()
                if len(vals) > 0:
                    personality[trait] = {
                        "mean": float(vals.mean()),
                        "std": float(vals.std()) if len(vals) > 1 else 0.0,
                        "n": int(len(vals)),
                    }
    else:
        for trait in traits:
            key = f"big_five_{trait}"
            values = [r.get(key, 5.0) for r in df if r.get(key) is not None]
            if values:
                mean_val = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
                    std_val = variance**0.5
                else:
                    std_val = 0.0
                personality[trait] = {"mean": mean_val, "std": std_val, "n": len(values)}

    # Rank traits by mean
    if personality:
        sorted_traits = sorted(personality.items(), key=lambda x: x[1]["mean"], reverse=True)
        personality["top_trait"] = sorted_traits[0][0] if sorted_traits else None
        personality["bottom_trait"] = sorted_traits[-1][0] if sorted_traits else None

    logger.debug(f"Aggregated personality data for {len(personality)} traits")
    return personality


# ---------------------------
# Step 6: Correlation Analysis
# ---------------------------


def compute_correlation_manual(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute Pearson correlation manually (fallback when scipy not available).

    Args:
        x, y: Lists of numeric values

    Returns:
        Tuple of (correlation, p_value_placeholder)
    """
    n = len(x)
    if n < 2:
        return 0.0, None

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    numerator = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))

    sum_sq_x = sum((xi - mean_x) ** 2 for xi in x)
    sum_sq_y = sum((yi - mean_y) ** 2 for yi in y)

    denominator = (sum_sq_x * sum_sq_y) ** 0.5

    if denominator == 0:
        return 0.0, None

    correlation = numerator / denominator

    # Simplified p-value estimation (not exact)
    # For exact p-values, scipy is needed
    return correlation, None


def compute_spearman_manual(x: List[float], y: List[float]) -> Tuple[float, float]:
    """
    Compute Spearman rank correlation manually.

    Args:
        x, y: Lists of numeric values

    Returns:
        Tuple of (correlation, p_value_placeholder)
    """
    n = len(x)
    if n < 2:
        return 0.0, None

    # Rank the values
    def rank_values(vals):
        sorted_vals = sorted(enumerate(vals), key=lambda v: v[1])
        ranks = [0] * len(vals)
        for rank, (idx, _) in enumerate(sorted_vals, start=1):
            ranks[idx] = rank
        return ranks

    ranks_x = rank_values(x)
    ranks_y = rank_values(y)

    # Compute Pearson correlation on ranks
    return compute_correlation_manual(ranks_x, ranks_y)


def correlation_analysis(df: Union["pd.DataFrame", List[Dict]]) -> Dict:
    """
    Compute correlations between personality traits and other metrics.

    Args:
        df: Cleaned records from step 3

    Returns:
        Dictionary with correlation results
    """
    logger.debug("Step 6: correlation_analysis")

    correlations = {}
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    # Extract data
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        data = df
    else:
        # Convert to simpler structure for correlation
        data = df

    # Define pairs to correlate
    pairs = []
    for trait in traits:
        pairs.append((f"big_five_{trait}", "emotional_reciprocity", f"{trait}_vs_reciprocity"))
        pairs.append(
            (f"big_five_{trait}", "response_time_topic_average", f"{trait}_vs_response_time")
        )

    # Add reciprocity vs response time
    pairs.append(
        ("emotional_reciprocity", "response_time_topic_average", "reciprocity_vs_response_time")
    )

    for var1, var2, label in pairs:
        if HAS_PANDAS and isinstance(data, pd.DataFrame):
            if var1 in data.columns and var2 in data.columns:
                x = data[var1].dropna()
                y = data[var2].dropna()

                # Align indices
                common_idx = x.index.intersection(y.index)
                x = x.loc[common_idx].values
                y = y.loc[common_idx].values

                if len(x) >= 5 and HAS_SCIPY:
                    # Use scipy for accurate correlations
                    pearson_r, pearson_p = scipy_stats.pearsonr(x, y)
                    spearman_r, spearman_p = scipy_stats.spearmanr(x, y)

                    correlations[label] = {
                        "pearson_r": float(pearson_r),
                        "pearson_p": float(pearson_p),
                        "spearman_r": float(spearman_r),
                        "spearman_p": float(spearman_p),
                        "n": len(x),
                    }
                elif len(x) >= 2:
                    # Fallback to manual computation
                    pearson_r, _ = compute_correlation_manual(list(x), list(y))
                    spearman_r, _ = compute_spearman_manual(list(x), list(y))

                    correlations[label] = {
                        "pearson_r": pearson_r,
                        "pearson_p": None,
                        "spearman_r": spearman_r,
                        "spearman_p": None,
                        "n": len(x),
                        "note": (
                            "Sample size too small for reliable p-values"
                            if len(x) < 5
                            else "scipy not available, p-values not computed"
                        ),
                    }
        else:
            # Manual extraction from list of dicts
            values_x = []
            values_y = []
            for record in data:
                val_x = record.get(var1)
                val_y = record.get(var2)
                if val_x is not None and val_y is not None:
                    values_x.append(safe_float(val_x))
                    values_y.append(safe_float(val_y))

            if len(values_x) >= 2:
                pearson_r, _ = compute_correlation_manual(values_x, values_y)
                spearman_r, _ = compute_spearman_manual(values_x, values_y)

                correlations[label] = {
                    "pearson_r": pearson_r,
                    "pearson_p": None,
                    "spearman_r": spearman_r,
                    "spearman_p": None,
                    "n": len(values_x),
                    "note": "Manual computation without scipy, p-values not available",
                }

    logger.debug(f"Computed {len(correlations)} correlation analyses")
    return correlations


# ---------------------------
# Step 7: Filter and Segment
# ---------------------------


def filter_and_segment(df: Union["pd.DataFrame", List[Dict]]) -> Dict:
    """
    Group by topic and MBTI, compute aggregated metrics per group.

    Args:
        df: Cleaned records from step 3

    Returns:
        Dictionary with segmented data
    """
    logger.debug("Step 7: filter_and_segment")

    segments = {"by_topic": {}, "by_mbti": {}}

    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # Group by topic
        if "topic" in df.columns:
            for topic, group in df.groupby("topic"):
                topic_normalized = str(topic).lower().split()[0] if topic else "unknown"

                segment_data = {
                    "count": len(group),
                    "mean_reciprocity": (
                        float(group["emotional_reciprocity"].mean())
                        if "emotional_reciprocity" in group.columns
                        else 0.5
                    ),
                    "mean_response_time": (
                        float(group["response_time_topic_average"].mean())
                        if "response_time_topic_average" in group.columns
                        else 0.0
                    ),
                    "dominant_emotions": (
                        dict(group["dominant_emotion"].value_counts())
                        if "dominant_emotion" in group.columns
                        else {}
                    ),
                }

                # Add mean Big Five per topic
                big_five_means = {}
                for trait in [
                    "openness",
                    "conscientiousness",
                    "extraversion",
                    "agreeableness",
                    "neuroticism",
                ]:
                    col = f"big_five_{trait}"
                    if col in group.columns:
                        big_five_means[trait] = float(group[col].mean())
                segment_data["mean_big_five"] = big_five_means

                segments["by_topic"][topic_normalized] = segment_data

        # Group by MBTI
        if "mbti" in df.columns:
            for mbti, group in df.groupby("mbti"):
                segment_data = {
                    "count": len(group),
                    "mean_reciprocity": (
                        float(group["emotional_reciprocity"].mean())
                        if "emotional_reciprocity" in group.columns
                        else 0.5
                    ),
                    "mean_response_time": (
                        float(group["response_time_topic_average"].mean())
                        if "response_time_topic_average" in group.columns
                        else 0.0
                    ),
                    "dominant_emotions": (
                        dict(group["dominant_emotion"].value_counts())
                        if "dominant_emotion" in group.columns
                        else {}
                    ),
                }
                segments["by_mbti"][str(mbti)] = segment_data
    else:
        # Manual grouping for list of dicts
        topic_groups = defaultdict(list)
        mbti_groups = defaultdict(list)

        for record in df:
            topic = str(record.get("topic", "unknown")).lower().split()[0]
            topic_groups[topic].append(record)

            mbti = str(record.get("mbti", "XXXX"))
            mbti_groups[mbti].append(record)

        # Process topic groups
        for topic, records in topic_groups.items():
            recip_vals = [r.get("emotional_reciprocity", 0.5) for r in records]
            rt_vals = [r.get("response_time_topic_average", 0.0) for r in records]
            emotions = [r.get("dominant_emotion", "neutral") for r in records]

            segment_data = {
                "count": len(records),
                "mean_reciprocity": sum(recip_vals) / len(recip_vals) if recip_vals else 0.5,
                "mean_response_time": sum(rt_vals) / len(rt_vals) if rt_vals else 0.0,
                "dominant_emotions": dict(Counter(emotions)),
            }

            # Mean Big Five per topic
            big_five_means = {}
            for trait in [
                "openness",
                "conscientiousness",
                "extraversion",
                "agreeableness",
                "neuroticism",
            ]:
                key = f"big_five_{trait}"
                vals = [r.get(key, 5.0) for r in records if r.get(key) is not None]
                if vals:
                    big_five_means[trait] = sum(vals) / len(vals)
            segment_data["mean_big_five"] = big_five_means

            segments["by_topic"][topic] = segment_data

        # Process MBTI groups
        for mbti, records in mbti_groups.items():
            recip_vals = [r.get("emotional_reciprocity", 0.5) for r in records]
            rt_vals = [r.get("response_time_topic_average", 0.0) for r in records]
            emotions = [r.get("dominant_emotion", "neutral") for r in records]

            segments["by_mbti"][mbti] = {
                "count": len(records),
                "mean_reciprocity": sum(recip_vals) / len(recip_vals) if recip_vals else 0.5,
                "mean_response_time": sum(rt_vals) / len(rt_vals) if rt_vals else 0.0,
                "dominant_emotions": dict(Counter(emotions)),
            }

    logger.debug(
        f"Segmented data: {len(segments['by_topic'])} topics, {len(segments['by_mbti'])} MBTI types"
    )
    return segments


# ---------------------------
# Step 8: Emotion Insights
# ---------------------------


def emotion_insights(df: Union["pd.DataFrame", List[Dict]]) -> Dict:
    """
    Identify emotion patterns and flag outlier conversations.

    Args:
        df: Cleaned records from step 3

    Returns:
        Dictionary with emotion insights and flagged conversations
    """
    logger.debug("Step 8: emotion_insights")

    insights = {}

    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # Most common dominant emotion
        if "dominant_emotion" in df.columns:
            insights["most_common_emotion"] = (
                df["dominant_emotion"].mode()[0] if len(df) > 0 else "neutral"
            )

        # Average emotion ratios
        emotion_cols = [col for col in df.columns if col.startswith("emotion_ratio_")]
        avg_ratios = {}
        for col in emotion_cols:
            emotion_name = col.replace("emotion_ratio_", "")
            avg_ratios[emotion_name] = float(df[col].mean())
        insights["average_emotion_ratios"] = avg_ratios

        # Flag conversations with unusual patterns
        flagged = []

        if "emotional_reciprocity" in df.columns:
            # Low reciprocity (below 10th percentile)
            recip_threshold = df["emotional_reciprocity"].quantile(0.10)
            low_recip = df[df["emotional_reciprocity"] < recip_threshold]

            for idx, row in low_recip.iterrows():
                flagged.append(
                    {
                        "conversation_id": str(row.get("conversation_id", idx)),
                        "reason": "low_reciprocity",
                        "value": float(row["emotional_reciprocity"]),
                        "threshold": float(recip_threshold),
                    }
                )

        # High sadness (above 90th percentile if sadness ratio exists)
        if "emotion_ratio_sadness" in df.columns:
            sadness_threshold = df["emotion_ratio_sadness"].quantile(0.90)
            high_sadness = df[df["emotion_ratio_sadness"] > sadness_threshold]

            for idx, row in high_sadness.iterrows():
                flagged.append(
                    {
                        "conversation_id": str(row.get("conversation_id", idx)),
                        "reason": "high_sadness",
                        "value": float(row["emotion_ratio_sadness"]),
                        "threshold": float(sadness_threshold),
                    }
                )

        insights["flagged_conversations"] = flagged
    else:
        # Manual computation for list of dicts
        emotions = [r.get("dominant_emotion", "neutral") for r in df]
        emotion_counter = Counter(emotions)
        insights["most_common_emotion"] = (
            emotion_counter.most_common(1)[0][0] if emotion_counter else "neutral"
        )

        # Average emotion ratios
        emotion_ratio_keys = set()
        for record in df:
            emotion_ratio_keys.update(k for k in record.keys() if k.startswith("emotion_ratio_"))

        avg_ratios = {}
        for key in emotion_ratio_keys:
            emotion_name = key.replace("emotion_ratio_", "")
            values = [r.get(key, 0.0) for r in df]
            avg_ratios[emotion_name] = sum(values) / len(values) if values else 0.0
        insights["average_emotion_ratios"] = avg_ratios

        # Flag conversations
        flagged = []

        # Low reciprocity
        recip_values = sorted([r.get("emotional_reciprocity", 0.5) for r in df])
        if len(recip_values) >= 10:
            recip_threshold = recip_values[len(recip_values) // 10]  # 10th percentile
            for record in df:
                recip = record.get("emotional_reciprocity", 0.5)
                if recip < recip_threshold:
                    flagged.append(
                        {
                            "conversation_id": str(record.get("conversation_id", "unknown")),
                            "reason": "low_reciprocity",
                            "value": recip,
                            "threshold": recip_threshold,
                        }
                    )

        # High sadness
        sadness_values = sorted([r.get("emotion_ratio_sadness", 0.0) for r in df])
        if len(sadness_values) >= 10:
            sadness_threshold = sadness_values[int(len(sadness_values) * 0.9)]  # 90th percentile
            for record in df:
                sadness = record.get("emotion_ratio_sadness", 0.0)
                if sadness > sadness_threshold and sadness > 0:
                    flagged.append(
                        {
                            "conversation_id": str(record.get("conversation_id", "unknown")),
                            "reason": "high_sadness",
                            "value": sadness,
                            "threshold": sadness_threshold,
                        }
                    )

        insights["flagged_conversations"] = flagged

    logger.debug(
        f"Generated emotion insights with {len(insights.get('flagged_conversations', []))} flagged conversations"
    )
    return insights


# ---------------------------
# Step 8.5: Highlights and Rankings Analysis
# ---------------------------


def highlights_and_rankings(
    records: Union["pd.DataFrame", List[Dict]],
    min_topic_n: int = 2,
    reciprocity_thresholds: Tuple[float, float] = (0.9, 0.8),
    include_final_insight: bool = True,
) -> Dict:
    """
    Compute topic-level aggregations and rankings for emotional reciprocity,
    response times, and emotional highlights.

    Args:
        records: Cleaned records (DataFrame or list of dicts) from clean_data
        min_topic_n: Minimum number of conversations per topic to include in top rankings
        reciprocity_thresholds: Tuple of (high_threshold, low_threshold) for reciprocity
        include_final_insight: Whether to include interpretive insight in summary

    Returns:
        Dictionary containing:
        - topics_aggregated: Per-topic statistics
        - reciprocity_ranking: Topics ranked by emotional reciprocity
        - response_time_ranking: Topics ranked by response speed
        - emotional_highlights: Aggregated emotion patterns
        - summary_text: Human-readable formatted summary
        - final_insight: Optional interpretive insight
    """
    logger.debug("Step 8.5: highlights_and_rankings")

    # Normalize topics and aggregate by topic
    topic_aggregation = defaultdict(
        lambda: {
            "conversations": [],
            "reciprocity_values": [],
            "response_time_values": [],
            "emotion_counts": Counter(),
            "emotion_ratio_sums": defaultdict(float),
            "n": 0,
        }
    )

    # Extract data from records
    if HAS_PANDAS and isinstance(records, pd.DataFrame):
        for _, row in records.iterrows():
            topic = str(row.get("topic", "unknown")).lower().strip()

            agg = topic_aggregation[topic]
            agg["conversations"].append(str(row.get("conversation_id", "")))
            agg["reciprocity_values"].append(float(row.get("emotional_reciprocity", 0.5)))
            agg["response_time_values"].append(float(row.get("response_time_topic_average", 0.0)))
            agg["emotion_counts"][str(row.get("dominant_emotion", "neutral"))] += 1
            agg["n"] += 1

            # Sum emotion ratios
            for col in records.columns:
                if col.startswith("emotion_ratio_"):
                    emotion_name = col.replace("emotion_ratio_", "")
                    agg["emotion_ratio_sums"][emotion_name] += float(row.get(col, 0.0))
    else:
        for record in records:
            topic = str(record.get("topic", "unknown")).lower().strip()

            agg = topic_aggregation[topic]
            agg["conversations"].append(str(record.get("conversation_id", "")))
            agg["reciprocity_values"].append(float(record.get("emotional_reciprocity", 0.5)))
            agg["response_time_values"].append(
                float(record.get("response_time_topic_average", 0.0))
            )
            agg["emotion_counts"][str(record.get("dominant_emotion", "neutral"))] += 1
            agg["n"] += 1

            # Sum emotion ratios
            for key in record.keys():
                if key.startswith("emotion_ratio_"):
                    emotion_name = key.replace("emotion_ratio_", "")
                    agg["emotion_ratio_sums"][emotion_name] += float(record.get(key, 0.0))

    # Compute per-topic statistics
    topics_aggregated = {}
    for topic, agg in topic_aggregation.items():
        n = agg["n"]
        if n == 0:
            continue

        recip_values = agg["reciprocity_values"]
        rt_values = [rt for rt in agg["response_time_values"] if rt > 0]

        # Compute mean and median
        mean_reciprocity = sum(recip_values) / len(recip_values) if recip_values else 0.5
        mean_response_time = sum(rt_values) / len(rt_values) if rt_values else 0.0
        median_response_time = sorted(rt_values)[len(rt_values) // 2] if rt_values else 0.0

        # Most common emotion
        most_common_emotion = (
            agg["emotion_counts"].most_common(1)[0][0] if agg["emotion_counts"] else "neutral"
        )

        # Average emotion ratios
        avg_emotion_ratios = {
            emotion: total / n for emotion, total in agg["emotion_ratio_sums"].items()
        }

        topics_aggregated[topic] = {
            "n": n,
            "mean_emotional_reciprocity": mean_reciprocity,
            "mean_response_time_minutes": mean_response_time,
            "median_response_time_minutes": median_response_time,
            "dominant_emotion": most_common_emotion,
            "emotion_counts": dict(agg["emotion_counts"]),
            "avg_emotion_ratios": avg_emotion_ratios,
        }

    # Ranking: Emotional Reciprocity (descending)
    reciprocity_ranked = sorted(
        topics_aggregated.items(), key=lambda x: x[1]["mean_emotional_reciprocity"], reverse=True
    )

    top_reciprocity = []
    low_reciprocity = []
    high_threshold, low_threshold = reciprocity_thresholds

    for topic, stats in reciprocity_ranked:
        if stats["n"] >= min_topic_n:
            if stats["mean_emotional_reciprocity"] >= high_threshold:
                top_reciprocity.append((topic, stats))
            elif stats["mean_emotional_reciprocity"] <= low_threshold:
                low_reciprocity.append((topic, stats))

    # Ranking: Response Time (ascending - faster is better)
    response_time_ranked = sorted(
        [(t, s) for t, s in topics_aggregated.items() if s["mean_response_time_minutes"] > 0],
        key=lambda x: x[1]["mean_response_time_minutes"],
    )

    fastest_topics = []
    slowest_topics = []
    for topic, stats in response_time_ranked:
        if stats["n"] >= min_topic_n:
            if len(fastest_topics) < 3:
                fastest_topics.append((topic, stats))

    # Get slowest (reverse order)
    for topic, stats in reversed(response_time_ranked):
        if stats["n"] >= min_topic_n:
            slowest_topics.append((topic, stats))
            break  # Just get the slowest

    # Emotional Highlights: aggregate across all topics
    all_emotion_counts = Counter()
    all_emotion_ratio_sums = defaultdict(float)
    total_conversations = 0

    for stats in topics_aggregated.values():
        all_emotion_counts.update(stats["emotion_counts"])
        total_conversations += stats["n"]
        for emotion, ratio in stats["avg_emotion_ratios"].items():
            all_emotion_ratio_sums[emotion] += ratio * stats["n"]

    # Calculate percentages
    emotion_percentages = {
        emotion: (count / total_conversations * 100) if total_conversations > 0 else 0
        for emotion, count in all_emotion_counts.items()
    }

    # Find topics with highest gratitude/sadness
    gratitude_topics = sorted(
        [
            (t, s)
            for t, s in topics_aggregated.items()
            if s["avg_emotion_ratios"].get("gratitude", 0) > 0
        ],
        key=lambda x: x[1]["avg_emotion_ratios"].get("gratitude", 0),
        reverse=True,
    )[:3]

    sadness_topics = sorted(
        [
            (t, s)
            for t, s in topics_aggregated.items()
            if s["avg_emotion_ratios"].get("sadness", 0) > 0
        ],
        key=lambda x: x[1]["avg_emotion_ratios"].get("sadness", 0),
        reverse=True,
    )[:3]

    # Generate combined summary text block
    summary_lines = []
    summary_lines.append("🔹 Emotional Reciprocity Ranking:")

    # Top reciprocity (limit to 3)
    top_3_recip = (
        top_reciprocity[:3]
        if top_reciprocity
        else [item for item in reciprocity_ranked if item[1]["n"] >= min_topic_n][:3]
    )

    for i, (topic, stats) in enumerate(top_3_recip, 1):
        summary_lines.append(
            f"   {i}. {topic} ({stats['mean_emotional_reciprocity']:.2f}) — n={stats['n']}"
        )

    if low_reciprocity:
        summary_lines.append("   ↓")
        topic, stats = low_reciprocity[0]
        summary_lines.append(
            f"   Lowest: {topic} ({stats['mean_emotional_reciprocity']:.2f}) — n={stats['n']}"
        )

    summary_lines.append("")
    summary_lines.append("🔹 Response Speed Ranking:")

    for i, (topic, stats) in enumerate(fastest_topics[:3], 1):
        summary_lines.append(
            f"   {i}. {topic} (~{stats['mean_response_time_minutes']:.0f} min) — n={stats['n']}"
        )

    if slowest_topics:
        summary_lines.append("   ↓")
        topic, stats = slowest_topics[0]
        summary_lines.append(
            f"   Slowest: {topic} (~{stats['mean_response_time_minutes']:.0f} min) — n={stats['n']}"
        )

    summary_lines.append("")
    summary_lines.append("🔹 Emotional Highlights:")

    # Dominant tone
    if emotion_percentages:
        dominant_emotion = max(emotion_percentages.items(), key=lambda x: x[1])
        summary_lines.append(
            f"   • Dominant tone: {dominant_emotion[0].title()} ({dominant_emotion[1]:.0f}%)"
        )

    # Gratitude expressions
    if gratitude_topics:
        gratitude_topic_names = ", ".join([t for t, _ in gratitude_topics])
        summary_lines.append(f"   • Strongest gratitude expressions in: {gratitude_topic_names}")

    # Emotional dips (sadness)
    if sadness_topics:
        sadness_topic_names = ", ".join([t for t, _ in sadness_topics])
        summary_lines.append(f"   • Emotional dip detected in: {sadness_topic_names}")

    # Final insight (optional)
    final_insight = ""
    if include_final_insight:
        # Generate a simple interpretive insight
        avg_reciprocity = (
            sum(s["mean_emotional_reciprocity"] for s in topics_aggregated.values())
            / len(topics_aggregated)
            if topics_aggregated
            else 0.5
        )
        avg_response = sum(
            s["mean_response_time_minutes"]
            for s in topics_aggregated.values()
            if s["mean_response_time_minutes"] > 0
        ) / max(
            1, len([s for s in topics_aggregated.values() if s["mean_response_time_minutes"] > 0])
        )

        reciprocity_desc = (
            "highly"
            if avg_reciprocity > 0.7
            else "moderately"
            if avg_reciprocity > 0.5
            else "somewhat"
        )
        emotion_desc = dominant_emotion[0] if emotion_percentages else "neutral"
        response_desc = (
            "quick" if avg_response < 60 else "moderate" if avg_response < 180 else "delayed"
        )

        final_insight = (
            f"\nFinal Insight:\n"
            f"Overall, conversations are emotionally rich and {reciprocity_desc} reciprocal, "
            f"with {emotion_desc} tone prevailing despite {response_desc} response times."
        )

        summary_lines.append("")
        summary_lines.append(final_insight.strip())

    summary_text = "\n".join(summary_lines)

    result = {
        "topics_aggregated": topics_aggregated,
        "reciprocity_ranking": {
            "top_topics": [(t, s["mean_emotional_reciprocity"], s["n"]) for t, s in top_3_recip],
            "lowest_topics": [
                (t, s["mean_emotional_reciprocity"], s["n"]) for t, s in low_reciprocity[:3]
            ],
        },
        "response_time_ranking": {
            "fastest_topics": [
                (t, s["mean_response_time_minutes"], s["n"]) for t, s in fastest_topics
            ],
            "slowest_topics": [
                (t, s["mean_response_time_minutes"], s["n"]) for t, s in slowest_topics
            ],
        },
        "emotional_highlights": {
            "dominant_emotion_percentages": emotion_percentages,
            "high_gratitude_topics": [t for t, _ in gratitude_topics],
            "high_sadness_topics": [t for t, _ in sadness_topics],
        },
        "summary_text": summary_text,
        "final_insight": final_insight if include_final_insight else None,
    }

    logger.debug(f"Generated highlights and rankings for {len(topics_aggregated)} topics")
    return result


# ---------------------------
# Step 9: Visualizations (Optional)
# ---------------------------


def visualizations(df: Union["pd.DataFrame", List[Dict]], results: Dict) -> Dict:
    """
    Placeholder for optional visualization generation.
    Returns empty dict for now (can be extended later).

    Args:
        df: Cleaned records
        results: Accumulated results so far

    Returns:
        Dictionary with visualization metadata (empty for now)
    """
    logger.debug("Step 9: visualizations (skipped in current implementation)")
    return {"note": "Visualizations skipped - can be added later with matplotlib/plotly"}


# ---------------------------
# Step 10: Advanced Analysis (Optional)
# ---------------------------


def advanced_analysis(df: Union["pd.DataFrame", List[Dict]], results: Dict) -> Dict:
    """
    Optional clustering analysis using sklearn (if available).

    Args:
        df: Cleaned records
        results: Accumulated results so far

    Returns:
        Dictionary with advanced analysis results
    """
    logger.debug("Step 10: advanced_analysis")

    advanced = {}

    if not HAS_SKLEARN or not HAS_PANDAS or not isinstance(df, pd.DataFrame):
        advanced["note"] = "sklearn/pandas not available, advanced analysis skipped"
        return advanced

    # Prepare features for clustering
    feature_cols = []
    for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
        col = f"big_five_{trait}"
        if col in df.columns:
            feature_cols.append(col)

    if "emotional_reciprocity" in df.columns:
        feature_cols.append("emotional_reciprocity")

    # Add emotion ratios
    emotion_cols = [col for col in df.columns if col.startswith("emotion_ratio_")]
    feature_cols.extend(emotion_cols)

    if len(feature_cols) < 2 or len(df) < 3:
        advanced["note"] = "Insufficient data for clustering"
        return advanced

    # Extract and normalize features
    X = df[feature_cols].fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try KMeans for k in [2, 3, 4]
    clustering_results = {}
    for k in [2, 3, 4]:
        if len(df) < k:
            continue

        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)

            # Compute silhouette score
            if len(set(labels)) > 1:
                silhouette = silhouette_score(X_scaled, labels)
                clustering_results[f"kmeans_k{k}"] = {
                    "n_clusters": k,
                    "silhouette_score": float(silhouette),
                    "cluster_sizes": dict(Counter(labels)),
                }
        except Exception as e:
            logger.debug(f"Clustering with k={k} failed: {e}")

    advanced["clustering"] = clustering_results
    logger.debug(f"Advanced analysis completed with {len(clustering_results)} clustering results")

    return advanced


# ---------------------------
# Step 11: Export Results
# ---------------------------


def export_results(results: Dict, records: Union["pd.DataFrame", List[Dict]]) -> Dict:
    """
    Prepare export data in JSON and CSV formats.

    Args:
        results: Accumulated results dictionary
        records: Cleaned records (DataFrame or list)

    Returns:
        Dictionary with JSON and CSV strings for download
    """
    logger.debug("Step 11: export_results")

    exports = {}

    # Prepare metrics JSON
    metrics_data = {
        "basic_metrics": results.get("basic_metrics", {}),
        "personality_aggregation": results.get("big_five_aggregation", {}),
        "correlations": results.get("correlations", {}),
        "topics_summary": results.get("topics_summary", {}),
        "mbti_summary": results.get("mbti_summary", {}),
        "emotion_insights": results.get("emotion_insights", {}),
        "highlights_and_rankings": results.get("highlights_and_rankings", {}),
        "advanced_analysis": results.get("advanced_analysis", {}),
    }
    exports["metrics_json"] = safe_json_dumps(metrics_data, ensure_ascii=False, indent=2)

    # Prepare per-conversation CSV
    if HAS_PANDAS and isinstance(records, pd.DataFrame):
        exports["per_conversation_csv"] = records.to_csv(index=False)
    else:
        # Manual CSV generation
        if records:
            # Get all keys from first record
            keys = list(records[0].keys())
            csv_lines = [",".join(keys)]

            for record in records:
                values = []
                for key in keys:
                    val = record.get(key, "")
                    # Simple CSV escaping
                    if isinstance(val, (list, dict)):
                        val = json.dumps(val)
                    val_str = str(val).replace('"', '""')
                    if "," in val_str or '"' in val_str:
                        val_str = f'"{val_str}"'
                    values.append(val_str)
                csv_lines.append(",".join(values))

            exports["per_conversation_csv"] = "\n".join(csv_lines)
        else:
            exports["per_conversation_csv"] = ""

    # Prepare flagged conversations JSON
    flagged = results.get("emotion_insights", {}).get("flagged_conversations", [])
    exports["flagged_json"] = safe_json_dumps(flagged, ensure_ascii=False, indent=2)

    logger.debug("Export data prepared")
    return exports


# ---------------------------
# Main Pipeline Function
# ---------------------------


def run_local_analysis(summary: Dict, matrix: Dict) -> Tuple[Dict, str]:
    """
    Run the complete local analysis pipeline (steps 1-11).

    Args:
        summary: Summary dictionary from streamlit_app.summarize_matrix
        matrix: Matrix dictionary from streamlit_app.run_analysis

    Returns:
        Tuple of (results_dict, profile_text)
        - results_dict: Complete analysis results with all metrics
        - profile_text: Human-readable summary for display
    """
    logger.info("Starting local analysis pipeline")

    try:
        # Step 1: Load and validate
        summary, matrix = load_and_validate(summary, matrix)

        # Check if matrix is empty
        if not matrix:
            logger.warning("Empty matrix provided to local analysis")
            return {
                "error": "No conversations to analyze",
                "basic_metrics": {},
                "big_five_aggregation": {},
                "correlations": {},
                "topics_summary": {},
                "mbti_summary": {},
                "emotion_insights": {},
                "highlights_and_rankings": {},
                "advanced_analysis": {},
                "per_conversation_table": [],
            }, "⚠️ No conversations available for analysis."

        # Step 2: Normalize structure
        records = normalize_structure(matrix)

        # Step 3: Clean data
        records = clean_data(records)

        # Step 4: Compute basic metrics
        basic_metrics = compute_basic_metrics(records)

        # Step 5: Aggregate personality data
        personality_agg = aggregate_personality_data(records)

        # Step 6: Correlation analysis
        correlations = correlation_analysis(records)

        # Step 7: Filter and segment
        segments = filter_and_segment(records)

        # Step 8: Emotion insights
        emotion_insights_data = emotion_insights(records)

        # Step 8.5: Highlights and rankings
        highlights_data = highlights_and_rankings(records)

        # Step 9: Visualizations (optional, skipped)
        visualizations(records, {})

        # Step 10: Advanced analysis (optional)
        advanced = advanced_analysis(records, {})

        # Compile results
        results = {
            "basic_metrics": basic_metrics,
            "big_five_aggregation": personality_agg,
            "correlations": correlations,
            "topics_summary": segments.get("by_topic", {}),
            "mbti_summary": segments.get("by_mbti", {}),
            "emotion_insights": emotion_insights_data,
            "highlights_and_rankings": highlights_data,
            "advanced_analysis": advanced,
            "per_conversation_table": (
                records.to_dict("records")
                if HAS_PANDAS and isinstance(records, pd.DataFrame)
                else records
            ),
        }

        # Step 11: Prepare exports
        exports = export_results(results, records)
        results["exports"] = exports

        # Generate human-readable profile text
        profile_text = generate_profile_text(results)

        logger.info("Local analysis pipeline completed successfully")
        return results, profile_text

    except Exception as e:
        logger.exception("Error in local analysis pipeline: %s", e)
        return {
            "error": str(e),
            "basic_metrics": {},
            "big_five_aggregation": {},
            "correlations": {},
            "topics_summary": {},
            "mbti_summary": {},
            "emotion_insights": {},
            "highlights_and_rankings": {},
            "advanced_analysis": {},
            "per_conversation_table": [],
        }, f"⚠️ Analysis failed: {str(e)}"


def generate_profile_text(results: Dict) -> str:
    """
    Generate human-readable profile summary from results.

    Args:
        results: Complete results dictionary

    Returns:
        Formatted profile text with rich descriptions and insights
    """
    lines = ["# 🧠 Comprehensive Psychological Profile\n"]
    lines.append("*An in-depth analysis of communication patterns, emotional intelligence, and personality traits*")
    lines.append("\n---\n")

    # ========== PERSONALITY OVERVIEW SECTION ==========
    lines.append("## 📊 Personality Overview")
    lines.append("")

    personality = results.get("big_five_aggregation", {})
    mbti_dist = results.get("basic_metrics", {}).get("mbti_distribution", {})
    
    # Big Five Traits Analysis
    if personality:
        lines.append("### The Big Five Personality Dimensions")
        lines.append("")
        
        traits_info = {
            "openness": {
                "name": "Openness to Experience",
                "high": "This indicates a strong appreciation for art, emotion, adventure, and unusual ideas. You're intellectually curious, creative, and open to new experiences.",
                "moderate": "You balance tradition with novelty, showing practical creativity and selective openness to new experiences.",
                "low": "You tend to prefer familiar routines and conventional approaches, valuing practicality over novelty."
            },
            "conscientiousness": {
                "name": "Conscientiousness",
                "high": "You demonstrate exceptional organization, dependability, and self-discipline. Goal-oriented behavior and strong planning skills are evident.",
                "moderate": "You show a balanced approach to organization and spontaneity, being reliable while remaining flexible.",
                "low": "You tend toward spontaneity and flexibility, preferring to go with the flow rather than strict planning."
            },
            "extraversion": {
                "name": "Extraversion",
                "high": "You're highly sociable and energetic, thriving in social interactions and seeking out engagement with others.",
                "moderate": "You display ambivert qualities, comfortable in both social and solitary settings depending on the context.",
                "low": "You prefer quieter, more introspective settings and recharge through solitude rather than social interaction."
            },
            "agreeableness": {
                "name": "Agreeableness",
                "high": "You show strong empathy, cooperation, and concern for harmony in relationships. Compassion and kindness are your strengths.",
                "moderate": "You balance assertiveness with cooperation, maintaining healthy boundaries while being considerate of others.",
                "low": "You tend to be more analytical and direct in communication, prioritizing truth and logic over social harmony."
            },
            "neuroticism": {
                "name": "Emotional Stability",
                "high": "You experience emotions intensely and may be more sensitive to stress. This emotional depth can fuel creativity and empathy.",
                "moderate": "You maintain generally stable emotions with occasional sensitivity to stressors, showing healthy emotional responsiveness.",
                "low": "You demonstrate remarkable emotional resilience and stability, remaining calm under pressure."
            }
        }
        
        # Get all trait scores and sort them
        trait_scores = []
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            trait_data = personality.get(trait, {})
            if trait_data:
                mean_score = trait_data.get("mean", 5.0)
                trait_scores.append((trait, mean_score))
        
        # Sort by score (highest first)
        trait_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Display traits in order of prominence
        for trait, score in trait_scores:
            trait_info = traits_info.get(trait, {})
            trait_name = trait_info.get("name", trait.title())
            
            # Determine interpretation based on score
            if score >= 7.0:
                interpretation = trait_info.get("high", "")
                level = "High"
                emoji = "🔴"
            elif score >= 4.0:
                interpretation = trait_info.get("moderate", "")
                level = "Moderate"
                emoji = "🟡"
            else:
                interpretation = trait_info.get("low", "")
                level = "Low"
                emoji = "🟢"
            
            # Create visual bar (clamp score between 0 and 10)
            bar_length = max(0, min(10, int(score)))
            bar = "█" * bar_length + "░" * (10 - bar_length)
            
            lines.append(f"**{emoji} {trait_name}** — {level} ({score:.1f}/10)")
            lines.append(f"`{bar}`")
            lines.append(f"*{interpretation}*")
            lines.append("")
    
    # MBTI Type Analysis
    if mbti_dist:
        lines.append("### Myers-Briggs Type Indicator (MBTI)")
        lines.append("")
        top_mbti = max(mbti_dist.items(), key=lambda x: x[1])[0] if mbti_dist else "XXXX"
        
        mbti_descriptions = {
            "INTJ": "The Architect - Strategic, analytical, and independent thinker",
            "INTP": "The Logician - Innovative, philosophical, and curious problem-solver",
            "ENTJ": "The Commander - Bold, decisive, and strategic leader",
            "ENTP": "The Debater - Quick-witted, resourceful, and intellectually curious",
            "INFJ": "The Advocate - Idealistic, empathetic, and purpose-driven",
            "INFP": "The Mediator - Poetic, creative, and values-driven",
            "ENFJ": "The Protagonist - Charismatic, inspiring, and natural leader",
            "ENFP": "The Campaigner - Enthusiastic, creative, and sociable free spirit",
            "ISTJ": "The Logistician - Practical, fact-minded, and reliable",
            "ISFJ": "The Defender - Dedicated, warm, and protector of traditions",
            "ESTJ": "The Executive - Organized, practical, and efficient administrator",
            "ESFJ": "The Consul - Caring, social, and community-minded",
            "ISTP": "The Virtuoso - Bold, practical, and experimental problem-solver",
            "ISFP": "The Adventurer - Flexible, charming, and artistic explorer",
            "ESTP": "The Entrepreneur - Energetic, perceptive, and action-oriented",
            "ESFP": "The Entertainer - Spontaneous, enthusiastic, and outgoing performer"
        }
        
        description = mbti_descriptions.get(top_mbti, "Unique personality pattern")
        count = mbti_dist.get(top_mbti, 0)
        total = sum(mbti_dist.values())
        percentage = (count / total * 100) if total > 0 else 0
        
        lines.append(f"**Primary Type: {top_mbti}**")
        lines.append(f"*{description}*")
        lines.append(f"Observed in {percentage:.0f}% of conversations ({count} of {total})")
        lines.append("")
    
    lines.append("---\n")

    # ========== EMOTIONAL INTELLIGENCE SECTION ==========
    lines.append("## 💝 Emotional Intelligence & Communication Dynamics")
    lines.append("")

    # Emotional Reciprocity Analysis
    recip = results.get("basic_metrics", {}).get("average_emotional_reciprocity", {})
    if recip:
        mean_recip = recip.get("mean", 0.5)
        std_recip = recip.get("std", 0.0)
        n_recip = recip.get("n", 0)
        
        lines.append("### Emotional Reciprocity")
        
        # Visual representation
        recip_percentage = int(mean_recip * 100)
        recip_bar = "█" * (recip_percentage // 10) + "░" * (10 - recip_percentage // 10)
        lines.append(f"`{recip_bar}` **{mean_recip:.2f}/1.0** ({recip_percentage}%)")
        lines.append("")
        
        # Detailed interpretation
        if mean_recip >= 0.75:
            lines.append("**Exceptional Emotional Attunement** 🌟")
            lines.append("Your conversations demonstrate outstanding mutual emotional engagement. You consistently match and respond to the emotional states of others, creating deeply connected and empathetic exchanges. This high reciprocity indicates:")
            lines.append("- Strong emotional intelligence and awareness")
            lines.append("- Ability to create safe spaces for emotional expression")
            lines.append("- Natural capacity for empathy and validation")
            lines.append("- Skilled at building and maintaining intimate connections")
        elif mean_recip >= 0.60:
            lines.append("**High Mutual Emotional Engagement** ✨")
            lines.append("You maintain strong emotional connections in your conversations. There's a healthy give-and-take of emotional expression, showing that you're both receptive to others' feelings and willing to share your own. This reflects:")
            lines.append("- Good emotional awareness and responsiveness")
            lines.append("- Balanced approach to emotional sharing")
            lines.append("- Ability to foster meaningful connections")
            lines.append("- Comfortable with emotional vulnerability")
        elif mean_recip >= 0.40:
            lines.append("**Moderate Emotional Exchange** 💭")
            lines.append("Your emotional reciprocity shows a balanced but somewhat reserved approach to emotional connection. While you engage emotionally, there may be selective sharing or varying comfort levels across different topics. Consider:")
            lines.append("- Opportunities to deepen emotional connections")
            lines.append("- Exploring comfort with vulnerability")
            lines.append("- Recognizing when emotional support is needed")
            lines.append("- Building trust for more open exchanges")
        else:
            lines.append("**Reserved Emotional Exchange** 🔒")
            lines.append("Your conversations tend toward practical or intellectual content with limited emotional reciprocity. This might indicate:")
            lines.append("- Preference for logical over emotional discourse")
            lines.append("- Possible emotional guardedness or boundaries")
            lines.append("- Opportunities for deeper emotional connection")
            lines.append("- Consider if this pattern aligns with your relationship goals")
        
        if std_recip > 0.2:
            lines.append(f"\n*Note: Reciprocity varies significantly across conversations (σ={std_recip:.2f}), suggesting context-dependent emotional engagement.*")
        
        lines.append("")

    # Dominant Emotions Analysis
    emotion_insights = results.get("emotion_insights", {})
    most_common = emotion_insights.get("most_common_emotion", "neutral")
    avg_ratios = emotion_insights.get("average_emotion_ratios", {})
    
    lines.append("### Emotional Landscape")
    lines.append("")
    
    emotion_emojis = {
        "joy": "😊",
        "happiness": "😊",
        "sadness": "😢",
        "anger": "😠",
        "fear": "😨",
        "surprise": "😲",
        "disgust": "🤢",
        "neutral": "😐",
        "love": "❤️",
        "gratitude": "🙏",
        "excitement": "🎉",
        "anxiety": "😰",
        "pride": "😌",
        "shame": "😔"
    }
    
    lines.append(f"**Primary Emotional Tone: {emotion_emojis.get(most_common.lower(), '💭')} {most_common.title()}**")
    lines.append("")
    
    # Show emotion distribution if available
    if avg_ratios:
        lines.append("**Emotional Distribution:**")
        # Sort emotions by ratio
        sorted_emotions = sorted(avg_ratios.items(), key=lambda x: x[1], reverse=True)
        for emotion, ratio in sorted_emotions[:5]:  # Top 5 emotions
            if ratio > 0.05:  # Only show emotions above 5%
                emoji = emotion_emojis.get(emotion.lower(), "•")
                percentage = ratio * 100
                bar_len = max(0, min(20, int(percentage / 5)))  # Scale for display, clamped to 0-20
                bar = "▓" * bar_len + "░" * (20 - bar_len)
                lines.append(f"{emoji} {emotion.title()}: `{bar}` {percentage:.1f}%")
        lines.append("")
    
    # Emotional complexity insight
    if avg_ratios:
        num_significant_emotions = sum(1 for ratio in avg_ratios.values() if ratio > 0.1)
        if num_significant_emotions >= 4:
            lines.append("*Your emotional expression is rich and diverse, showing emotional depth and complexity in your communications.*")
        elif num_significant_emotions >= 2:
            lines.append("*You express a balanced range of emotions, contributing to authentic and relatable interactions.*")
        else:
            lines.append("*Your emotional expression tends toward consistency and stability, with clear dominant themes.*")
        lines.append("")

    # Response Time Analysis
    rt_stats = results.get("basic_metrics", {}).get("response_time_stats", {})
    if rt_stats and rt_stats.get("n", 0) > 0:
        mean_rt = rt_stats.get("mean", 0.0)
        std_rt = rt_stats.get("std", 0.0)
        
        lines.append("### Communication Responsiveness")
        lines.append("")
        
        # Convert to hours/minutes for better readability
        if mean_rt < 60:
            time_str = f"{mean_rt:.0f} minutes"
        elif mean_rt < 1440:  # Less than 24 hours
            hours = mean_rt / 60
            time_str = f"{hours:.1f} hours"
        else:
            days = mean_rt / 1440
            time_str = f"{days:.1f} days"
        
        lines.append(f"**Average Response Time: {time_str}**")
        lines.append("")
        
        if mean_rt < 15:
            lines.append("**Highly Responsive** ⚡")
            lines.append("You respond almost immediately to messages, indicating high availability and prioritization of communication. This shows:")
            lines.append("- Strong engagement and attentiveness")
            lines.append("- Immediate accessibility to conversation partners")
            lines.append("- Possible real-time conversation flow")
        elif mean_rt < 60:
            lines.append("**Quick Engagement** 🚀")
            lines.append("Your response times are impressively quick, showing good availability and interest in maintaining active conversations.")
        elif mean_rt < 180:
            lines.append("**Moderate Pace** ⏱️")
            lines.append("You maintain a balanced response pattern, allowing time for thoughtful replies while staying engaged.")
        elif mean_rt < 1440:
            lines.append("**Thoughtful Responses** 🤔")
            lines.append("Your measured response times suggest careful consideration and may reflect a busy schedule or preference for asynchronous communication.")
        else:
            lines.append("**Delayed Communication** 📅")
            lines.append("Extended response times may indicate competing priorities or preference for less frequent, more substantial exchanges.")
        
        if std_rt > mean_rt * 0.5:
            lines.append(f"\n*Response times vary considerably (σ={std_rt:.1f}), suggesting different availability or engagement levels across contexts.*")
        
        lines.append("")

    lines.append("---\n")

    # ========== CONVERSATION PATTERNS SECTION ==========
    highlights = results.get("highlights_and_rankings", {})
    if highlights:
        lines.append("## 🎯 Conversation Patterns & Topic Analysis")
        lines.append("")
        
        # Topic-based reciprocity
        recip_ranking = highlights.get("reciprocity_ranking", {})
        top_topics = recip_ranking.get("top_topics", [])
        low_topics = recip_ranking.get("lowest_topics", [])
        
        if top_topics:
            lines.append("### Topics with Strongest Emotional Connection")
            lines.append("")
            for i, (topic, recip_score, n) in enumerate(top_topics[:3], 1):
                lines.append(f"{i}. **{topic.title()}** ({recip_score:.2f}) — {n} conversation{'s' if n > 1 else ''}")
                if recip_score >= 0.8:
                    lines.append(f"   *This topic elicits deep, authentic emotional sharing and mutual understanding.*")
                else:
                    lines.append(f"   *Strong emotional engagement and reciprocal communication patterns observed.*")
            lines.append("")
        
        if low_topics and low_topics[0][1] < 0.6:  # Only show if actually low
            lines.append("### Topics with Growth Opportunities")
            lines.append("")
            for topic, recip_score, n in low_topics[:2]:
                lines.append(f"• **{topic.title()}** ({recip_score:.2f}) — {n} conversation{'s' if n > 1 else ''}")
                lines.append(f"   *Consider exploring emotional dimensions of this topic for deeper connection.*")
            lines.append("")
        
        # Response speed by topic
        rt_ranking = highlights.get("response_time_ranking", {})
        fastest_topics = rt_ranking.get("fastest_topics", [])
        slowest_topics = rt_ranking.get("slowest_topics", [])
        
        if fastest_topics:
            lines.append("### High-Priority Topics (Fastest Responses)")
            lines.append("")
            for topic, rt_minutes, n in fastest_topics[:3]:
                if rt_minutes < 60:
                    time_str = f"{rt_minutes:.0f} min"
                else:
                    time_str = f"{rt_minutes/60:.1f} hrs"
                lines.append(f"• **{topic.title()}** — {time_str} average response")
            lines.append("")
            lines.append("*These topics appear to capture your immediate attention and engagement.*")
            lines.append("")
        
        # Emotional highlights
        emotional_hl = highlights.get("emotional_highlights", {})
        gratitude_topics = emotional_hl.get("high_gratitude_topics", [])
        sadness_topics = emotional_hl.get("high_sadness_topics", [])
        
        if gratitude_topics:
            lines.append(f"### Expressions of Appreciation 🙏")
            lines.append(f"*Strong gratitude detected in: {', '.join(t.title() for t in gratitude_topics[:3])}*")
            lines.append("")
        
        if sadness_topics:
            lines.append(f"### Emotional Vulnerability Noted 💙")
            lines.append(f"*Sadness or concern expressed in: {', '.join(t.title() for t in sadness_topics[:3])}*")
            lines.append("*These moments of vulnerability can deepen relationships when met with empathy.*")
            lines.append("")
        
        lines.append("---\n")

    # ========== INSIGHTS & RECOMMENDATIONS SECTION ==========
    lines.append("## 💡 Key Insights & Recommendations")
    lines.append("")
    
    # Flagged conversations
    flagged = emotion_insights.get("flagged_conversations", [])
    if flagged:
        low_recip_count = sum(1 for f in flagged if f.get("reason") == "low_reciprocity")
        high_sadness_count = sum(1 for f in flagged if f.get("reason") == "high_sadness")
        
        if low_recip_count > 0:
            lines.append(f"⚠️ **Attention Areas:** {low_recip_count} conversation{'s' if low_recip_count > 1 else ''} showed lower-than-usual emotional reciprocity.")
            lines.append("   *Consider reaching out to strengthen these connections or reassess relationship dynamics.*")
            lines.append("")
        
        if high_sadness_count > 0:
            lines.append(f"💙 **Emotional Support Opportunity:** {high_sadness_count} conversation{'s' if high_sadness_count > 1 else ''} contained elevated sadness.")
            lines.append("   *These may benefit from additional empathy, check-ins, or supportive follow-up.*")
            lines.append("")
    
    # Generate overall synthesis
    lines.append("### Overall Communication Profile")
    lines.append("")
    
    # Synthesize personality and communication style
    if personality:
        top_trait = personality.get("top_trait", "")
        if top_trait == "extraversion":
            lines.append("Your **socially energized** communication style, combined with emotional awareness, suggests you thrive in dynamic, interactive conversations and bring enthusiasm to your relationships.")
        elif top_trait == "agreeableness":
            lines.append("Your **harmony-focused** approach, characterized by empathy and cooperation, makes you a natural relationship builder who prioritizes others' feelings and collective wellbeing.")
        elif top_trait == "conscientiousness":
            lines.append("Your **structured and reliable** communication pattern reflects thoughtfulness and dependability, making you a trustworthy and consistent conversation partner.")
        elif top_trait == "openness":
            lines.append("Your **intellectually curious** style brings creativity and depth to conversations, showing willingness to explore new ideas and emotional territories.")
        elif top_trait == "neuroticism":
            lines.append("Your **emotionally rich** communication reveals depth of feeling and sensitivity, which can foster profound connections when paired with emotional regulation strategies.")
    
    lines.append("")
    
    # Conversation count context
    n_conversations = results.get("basic_metrics", {}).get("per_conversation_count", 0)
    if n_conversations > 0:
        lines.append(f"*This analysis is based on {n_conversations} conversation{'s' if n_conversations != 1 else ''}, providing a comprehensive view of your communication patterns and relational dynamics.*")
    
    lines.append("\n---\n")
    lines.append("*Generated using local deterministic analysis • Privacy-preserving • No external AI calls*")

    return "\n".join(lines)


# ---------------------------
# Merge Multiple Profiles Function
# ---------------------------


def merge_local_profiles(
    profiles_list: List[Tuple[Dict, str, str]]
) -> Tuple[Dict, str]:
    """
    Merge multiple per-file local profile results into a single aggregated profile.

    This function combines personality profiles from multiple files by:
    - Averaging Big Five trait scores (weighted by conversation count)
    - Merging MBTI distributions
    - Combining emotion insights
    - Aggregating topic summaries
    - Concatenating per-conversation data
    - Merging export formats (JSON, CSV)

    Args:
        profiles_list: List of tuples (results_dict, profile_text, filename) for each file
                       where results_dict is from run_local_analysis

    Returns:
        Tuple of (merged_results_dict, merged_profile_text)
        - merged_results_dict: Combined analysis with aggregated metrics
        - merged_profile_text: Human-readable merged profile summary
    """
    logger.info(f"merge_local_profiles: merging {len(profiles_list)} profiles")

    if not profiles_list:
        return {
            "error": "No profiles to merge",
            "basic_metrics": {},
            "big_five_aggregation": {},
            "correlations": {},
            "topics_summary": {},
            "mbti_summary": {},
            "emotion_insights": {},
            "highlights_and_rankings": {},
            "advanced_analysis": {},
            "per_conversation_table": [],
        }, "⚠️ No profiles available to merge."

    if len(profiles_list) == 1:
        # Single profile - return as-is with note
        results, profile_text, filename = profiles_list[0]
        merged_text = f"# 🔀 Merged Profile (Single File)\n\n*Source: {filename}*\n\n{profile_text}"
        return results, merged_text

    # Extract data from all profiles
    all_results = []
    all_filenames = []
    total_conversations = 0

    for results, profile_text, filename in profiles_list:
        if "error" in results:
            logger.warning(f"Skipping profile with error from {filename}: {results.get('error')}")
            continue
        all_results.append(results)
        all_filenames.append(filename)
        n = results.get("basic_metrics", {}).get("per_conversation_count", 0)
        total_conversations += n

    if not all_results:
        return {
            "error": "All profiles had errors",
            "basic_metrics": {},
            "big_five_aggregation": {},
            "correlations": {},
            "topics_summary": {},
            "mbti_summary": {},
            "emotion_insights": {},
            "highlights_and_rankings": {},
            "advanced_analysis": {},
            "per_conversation_table": [],
        }, "⚠️ All profiles had errors and could not be merged."

    logger.debug(
        f"Merging {len(all_results)} valid profiles with {total_conversations} total conversations"
    )

    # ========== MERGE BASIC METRICS ==========
    merged_basic_metrics = {}

    # Merge emotional reciprocity (weighted average by conversation count)
    recip_sum = 0.0
    recip_weight = 0
    for r in all_results:
        recip_data = r.get("basic_metrics", {}).get("average_emotional_reciprocity", {})
        if recip_data and "mean" in recip_data:
            n = recip_data.get("n", 0)
            recip_sum += recip_data["mean"] * n
            recip_weight += n

    if recip_weight > 0:
        merged_basic_metrics["average_emotional_reciprocity"] = {
            "mean": recip_sum / recip_weight,
            "n": recip_weight,
            "std": 0.0,  # Recompute if needed; for simplicity set to 0
        }

    # Merge dominant emotion counts
    all_emotion_counts = Counter()
    for r in all_results:
        emotion_counts = r.get("basic_metrics", {}).get("dominant_emotion_counts", {})
        all_emotion_counts.update(emotion_counts)
    merged_basic_metrics["dominant_emotion_counts"] = dict(all_emotion_counts)

    # Merge MBTI distribution
    all_mbti_counts = Counter()
    for r in all_results:
        mbti_dist = r.get("basic_metrics", {}).get("mbti_distribution", {})
        all_mbti_counts.update(mbti_dist)
    merged_basic_metrics["mbti_distribution"] = dict(all_mbti_counts)

    # Merge response time stats (weighted average)
    rt_sum = 0.0
    rt_weight = 0
    for r in all_results:
        rt_stats = r.get("basic_metrics", {}).get("response_time_stats", {})
        if rt_stats and "mean" in rt_stats:
            n = rt_stats.get("n", 0)
            rt_sum += rt_stats["mean"] * n
            rt_weight += n

    if rt_weight > 0:
        merged_basic_metrics["response_time_stats"] = {
            "mean": rt_sum / rt_weight,
            "n": rt_weight,
            "std": 0.0,
        }

    merged_basic_metrics["per_conversation_count"] = total_conversations

    # ========== MERGE BIG FIVE AGGREGATION ==========
    merged_big_five = {}
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    for trait in traits:
        trait_sum = 0.0
        trait_weight = 0
        for r in all_results:
            trait_data = r.get("big_five_aggregation", {}).get(trait, {})
            if trait_data and "mean" in trait_data:
                n = trait_data.get("n", 0)
                trait_sum += trait_data["mean"] * n
                trait_weight += n

        if trait_weight > 0:
            merged_big_five[trait] = {
                "mean": trait_sum / trait_weight,
                "n": trait_weight,
                "std": 0.0,
            }

    # Rank traits by mean
    if merged_big_five:
        sorted_traits = sorted(merged_big_five.items(), key=lambda x: x[1]["mean"], reverse=True)
        merged_big_five["top_trait"] = sorted_traits[0][0] if sorted_traits else None
        merged_big_five["bottom_trait"] = sorted_traits[-1][0] if sorted_traits else None

    # ========== MERGE EMOTION INSIGHTS ==========
    # Merge flagged conversations (union of all flagged)
    all_flagged = []
    for idx, r in enumerate(all_results):
        flagged = r.get("emotion_insights", {}).get("flagged_conversations", [])
        # Add source file info to each flagged conversation
        for flag in flagged:
            flag_copy = flag.copy()
            flag_copy["source_file"] = all_filenames[idx]
            all_flagged.append(flag_copy)

    # Most common emotion across all profiles
    if merged_basic_metrics.get("dominant_emotion_counts"):
        most_common_emotion = max(
            merged_basic_metrics["dominant_emotion_counts"].items(), key=lambda x: x[1]
        )[0]
    else:
        most_common_emotion = "neutral"

    # Average emotion ratios (weighted by conversation count)
    all_emotion_ratio_sums = defaultdict(float)
    for r in all_results:
        avg_ratios = r.get("emotion_insights", {}).get("average_emotion_ratios", {})
        n = r.get("basic_metrics", {}).get("per_conversation_count", 0)
        for emotion, ratio in avg_ratios.items():
            all_emotion_ratio_sums[emotion] += ratio * n

    avg_emotion_ratios = {}
    if total_conversations > 0:
        for emotion, total in all_emotion_ratio_sums.items():
            avg_emotion_ratios[emotion] = total / total_conversations

    merged_emotion_insights = {
        "flagged_conversations": all_flagged,
        "most_common_emotion": most_common_emotion,
        "average_emotion_ratios": avg_emotion_ratios,
    }

    # ========== MERGE TOPICS SUMMARY ==========
    # Combine topics from all files
    all_topics = {}
    for r in all_results:
        topics = r.get("topics_summary", {})
        for topic, data in topics.items():
            if topic not in all_topics:
                all_topics[topic] = data.copy()
            else:
                # Merge topic data (sum counts, recompute means)
                existing = all_topics[topic]
                existing["count"] = existing.get("count", 0) + data.get("count", 0)
                # For means, we'd need to track totals; simplified: just keep first
                # In a more sophisticated version, track per-topic totals and recompute

    # ========== MERGE MBTI SUMMARY ==========
    all_mbti_summary = {}
    for r in all_results:
        mbti_summary = r.get("mbti_summary", {})
        for mbti, data in mbti_summary.items():
            if mbti not in all_mbti_summary:
                all_mbti_summary[mbti] = data.copy()
            else:
                # Merge MBTI data
                existing = all_mbti_summary[mbti]
                existing["count"] = existing.get("count", 0) + data.get("count", 0)

    # ========== MERGE PER-CONVERSATION TABLE ==========
    all_conversations = []
    for idx, r in enumerate(all_results):
        convs = r.get("per_conversation_table", [])
        # Add source file info
        for conv in convs:
            conv_copy = conv.copy() if isinstance(conv, dict) else conv
            if isinstance(conv_copy, dict):
                conv_copy["source_file"] = all_filenames[idx]
            all_conversations.append(conv_copy)

    # ========== COMPILE MERGED RESULTS ==========
    merged_results = {
        "basic_metrics": merged_basic_metrics,
        "big_five_aggregation": merged_big_five,
        "correlations": {},  # Correlations not merged (too complex without raw data)
        "topics_summary": all_topics,
        "mbti_summary": all_mbti_summary,
        "emotion_insights": merged_emotion_insights,
        "highlights_and_rankings": {},  # Re-compute if needed
        "advanced_analysis": {},  # Advanced analysis not merged
        "per_conversation_table": all_conversations,
        "merged_from_files": all_filenames,
        "merged_file_count": len(all_results),
    }

    # ========== PREPARE MERGED EXPORTS ==========
    # Merge JSON exports
    merged_metrics_json = safe_json_dumps(
        {
            "basic_metrics": merged_basic_metrics,
            "personality_aggregation": merged_big_five,
            "topics_summary": all_topics,
            "mbti_summary": all_mbti_summary,
            "emotion_insights": merged_emotion_insights,
            "merged_from_files": all_filenames,
        },
        ensure_ascii=False,
        indent=2,
    )

    # Merge CSV exports (concatenate all per-conversation data)
    if HAS_PANDAS:
        all_df_records = []
        for conv in all_conversations:
            if isinstance(conv, dict):
                all_df_records.append(conv)

        if all_df_records:
            merged_df = pd.DataFrame(all_df_records)
            merged_csv = merged_df.to_csv(index=False)
        else:
            merged_csv = ""
    else:
        # Manual CSV generation
        if all_conversations and isinstance(all_conversations[0], dict):
            keys = set()
            for conv in all_conversations:
                if isinstance(conv, dict):
                    keys.update(conv.keys())
            keys = sorted(keys)

            csv_lines = [",".join(keys)]
            for conv in all_conversations:
                if isinstance(conv, dict):
                    values = []
                    for key in keys:
                        val = conv.get(key, "")
                        if isinstance(val, (list, dict)):
                            val = json.dumps(val)
                        val_str = str(val).replace('"', '""')
                        if "," in val_str or '"' in val_str:
                            val_str = f'"{val_str}"'
                        values.append(val_str)
                    csv_lines.append(",".join(values))
            merged_csv = "\n".join(csv_lines)
        else:
            merged_csv = ""

    # Merge flagged JSON
    merged_flagged_json = safe_json_dumps(all_flagged, ensure_ascii=False, indent=2)

    merged_results["exports"] = {
        "metrics_json": merged_metrics_json,
        "per_conversation_csv": merged_csv,
        "flagged_json": merged_flagged_json,
    }

    # ========== GENERATE MERGED PROFILE TEXT ==========
    merged_profile_text = _generate_merged_profile_text(merged_results, all_filenames)

    logger.info(f"merge_local_profiles completed: {len(all_results)} profiles merged")
    return merged_results, merged_profile_text


def _generate_merged_profile_text(results: Dict, filenames: List[str]) -> str:
    """
    Generate human-readable merged profile summary.

    Args:
        results: Merged results dictionary
        filenames: List of source filenames

    Returns:
        Formatted merged profile text
    """
    lines = ["# 🔀 Merged Psychological Profile\n"]
    lines.append(
        f"*Combined analysis from {len(filenames)} file(s): {', '.join(filenames)}*"
    )
    lines.append("\n---\n")

    lines.append("## 📊 Aggregated Personality Overview")
    lines.append("")

    personality = results.get("big_five_aggregation", {})
    mbti_dist = results.get("basic_metrics", {}).get("mbti_distribution", {})

    # Big Five Traits Analysis
    if personality:
        lines.append("### The Big Five Personality Dimensions (Averaged Across Files)")
        lines.append("")

        trait_scores = []
        for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            trait_data = personality.get(trait, {})
            if trait_data:
                mean_score = trait_data.get("mean", 5.0)
                trait_scores.append((trait, mean_score))

        trait_scores.sort(key=lambda x: x[1], reverse=True)

        for trait, score in trait_scores:
            bar_length = max(0, min(10, int(score)))
            bar = "█" * bar_length + "░" * (10 - bar_length)
            lines.append(f"**{trait.title()}**: `{bar}` {score:.1f}/10")

        lines.append("")

    # MBTI Type Analysis
    if mbti_dist:
        lines.append("### Myers-Briggs Type Indicator (MBTI) Distribution")
        lines.append("")
        top_mbti = max(mbti_dist.items(), key=lambda x: x[1])[0] if mbti_dist else "XXXX"
        count = mbti_dist.get(top_mbti, 0)
        total = sum(mbti_dist.values())
        percentage = (count / total * 100) if total > 0 else 0

        lines.append(f"**Most Frequent Type: {top_mbti}** ({percentage:.0f}%, {count}/{total} conversations)")

        # Show top 3 MBTI types
        sorted_mbti = sorted(mbti_dist.items(), key=lambda x: x[1], reverse=True)[:3]
        if len(sorted_mbti) > 1:
            lines.append("\nTop MBTI Types:")
            for mbti, cnt in sorted_mbti:
                pct = (cnt / total * 100) if total > 0 else 0
                lines.append(f"- {mbti}: {pct:.0f}% ({cnt} conversations)")
        lines.append("")

    lines.append("---\n")

    # ========== EMOTIONAL INTELLIGENCE SECTION ==========
    lines.append("## 💝 Merged Emotional Intelligence Metrics")
    lines.append("")

    # Emotional Reciprocity Analysis
    recip = results.get("basic_metrics", {}).get("average_emotional_reciprocity", {})
    if recip:
        mean_recip = recip.get("mean", 0.5)
        n_recip = recip.get("n", 0)

        lines.append("### Emotional Reciprocity (Aggregated)")

        recip_percentage = int(mean_recip * 100)
        recip_bar = "█" * (recip_percentage // 10) + "░" * (10 - recip_percentage // 10)
        lines.append(f"`{recip_bar}` **{mean_recip:.2f}/1.0** ({recip_percentage}%)")
        lines.append(f"*Based on {n_recip} conversations across all files*")
        lines.append("")

    # Dominant Emotions
    emotion_insights = results.get("emotion_insights", {})
    most_common = emotion_insights.get("most_common_emotion", "neutral")
    avg_ratios = emotion_insights.get("average_emotion_ratios", {})

    lines.append("### Emotional Landscape (Combined)")
    lines.append("")
    lines.append(f"**Primary Emotional Tone: {most_common.title()}**")

    if avg_ratios:
        lines.append("\n**Emotion Distribution (Averaged):**")
        sorted_emotions = sorted(avg_ratios.items(), key=lambda x: x[1], reverse=True)
        for emotion, ratio in sorted_emotions[:5]:
            if ratio > 0.05:
                percentage = ratio * 100
                bar_len = max(0, min(20, int(percentage / 5)))
                bar = "▓" * bar_len + "░" * (20 - bar_len)
                lines.append(f"• {emotion.title()}: `{bar}` {percentage:.1f}%")
        lines.append("")

    # Response Time Analysis
    rt_stats = results.get("basic_metrics", {}).get("response_time_stats", {})
    if rt_stats and rt_stats.get("n", 0) > 0:
        mean_rt = rt_stats.get("mean", 0.0)

        lines.append("### Communication Responsiveness (Aggregated)")
        lines.append("")

        if mean_rt < 60:
            time_str = f"{mean_rt:.0f} minutes"
        elif mean_rt < 1440:
            hours = mean_rt / 60
            time_str = f"{hours:.1f} hours"
        else:
            days = mean_rt / 1440
            time_str = f"{days:.1f} days"

        lines.append(f"**Average Response Time: {time_str}**")
        lines.append(f"*Computed from {rt_stats.get('n', 0)} conversation exchanges*")
        lines.append("")

    lines.append("---\n")

    # ========== FLAGGED CONVERSATIONS ==========
    flagged = emotion_insights.get("flagged_conversations", [])
    if flagged:
        lines.append("## ⚠️ Flagged Conversations (All Files)")
        lines.append("")
        lines.append(f"Found {len(flagged)} flagged conversation(s) across all files:")

        # Group by reason
        by_reason = defaultdict(list)
        for flag in flagged:
            by_reason[flag.get("reason", "unknown")].append(flag)

        for reason, flags in by_reason.items():
            lines.append(f"\n**{reason.replace('_', ' ').title()}**: {len(flags)} conversation(s)")

        lines.append("")

    # ========== SUMMARY ==========
    lines.append("## 📈 Merge Summary")
    lines.append("")
    n_conversations = results.get("basic_metrics", {}).get("per_conversation_count", 0)
    lines.append(f"- **Total Conversations Analyzed**: {n_conversations}")
    lines.append(f"- **Files Merged**: {len(filenames)}")
    lines.append(f"- **Source Files**: {', '.join(filenames)}")
    lines.append("")

    lines.append("---\n")
    lines.append(
        "*Merged profile generated from multiple local analyses • Privacy-preserving • No external AI calls*"
    )

    return "\n".join(lines)
