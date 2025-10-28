"""
Local psychological profile generator module.

Provides deterministic analysis pipeline (steps 1-11) to analyze WhatsApp conversations
without external AI calls. Uses in-app data structures from streamlit_app.py.
"""

import json
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Union, Optional, Any

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
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


logger = logging.getLogger("whatsapp_analyzer.local_profile")


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
    required_summary_keys = ["positive_topics", "negative_topics", "emotion_variability", "analysis"]
    for key in required_summary_keys:
        if key not in summary:
            logger.warning(f"Missing key in summary: {key}")
    
    normalized_summary = summary.copy()
    
    logger.debug(f"Validated {len(normalized_matrix)} conversations")
    return normalized_summary, normalized_matrix


# ---------------------------
# Step 2: Normalize Structure
# ---------------------------

def normalize_structure(matrix: Dict) -> Union[List[Dict], 'pd.DataFrame']:
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
        
        record = {
            "conversation_id": str(conv_id)
        }
        
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


def clean_data(records: Union['pd.DataFrame', List[Dict]]) -> Union['pd.DataFrame', List[Dict]]:
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
            for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
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

def compute_basic_metrics(df: Union['pd.DataFrame', List[Dict]]) -> Dict:
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
                "n": int(len(recip_vals))
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
                    "n": int(len(rt_vals))
                }
            else:
                metrics["response_time_stats"] = {"mean": 0.0, "std": 0.0, "n": 0}
        
        # Per-conversation summary
        metrics["per_conversation_count"] = len(df)
        
    else:
        # Manual computation for list of dicts
        recip_values = [r.get("emotional_reciprocity", 0.5) for r in df 
                       if r.get("emotional_reciprocity") is not None]
        if recip_values:
            mean_recip = sum(recip_values) / len(recip_values)
            if len(recip_values) > 1:
                variance = sum((x - mean_recip) ** 2 for x in recip_values) / (len(recip_values) - 1)
                std_recip = variance ** 0.5
            else:
                std_recip = 0.0
            metrics["average_emotional_reciprocity"] = {
                "mean": mean_recip,
                "std": std_recip,
                "n": len(recip_values)
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
                std_rt = variance_rt ** 0.5
            else:
                std_rt = 0.0
            metrics["response_time_stats"] = {
                "mean": mean_rt,
                "std": std_rt,
                "n": len(rt_values)
            }
        else:
            metrics["response_time_stats"] = {"mean": 0.0, "std": 0.0, "n": 0}
        
        metrics["per_conversation_count"] = len(df)
    
    logger.debug(f"Computed basic metrics: {len(metrics)} categories")
    return metrics


# ---------------------------
# Step 5: Aggregate Personality Data
# ---------------------------

def aggregate_personality_data(df: Union['pd.DataFrame', List[Dict]]) -> Dict:
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
                        "n": int(len(vals))
                    }
    else:
        for trait in traits:
            key = f"big_five_{trait}"
            values = [r.get(key, 5.0) for r in df if r.get(key) is not None]
            if values:
                mean_val = sum(values) / len(values)
                if len(values) > 1:
                    variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
                    std_val = variance ** 0.5
                else:
                    std_val = 0.0
                personality[trait] = {
                    "mean": mean_val,
                    "std": std_val,
                    "n": len(values)
                }
    
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


def correlation_analysis(df: Union['pd.DataFrame', List[Dict]]) -> Dict:
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
        pairs.append((f"big_five_{trait}", "response_time_topic_average", f"{trait}_vs_response_time"))
    
    # Add reciprocity vs response time
    pairs.append(("emotional_reciprocity", "response_time_topic_average", "reciprocity_vs_response_time"))
    
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
                        "n": len(x)
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
                        "note": "Sample size too small for reliable p-values" if len(x) < 5 else "scipy not available, p-values not computed"
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
                    "note": "Manual computation without scipy, p-values not available"
                }
    
    logger.debug(f"Computed {len(correlations)} correlation analyses")
    return correlations


# ---------------------------
# Step 7: Filter and Segment
# ---------------------------

def filter_and_segment(df: Union['pd.DataFrame', List[Dict]]) -> Dict:
    """
    Group by topic and MBTI, compute aggregated metrics per group.
    
    Args:
        df: Cleaned records from step 3
        
    Returns:
        Dictionary with segmented data
    """
    logger.debug("Step 7: filter_and_segment")
    
    segments = {
        "by_topic": {},
        "by_mbti": {}
    }
    
    if HAS_PANDAS and isinstance(df, pd.DataFrame):
        # Group by topic
        if "topic" in df.columns:
            for topic, group in df.groupby("topic"):
                topic_normalized = str(topic).lower().split()[0] if topic else "unknown"
                
                segment_data = {
                    "count": len(group),
                    "mean_reciprocity": float(group["emotional_reciprocity"].mean()) if "emotional_reciprocity" in group.columns else 0.5,
                    "mean_response_time": float(group["response_time_topic_average"].mean()) if "response_time_topic_average" in group.columns else 0.0,
                    "dominant_emotions": dict(group["dominant_emotion"].value_counts()) if "dominant_emotion" in group.columns else {}
                }
                
                # Add mean Big Five per topic
                big_five_means = {}
                for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
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
                    "mean_reciprocity": float(group["emotional_reciprocity"].mean()) if "emotional_reciprocity" in group.columns else 0.5,
                    "mean_response_time": float(group["response_time_topic_average"].mean()) if "response_time_topic_average" in group.columns else 0.0,
                    "dominant_emotions": dict(group["dominant_emotion"].value_counts()) if "dominant_emotion" in group.columns else {}
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
                "dominant_emotions": dict(Counter(emotions))
            }
            
            # Mean Big Five per topic
            big_five_means = {}
            for trait in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
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
                "dominant_emotions": dict(Counter(emotions))
            }
    
    logger.debug(f"Segmented data: {len(segments['by_topic'])} topics, {len(segments['by_mbti'])} MBTI types")
    return segments


# ---------------------------
# Step 8: Emotion Insights
# ---------------------------

def emotion_insights(df: Union['pd.DataFrame', List[Dict]]) -> Dict:
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
            insights["most_common_emotion"] = df["dominant_emotion"].mode()[0] if len(df) > 0 else "neutral"
        
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
                flagged.append({
                    "conversation_id": str(row.get("conversation_id", idx)),
                    "reason": "low_reciprocity",
                    "value": float(row["emotional_reciprocity"]),
                    "threshold": float(recip_threshold)
                })
        
        # High sadness (above 90th percentile if sadness ratio exists)
        if "emotion_ratio_sadness" in df.columns:
            sadness_threshold = df["emotion_ratio_sadness"].quantile(0.90)
            high_sadness = df[df["emotion_ratio_sadness"] > sadness_threshold]
            
            for idx, row in high_sadness.iterrows():
                flagged.append({
                    "conversation_id": str(row.get("conversation_id", idx)),
                    "reason": "high_sadness",
                    "value": float(row["emotion_ratio_sadness"]),
                    "threshold": float(sadness_threshold)
                })
        
        insights["flagged_conversations"] = flagged
    else:
        # Manual computation for list of dicts
        emotions = [r.get("dominant_emotion", "neutral") for r in df]
        emotion_counter = Counter(emotions)
        insights["most_common_emotion"] = emotion_counter.most_common(1)[0][0] if emotion_counter else "neutral"
        
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
                    flagged.append({
                        "conversation_id": str(record.get("conversation_id", "unknown")),
                        "reason": "low_reciprocity",
                        "value": recip,
                        "threshold": recip_threshold
                    })
        
        # High sadness
        sadness_values = sorted([r.get("emotion_ratio_sadness", 0.0) for r in df])
        if len(sadness_values) >= 10:
            sadness_threshold = sadness_values[int(len(sadness_values) * 0.9)]  # 90th percentile
            for record in df:
                sadness = record.get("emotion_ratio_sadness", 0.0)
                if sadness > sadness_threshold and sadness > 0:
                    flagged.append({
                        "conversation_id": str(record.get("conversation_id", "unknown")),
                        "reason": "high_sadness",
                        "value": sadness,
                        "threshold": sadness_threshold
                    })
        
        insights["flagged_conversations"] = flagged
    
    logger.debug(f"Generated emotion insights with {len(insights.get('flagged_conversations', []))} flagged conversations")
    return insights


# ---------------------------
# Step 9: Visualizations (Optional)
# ---------------------------

def visualizations(df: Union['pd.DataFrame', List[Dict]], results: Dict) -> Dict:
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

def advanced_analysis(df: Union['pd.DataFrame', List[Dict]], results: Dict) -> Dict:
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
                    "cluster_sizes": dict(Counter(labels))
                }
        except Exception as e:
            logger.debug(f"Clustering with k={k} failed: {e}")
    
    advanced["clustering"] = clustering_results
    logger.debug(f"Advanced analysis completed with {len(clustering_results)} clustering results")
    
    return advanced


# ---------------------------
# Step 11: Export Results
# ---------------------------

def export_results(results: Dict, records: Union['pd.DataFrame', List[Dict]]) -> Dict:
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
        "segments": results.get("segments", {}),
        "emotion_insights": results.get("emotion_insights", {}),
        "advanced_analysis": results.get("advanced_analysis", {})
    }
    exports["metrics_json"] = json.dumps(metrics_data, ensure_ascii=False, indent=2)
    
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
    exports["flagged_json"] = json.dumps(flagged, ensure_ascii=False, indent=2)
    
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
                "segments": {},
                "emotion_insights": {},
                "advanced_analysis": {},
                "per_conversation_table": []
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
        
        # Step 9: Visualizations (optional, skipped)
        viz_data = visualizations(records, {})
        
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
            "advanced_analysis": advanced,
            "per_conversation_table": records.to_dict('records') if HAS_PANDAS and isinstance(records, pd.DataFrame) else records
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
            "segments": {},
            "emotion_insights": {},
            "advanced_analysis": {},
            "per_conversation_table": []
        }, f"⚠️ Analysis failed: {str(e)}"


def generate_profile_text(results: Dict) -> str:
    """
    Generate human-readable profile summary from results.
    
    Args:
        results: Complete results dictionary
        
    Returns:
        Formatted profile text (3-6 bullet points)
    """
    lines = ["## Local Psychological Profile\n"]
    
    # Average reciprocity
    recip = results.get("basic_metrics", {}).get("average_emotional_reciprocity", {})
    if recip:
        mean_recip = recip.get("mean", 0.5)
        lines.append(f"• **Emotional Reciprocity**: {mean_recip:.2f}/1.0 - "
                    f"{'High mutual emotional engagement' if mean_recip > 0.6 else 'Moderate emotional exchange' if mean_recip > 0.4 else 'Limited emotional reciprocity'}")
    
    # Top dominant emotion
    emotion_insights = results.get("emotion_insights", {})
    most_common = emotion_insights.get("most_common_emotion", "neutral")
    lines.append(f"• **Dominant Emotion**: {most_common.title()} - "
                f"Most frequently expressed emotional state")
    
    # Top personality trait
    personality = results.get("big_five_aggregation", {})
    top_trait = personality.get("top_trait")
    if top_trait:
        trait_data = personality.get(top_trait, {})
        mean_score = trait_data.get("mean", 5.0)
        lines.append(f"• **Prominent Trait**: {top_trait.title()} (score: {mean_score:.1f}/10) - "
                    f"Most pronounced Big Five personality characteristic")
    
    # MBTI distribution
    mbti_dist = results.get("basic_metrics", {}).get("mbti_distribution", {})
    if mbti_dist:
        top_mbti = max(mbti_dist.items(), key=lambda x: x[1])[0] if mbti_dist else "XXXX"
        lines.append(f"• **MBTI Pattern**: {top_mbti} - "
                    f"Most common personality type in conversations")
    
    # Flagged conversations
    flagged = emotion_insights.get("flagged_conversations", [])
    if flagged:
        low_recip_count = sum(1 for f in flagged if f.get("reason") == "low_reciprocity")
        if low_recip_count > 0:
            lines.append(f"• **Attention Needed**: {low_recip_count} conversation(s) with low emotional reciprocity detected")
    
    # Response time
    rt_stats = results.get("basic_metrics", {}).get("response_time_stats", {})
    if rt_stats and rt_stats.get("n", 0) > 0:
        mean_rt = rt_stats.get("mean", 0.0)
        lines.append(f"• **Response Pattern**: Average response time of {mean_rt:.1f} minutes - "
                    f"{'Quick engagement' if mean_rt < 30 else 'Moderate pace' if mean_rt < 120 else 'Delayed responses'}")
    
    return "\n".join(lines)
