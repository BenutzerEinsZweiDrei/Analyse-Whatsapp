"""
Correlation analysis module for Profile Fusion.

Computes Pearson and Spearman correlations with statistical significance tests.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    scipy_stats = None

from .io import normalize_data


@st.cache_data
def compute_pearson_correlation(x: List[float], y: List[float]) -> Optional[Tuple[float, float]]:
    """
    Compute Pearson correlation coefficient and p-value.

    Args:
        x: First variable values
        y: Second variable values

    Returns:
        Tuple of (correlation coefficient, p-value) or None if insufficient data
    """
    if not HAS_SCIPY:
        return None

    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None

    try:
        correlation, pvalue = scipy_stats.pearsonr(x, y)
        return (float(correlation), float(pvalue))
    except Exception:
        return None


@st.cache_data
def compute_spearman_correlation(x: List[float], y: List[float]) -> Optional[Tuple[float, float]]:
    """
    Compute Spearman correlation coefficient and p-value.

    Args:
        x: First variable values
        y: Second variable values

    Returns:
        Tuple of (correlation coefficient, p-value) or None if insufficient data
    """
    if not HAS_SCIPY:
        return None

    if len(x) < 2 or len(y) < 2 or len(x) != len(y):
        return None

    try:
        correlation, pvalue = scipy_stats.spearmanr(x, y)
        return (float(correlation), float(pvalue))
    except Exception:
        return None


@st.cache_data
def compute_correlations(
    data_list: List[Dict[str, Any]], p_threshold: float = 0.05
) -> Dict[str, Any]:
    """
    Compute correlations between personality traits and behavioral metrics.

    Analyzes correlations found in each dataset and aggregates across files.

    Args:
        data_list: List of personality result dictionaries
        p_threshold: P-value threshold for statistical significance

    Returns:
        Dictionary with correlation results including:
        - all_correlations: List of all computed correlations
        - significant_correlations: Filtered by p-value threshold
        - trait_behavior_summary: Summary of trait-behavior associations
    """
    if not HAS_SCIPY:
        return {
            "error": "scipy not available - correlations cannot be computed",
            "all_correlations": [],
            "significant_correlations": [],
            "trait_behavior_summary": {},
        }

    # Normalize data first
    normalized_list = [normalize_data(data) for data in data_list]

    all_correlations = []

    # Extract correlations from each file
    for file_idx, data in enumerate(normalized_list):
        if "correlations" not in data:
            continue

        correlations = data["correlations"]

        # Process each correlation entry
        for corr_key, corr_data in correlations.items():
            # Parse correlation key (e.g., "openness_vs_reciprocity")
            parts = corr_key.split("_vs_")
            if len(parts) != 2:
                continue

            var1, var2 = parts

            # Extract coefficients
            pearson_r = corr_data.get("pearson_r")
            pearson_p = corr_data.get("pearson_p")
            spearman_r = corr_data.get("spearman_r")
            spearman_p = corr_data.get("spearman_p")

            if pearson_r is not None or spearman_r is not None:
                all_correlations.append(
                    {
                        "file_index": file_idx,
                        "variable1": var1,
                        "variable2": var2,
                        "pearson_r": pearson_r,
                        "pearson_p": pearson_p,
                        "spearman_r": spearman_r,
                        "spearman_p": spearman_p,
                    }
                )

    # Also compute new correlations from raw data
    # Correlate each Big Five trait with reciprocity and response_time
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]

    for trait in traits:
        # Trait vs reciprocity
        trait_vals = []
        reciprocity_vals = []

        for data in normalized_list:
            personality_agg = data.get("personality_aggregation", {})
            basic_metrics = data.get("basic_metrics", {})

            trait_val = personality_agg.get(trait)
            reciprocity_val = basic_metrics.get("emotional_reciprocity")

            if trait_val is not None and reciprocity_val is not None:
                trait_vals.append(trait_val)
                reciprocity_vals.append(reciprocity_val)

        if len(trait_vals) >= 2:
            pearson_result = compute_pearson_correlation(trait_vals, reciprocity_vals)
            spearman_result = compute_spearman_correlation(trait_vals, reciprocity_vals)

            all_correlations.append(
                {
                    "file_index": "aggregated",
                    "variable1": trait,
                    "variable2": "reciprocity",
                    "pearson_r": pearson_result[0] if pearson_result else None,
                    "pearson_p": pearson_result[1] if pearson_result else None,
                    "spearman_r": spearman_result[0] if spearman_result else None,
                    "spearman_p": spearman_result[1] if spearman_result else None,
                }
            )

        # Trait vs response_time
        trait_vals = []
        response_time_vals = []

        for data in normalized_list:
            personality_agg = data.get("personality_aggregation", {})
            basic_metrics = data.get("basic_metrics", {})

            trait_val = personality_agg.get(trait)
            response_time_val = basic_metrics.get("mean_response_time")

            if trait_val is not None and response_time_val is not None:
                trait_vals.append(trait_val)
                response_time_vals.append(response_time_val)

        if len(trait_vals) >= 2:
            pearson_result = compute_pearson_correlation(trait_vals, response_time_vals)
            spearman_result = compute_spearman_correlation(trait_vals, response_time_vals)

            all_correlations.append(
                {
                    "file_index": "aggregated",
                    "variable1": trait,
                    "variable2": "response_time",
                    "pearson_r": pearson_result[0] if pearson_result else None,
                    "pearson_p": pearson_result[1] if pearson_result else None,
                    "spearman_r": spearman_result[0] if spearman_result else None,
                    "spearman_p": spearman_result[1] if spearman_result else None,
                }
            )

    # Filter for significant correlations
    significant_correlations = []
    for corr in all_correlations:
        is_significant = False

        if corr["pearson_p"] is not None and corr["pearson_p"] < p_threshold:
            is_significant = True
        if corr["spearman_p"] is not None and corr["spearman_p"] < p_threshold:
            is_significant = True

        if is_significant:
            significant_correlations.append(corr)

    # Build trait-behavior summary
    trait_behavior_summary = {}

    for trait in traits:
        trait_behavior_summary[trait] = {
            "reciprocity": {"correlations": [], "avg_pearson_r": None, "consistency": None},
            "response_time": {"correlations": [], "avg_pearson_r": None, "consistency": None},
        }

    for corr in all_correlations:
        var1 = corr["variable1"]
        var2 = corr["variable2"]

        if var1 in traits and var2 in ["reciprocity", "response_time"]:
            trait_behavior_summary[var1][var2]["correlations"].append(corr)

    # Compute averages and consistency
    for trait, behaviors in trait_behavior_summary.items():
        for behavior, data in behaviors.items():
            correlations = data["correlations"]
            if correlations:
                pearson_rs = [c["pearson_r"] for c in correlations if c["pearson_r"] is not None]
                if pearson_rs:
                    data["avg_pearson_r"] = round(sum(pearson_rs) / len(pearson_rs), 4)

                    # Consistency: how many have same sign as average
                    avg_sign = 1 if data["avg_pearson_r"] >= 0 else -1
                    same_sign_count = sum(1 for r in pearson_rs if (r >= 0) == (avg_sign >= 0))
                    data["consistency"] = f"{same_sign_count}/{len(pearson_rs)}"

    return {
        "all_correlations": all_correlations,
        "significant_correlations": significant_correlations,
        "trait_behavior_summary": trait_behavior_summary,
    }


def get_top_correlations(
    correlations: List[Dict[str, Any]], top_n: int = 5
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Get top N positive and negative correlations by effect size.

    Args:
        correlations: List of correlation dictionaries
        top_n: Number of top correlations to return

    Returns:
        Dictionary with 'top_positive' and 'top_negative' correlation lists
    """
    # Sort by absolute Pearson r (use Spearman if Pearson unavailable)
    correlations_with_effect = []
    for corr in correlations:
        effect_size = corr.get("pearson_r") or corr.get("spearman_r")
        if effect_size is not None:
            correlations_with_effect.append({**corr, "effect_size": effect_size})

    # Sort by effect size
    sorted_corrs = sorted(correlations_with_effect, key=lambda x: x["effect_size"], reverse=True)

    return {
        "top_positive": sorted_corrs[:top_n],
        "top_negative": sorted_corrs[-top_n:][::-1],  # Reverse to show most negative first
    }
