"""
Analysis package for Profile Fusion v2.0.

This package provides modular components for personality profile merging,
statistical analysis, correlation computation, and insight generation.
"""

__version__ = "2.0"

from .correlations import compute_correlations, get_top_correlations
from .insights import generate_emotional_insights, generate_topic_insights
from .io import load_and_validate_json_files, normalize_data
from .merge import merge_all_data, merge_big_five, merge_emotions, merge_mbti
from .narrative import generate_natural_language_summary
from .profile_matrix import create_profile_matrix, normalize_matrix
from .stats import compute_aggregated_statistics

__all__ = [
    "load_and_validate_json_files",
    "normalize_data",
    "merge_big_five",
    "merge_emotions",
    "merge_mbti",
    "merge_all_data",
    "compute_aggregated_statistics",
    "compute_correlations",
    "get_top_correlations",
    "create_profile_matrix",
    "normalize_matrix",
    "generate_emotional_insights",
    "generate_topic_insights",
    "generate_natural_language_summary",
]
