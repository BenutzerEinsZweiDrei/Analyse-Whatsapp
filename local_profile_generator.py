"""
Local profile generator adapter module.

DEPRECATED: This module is kept for backward compatibility only.
Please use app.core.local_profile instead.

This adapter forwards all calls to the canonical implementation at app.core.local_profile.
All functionality has been consolidated into the canonical module.
"""

# Import everything from the canonical module
from app.core.local_profile import (
    NumpyEncoder,
    advanced_analysis,
    aggregate_personality_data,
    clean_data,
    compute_basic_metrics,
    compute_correlation_manual,
    compute_spearman_manual,
    correlation_analysis,
    emotion_insights,
    export_results,
    filter_and_segment,
    generate_profile_text,
    highlights_and_rankings,
    load_and_validate,
    normalize_structure,
    run_local_analysis,
    safe_float,
    safe_json_dumps,
    visualizations,
)

# Re-export __all__ for discoverability
__all__ = [
    "run_local_analysis",
    "generate_profile_text",
    "load_and_validate",
    "normalize_structure",
    "clean_data",
    "compute_basic_metrics",
    "aggregate_personality_data",
    "correlation_analysis",
    "filter_and_segment",
    "emotion_insights",
    "highlights_and_rankings",
    "visualizations",
    "advanced_analysis",
    "export_results",
    "safe_float",
    "safe_json_dumps",
    "NumpyEncoder",
    "compute_correlation_manual",
    "compute_spearman_manual",
]
