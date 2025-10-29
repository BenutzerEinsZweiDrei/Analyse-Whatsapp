"""
Personality Profile Fusion v2.0 - Advanced Streamlit App

NEW in v2.0:
- Comprehensive profile matrix with heatmap visualization
- Statistical correlation analysis (Pearson & Spearman) with p-values
- Rich emotional and topic-level insights
- Natural language summary generation
- Trait-behavior association analysis
- Aggregated statistics with variability measures
- Multiple export formats (JSON, CSV for matrix)
- Enhanced UI with configurable controls

This app merges up to 5 JSON personality profile files into a comprehensive
fusion analysis with backward-compatible v1.0 output plus extensive v2.0 enhancements.

Version: 2.0
"""

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

# Import analysis modules
from analysis import (
    compute_aggregated_statistics,
    compute_correlations,
    create_profile_matrix,
    generate_emotional_insights,
    generate_natural_language_summary,
    generate_topic_insights,
    load_and_validate_json_files,
    merge_all_data,
    normalize_matrix,
)
from analysis.ui_components import (
    create_merged_json_export,
    display_correlation_table,
    display_emotional_insights,
    display_profile_matrix_heatmap,
    display_topic_insights,
    display_trait_behavior_summary,
)


def main():
    """Main application logic with v2.0 enhancements."""
    st.title("🔀 Personality Profile Fusion v2.0")
    st.write(
        "Upload up to 5 personality result JSON files and create a comprehensive "
        "fusion analysis with rich insights, correlations, and visualizations."
    )

    # Sidebar controls
    st.sidebar.header("⚙️ Analysis Settings")

    p_threshold = st.sidebar.slider(
        "P-value threshold",
        min_value=0.01,
        max_value=0.10,
        value=0.05,
        step=0.01,
        help="Statistical significance threshold for correlations",
    )

    normalization_method = st.sidebar.selectbox(
        "Matrix normalization",
        options=["none", "minmax", "zscore"],
        index=0,
        help="Normalization method for profile matrix (None, Min-Max, or Z-score)",
    )

    summary_format = st.sidebar.radio(
        "Summary format", options=["bullet", "paragraph"], index=0, help="Natural language summary format"
    )

    show_per_topic = st.sidebar.checkbox(
        "Show per-topic analysis",
        value=False,
        help="Display topic-level insights (if available)",
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload up to 5 personality result files",
        type="json",
        accept_multiple_files=True,
        help="Select JSON files exported from the WhatsApp personality analyzer",
    )

    if not uploaded_files:
        st.info("👆 Please upload at least 2 JSON files to begin fusion analysis.")
        return

    # Enforce maximum of 5 files
    if len(uploaded_files) > 5:
        st.warning(
            f"⚠️ You uploaded {len(uploaded_files)} files, but only the first 5 will be processed."
        )
        uploaded_files = uploaded_files[:5]

    # Load and validate JSON files
    with st.spinner("Loading and validating JSON files..."):
        data_list, filenames = load_and_validate_json_files(uploaded_files)

    # Validation: require at least 2 valid files
    if len(data_list) < 2:
        st.warning("⚠️ Please upload at least 2 valid files to perform fusion analysis.")
        st.info(
            f"📊 Status: {len(data_list)} valid file(s) loaded out of {len(uploaded_files)} uploaded."
        )
        return

    st.success(f"✅ Successfully loaded {len(data_list)} valid JSON file(s)")

    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Merge data
    status_text.text("Merging personality data...")
    progress_bar.progress(20)
    merged_data = merge_all_data(data_list)
    big_five_df = merged_data["big_five"]
    emotions_df = merged_data["emotions"]
    mbti_df = merged_data["mbti"]

    # Compute aggregated statistics
    status_text.text("Computing aggregated statistics...")
    progress_bar.progress(40)
    aggregated_stats = compute_aggregated_statistics(data_list)

    # Create profile matrix
    status_text.text("Creating personality profile matrix...")
    progress_bar.progress(60)
    profile_matrix = create_profile_matrix(data_list, filenames)
    normalized_matrix = normalize_matrix(profile_matrix, method=normalization_method)

    # Compute correlations
    status_text.text("Analyzing correlations...")
    progress_bar.progress(70)
    correlations = compute_correlations(data_list, p_threshold=p_threshold)

    # Generate insights
    status_text.text("Generating insights...")
    progress_bar.progress(85)
    emotional_insights = generate_emotional_insights(data_list)
    topic_insights = generate_topic_insights(data_list)

    # Complete
    progress_bar.progress(100)
    status_text.text("Analysis complete!")

    # Display results
    st.divider()

    # === Basic Summaries (V1.0 Compatible) ===
    st.header("📊 Basic Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🧠 Big Five Traits")
        if not big_five_df.empty:
            # Show mean and std
            display_df = big_five_df[["trait", "mean", "std"]].copy()
            display_df.columns = ["Trait", "Mean", "Std Dev"]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No Big Five data found.")

    with col2:
        st.subheader("🔤 MBTI Distribution")
        if not mbti_df.empty:
            # Calculate percentages
            total_count = mbti_df["count"].sum()
            mbti_display = mbti_df.copy()
            mbti_display["percentage"] = (mbti_display["count"] / total_count * 100).round(2)
            st.dataframe(
                mbti_display[["mbti_type", "count", "percentage"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No MBTI data found.")

    # Emotions
    st.subheader("😊 Emotion Distribution")
    if not emotions_df.empty:
        st.bar_chart(emotions_df["count"], use_container_width=True)
        st.caption(f"Combined emotion frequencies across {len(data_list)} profile(s).")
    else:
        st.info("No emotion data found.")

    st.divider()

    # === V2.0 Enhanced Features ===
    st.header("🚀 Advanced Analysis (v2.0)")

    # Profile Matrix
    with st.expander("🎨 Personality Profile Matrix", expanded=True):
        display_profile_matrix_heatmap(normalized_matrix)

    # Aggregated Statistics
    with st.expander("📈 Aggregated Statistics", expanded=False):
        st.write("**Behavioral Metrics:**")
        if aggregated_stats["reciprocity"]["mean"] is not None:
            st.metric(
                "Mean Reciprocity",
                f"{aggregated_stats['reciprocity']['mean']:.4f}",
                delta=f"± {aggregated_stats['reciprocity'].get('std', 0):.4f}",
            )
        if aggregated_stats["response_time"]["mean"] is not None:
            st.metric(
                "Mean Response Time",
                f"{aggregated_stats['response_time']['mean']:.2f}s",
                delta=f"± {aggregated_stats['response_time'].get('std', 0):.2f}",
            )

        st.write("**Top 3 Emotions Overall:**")
        for i, emotion_info in enumerate(aggregated_stats.get("top_emotions", []), 1):
            st.write(
                f"{i}. {emotion_info['emotion'].capitalize()}: {emotion_info['count']} occurrences"
            )

    # Correlation Analysis
    with st.expander("🔗 Correlation Analysis", expanded=True):
        display_correlation_table(correlations, p_threshold)
        display_trait_behavior_summary(correlations.get("trait_behavior_summary", {}))

    # Emotional Insights
    with st.expander("😊 Emotional Patterns", expanded=False):
        display_emotional_insights(emotional_insights)

    # Topic Insights
    if show_per_topic:
        with st.expander("📊 Topic-Level Insights", expanded=False):
            display_topic_insights(topic_insights)

    # Natural Language Summary
    st.divider()
    st.header("💬 Natural Language Summary")
    summary = generate_natural_language_summary(
        aggregated_stats=aggregated_stats,
        correlations=correlations,
        emotional_insights=emotional_insights,
        topic_insights=topic_insights,
        num_files=len(data_list),
        format_style=summary_format,
    )
    st.markdown(summary)

    # Download Section
    st.divider()
    st.header("💾 Download Results")

    col1, col2, col3 = st.columns(3)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Comprehensive JSON export
    with col1:
        merged_json = create_merged_json_export(
            big_five_df,
            emotions_df,
            mbti_df,
            normalized_matrix,
            aggregated_stats,
            correlations,
            emotional_insights,
            topic_insights,
        )
        st.download_button(
            label="📥 Full Analysis (JSON)",
            data=merged_json,
            file_name=f"personality_fusion_{timestamp}.json",
            mime="application/json",
            help="Comprehensive analysis with v1.0 backward compatibility + v2.0 enhancements",
        )

    # Profile matrix CSV
    with col2:
        if not normalized_matrix.empty:
            matrix_csv = normalized_matrix.to_csv(index=True)
            st.download_button(
                label="📊 Profile Matrix (CSV)",
                data=matrix_csv,
                file_name=f"profile_matrix_{timestamp}.csv",
                mime="text/csv",
                help="Personality profile matrix in CSV format",
            )

    # Summary text
    with col3:
        st.download_button(
            label="📝 Summary (TXT)",
            data=summary,
            file_name=f"personality_summary_{timestamp}.txt",
            mime="text/plain",
            help="Natural language summary as text file",
        )


if __name__ == "__main__":
    main()
