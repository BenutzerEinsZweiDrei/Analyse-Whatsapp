"""
Personality Profile Merger - Streamlit App

This app allows users to upload up to 5 JSON personality result files
(produced by the main WhatsApp analyzer app) and merge them into a single
comprehensive personality profile summary.
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st


def load_json_files(uploaded_files):
    """
    Load and parse JSON files from uploaded file objects.

    Args:
        uploaded_files: List of uploaded file objects from st.file_uploader

    Returns:
        List of parsed JSON dictionaries, skipping any files that fail to parse
    """
    data_list = []
    for file in uploaded_files:
        try:
            # Reset file pointer to beginning
            file.seek(0)
            # Read and parse JSON
            content = json.load(file)
            data_list.append(content)
        except json.JSONDecodeError as e:
            st.error(f"❌ Failed to parse {file.name}: Invalid JSON format - {str(e)}")
        except Exception as e:
            st.error(f"❌ Error reading {file.name}: {str(e)}")
    return data_list


def merge_big_five(data_list):
    """
    Merge Big Five personality traits across multiple profiles.

    Computes the average for each trait across all files that contain that trait.
    If a trait is missing in some files, it's averaged only over files where it exists.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        pandas DataFrame with columns ["trait", "average_score"] sorted by score descending
    """
    traits = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]
    trait_values = {trait: [] for trait in traits}

    # Collect trait values from all files
    for data in data_list:
        personality_agg = data.get("personality_aggregation", {})
        for trait in traits:
            # Check if trait exists and has a numeric value
            if trait in personality_agg:
                value = personality_agg[trait]
                # Handle both direct float values and dict with 'mean' key
                if isinstance(value, dict):
                    value = value.get("mean", value.get("value"))
                if value is not None and isinstance(value, (int, float)):
                    trait_values[trait].append(float(value))

    # Compute averages
    results = []
    for trait, values in trait_values.items():
        if values:  # Only include traits that have at least one value
            avg_score = sum(values) / len(values)
            results.append({"trait": trait, "average_score": round(avg_score, 3)})

    # Sort by average score descending
    results.sort(key=lambda x: x["average_score"], reverse=True)

    return pd.DataFrame(results)


def merge_emotions(data_list):
    """
    Merge emotion counts across multiple profiles.

    Sums up the counts for each emotion type across all files.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        pandas DataFrame with emotion names as index and counts as values,
        sorted by count descending
    """
    emotion_totals = {}

    # Combine emotion counts from all files
    for data in data_list:
        basic_metrics = data.get("basic_metrics", {})
        emotion_counts = basic_metrics.get("dominant_emotion_counts", {})

        for emotion, count in emotion_counts.items():
            if isinstance(count, (int, float)):
                emotion_totals[emotion] = emotion_totals.get(emotion, 0) + count

    # Sort by count descending
    sorted_emotions = dict(sorted(emotion_totals.items(), key=lambda x: x[1], reverse=True))

    # Convert to DataFrame for bar chart
    if sorted_emotions:
        return pd.DataFrame.from_dict(sorted_emotions, orient="index", columns=["count"])
    else:
        return pd.DataFrame()


def merge_mbti(data_list):
    """
    Merge MBTI type counts across multiple profiles.

    Sums up the counts for each MBTI type across all files.

    Args:
        data_list: List of personality result dictionaries

    Returns:
        pandas DataFrame with columns ["mbti_type", "count"] sorted by count descending
    """
    mbti_totals = {}

    # Combine MBTI counts from all files
    for data in data_list:
        mbti_summary = data.get("mbti_summary", {})

        for mbti_type, count in mbti_summary.items():
            if isinstance(count, (int, float)):
                mbti_totals[mbti_type] = mbti_totals.get(mbti_type, 0) + count

    # Convert to list of dicts and sort by count descending
    results = [{"mbti_type": mbti_type, "count": count} for mbti_type, count in mbti_totals.items()]
    results.sort(key=lambda x: x["count"], reverse=True)

    return pd.DataFrame(results)


def create_merged_json(big_five_df, emotions_df, mbti_df):
    """
    Create a JSON export of the merged personality profile.

    Args:
        big_five_df: DataFrame with Big Five traits
        emotions_df: DataFrame with emotion counts
        mbti_df: DataFrame with MBTI counts

    Returns:
        JSON string of the merged profile
    """
    merged_data = {
        "personality_aggregation": {},
        "basic_metrics": {"dominant_emotion_counts": {}},
        "mbti_summary": {},
        "metadata": {
            "merged_at": datetime.now().isoformat(),
            "merger_version": "1.0",
        },
    }

    # Add Big Five data
    if not big_five_df.empty:
        for _, row in big_five_df.iterrows():
            merged_data["personality_aggregation"][row["trait"]] = row["average_score"]

    # Add emotion data
    if not emotions_df.empty:
        merged_data["basic_metrics"]["dominant_emotion_counts"] = emotions_df["count"].to_dict()

    # Add MBTI data
    if not mbti_df.empty:
        for _, row in mbti_df.iterrows():
            merged_data["mbti_summary"][row["mbti_type"]] = int(row["count"])

    return json.dumps(merged_data, indent=2, ensure_ascii=False)


def main():
    """Main application logic."""
    st.title("🔀 Personality Profile Merger")
    st.write(
        "Upload up to 5 personality result JSON files from the main analyzer "
        "and merge them into a single comprehensive profile."
    )

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload up to 5 personality result files",
        type="json",
        accept_multiple_files=True,
        help="Select JSON files exported from the WhatsApp personality analyzer",
    )

    if not uploaded_files:
        st.info("👆 Please upload at least 2 JSON files to begin merging profiles.")
        return

    # Enforce maximum of 5 files
    if len(uploaded_files) > 5:
        st.warning(
            f"⚠️ You uploaded {len(uploaded_files)} files, but only the first 5 will be processed. "
            "The remaining files have been trimmed."
        )
        uploaded_files = uploaded_files[:5]

    # Load JSON files
    with st.spinner("Loading JSON files..."):
        data_list = load_json_files(uploaded_files)

    # Validation: require at least 2 valid files
    if len(data_list) < 2:
        st.warning("⚠️ Please upload at least 2 files to merge profiles.")
        st.info(
            f"📊 Status: {len(data_list)} valid file(s) loaded out of {len(uploaded_files)} uploaded."
        )
        return

    st.success(f"✅ Successfully loaded {len(data_list)} valid JSON file(s)")

    # Merge sections
    st.divider()

    # Big Five Summary
    st.subheader("🧠 Big Five Summary")
    big_five_df = merge_big_five(data_list)

    if not big_five_df.empty:
        st.dataframe(big_five_df, use_container_width=True, hide_index=True)
        st.caption(
            "Average personality trait scores across all uploaded profiles. "
            "Scores range from 0 (low) to 1 (high)."
        )
    else:
        st.info("No Big Five personality data found in the uploaded files.")

    # Emotion Summary
    st.subheader("😊 Emotion Summary")
    emotions_df = merge_emotions(data_list)

    if not emotions_df.empty:
        st.bar_chart(emotions_df, use_container_width=True)
        st.caption(
            f"Combined emotion frequencies across {len(data_list)} profile(s). "
            "Higher bars indicate more frequent emotional expressions."
        )
    else:
        st.info("No emotion data found in the uploaded files.")

    # MBTI Overview
    st.subheader("🔤 MBTI Overview")
    mbti_df = merge_mbti(data_list)

    if not mbti_df.empty:
        st.dataframe(mbti_df, use_container_width=True, hide_index=True)
        st.caption(
            "Distribution of MBTI personality types across all conversations in the merged profiles."
        )
    else:
        st.info("No MBTI data found in the uploaded files.")

    # Final Personality Insight
    st.divider()
    st.subheader("💡 Final Personality Insight")

    # Determine dominant trait and emotion
    dominant_trait = None
    dominant_trait_score = None
    if not big_five_df.empty:
        dominant_trait = big_five_df.iloc[0]["trait"]
        dominant_trait_score = big_five_df.iloc[0]["average_score"]

    dominant_emotion = None
    dominant_emotion_count = None
    if not emotions_df.empty:
        dominant_emotion = emotions_df.index[0]
        dominant_emotion_count = int(emotions_df.iloc[0]["count"])

    # Generate summary text
    summary_parts = []
    summary_parts.append(
        f"**Analysis Summary:** Across **{len(data_list)}** uploaded personality profile(s):"
    )

    if dominant_trait and dominant_trait_score is not None:
        summary_parts.append(
            f"- The most prominent Big Five personality trait is **{dominant_trait.capitalize()}** "
            f"with an average score of **{dominant_trait_score}**."
        )

    if dominant_emotion and dominant_emotion_count is not None:
        summary_parts.append(
            f"- The most frequently expressed emotion is **{dominant_emotion.capitalize()}** "
            f"with a total count of **{dominant_emotion_count}**."
        )

    if not mbti_df.empty:
        top_mbti = mbti_df.iloc[0]["mbti_type"]
        top_mbti_count = int(mbti_df.iloc[0]["count"])
        summary_parts.append(
            f"- The dominant MBTI type is **{top_mbti}** with **{top_mbti_count}** occurrence(s)."
        )

    if len(summary_parts) > 1:
        st.markdown("\n".join(summary_parts))
    else:
        st.info("Not enough data to generate a comprehensive summary.")

    # Download merged result
    st.divider()
    st.subheader("💾 Download Merged Profile")

    merged_json = create_merged_json(big_five_df, emotions_df, mbti_df)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"merged_personality_{timestamp}.json"

    st.download_button(
        label="📥 Download Merged Profile (JSON)",
        data=merged_json,
        file_name=filename,
        mime="application/json",
        help="Download the merged personality profile as a JSON file",
    )
    st.caption(
        "Download the combined analysis data for archival, comparison, or further processing."
    )


if __name__ == "__main__":
    main()
