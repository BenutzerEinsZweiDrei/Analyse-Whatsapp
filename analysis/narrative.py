"""
Narrative module for Profile Fusion.

Generates human-readable natural language summaries of personality profiles.
"""

from typing import Any, Dict, List, Literal

import streamlit as st


@st.cache_data
def generate_natural_language_summary(
    aggregated_stats: Dict[str, Any],
    correlations: Dict[str, Any],
    emotional_insights: Dict[str, Any],
    topic_insights: Dict[str, Any],
    num_files: int,
    format_style: Literal["bullet", "paragraph"] = "bullet",
) -> str:
    """
    Generate natural language summary of personality fusion analysis.

    Args:
        aggregated_stats: Aggregated statistics dictionary
        correlations: Correlation analysis dictionary
        emotional_insights: Emotional insights dictionary
        topic_insights: Topic-level insights dictionary
        num_files: Number of files analyzed
        format_style: Output format ('bullet' or 'paragraph')

    Returns:
        Human-readable text summary
    """
    sections = []

    # Header
    sections.append(f"## Personality Fusion Analysis Summary\n")
    sections.append(f"*Analysis of {num_files} personality profile(s)*\n")

    # Emotional tone summary
    if emotional_insights.get("top_emotions"):
        sections.append("### Emotional Tone")

        top_emotions = emotional_insights["top_emotions"]
        emotion_names = [e["emotion"] for e in top_emotions]
        emotion_text = ", ".join(emotion_names[:-1]) + f", and {emotion_names[-1]}"

        total_count = sum(e["count"] for e in top_emotions)
        percentages = [f"{e['emotion']} ({e['percentage']}%)" for e in top_emotions]

        if format_style == "bullet":
            sections.append(
                f"- The emotional tone is predominantly characterized by **{emotion_text}**."
            )
            sections.append(
                f"- Top emotions: {', '.join(percentages)} accounting for a significant portion of emotional expressions."
            )
        else:
            sections.append(
                f"The emotional tone is predominantly characterized by {emotion_text}. "
                f"These top emotions ({', '.join(percentages)}) account for a significant portion of emotional expressions."
            )

        # Reciprocity context for top emotion
        top_emotion = top_emotions[0]["emotion"]
        emotion_details = emotional_insights["emotion_details"].get(top_emotion, {})

        if emotion_details.get("mean_reciprocity") is not None:
            reciprocity = emotion_details["mean_reciprocity"]
            if format_style == "bullet":
                sections.append(
                    f"- When **{top_emotion}** is dominant, the average reciprocity is **{reciprocity:.3f}**."
                )
            else:
                sections.append(
                    f"When {top_emotion} is dominant, conversations show an average reciprocity of {reciprocity:.3f}."
                )

        sections.append("")

    # Personality summary (Big Five)
    if aggregated_stats.get("trait_rankings"):
        sections.append("### Personality Profile")

        most_common_top = aggregated_stats["trait_rankings"].get("most_common_top")
        most_common_bottom = aggregated_stats["trait_rankings"].get("most_common_bottom")

        if most_common_top:
            if format_style == "bullet":
                sections.append(
                    f"- **{most_common_top.capitalize()}** is the most commonly highest-scoring personality trait."
                )
            else:
                sections.append(
                    f"{most_common_top.capitalize()} consistently emerges as the highest-scoring personality trait across profiles."
                )

        if most_common_bottom:
            if format_style == "bullet":
                sections.append(
                    f"- **{most_common_bottom.capitalize()}** is the most commonly lowest-scoring trait."
                )
            else:
                sections.append(
                    f"Conversely, {most_common_bottom.capitalize()} typically scores lowest."
                )

        sections.append("")

    # Behavioral metrics
    sections.append("### Behavioral Patterns")

    if aggregated_stats.get("reciprocity", {}).get("mean") is not None:
        mean_r = aggregated_stats["reciprocity"]["mean"]
        std_r = aggregated_stats["reciprocity"].get("std")

        if format_style == "bullet":
            sections.append(f"- **Emotional reciprocity** averages **{mean_r:.3f}**", end="")
            if std_r:
                sections.append(f" (± {std_r:.3f})")
            else:
                sections.append(".")
        else:
            text = f"Emotional reciprocity averages {mean_r:.3f}"
            if std_r:
                text += f" with a standard deviation of {std_r:.3f}"
            text += "."
            sections.append(text)

    if aggregated_stats.get("response_time", {}).get("mean") is not None:
        mean_rt = aggregated_stats["response_time"]["mean"]
        std_rt = aggregated_stats["response_time"].get("std")

        if format_style == "bullet":
            sections.append(f"- **Response time** averages **{mean_rt:.2f} seconds**", end="")
            if std_rt:
                sections.append(f" (± {std_rt:.2f})")
            else:
                sections.append(".")
        else:
            text = f"The average response time is {mean_rt:.2f} seconds"
            if std_rt:
                text += f" with variability of {std_rt:.2f} seconds"
            text += "."
            sections.append(text)

    sections.append("")

    # Strongest correlations
    if correlations.get("significant_correlations"):
        sections.append("### Key Associations")

        sig_corrs = correlations["significant_correlations"]

        # Find strongest positive and negative
        strongest_positive = None
        strongest_negative = None

        for corr in sig_corrs:
            r = corr.get("pearson_r") or corr.get("spearman_r")
            if r is not None:
                if strongest_positive is None or r > (
                    strongest_positive.get("pearson_r") or strongest_positive.get("spearman_r")
                ):
                    if r > 0:
                        strongest_positive = corr

                if strongest_negative is None or r < (
                    strongest_negative.get("pearson_r") or strongest_negative.get("spearman_r")
                ):
                    if r < 0:
                        strongest_negative = corr

        if strongest_positive:
            var1 = strongest_positive["variable1"]
            var2 = strongest_positive["variable2"]
            r = strongest_positive.get("pearson_r") or strongest_positive.get("spearman_r")
            p = strongest_positive.get("pearson_p") or strongest_positive.get("spearman_p")

            if format_style == "bullet":
                sections.append(
                    f"- **Strongest positive association**: {var1} and {var2} (r={r:.3f}, p={p:.4f})"
                )
            else:
                sections.append(
                    f"The strongest positive association is between {var1} and {var2} "
                    f"with a correlation of {r:.3f} (p={p:.4f})."
                )

        if strongest_negative:
            var1 = strongest_negative["variable1"]
            var2 = strongest_negative["variable2"]
            r = strongest_negative.get("pearson_r") or strongest_negative.get("spearman_r")
            p = strongest_negative.get("pearson_p") or strongest_negative.get("spearman_p")

            if format_style == "bullet":
                sections.append(
                    f"- **Strongest negative association**: {var1} and {var2} (r={r:.3f}, p={p:.4f})"
                )
            else:
                sections.append(
                    f"The strongest negative association is between {var1} and {var2} "
                    f"with a correlation of {r:.3f} (p={p:.4f})."
                )

        sections.append("")

    # Topic insights
    if topic_insights.get("highest_reciprocity_topics"):
        sections.append("### Topic-Level Insights")

        highest_r_topics = topic_insights["highest_reciprocity_topics"]
        if highest_r_topics:
            top_topic = highest_r_topics[0]
            if format_style == "bullet":
                sections.append(
                    f"- **Highest reciprocity topic**: {top_topic['topic']} ({top_topic['reciprocity']:.3f})"
                )
            else:
                sections.append(
                    f"The topic '{top_topic['topic']}' shows the highest emotional reciprocity at {top_topic['reciprocity']:.3f}."
                )

        most_positive = topic_insights.get("most_positive_topics")
        if most_positive:
            top_positive = most_positive[0]
            if format_style == "bullet":
                sections.append(
                    f"- **Most positive topic**: {top_positive['topic']} ({top_positive['positive_count']} positive expressions)"
                )
            else:
                sections.append(
                    f"The topic '{top_positive['topic']}' exhibits the most positive emotional tone "
                    f"with {top_positive['positive_count']} positive expressions."
                )

        sections.append("")

    # Caveats and recommendations
    sections.append("### Caveats and Recommendations")

    if format_style == "bullet":
        sections.append(
            f"- **Sample size**: This analysis is based on {num_files} file(s). Larger samples increase reliability."
        )
        sections.append(
            "- **Statistical significance**: Correlations with p < 0.05 are highlighted, but multiple comparisons may inflate Type I error."
        )
        sections.append(
            "- **Causality**: Correlations do not imply causation; associations may be influenced by confounding variables."
        )

        if topic_insights.get("outlier_topics"):
            outliers = topic_insights["outlier_topics"]
            outlier_names = [o["topic"] for o in outliers[:2]]
            sections.append(
                f"- **Recommended follow-up**: Investigate outlier topics ({', '.join(outlier_names)}) for unusual patterns."
            )
    else:
        sections.append(
            f"This analysis is based on {num_files} personality profile(s). "
            "Larger sample sizes generally increase the reliability of findings. "
            "Correlations with p-values below 0.05 are highlighted as statistically significant; "
            "however, multiple comparisons may inflate Type I error rates. "
            "Remember that correlations do not imply causation—associations may be influenced by confounding variables or other factors. "
        )

        if topic_insights.get("outlier_topics"):
            outliers = topic_insights["outlier_topics"]
            outlier_names = [o["topic"] for o in outliers[:2]]
            sections.append(
                f"For follow-up analysis, we recommend investigating outlier topics such as {', '.join(outlier_names)} "
                "which show unusual patterns that merit deeper exploration."
            )

    # Join all sections
    if format_style == "bullet":
        return "\n".join(sections)
    else:
        return "\n\n".join(sections)
