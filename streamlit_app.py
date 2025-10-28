"""
WhatsApp Conversation Analyzer - Streamlit UI

This is the main Streamlit application entry point.
All core logic has been moved to the app/ package.
"""

import importlib
import importlib.metadata
import json
import os
import platform
import time

import streamlit as st

# Import from app package
from app.config import get_settings, mask_key
from app.core.local_profile import run_local_analysis
from app.core.preprocessing import init_nltk
from app.core.summarizer import summarize_matrix
from app.logging_config import configure_logging, get_logs, set_debug_mode
from app.run_analysis import cached_run_analysis
from app.services.g4f_client import generate_profile, handle_g4f_error

# ---------------------------
# Initialize Application
# ---------------------------

# Configure logging
logger = configure_logging(debug=False)

# Initialize NLTK resources (downloads if needed)
# This is done at startup to avoid delays during first analysis
try:
    init_nltk()
    logger.info("NLTK resources initialized")
except Exception as e:
    logger.warning(f"NLTK initialization warning: {e}")

# Get settings (API keys from environment or secrets)
settings = get_settings()

# ---------------------------
# Streamlit UI
# ---------------------------

st.title("WhatsApp Conversation Analyzer (with Debug Info)")

# Initialize session state for persistence across reruns
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
    st.session_state.matrix = None
    st.session_state.summary = None
    st.session_state.conversation_messages = None
    st.session_state.file_content = None
    st.session_state.username = ""

# Debug toggle in UI
debug_mode = st.checkbox("Enable debug mode (show logs and detailed info)", value=False)
set_debug_mode(debug_mode)

# Show environment info in debug mode
if debug_mode:
    with st.expander("Environment & Dependency Info (debug)"):
        # Core environment
        env_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "cwd": os.getcwd(),
        }
        st.write(env_info)

        # Package versions
        key_packages = [
            "streamlit",
            "nltk",
            "textrazor",
            "empath",
            "emot",
            "g4f",
            "requests",
            "regex",
            "vaderSentiment",
        ]
        pkg_versions = {}
        for pkg in key_packages:
            try:
                pkg_versions[pkg] = importlib.metadata.version(pkg)
            except Exception:
                try:
                    pkg_versions[pkg] = importlib.import_module(pkg).__version__
                except Exception:
                    pkg_versions[pkg] = "not installed / unknown"
        st.write(pkg_versions)

        # Masked API keys
        st.write(
            {
                "jina_key": mask_key(settings.jina_api_key),
                "textrazor_key": mask_key(settings.textrazor_api_key),
            }
        )
        st.write(
            "Note: For production, configure API keys via Streamlit secrets or environment variables."
        )

# File upload and analysis form
st.info(
    "📋 **How to get started:**\n"
    "1. Export your WhatsApp chat (Settings → Export chat → Without media)\n"
    "2. Upload the .txt file below\n"
    "3. Enter the username you want to analyze (as it appears in the chat)\n"
    "4. Click 'Start Analysis' to begin processing"
)

with st.form("analysis_form"):
    uploaded_file = st.file_uploader("Upload your whatsapp.txt file", type=["txt"])
    st.caption("Upload the exported WhatsApp chat file in .txt format")

    username = st.text_input("Enter the username to analyze")
    st.caption("Enter the exact username as it appears in the chat messages")

    submit_analysis = st.form_submit_button("Start Analysis")
    st.caption("⏱️ Analysis typically takes 30-60 seconds depending on chat size")

if submit_analysis:
    try:
        if not uploaded_file:
            st.error("Please upload a whatsapp.txt file.")
            logger.warning("Start Analysis pressed but no file uploaded.")
        elif not username.strip():
            st.error("Please enter a username.")
            logger.warning("Start Analysis pressed but username is empty.")
        else:
            # Read and decode file
            file_bytes = uploaded_file.read()
            file_size = len(file_bytes)
            logger.info(
                f"File uploaded: filename={getattr(uploaded_file, 'name', '<unknown>')}, size={file_size} bytes"
            )

            try:
                file_content = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                try:
                    file_content = file_bytes.decode("latin-1")
                    logger.warning("File decoded with latin-1 fallback.")
                except Exception:
                    st.error(
                        "Could not decode uploaded file. Please provide a UTF-8 encoded text file."
                    )
                    logger.exception("Failed to decode uploaded file.")
                    raise

            # Run analysis
            with st.spinner("Analyzing conversations... This may take a while."):
                total_start = time.time()

                # Use cached analysis function from app.run_analysis
                matrix, conversation_messages = cached_run_analysis(file_content, username.strip())

                # Summarize results
                summary = summarize_matrix(matrix)

                total_time = time.time() - total_start
                logger.info(f"Full analysis completed in {total_time:.2f}s")

            # Store results in session state
            st.session_state.matrix = matrix
            st.session_state.summary = summary
            st.session_state.conversation_messages = conversation_messages
            st.session_state.file_content = file_content
            st.session_state.username = username.strip()
            st.session_state.analysis_done = True
            st.session_state.file_size = file_size
            st.session_state.total_time = total_time

            st.success("Analysis completed!")

    except Exception as main_e:
        logger.exception(f"Unhandled exception during analysis: {main_e}")
        st.error(f"An unexpected error occurred: {main_e}")
        if debug_mode:
            st.exception(main_e)

# Display results if analysis has been done
if st.session_state.analysis_done:
    summary = st.session_state.summary
    matrix = st.session_state.matrix
    file_size = st.session_state.get("file_size", 0)
    total_time = st.session_state.get("total_time", 0)

    # Summary section
    st.subheader("Summary")
    st.write(
        f"**Positive Topics ({len(summary['positive_topics'])}):** {', '.join(summary['positive_topics'])}"
    )
    st.write(
        f"**Negative Topics ({len(summary['negative_topics'])}):** {', '.join(summary['negative_topics'])}"
    )
    st.write(f"**Emotional Variability:** {summary['emotion_variability']:.3f}")

    # Analysis summary
    st.subheader("Analysis Summary")
    st.json(summary["analysis"])

    # Download debug info button
    debug_info = {
        "summary": {
            "positive_topics": summary["positive_topics"],
            "negative_topics": summary["negative_topics"],
            "emotion_variability": summary["emotion_variability"],
        },
        "analysis": summary["analysis"],
        "matrix": summary["matrix"],
        "metadata": {
            "username": st.session_state.username,
            "file_size_bytes": file_size,
            "total_conversations": len(summary["matrix"]),
            "analysis_time_seconds": total_time,
        },
    }
    debug_json = json.dumps(debug_info, ensure_ascii=False, indent=2)
    st.download_button(
        label="Download Complete Analysis (Debug Info)",
        data=debug_json,
        file_name="whatsapp_analysis_debug.json",
        mime="application/json",
        key="download_debug_info",
    )
    st.caption("💾 Download complete analysis data including all metrics and metadata for debugging")

    # Profile generation options
    st.divider()
    st.subheader("🧠 Psychological Profile Generation")
    st.info(
        "**Choose a profile generation method:**\n\n"
        "🔒 **Local Profile** (Recommended): Generates a comprehensive psychological profile "
        "using local algorithms. All processing happens on your device - no data is sent to external servers. "
        "Includes Big Five personality traits, MBTI analysis, emotion insights, and detailed metrics.\n\n"
        "🤖 **AI Profile**: Uses artificial intelligence to generate a natural language profile summary. "
        "Requires internet connection and may send data to external AI services. Results may vary."
    )

    # Local psychological profile generation button
    if st.button("Generate Local Psychological Profile", key="generate_local"):
        with st.spinner("Generating local profile..."):
            try:
                logger.info("Starting local profile generation")

                # Call local analysis pipeline from app.core.local_profile
                results, profile_text = run_local_analysis(
                    st.session_state.summary, st.session_state.matrix
                )

                # Display profile
                st.markdown(profile_text)

                # Show detailed results in expander
                with st.expander("View Detailed Analysis Results (JSON)", expanded=False):
                    st.json(
                        {
                            "basic_metrics": results.get("basic_metrics", {}),
                            "big_five_aggregation": results.get("big_five_aggregation", {}),
                            "emotion_insights": results.get("emotion_insights", {}),
                            "topics_summary": results.get("topics_summary", {}),
                            "mbti_summary": results.get("mbti_summary", {}),
                        }
                    )

                # Download buttons
                st.subheader("Download Results")
                st.caption(
                    "Export your analysis results in different formats for further analysis or record keeping"
                )

                if "exports" in results and "metrics_json" in results["exports"]:
                    st.download_button(
                        label="📥 Download Complete Analysis (JSON)",
                        data=results["exports"]["metrics_json"],
                        file_name="analysis_local_results.json",
                        mime="application/json",
                        key="download_local_json",
                    )
                    st.caption("Complete analysis with all metrics, personality traits, and aggregated data")

                if "exports" in results and "per_conversation_csv" in results["exports"]:
                    st.download_button(
                        label="📥 Download Per-Conversation Data (CSV)",
                        data=results["exports"]["per_conversation_csv"],
                        file_name="per_conversation.csv",
                        mime="text/csv",
                        key="download_local_csv",
                    )
                    st.caption("Individual conversation metrics in spreadsheet format - great for custom analysis")

                if "exports" in results and "flagged_json" in results["exports"]:
                    flagged_data = results["exports"]["flagged_json"]
                    if flagged_data and flagged_data != "[]":
                        st.download_button(
                            label="📥 Download Flagged Conversations (JSON)",
                            data=flagged_data,
                            file_name="flagged_conversations.json",
                            mime="application/json",
                            key="download_flagged_json",
                        )
                        st.caption("Conversations with unusual emotional patterns or outlier metrics")

                logger.info("Local profile generation completed successfully")

            except Exception as e:
                logger.exception(f"Error generating local profile: {e}")
                st.error(
                    f"⚠️ Local Profile Generation Failed\n\n"
                    f"An error occurred while generating the local psychological profile:\n"
                    f"{str(e)}\n\n"
                    f"Please check the analysis data and try again."
                )
                if debug_mode:
                    st.exception(e)

    # AI profile generation button
    if st.button("Generate Psychological Profile with AI", key="generate_ai"):
        with st.spinner("Generating AI profile..."):
            st.subheader("AI-Generated Psychological Profile")

            try:
                logger.debug("Sending prompt to g4f model")

                # Generate profile using app.services.g4f_client
                response = generate_profile(st.session_state.summary["analysis"])

                if response:
                    logger.debug(f"g4f response type: {type(response)}")
                    st.write(response)
                else:
                    logger.error("No response content generated from g4f")
                    st.warning(
                        "⚠️ No AI Profile Generated\n\n"
                        "The AI service responded but did not generate any content. "
                        "Please try again or review the detailed analysis data above."
                    )

            except Exception as e:
                # Use error handler from app.services.g4f_client
                error_message = handle_g4f_error(e)
                st.error(
                    f"⚠️ AI Profile Generation Unavailable\n\n"
                    f"{error_message}\n\n"
                    f"You can still download and review the detailed analysis data above."
                )
                if debug_mode:
                    st.exception(e)

    # Detailed conversation matrix
    st.divider()
    st.subheader("Detailed Conversation Matrix")
    st.caption(
        "Raw analysis data showing metrics for each conversation: sentiment scores, "
        "personality traits (Big Five), MBTI indicators, emotions, topics, and response times"
    )
    st.json(summary["matrix"])

    # Download matrix JSON
    json_data = json.dumps(summary["matrix"], ensure_ascii=False, indent=4)
    st.download_button(
        label="Download conv_matrix.json (Matrix Only)",
        data=json_data,
        file_name="conv_matrix.json",
        mime="application/json",
        key="download_matrix_json",
    )
    st.caption("Download just the conversation matrix data without metadata")

    # Show logs in debug mode
    if debug_mode:
        st.subheader("Debug Logs")
        logs = get_logs()
        st.text_area("Logs", value=logs, height=300, key="debug_logs_area")

    # Always offer log download
    st.divider()
    st.subheader("Debug Information")
    st.caption("Technical logs for troubleshooting issues or understanding the analysis process")
    logs = get_logs()
    if logs.strip():
        st.download_button(
            label="Download Analysis Logs",
            data=logs,
            file_name="analysis_logs.txt",
            mime="text/plain",
            key="download_logs",
        )
        st.caption("Download detailed processing logs including timing information and any warnings")
    else:
        st.info(
            "No logs available. Enable the 'Enable debug mode' checkbox above to capture detailed logs."
        )
